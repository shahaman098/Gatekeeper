import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency for local-only runs
    genai = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_VAULT_PATH = Path(__file__).resolve().with_name("context_vault.json")
TELEMETRY_LOG_PATH = Path(__file__).resolve().with_name("events.jsonl")

DEFAULT_FUNCTIONGEMMA_PATH = "cactus/weights/functiongemma-270m-it"
FUNCTIONGEMMA_PATH = os.getenv("FUNCTIONGEMMA_PATH", DEFAULT_FUNCTIONGEMMA_PATH)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
CONFIDENCE_THRESHOLD = float(os.getenv("LOCAL_CONFIDENCE_THRESHOLD", "0.74"))
ASSISTANT_DRY_RUN = os.getenv("ASSISTANT_DRY_RUN", "1") not in {"0", "false", "False"}
AUDIO_IDLE_SECONDS = float(os.getenv("AUDIO_IDLE_SECONDS", "1.0"))
MIN_AUDIO_BYTES = int(os.getenv("MIN_AUDIO_BYTES", "1024"))
ENABLE_AUDIO_TRANSCRIPTION = os.getenv("ENABLE_AUDIO_TRANSCRIPTION", "1") not in {"0", "false", "False"}
DEFAULT_WHISPER_PATH = "cactus/weights/whisper-small"
WHISPER_PATH = os.getenv("WHISPER_PATH", DEFAULT_WHISPER_PATH)
WHISPER_PROMPT = os.getenv(
    "WHISPER_PROMPT",
    "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
)
STATE_LISTENING = "state-listening"
STATE_THINKING = "state-thinking"
STATE_RESPONDING = "state-responding"
STATE_IDLE = "state-idle"

if (PROJECT_ROOT / "cactus/python/src").exists():
    sys.path.insert(0, str(PROJECT_ROOT / "cactus/python/src"))

try:
    from cactus import cactus_complete, cactus_destroy, cactus_init, cactus_reset, cactus_transcribe
except Exception:  # pragma: no cover - optional for environments without Cactus installed
    cactus_complete = None
    cactus_destroy = None
    cactus_init = None
    cactus_reset = None
    cactus_transcribe = None

app = FastAPI(title="Siri++ Hybrid Bridge", version="0.2.0")


class IncomingMessage(BaseModel):
    type: Literal["text_command", "audio_chunk"]
    payload: str


@dataclass
class ToolExecution:
    response: str
    command: dict[str, Any] | None = None
    escalate: bool = False
    escalate_reason: str | None = None


@dataclass
class RouterDecision:
    tier: Literal["local", "cloud"]
    response: str = ""
    reason: str = ""
    confidence: float = 0.0


class AudioBuffer:
    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._last_chunk_at: float | None = None

    def add_chunk(self, chunk: bytes) -> None:
        self._chunks.append(chunk)
        self._last_chunk_at = time.time()

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def total_bytes(self) -> int:
        return sum(len(chunk) for chunk in self._chunks)

    def should_flush(self) -> bool:
        if not self._chunks or self._last_chunk_at is None:
            return False
        return (time.time() - self._last_chunk_at) >= AUDIO_IDLE_SECONDS

    def pop_audio(self) -> bytes:
        data = b"".join(self._chunks)
        self._chunks.clear()
        self._last_chunk_at = None
        return data


class SpeechBrain:
    def __init__(self) -> None:
        self.model: Any = None
        self.model_error: str | None = None

    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        if not ENABLE_AUDIO_TRANSCRIPTION:
            self.model_error = "Audio transcription disabled by ENABLE_AUDIO_TRANSCRIPTION=0."
            return False
        if cactus_init is None or cactus_transcribe is None:
            self.model_error = "Cactus transcription runtime is unavailable."
            return False

        model_path = resolve_model_path(WHISPER_PATH)
        if not Path(model_path).exists():
            self.model_error = f"Whisper model path does not exist: {model_path}"
            return False

        try:
            self.model = cactus_init(model_path)
            self.model_error = None
            return True
        except Exception as exc:
            self.model_error = f"Failed to initialize Whisper model: {exc}"
            return False

    def _bytes_to_wav_path(self, audio_data: bytes, tmpdir: str) -> tuple[str | None, str | None]:
        if audio_data.startswith(b"RIFF") and b"WAVE" in audio_data[:64]:
            wav_path = str(Path(tmpdir) / "audio.wav")
            Path(wav_path).write_bytes(audio_data)
            return wav_path, None

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            return None, "Audio is not WAV and ffmpeg is unavailable for conversion."

        src_path = str(Path(tmpdir) / "audio_input")
        wav_path = str(Path(tmpdir) / "audio.wav")
        Path(src_path).write_bytes(audio_data)
        ok, err = run_command(
            [ffmpeg_bin, "-y", "-i", src_path, "-ar", "16000", "-ac", "1", wav_path]
        )
        if not ok:
            return None, f"ffmpeg conversion failed: {err}"
        if not Path(wav_path).exists():
            return None, "ffmpeg conversion did not produce output WAV."
        return wav_path, None

    def transcribe_audio(self, audio_data: bytes) -> tuple[str | None, str | None]:
        if len(audio_data) < MIN_AUDIO_BYTES:
            return None, "Audio chunk too short to transcribe."
        if not self._ensure_model():
            return None, self.model_error or "Whisper model unavailable."

        with tempfile.TemporaryDirectory(prefix="bridge_audio_") as tmpdir:
            wav_path, err = self._bytes_to_wav_path(audio_data, tmpdir)
            if not wav_path:
                return None, err or "Could not prepare WAV audio."

            try:
                raw = cactus_transcribe(self.model, wav_path, prompt=WHISPER_PROMPT)
                parsed = json.loads(raw)
            except Exception as exc:
                return None, f"Transcription failed: {exc}"

        text = str(parsed.get("response", "")).strip()
        if not text:
            return None, "No speech detected."
        return text, None

    def destroy(self) -> None:
        if self.model is not None and cactus_destroy is not None:
            try:
                cactus_destroy(self.model)
            except Exception:
                pass
            self.model = None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def resolve_model_path(path_value: str) -> str:
    model_path = Path(path_value)
    if model_path.is_absolute():
        return str(model_path)
    return str((PROJECT_ROOT / model_path).resolve())


def clamp_percent(value: Any, default: int = 50) -> int:
    try:
        return max(0, min(100, int(value)))
    except Exception:
        return default


def run_command(args: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, check=True)
        return True, completed.stdout.strip()
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip() or str(exc)


def log_event(event: dict[str, Any]) -> None:
    try:
        payload = {"timestamp": datetime.utcnow().isoformat() + "Z", **event}
        with TELEMETRY_LOG_PATH.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload) + "\n")
    except Exception:
        return


def ensure_context_vault() -> dict[str, Any]:
    if CONTEXT_VAULT_PATH.exists():
        try:
            return json.loads(CONTEXT_VAULT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    default_vault = {
        "profile": {
            "name": "Alex",
            "coffee_order": "flat white, oat milk, no sugar",
            "home_city": "San Francisco",
        },
        "people": [
            {"name": "Sister", "notes": "Often recommends books in iMessage."},
            {"name": "Milo", "notes": "Family dog. Vet on Tuesdays."},
        ],
        "recent_documents": [
            "Garage-dimensions.pdf",
            "Car-comparison-notes.md",
            "Hackathon-pitch-outline.docx",
        ],
        "memory": [
            "User prefers concise responses.",
            "User has two kids: Emma and Lucas.",
            "Favorite coding editor is VS Code.",
        ],
    }
    CONTEXT_VAULT_PATH.write_text(json.dumps(default_vault, indent=2), encoding="utf-8")
    return default_vault


def flatten_context_entries(vault: dict[str, Any]) -> list[str]:
    entries: list[str] = []

    profile = vault.get("profile", {})
    if isinstance(profile, dict):
        entries.extend([f"profile.{k}: {v}" for k, v in profile.items()])

    for key in ("people", "recent_documents", "memory"):
        value = vault.get(key, [])
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    entries.append(json.dumps(item, ensure_ascii=True))
                else:
                    entries.append(str(item))

    return entries


def retrieve_context(query: str, top_k: int = 3) -> list[str]:
    vault = ensure_context_vault()
    entries = flatten_context_entries(vault)

    query_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    if not query_tokens:
        return entries[:top_k]

    scored: list[tuple[int, str]] = []
    for entry in entries:
        entry_tokens = set(re.findall(r"[a-zA-Z0-9]+", entry.lower()))
        score = len(query_tokens.intersection(entry_tokens))
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    if scored:
        return [entry for _, entry in scored[:top_k]]
    return entries[:top_k]


def tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "set_volume",
            "description": "Set system output volume on Mac in percent from 0 to 100.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "description": "Volume percent (0-100)."}
                },
                "required": ["level"],
            },
        },
        {
            "name": "set_brightness",
            "description": "Set screen brightness in percent from 0 to 100.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "description": "Brightness percent (0-100)."}
                },
                "required": ["level"],
            },
        },
        {
            "name": "set_do_not_disturb",
            "description": "Enable or disable do not disturb mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "description": "True to enable, false to disable."}
                },
                "required": ["enabled"],
            },
        },
        {
            "name": "launch_app",
            "description": "Launch an application by name on macOS.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Application name, e.g. Safari."}
                },
                "required": ["app_name"],
            },
        },
        {
            "name": "search_local_files",
            "description": "Search local files by query string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search phrase."},
                    "limit": {"type": "integer", "description": "Max number of results (default 5)."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "lookup_context",
            "description": "Retrieve relevant personal context from the local context vault.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Context query."}
                },
                "required": ["query"],
            },
        },
        {
            "name": "create_reminder",
            "description": "Create a local reminder in Apple Reminders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Reminder title."}
                },
                "required": ["title"],
            },
        },
        {
            "name": "escalate_to_cloud",
            "description": "Escalate the request to Gemini cloud for deeper reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why local tier is insufficient."}
                },
                "required": ["reason"],
            },
        },
    ]


def cactus_tool_specs() -> list[dict[str, Any]]:
    return [{"type": "function", "function": t} for t in tool_specs()]


def parse_tool_call(raw_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "function" in raw_call and isinstance(raw_call["function"], dict):
        name = str(raw_call["function"].get("name", ""))
        args = raw_call["function"].get("arguments", {})
    else:
        name = str(raw_call.get("name", ""))
        args = raw_call.get("arguments", raw_call.get("args", {}))

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    if not isinstance(args, dict):
        args = {}

    return name, args


def extract_first_url(text: str) -> str | None:
    match = re.search(r"https?://\\S+", text)
    if not match:
        return None
    return match.group(0).rstrip(").,")


def command_to_action(command: dict[str, Any]) -> str:
    name = str(command.get("name", ""))
    arguments = command.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}

    if name == "set_volume":
        return f"Set volume to {arguments.get('level', 'unknown')}"
    if name == "set_brightness":
        return f"Set brightness to {arguments.get('level', 'unknown')}"
    if name == "set_do_not_disturb":
        enabled = bool(arguments.get("enabled", True))
        return "Enabled Do Not Disturb" if enabled else "Disabled Do Not Disturb"
    if name == "launch_app":
        return f"Opened {arguments.get('app_name', 'application')}"
    if name == "search_local_files":
        return f"Searched local files for '{arguments.get('query', '')}'"
    if name == "lookup_context":
        return "Retrieved local context"
    if name == "create_reminder":
        return f"Created reminder '{arguments.get('title', '')}'"
    if name == "escalate_to_cloud":
        return "Escalated to Gemini cloud reasoning"
    return f"Executed {name}"


def build_response_payload(
    text: str,
    actions_taken: list[str] | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": text,
        "actions_taken": actions_taken or [],
    }
    if source_url:
        payload["source_url"] = source_url
    return payload


def build_escalation_call(reason: str) -> dict[str, Any]:
    return {
        "name": "escalate_to_cloud",
        "arguments": {"reason": reason},
    }


class LocalBrain:
    def __init__(self) -> None:
        self.model: Any = None
        self.model_error: str | None = None

    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        if cactus_init is None:
            self.model_error = "Cactus runtime is unavailable in this environment."
            return False

        model_path = resolve_model_path(FUNCTIONGEMMA_PATH)
        if not Path(model_path).exists():
            self.model_error = f"FunctionGemma model path does not exist: {model_path}"
            return False

        try:
            self.model = cactus_init(model_path)
            self.model_error = None
            return True
        except Exception as exc:
            self.model_error = f"Failed to initialize FunctionGemma: {exc}"
            return False

    def decide(self, command: str, context_snippets: list[str]) -> tuple[RouterDecision, list[dict[str, Any]]]:
        if "escalate_to_cloud" in normalize_text(command):
            return (
                RouterDecision(tier="local", reason="explicit_escalate_keyword", confidence=0.0),
                [build_escalation_call("explicit_escalate_keyword")],
            )

        if not self._ensure_model():
            reason = self.model_error or "router_unavailable"
            return (
                RouterDecision(tier="local", reason="router_unavailable", confidence=0.0),
                [build_escalation_call(reason)],
            )

        system_prompt = (
            "You are the local gatekeeper model for a hybrid assistant. "
            "Always prefer local tools for simple device actions and local retrieval. "
            "Call escalate_to_cloud only when deep reasoning is truly needed."
        )

        context_block = "\n".join(f"- {item}" for item in context_snippets)
        user_prompt = (
            f"User command: {command}\n"
            "Local context snippets:\n"
            f"{context_block}\n"
            "Choose tool calls to fulfill the request."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw = cactus_complete(
                self.model,
                messages,
                tools=cactus_tool_specs(),
                force_tools=True,
                max_tokens=256,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                stop_sequences=["<|im_end|>", "<end_of_turn>"],
            )
            parsed = json.loads(raw)
        except Exception as exc:
            return (
                RouterDecision(tier="local", reason="local_model_error", confidence=0.0),
                [build_escalation_call(f"local_model_error:{exc}")],
            )

        calls = parsed.get("function_calls") or []
        confidence = float(parsed.get("confidence", 0.0) or 0.0)

        if parsed.get("cloud_handoff") is True:
            if not calls:
                calls = [build_escalation_call("cloud_handoff")]
            return RouterDecision(tier="local", reason="tool_calls", confidence=confidence), calls

        if confidence < CONFIDENCE_THRESHOLD and not calls:
            calls = [build_escalation_call("low_confidence")]
            return RouterDecision(tier="local", reason="tool_calls", confidence=confidence), calls

        for call in calls:
            name, _args = parse_tool_call(call)
            if name == "escalate_to_cloud":
                return RouterDecision(tier="local", reason="tool_calls", confidence=confidence), calls

        if not calls:
            # If no call is emitted, default local for simple reply path.
            model_text = (parsed.get("response") or "I handled this locally.").strip()
            return RouterDecision(tier="local", response=model_text, reason="local_text_response", confidence=confidence), []

        return RouterDecision(tier="local", reason="tool_calls", confidence=confidence), calls

    def reset(self) -> None:
        if self.model is not None and cactus_reset is not None:
            try:
                cactus_reset(self.model)
            except Exception:
                return

    def destroy(self) -> None:
        if self.model is not None and cactus_destroy is not None:
            try:
                cactus_destroy(self.model)
            except Exception:
                pass
            self.model = None


LOCAL_BRAIN = LocalBrain()
SPEECH_BRAIN = SpeechBrain()


def execute_local_tool(name: str, arguments: dict[str, Any]) -> ToolExecution:
    if name == "set_volume":
        level = clamp_percent(arguments.get("level"))
        command = {"name": "set_volume", "arguments": {"level": level}}
        if ASSISTANT_DRY_RUN:
            return ToolExecution(response=f"[dry-run] Volume would be set to {level}.", command=command)
        cmd = ["osascript", "-e", f"set volume output volume {level}"]
        ok, err = run_command(cmd)
        if ok:
            return ToolExecution(response=f"Volume set to {level}.", command=command)
        return ToolExecution(response=f"Failed to set volume: {err}", command=command)

    if name == "set_brightness":
        level = clamp_percent(arguments.get("level"))
        command = {"name": "set_brightness", "arguments": {"level": level}}
        brightness_bin = shutil.which("brightness")
        if ASSISTANT_DRY_RUN:
            return ToolExecution(response=f"[dry-run] Brightness would be set to {level}.", command=command)
        if brightness_bin:
            ok, err = run_command([brightness_bin, f"{level / 100:.2f}"])
            if ok:
                return ToolExecution(response=f"Brightness set to {level}.", command=command)
            return ToolExecution(response=f"Failed to set brightness: {err}", command=command)
        return ToolExecution(
            response=(
                "Brightness CLI not found. Install `brightness` (brew) or keep ASSISTANT_DRY_RUN=1."
            ),
            command=command,
        )

    if name == "set_do_not_disturb":
        enabled = bool(arguments.get("enabled", True))
        command = {"name": "set_do_not_disturb", "arguments": {"enabled": enabled}}
        if ASSISTANT_DRY_RUN:
            state = "enabled" if enabled else "disabled"
            return ToolExecution(response=f"[dry-run] Do Not Disturb would be {state}.", command=command)

        on_shortcut = os.getenv("DND_ON_SHORTCUT", "DND On")
        off_shortcut = os.getenv("DND_OFF_SHORTCUT", "DND Off")
        shortcuts_bin = shutil.which("shortcuts")
        if shortcuts_bin:
            target = on_shortcut if enabled else off_shortcut
            ok, err = run_command([shortcuts_bin, "run", target])
            if ok:
                state = "enabled" if enabled else "disabled"
                return ToolExecution(response=f"Do Not Disturb {state}.", command=command)
            return ToolExecution(response=f"Failed running shortcut '{target}': {err}", command=command)

        return ToolExecution(
            response="Do Not Disturb automation is not configured. Set DND shortcuts or use dry-run mode.",
            command=command,
        )

    if name == "launch_app":
        app_name = str(arguments.get("app_name", "")).strip() or "Finder"
        command = {"name": "launch_app", "arguments": {"app_name": app_name}}
        if ASSISTANT_DRY_RUN:
            return ToolExecution(response=f"[dry-run] Would open {app_name}.", command=command)
        ok, err = run_command(["open", "-a", app_name])
        if ok:
            return ToolExecution(response=f"Opening {app_name}.", command=command)
        return ToolExecution(response=f"Failed to open {app_name}: {err}", command=command)

    if name == "search_local_files":
        query = str(arguments.get("query", "")).strip()
        limit = clamp_percent(arguments.get("limit", 5), default=5)
        limit = max(1, min(10, limit))
        command = {"name": "search_local_files", "arguments": {"query": query, "limit": limit}}

        if not query:
            return ToolExecution(response="Please provide a file search query.", command=command)

        mdfind_bin = shutil.which("mdfind")
        if mdfind_bin:
            ok, out = run_command([mdfind_bin, query])
            if not ok:
                return ToolExecution(response=f"File search failed: {out}", command=command)
            lines = [line for line in out.splitlines() if line.strip()][:limit]
            if not lines:
                return ToolExecution(response=f"No local files found for '{query}'.", command=command)
            return ToolExecution(response="\n".join(["Top local file matches:", *lines]), command=command)

        rg_bin = shutil.which("rg")
        if rg_bin:
            ok, out = run_command([rg_bin, "--files", str(PROJECT_ROOT)])
            if not ok:
                return ToolExecution(response=f"Fallback file listing failed: {out}", command=command)
            matches = [line for line in out.splitlines() if query.lower() in line.lower()][:limit]
            if not matches:
                return ToolExecution(response=f"No local files found for '{query}'.", command=command)
            return ToolExecution(response="\n".join(["Top local file matches:", *matches]), command=command)

        return ToolExecution(response="No local file search binary available (mdfind or rg).", command=command)

    if name == "lookup_context":
        query = str(arguments.get("query", "")).strip()
        command = {"name": "lookup_context", "arguments": {"query": query}}
        if not query:
            return ToolExecution(response="Please provide a context query.", command=command)
        matches = retrieve_context(query, top_k=3)
        return ToolExecution(response="\n".join(["Context matches:", *matches]), command=command)

    if name == "create_reminder":
        title = str(arguments.get("title", "")).strip()
        command = {"name": "create_reminder", "arguments": {"title": title}}
        if not title:
            return ToolExecution(response="Please provide a reminder title.", command=command)

        if ASSISTANT_DRY_RUN:
            return ToolExecution(response=f"[dry-run] Reminder would be created: {title}", command=command)

        applescript = (
            'tell application "Reminders"\n'
            'tell list "Reminders"\n'
            f'make new reminder with properties {{name:"{title}"}}\n'
            "end tell\n"
            "end tell"
        )
        ok, err = run_command(["osascript", "-e", applescript])
        if ok:
            return ToolExecution(response=f"Reminder created: {title}", command=command)
        return ToolExecution(response=f"Failed to create reminder: {err}", command=command)

    if name == "escalate_to_cloud":
        reason = str(arguments.get("reason", "local model requested cloud escalation")).strip()
        command = {"name": "escalate_to_cloud", "arguments": {"reason": reason}}
        return ToolExecution(
            response="Escalating to cloud.",
            command=command,
            escalate=True,
            escalate_reason=reason,
        )

    return ToolExecution(response=f"Unknown tool call '{name}'.", escalate=True, escalate_reason="unknown_tool")


def run_local_pipeline(command: str) -> tuple[RouterDecision, str, list[str], list[dict[str, Any]]]:
    context_snippets = retrieve_context(command, top_k=3)
    decision, calls = LOCAL_BRAIN.decide(command, context_snippets)

    if decision.tier == "cloud":
        return decision, "", context_snippets, []

    if decision.response and not calls:
        return decision, decision.response, context_snippets, []

    tool_responses: list[str] = []
    executed_commands: list[dict[str, Any]] = []
    escalate_reason = ""

    for raw_call in calls:
        name, args = parse_tool_call(raw_call)
        execution = execute_local_tool(name, args)
        tool_responses.append(execution.response)
        if execution.command is not None:
            executed_commands.append(execution.command)
        if execution.escalate:
            escalate_reason = execution.escalate_reason or "tool_execution_escalation"
            return (
                RouterDecision(
                    tier="cloud",
                    reason=escalate_reason,
                    confidence=decision.confidence,
                ),
                "",
                context_snippets,
                executed_commands,
            )

    if not tool_responses:
        return (
            RouterDecision(
                tier="local",
                response="Handled locally.",
                reason="no_tool_response",
                confidence=decision.confidence,
            ),
            "Handled locally.",
            context_snippets,
            executed_commands,
        )

    summary = "\n".join(tool_responses)
    return (
        RouterDecision(
            tier="local",
            response=summary,
            reason=decision.reason,
            confidence=decision.confidence,
        ),
        summary,
        context_snippets,
        executed_commands,
    )


def run_cloud_reasoning(command: str, context_snippets: list[str], reason: str) -> str:
    if genai is None or not os.getenv("GEMINI_API_KEY"):
        if reason == "unknown_tool":
            return "Cloud fallback requested because local tool was unavailable, but GEMINI_API_KEY is missing."
        return (
            "Cloud routing requested, but GEMINI_API_KEY is not configured. "
            "Set GEMINI_API_KEY to enable Gemini 2.0 Flash responses."
        )

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = (
        "You are the cloud reasoning tier for a hybrid desktop assistant. "
        "Respond concisely with actionable output.\n\n"
        f"Escalation reason: {reason}\n"
        f"User command: {command}\n"
        "Relevant local context snippets (already filtered on-device):\n"
        + "\n".join(f"- {item}" for item in context_snippets)
    )

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    return "Cloud tier returned no text response."


async def send_status(websocket: WebSocket, state: str) -> None:
    await websocket.send_text(json.dumps({"type": "status", "state": state}))


async def send_response(websocket: WebSocket, payload: Any) -> None:
    normalized_payload: dict[str, Any]
    if isinstance(payload, dict):
        normalized_payload = dict(payload)
        normalized_payload.setdefault("text", "")
        normalized_payload.setdefault("actions_taken", [])
    else:
        normalized_payload = build_response_payload(str(payload))

    await websocket.send_text(
        json.dumps(
            {
                "type": "response",
                "state": STATE_RESPONDING,
                "payload": normalized_payload,
            }
        )
    )


async def send_progress(
    websocket: WebSocket,
    step: str,
    percentage: int | None = None,
) -> None:
    payload: dict[str, Any] = {"step": step}
    if percentage is not None:
        payload["percentage"] = max(0, min(100, int(percentage)))

    await websocket.send_text(
        json.dumps(
            {
                "type": "progress",
                "state": STATE_THINKING,
                "payload": payload,
            }
        )
    )


async def process_command(websocket: WebSocket, command: str, source: str) -> None:
    started = time.time()
    await send_status(websocket, state=STATE_THINKING)
    await send_progress(
        websocket,
        step="Routing request through local gatekeeper...",
        percentage=10,
    )

    local_decision, local_response, context_snippets, executed_commands = await asyncio.to_thread(
        run_local_pipeline, command
    )
    actions_taken = [command_to_action(item) for item in executed_commands]
    local_text = local_response or local_decision.response or "Done."

    if local_decision.tier == "local":
        if executed_commands:
            await send_progress(
                websocket,
                step="Executing local actions...",
                percentage=70,
            )
        else:
            await send_progress(
                websocket,
                step="Preparing local response...",
                percentage=70,
            )

        response_payload = build_response_payload(
            text=local_text,
            actions_taken=actions_taken,
        )
        await send_progress(
            websocket,
            step="Formatting final response...",
            percentage=95,
        )
        await send_response(websocket, response_payload)
        await send_status(websocket, state=STATE_IDLE)
        log_event(
            {
                "source": source,
                "command": command,
                "tier": "local",
                "reason": local_decision.reason,
                "confidence": local_decision.confidence,
                "actions_taken": actions_taken,
                "latency_ms": round((time.time() - started) * 1000, 2),
            }
        )
        LOCAL_BRAIN.reset()
        return

    await send_progress(
        websocket,
        step="Escalating to cloud reasoning...",
        percentage=60,
    )
    await send_progress(
        websocket,
        step="Analyzing with Gemini...",
        percentage=85,
    )
    cloud_text = await asyncio.to_thread(
        run_cloud_reasoning,
        command,
        context_snippets,
        local_decision.reason,
    )
    cloud_url = extract_first_url(cloud_text)
    cloud_actions = [*actions_taken, "Generated cloud reasoning response"]
    cloud_payload = build_response_payload(
        text=cloud_text,
        source_url=cloud_url,
        actions_taken=cloud_actions,
    )
    await send_progress(
        websocket,
        step="Formatting final response...",
        percentage=95,
    )
    await send_response(websocket, cloud_payload)
    await send_status(websocket, state=STATE_IDLE)
    log_event(
        {
            "source": source,
            "command": command,
            "tier": "cloud",
            "reason": local_decision.reason,
            "confidence": local_decision.confidence,
            "actions_taken": cloud_actions,
            "latency_ms": round((time.time() - started) * 1000, 2),
        }
    )
    LOCAL_BRAIN.reset()


@app.websocket("/ws")
async def websocket_bridge(websocket: WebSocket) -> None:
    await websocket.accept()
    audio_buffer = AudioBuffer()

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.25)
            except asyncio.TimeoutError:
                if not audio_buffer.should_flush():
                    continue

                audio_data = audio_buffer.pop_audio()
                await send_progress(
                    websocket,
                    step="Transcribing audio input...",
                    percentage=25,
                )
                transcript, err = await asyncio.to_thread(SPEECH_BRAIN.transcribe_audio, audio_data)
                if not transcript:
                    await send_response(websocket, err or "Unable to transcribe audio.")
                    await send_status(websocket, state=STATE_IDLE)
                    log_event(
                        {
                            "source": "audio",
                            "tier": "local",
                            "reason": "transcription_failed",
                            "error": err or "unknown",
                        }
                    )
                    continue

                await process_command(websocket, transcript, source="audio")
                continue

            try:
                message = IncomingMessage.model_validate_json(raw)
            except ValidationError:
                await send_response(websocket, "Invalid message format. Expected JSON with type and payload.")
                continue

            if message.type == "audio_chunk":
                try:
                    chunk = base64.b64decode(message.payload.encode("utf-8"), validate=True)
                except Exception:
                    await send_response(websocket, "Invalid audio_chunk payload. Expected base64 string.")
                    continue

                audio_buffer.add_chunk(chunk)
                await send_status(websocket, state=STATE_LISTENING)
                continue

            command = message.payload.strip()
            if not command:
                await send_response(websocket, "Empty text command.")
                continue

            if audio_buffer.chunk_count:
                audio_buffer.pop_audio()
            await process_command(websocket, command, source="text")

    except WebSocketDisconnect:
        return


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "dry_run": ASSISTANT_DRY_RUN,
        "functiongemma_path": resolve_model_path(FUNCTIONGEMMA_PATH),
        "whisper_path": resolve_model_path(WHISPER_PATH),
        "audio_transcription_enabled": ENABLE_AUDIO_TRANSCRIPTION,
        "audio_idle_seconds": AUDIO_IDLE_SECONDS,
        "min_audio_bytes": MIN_AUDIO_BYTES,
        "local_confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.on_event("shutdown")
def shutdown() -> None:
    LOCAL_BRAIN.destroy()
    SPEECH_BRAIN.destroy()
