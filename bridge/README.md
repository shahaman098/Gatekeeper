# Siri++ Bridge Layer

This bridge implements the exact Electron <-> Python WebSocket contract and a full hybrid pipeline:
- The Gatekeeper: FunctionGemma on Cactus (tool-calling)
- The Context Vault: local retrieval + memory (`context_vault.json`)
- The Genius Escalation: Gemini 2.0 Flash via `escalate_to_cloud`
- Local tool execution on macOS (volume, app launch, file search, reminders, etc.)

## Contract (exact)

Electron -> Python:

```json
{
  "type": "text_command",
  "payload": "Turn my volume down to 20"
}
```

```json
{
  "type": "audio_chunk",
  "payload": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA..."
}
```

Python -> Electron:

```json
{
  "type": "status",
  "state": "state-listening"
}
```

```json
{
  "type": "progress",
  "state": "state-thinking",
  "payload": {
    "step": "Opening Google Chrome...",
    "percentage": 25
  }
}
```

```json
{
  "type": "response",
  "state": "state-responding",
  "payload": {
    "text": "Volume set to 20.",
    "source_url": "https://apple.com/apple-intelligence",
    "actions_taken": ["Opened Chrome", "Read 2 articles"]
  }
}
```

## Backend (FastAPI)

```bash
cd bridge/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Optional env vars:

```bash
export GEMINI_API_KEY="your-key"
export GEMINI_MODEL="gemini-2.0-flash"
export FUNCTIONGEMMA_PATH="cactus/weights/functiongemma-270m-it"
export WHISPER_PATH="cactus/weights/whisper-small"
export LOCAL_CONFIDENCE_THRESHOLD="0.74"
export ASSISTANT_DRY_RUN="1"
export ENABLE_AUDIO_TRANSCRIPTION="1"
export AUDIO_IDLE_SECONDS="1.0"
export MIN_AUDIO_BYTES="1024"
```

Notes:
- `ASSISTANT_DRY_RUN=1` (default): no real OS changes, safe demo mode.
- Set `ASSISTANT_DRY_RUN=0` to execute supported system actions for real.
- `context_vault.json` keeps local memory and is never sent wholesale to cloud; only relevant snippets are included on escalation.
- `audio_chunk` now supports idle-flush transcription: once chunks stop for `AUDIO_IDLE_SECONDS`, backend transcribes and routes as a normal command.
- Non-WAV audio uses `ffmpeg` for conversion if available. Install ffmpeg to support browser-recorded formats.

## Frontend (Electron)

Frontend is owned by a separate teammate. Backend guarantees the contract above.

```bash
cd bridge/electron
npm install
npm start
```

## Routing behavior

- Tier 1 local:
  - FunctionGemma is the only gatekeeper for routing/tool calls.
  - Local tools run on-device.
  - Context retrieval runs from `bridge/backend/context_vault.json`.
- Tier 2 cloud:
  - Triggered through the `escalate_to_cloud` tool call path.
  - If confidence is low or router/model errors occur, backend converts that into `escalate_to_cloud` to preserve one escalation path.
  - Gemini receives only filtered context snippets.

## Backend Design

- Backend-only technical design is documented in `bridge/backend/TECHNICAL_DESIGN.md`.

## Telemetry

Each request is logged to:
- `bridge/backend/events.jsonl`

Logged fields include: command, selected tier, reason, confidence, latency.

## macOS tool coverage

Implemented tools:
- `set_volume`
- `set_brightness` (uses `brightness` CLI if installed)
- `set_do_not_disturb` (via `shortcuts`, configurable)
- `launch_app`
- `search_local_files`
- `lookup_context`
- `create_reminder`
- `escalate_to_cloud`

For DND shortcuts, optionally configure:

```bash
export DND_ON_SHORTCUT="DND On"
export DND_OFF_SHORTCUT="DND Off"
```
