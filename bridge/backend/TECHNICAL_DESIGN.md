# Backend Technical Design (Processing Layer Only)

## Scope
This document covers the Python backend only. Frontend implementation is owned by a separate teammate.

## Target Flow
1. Frontend sends WebSocket `text_command` or streaming `audio_chunk`.
2. Backend emits `status: state-listening` while audio is being received.
3. Backend runs local STT when audio stream goes idle.
4. Backend routes text through FunctionGemma (Gatekeeper) on Cactus.
5. Gatekeeper selects:
- Tier 1 local tools (macOS actions/local retrieval), or
- Tier 2 cloud escalation via `escalate_to_cloud` tool.
6. Backend emits `progress` events during each sub-step.
7. Backend emits final structured `response` with `state-responding`.
8. Backend emits `status: state-idle`.

## WebSocket Contract

### Inbound (Electron -> Python)
- `{"type":"text_command","payload":"..."}`
- `{"type":"audio_chunk","payload":"<base64>"}`

### Outbound (Python -> Electron)
- `{"type":"status","state":"state-listening|state-thinking|state-idle"}`
- `{"type":"progress","state":"state-thinking","payload":{"step":"...","percentage":25}}`
- `{"type":"response","state":"state-responding","payload":{"text":"...","source_url":"...","actions_taken":["..."]}}`

## Components
- `websocket_bridge`: session loop, input validation, stream handling.
- `AudioBuffer`: collects chunks and detects idle flush.
- `SpeechBrain`: local transcription (Whisper via Cactus).
- `LocalBrain`: FunctionGemma tool routing only.
- `execute_local_tool`: Tier 1 action executor.
- `run_cloud_reasoning`: Tier 2 Gemini API executor.
- `process_command`: orchestrates status/progress/response lifecycle.

## State Machine
- `state-listening`: audio chunks are being streamed.
- `state-thinking`: backend is transcribing/routing/executing.
- `state-responding`: included in final `response` message.
- `state-idle`: request finished.

## Routing Rules
- Local tool execution is preferred by Gatekeeper.
- Complex requests are escalated only via `escalate_to_cloud` tool call.
- If Gatekeeper model is unavailable or errors, backend emits escalation tool call path to cloud.

## Observability
- Request telemetry is appended to `events.jsonl` with command, route, confidence, reason, latency, actions.

## Runtime Requirements
- FunctionGemma weights (`FUNCTIONGEMMA_PATH`)
- Whisper weights (`WHISPER_PATH`) for audio path
- Gemini key (`GEMINI_API_KEY`) for cloud path
- Optional `ffmpeg` for non-WAV audio conversion
