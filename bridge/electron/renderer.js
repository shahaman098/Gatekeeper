const pillEl = document.getElementById('pill');
const uiStateEl = document.getElementById('ui-state');
const tierChipEl = document.getElementById('tier-chip');
const micChipEl = document.getElementById('mic-chip');
const socketChipEl = document.getElementById('socket-chip');
const statusLineEl = document.getElementById('status-line');
const transcriptLineEl = document.getElementById('transcript-line');
const responseTextEl = document.getElementById('response-text');
const sourceLinkEl = document.getElementById('source-link');
const actionsListEl = document.getElementById('actions-list');
const formEl = document.getElementById('command-form');
const inputEl = document.getElementById('command-input');
const micBtnEl = document.getElementById('mic-btn');
const cloudBtnEl = document.getElementById('cloud-btn');

const WAKE_WORD = 'hey moonwalk';
const COMMAND_SILENCE_MS = 1300;
const IDLE_RESET_MS = 2400;

const UI_STATES = {
  IDLE: 'IDLE',
  LISTENING: 'LISTENING',
  LOADING: 'LOADING',
  DOING: 'DOING',
};

const MIC_MODES = {
  WAKEWORD: 'wakeword',
  STREAM: 'stream',
};

let ws = null;
let reconnectTimer = null;
let currentTier = 'local';
let uiState = UI_STATES.IDLE;
let isMicActive = false;
let pendingCommand = false;

let micMode = MIC_MODES.WAKEWORD;
let recognition = null;
let captureMode = false;
let commandBuffer = '';
let silenceTimer = null;
let idleResetTimer = null;

let mediaStream = null;
let mediaRecorder = null;

function setUiState(nextState) {
  uiState = nextState;
  uiStateEl.textContent = nextState;
  pillEl.dataset.uiState = nextState;
}

function setTier(nextTier) {
  currentTier = nextTier === 'cloud' ? 'cloud' : 'local';
  tierChipEl.textContent = currentTier.toUpperCase();
  tierChipEl.className = `chip tier-chip ${currentTier}`;
}

function setSocketStatus(connected) {
  socketChipEl.textContent = connected ? 'SOCKET ON' : 'SOCKET OFF';
  socketChipEl.className = `chip socket-chip ${connected ? 'connected' : 'disconnected'}`;
}

function setMicStatus(active) {
  isMicActive = active;
  micChipEl.textContent = active ? 'MIC ON' : 'MIC OFF';
  micChipEl.className = `chip mic-chip ${active ? 'on' : 'off'}`;
  micBtnEl.textContent = active ? 'Stop Mic' : 'Start Mic';
}

function setResponse(payload) {
  const text = typeof payload === 'string' ? payload : payload?.text || '';
  responseTextEl.textContent = text || 'No response text.';

  const sourceUrl = payload && typeof payload === 'object' ? payload.source_url : null;
  if (sourceUrl) {
    sourceLinkEl.hidden = false;
    sourceLinkEl.href = sourceUrl;
    sourceLinkEl.textContent = sourceUrl;
  } else {
    sourceLinkEl.hidden = true;
    sourceLinkEl.href = '#';
    sourceLinkEl.textContent = '';
  }

  actionsListEl.innerHTML = '';
  const actions = payload && typeof payload === 'object' && Array.isArray(payload.actions_taken)
    ? payload.actions_taken
    : [];
  for (const action of actions) {
    const item = document.createElement('li');
    item.textContent = action;
    actionsListEl.appendChild(item);
  }
}

function normalizeText(text) {
  return String(text || '').toLowerCase().trim().replace(/\s+/g, ' ');
}

function extractCommandAfterWakeWord(text) {
  const lower = normalizeText(text);
  const idx = lower.indexOf(WAKE_WORD);
  if (idx === -1) {
    return '';
  }
  const rawSlice = text.slice(idx + WAKE_WORD.length);
  return rawSlice.replace(/^[,\s.:;-]+/, '').trim();
}

function clearSilenceTimer() {
  if (silenceTimer) {
    clearTimeout(silenceTimer);
    silenceTimer = null;
  }
}

function scheduleIdleReset() {
  if (idleResetTimer) {
    clearTimeout(idleResetTimer);
  }
  idleResetTimer = setTimeout(() => {
    setUiState(UI_STATES.IDLE);
    setTier('local');
    pendingCommand = false;
    statusLineEl.textContent = micMode === MIC_MODES.WAKEWORD
      ? 'Resume listening for wake word...'
      : 'Local audio stream ready.';
    transcriptLineEl.textContent = micMode === MIC_MODES.WAKEWORD
      ? `Wake phrase: "${WAKE_WORD}"`
      : 'Mic mode: local streaming to backend Whisper.';
  }, IDLE_RESET_MS);
}

function queueCommandFinalization() {
  clearSilenceTimer();
  silenceTimer = setTimeout(() => finalizeCapturedCommand(), COMMAND_SILENCE_MS);
}

function sendJson(payload) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    setResponse('WebSocket not connected.');
    return false;
  }
  ws.send(JSON.stringify(payload));
  return true;
}

function sendTextCommand(command) {
  const ok = sendJson({ type: 'text_command', payload: command });
  if (!ok) {
    setUiState(UI_STATES.IDLE);
    return;
  }
  pendingCommand = true;
  setUiState(UI_STATES.LOADING);
  statusLineEl.textContent = 'Sending command to backend...';
  transcriptLineEl.textContent = `Final transcript: "${command}"`;
}

function finalizeCapturedCommand() {
  clearSilenceTimer();
  if (!captureMode) {
    return;
  }
  captureMode = false;
  const command = commandBuffer.trim();
  commandBuffer = '';

  if (!command) {
    setUiState(UI_STATES.IDLE);
    statusLineEl.textContent = 'Wake word heard but no command captured.';
    transcriptLineEl.textContent = `Wake phrase: "${WAKE_WORD}"`;
    return;
  }

  sendTextCommand(command);
}

function handleWakeWordDetection(transcript) {
  captureMode = true;
  commandBuffer = '';
  setUiState(UI_STATES.LISTENING);
  statusLineEl.textContent = 'Wake word detected. Listening for command...';
  transcriptLineEl.textContent = 'Capturing command after wake phrase...';

  const remainder = extractCommandAfterWakeWord(transcript);
  if (remainder) {
    commandBuffer = remainder;
    transcriptLineEl.textContent = `Capturing: "${commandBuffer}"`;
  }
  queueCommandFinalization();
}

function appendCommandFragment(fragment) {
  if (!captureMode || pendingCommand) {
    return;
  }
  const text = String(fragment || '').trim();
  if (!text) {
    return;
  }

  commandBuffer = commandBuffer ? `${commandBuffer} ${text}` : text;
  transcriptLineEl.textContent = `Capturing: "${commandBuffer}"`;
  queueCommandFinalization();
}

function switchMicMode(nextMode, reasonLine) {
  micMode = nextMode;
  if (reasonLine) {
    statusLineEl.textContent = reasonLine;
  }
  transcriptLineEl.textContent = nextMode === MIC_MODES.WAKEWORD
    ? `Wake phrase: "${WAKE_WORD}"`
    : 'Mic mode: local streaming to backend Whisper.';
}

async function blobToBase64(blob) {
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

async function startStreamMic() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === 'undefined') {
    statusLineEl.textContent = 'Local mic streaming unsupported in this environment.';
    setMicStatus(false);
    return;
  }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeCandidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
    ];
    const mimeType = mimeCandidates.find((candidate) => {
      try {
        return typeof MediaRecorder.isTypeSupported === 'function'
          ? MediaRecorder.isTypeSupported(candidate)
          : false;
      } catch (_) {
        return false;
      }
    });

    mediaRecorder = mimeType
      ? new MediaRecorder(mediaStream, { mimeType })
      : new MediaRecorder(mediaStream);

    mediaRecorder.ondataavailable = async (event) => {
      if (!isMicActive || !event.data || event.data.size === 0) {
        return;
      }
      try {
        const b64 = await blobToBase64(event.data);
        sendJson({ type: 'audio_chunk', payload: b64 });
      } catch (_) {
        // ignore chunk conversion failures
      }
    };

    mediaRecorder.onerror = (event) => {
      const err = event && event.error ? event.error.name || event.error.message : 'unknown';
      statusLineEl.textContent = `Mic stream error: ${err}`;
      stopStreamMic();
    };

    mediaRecorder.onstop = () => {
      if (mediaStream) {
        for (const track of mediaStream.getTracks()) {
          track.stop();
        }
      }
      mediaStream = null;
      mediaRecorder = null;
    };

    mediaRecorder.start(250);
    setMicStatus(true);
    setUiState(UI_STATES.LISTENING);
    statusLineEl.textContent = 'Streaming mic audio to backend...';
    transcriptLineEl.textContent = 'Say your command naturally.';
  } catch (error) {
    setMicStatus(false);
    statusLineEl.textContent = `Mic stream error: ${error && error.message ? error.message : String(error)}`;
  }
}

function stopStreamMic() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  if (mediaStream) {
    for (const track of mediaStream.getTracks()) {
      track.stop();
    }
  }
  mediaStream = null;
  mediaRecorder = null;
}

function initSpeechRecognition() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    switchMicMode(MIC_MODES.STREAM, 'Wake-word API unavailable. Switching to local stream mode.');
    return null;
  }

  const rec = new Recognition();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = 'en-US';

  rec.onstart = () => {
    setMicStatus(true);
    setUiState(UI_STATES.IDLE);
    statusLineEl.textContent = `Continuously listening for "${WAKE_WORD}"...`;
  };

  rec.onend = () => {
    if (micMode !== MIC_MODES.WAKEWORD) {
      return;
    }
    if (isMicActive) {
      setTimeout(() => {
        try {
          rec.start();
        } catch (_) {
          // no-op
        }
      }, 300);
    } else {
      setMicStatus(false);
    }
  };

  rec.onerror = (event) => {
    const err = event && event.error ? event.error : 'unknown';
    if (err === 'network') {
      switchMicMode(MIC_MODES.STREAM, 'Wake-word speech API unavailable. Using local stream mode.');
      try {
        rec.stop();
      } catch (_) {
        // no-op
      }
      recognition = null;
      if (isMicActive) {
        startStreamMic();
      }
      return;
    }
    statusLineEl.textContent = `Mic error: ${err}`;
  };

  rec.onresult = (event) => {
    if (pendingCommand) {
      return;
    }

    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const result = event.results[i];
      const transcript = result[0]?.transcript || '';
      const normalized = normalizeText(transcript);
      const hasWakeWord = normalized.includes(WAKE_WORD);

      if (!captureMode && hasWakeWord) {
        handleWakeWordDetection(transcript);
        continue;
      }

      if (captureMode) {
        const fragment = hasWakeWord ? extractCommandAfterWakeWord(transcript) : transcript;
        appendCommandFragment(fragment);
      }
    }
  };

  return rec;
}

async function startListening() {
  if (micMode === MIC_MODES.STREAM) {
    await startStreamMic();
    return;
  }

  if (!recognition) {
    recognition = initSpeechRecognition();
  }
  if (!recognition) {
    await startStreamMic();
    return;
  }
  if (isMicActive) {
    return;
  }

  isMicActive = true;
  try {
    recognition.start();
  } catch (_) {
    // no-op
  }
}

function stopListening() {
  isMicActive = false;
  captureMode = false;
  pendingCommand = false;
  clearSilenceTimer();

  if (recognition) {
    try {
      recognition.stop();
    } catch (_) {
      // no-op
    }
  }
  stopStreamMic();
  setMicStatus(false);
  setUiState(UI_STATES.IDLE);
  statusLineEl.textContent = 'Microphone stopped.';
}

function onBackendMessage(message) {
  let data;
  try {
    data = JSON.parse(message.data);
  } catch (_) {
    return;
  }

  if (data.type === 'status') {
    const backendState = String(data.state || '');

    if (backendState === 'state-listening') {
      setUiState(UI_STATES.LISTENING);
      statusLineEl.textContent = 'Capturing command audio...';
      return;
    }

    if (backendState === 'state-thinking') {
      setUiState(UI_STATES.DOING);
      statusLineEl.textContent = 'Backend is processing...';
      return;
    }

    if (backendState === 'state-idle' && !pendingCommand) {
      scheduleIdleReset();
    }
    return;
  }

  if (data.type === 'progress') {
    const step = data?.payload?.step || 'Working...';
    setUiState(UI_STATES.DOING);
    statusLineEl.textContent = step;
    const text = normalizeText(step);
    if (text.includes('cloud') || text.includes('gemini')) {
      setTier('cloud');
    } else if (text.includes('local') || text.includes('gatekeeper')) {
      setTier('local');
    }
    return;
  }

  if (data.type === 'response') {
    setUiState(UI_STATES.DOING);
    statusLineEl.textContent = 'Action complete.';
    setResponse(data.payload);
    pendingCommand = false;
    scheduleIdleReset();
  }
}

function setupSocket() {
  ws = new WebSocket(window.bridgeConfig.websocketUrl);
  ws.addEventListener('open', () => {
    setSocketStatus(true);
    responseTextEl.textContent = 'Backend connected.';
  });
  ws.addEventListener('message', onBackendMessage);
  ws.addEventListener('close', () => {
    setSocketStatus(false);
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
    }
    reconnectTimer = setTimeout(setupSocket, 1200);
  });
  ws.addEventListener('error', () => {
    setSocketStatus(false);
  });
}

formEl.addEventListener('submit', (event) => {
  event.preventDefault();
  const command = inputEl.value.trim();
  if (!command) {
    return;
  }
  sendTextCommand(command);
  inputEl.value = '';
});

micBtnEl.addEventListener('click', async () => {
  if (isMicActive) {
    stopListening();
  } else {
    await startListening();
  }
});

cloudBtnEl.addEventListener('click', () => {
  sendTextCommand('escalate_to_cloud: in one sentence, define a hybrid local-first assistant');
});

setUiState(UI_STATES.IDLE);
setTier('local');
setMicStatus(false);
setSocketStatus(false);
switchMicMode(MIC_MODES.WAKEWORD, 'Continuously listening for wake word...');
setupSocket();
startListening();
