const bubble = document.getElementById('bubble');
const stateEl = document.getElementById('state');
const tierEl = document.getElementById('tier');
const responseEl = document.getElementById('response');
const form = document.getElementById('command-form');
const input = document.getElementById('command-input');
const micBtn = document.getElementById('mic-btn');

const ws = new WebSocket(window.bridgeConfig.websocketUrl);

let micInterval = null;
let isMicOn = false;

function setVisualState(state, tier) {
  stateEl.textContent = state;
  tierEl.textContent = tier;
  tierEl.className = tier;

  bubble.classList.remove('listening', 'thinking-local', 'thinking-cloud');
  if (state === 'listening') {
    bubble.classList.add('listening');
  }
  if (state === 'thinking' && tier === 'local') {
    bubble.classList.add('thinking-local');
  }
  if (state === 'thinking' && tier === 'cloud') {
    bubble.classList.add('thinking-cloud');
  }
}

function sendJson(message) {
  if (ws.readyState !== WebSocket.OPEN) {
    responseEl.textContent = 'WebSocket not connected.';
    return;
  }
  ws.send(JSON.stringify(message));
}

function sendTextCommand(payload) {
  sendJson({
    type: 'text_command',
    payload,
  });
}

function startMicSimulation() {
  if (isMicOn) {
    clearInterval(micInterval);
    micInterval = null;
    isMicOn = false;
    micBtn.textContent = 'Mic';
    setVisualState('idle', tierEl.textContent || 'local');
    return;
  }

  isMicOn = true;
  micBtn.textContent = 'Stop';
  const sampleChunk = 'UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=';

  micInterval = setInterval(() => {
    sendJson({
      type: 'audio_chunk',
      payload: sampleChunk,
    });
  }, 250);
}

ws.addEventListener('open', () => {
  responseEl.textContent = 'Connected. Ask something.';
});

ws.addEventListener('message', (event) => {
  let data;
  try {
    data = JSON.parse(event.data);
  } catch {
    return;
  }

  if (data.type === 'status') {
    setVisualState(data.state, data.tier);
    return;
  }

  if (data.type === 'response') {
    responseEl.textContent = data.payload;
  }
});

ws.addEventListener('close', () => {
  responseEl.textContent = 'Socket closed. Restart backend and app.';
});

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const payload = input.value.trim();
  if (!payload) {
    return;
  }
  sendTextCommand(payload);
  input.value = '';
});

micBtn.addEventListener('click', startMicSimulation);
