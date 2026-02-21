const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('bridgeConfig', {
  websocketUrl: 'ws://127.0.0.1:8000/ws',
});
