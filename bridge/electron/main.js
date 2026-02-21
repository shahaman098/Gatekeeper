const path = require('path');

function loadElectronMain() {
  try {
    const electronMain = require('electron/main');
    if (electronMain && electronMain.app) {
      return electronMain;
    }
  } catch (_) {
    // Fallback below.
  }

  const electron = require('electron');
  if (electron && typeof electron !== 'string' && electron.app) {
    return electron;
  }
  throw new Error('Electron main process APIs are unavailable.');
}

const { app, BrowserWindow, screen } = loadElectronMain();

function createWindow() {
  const width = 820;
  const height = 360;
  const { width: displayWidth } = screen.getPrimaryDisplay().workAreaSize;

  const window = new BrowserWindow({
    width,
    height,
    x: Math.max(0, Math.floor((displayWidth - width) / 2)),
    y: 20,
    frame: false,
    transparent: true,
    resizable: true,
    alwaysOnTop: true,
    titleBarStyle: 'hidden',
    backgroundColor: '#00000000',
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  window.loadFile(path.join(__dirname, 'index.html'));
}

app.whenReady().then(() => {
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
