import { KongmingGame } from './Game';
import { createMenuControls } from './menu/menuControls';

const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const boardWrapper = document.querySelector('.board-wrapper');

if (!boardEl || !statusEl || !boardWrapper) {
  throw new Error('Required DOM elements not found');
}

createMenuControls();
const game = new KongmingGame(boardEl, statusEl, boardWrapper as HTMLElement);

const params = new URLSearchParams(window.location.search);
if (params.get('debugSolved') === '1') {
  game.forceWinState();
}

const resetButton = document.getElementById('reset');
const themeButton = document.getElementById('theme');
resetButton?.addEventListener('click', () => {
  game.setup();
});

const infoPanel = document.getElementById('info-panel');
type MenuActivateEvent = CustomEvent<{ menu?: string }>;
document.addEventListener('menu:activate', (event: Event) => {
  const menu = (event as MenuActivateEvent).detail.menu;
  if (menu === 'info') {
    infoPanel?.classList.add('visible');
  } else {
    infoPanel?.classList.remove('visible');
  }
});

const themes: Array<{ name: string; label: string; className: string }> = [
  { name: 'default', label: 'Theme', className: '' },
  { name: 'light', label: 'Light', className: 'theme-light' },
  { name: 'dark', label: 'Dark', className: 'theme-dark' },
];
let themeIndex = 0;

function applyTheme(index: number): void {
  themeIndex = (index + themes.length) % themes.length;
  document.body.classList.remove('theme-light', 'theme-dark');
  const theme = themes[themeIndex];
  if (theme.className) {
    document.body.classList.add(theme.className);
  }
  if (themeButton) {
    themeButton.textContent = 'Theme';
  }
  localStorage.setItem('kongming-theme', theme.name);
}

themeButton?.addEventListener('click', () => {
  applyTheme(themeIndex + 1);
});

const storedTheme = localStorage.getItem('kongming-theme');
const startIndex = themes.findIndex(t => t.name === storedTheme);
applyTheme(startIndex >= 0 ? startIndex : 0);
