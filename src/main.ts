import { KongmingGame } from './Game';
import { createMenuControls } from './menu/menuControls';
import { shapes } from './shapes';

const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const boardWrapper = document.querySelector('.board-wrapper');

if (!boardEl || !statusEl || !boardWrapper) {
  throw new Error('Required DOM elements not found');
}

createMenuControls();

const storedShapeId = localStorage.getItem('kongming-shape');
const startShapeIndex = shapes.findIndex(shape => shape.id === storedShapeId);
let currentShapeIndex = startShapeIndex >= 0 ? startShapeIndex : 0;
localStorage.setItem('kongming-shape', shapes[currentShapeIndex].id);
const game = new KongmingGame(boardEl, statusEl, boardWrapper as HTMLElement, shapes[currentShapeIndex]);

const params = new URLSearchParams(window.location.search);
if (params.get('debugSolved') === '1') {
  game.forceWinState();
}

const resetButton = document.getElementById('reset');
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
  const themeButton = document.getElementById('theme');
  if (themeButton) {
    themeButton.textContent = 'Theme';
  }
  localStorage.setItem('kongming-theme', theme.name);
}

const themeButton = document.getElementById('theme');
themeButton?.addEventListener('click', () => {
  applyTheme(themeIndex + 1);
});

const storedTheme = localStorage.getItem('kongming-theme');
const startIndex = themes.findIndex(t => t.name === storedTheme);
applyTheme(startIndex >= 0 ? startIndex : 0);

const boardMenuButton = document.querySelector<HTMLButtonElement>('.menu-panel.left .menu-item[data-menu="board"]');
const shapeButton = document.getElementById('shape');

function cycleShape(): void {
  const nextIndex = (currentShapeIndex + 1) % shapes.length;
  const nextShape = shapes[nextIndex];
  if (game.hasProgress()) {
    const proceed = window.confirm('Changing shapes will reset the board. Continue?');
    if (!proceed) return;
  }
  currentShapeIndex = nextIndex;
  game.changeShape(nextShape);
  localStorage.setItem('kongming-shape', shapes[currentShapeIndex].id);
}

boardMenuButton?.addEventListener('click', cycleShape);
shapeButton?.addEventListener('click', cycleShape);
