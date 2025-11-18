import { KongmingGame } from './Game';
import { initPlayStats } from './playStats';
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
const game = new KongmingGame(
  boardEl,
  statusEl,
  boardWrapper as HTMLElement,
  shapes[currentShapeIndex],
);
const playStatsController = initPlayStats(game);
const solverOverlay = document.querySelector<SVGSVGElement>('.solver-overlay');
if (solverOverlay) {
  game.setSolverOverlay(solverOverlay);
}

const params = new URLSearchParams(window.location.search);
if (params.get('debugSolved') === '1') {
  game.forceWinState();
}

const resetButton = document.getElementById('reset');
resetButton?.addEventListener('click', () => {
  game.setup();
  resetAutoSolveState();
});

const undoButton = document.getElementById('undo');
undoButton?.addEventListener('click', () => {
  game.undo();
});

const boardActions = document.getElementById('board-actions');
const helperActions = document.getElementById('helper-actions');
const autoActions = document.getElementById('auto-actions');
const autoSolveButton = document.getElementById('auto-solve') as HTMLButtonElement | null;
const infoPanel = document.getElementById('info-panel');
type MenuActivateEvent = CustomEvent<{ menu?: string }>;
document.addEventListener('menu:activate', (event: Event) => {
  const menu = (event as MenuActivateEvent).detail.menu;
  if (menu === 'info') {
    infoPanel?.classList.add('visible');
  } else {
    infoPanel?.classList.remove('visible');
  }
  updateActionVisibility(menu);
});
document.addEventListener('solver:cleared', () => {
  resetAutoSolveState();
});
updateActionVisibility('play');
function updateActionVisibility(menu?: string): void {
  const showBoard = menu === 'play' || !menu;
  boardActions?.classList.toggle('hidden', !showBoard);
  helperActions?.classList.toggle('hidden', menu !== 'helper');
  autoActions?.classList.toggle('hidden', menu !== 'auto');
  if (menu !== 'auto') {
    game.clearSolverVisualization();
    resetAutoSolveState();
  }
}

const themes: Array<{ name: string; label: string; className: string }> = [
  { name: 'default', label: 'Theme', className: '' },
  { name: 'light', label: 'Light', className: 'theme-light' },
  { name: 'dark', label: 'Dark', className: 'theme-dark' },
];
let themeIndex = 0;

function applyTheme(index: number, announce = false): void {
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
  if (announce) {
    game.updateStatusMessage(`Theme: ${theme.label}`);
  }
}

const themeButton = document.getElementById('theme');
themeButton?.addEventListener('click', () => {
  applyTheme(themeIndex + 1, true);
});

const storedTheme = localStorage.getItem('kongming-theme');
const startIndex = themes.findIndex(t => t.name === storedTheme);
applyTheme(startIndex >= 0 ? startIndex : 0, false);

const autoPlayButton = document.getElementById('auto-play') as HTMLButtonElement | null;
const hintButton = document.getElementById('hint');
hintButton?.addEventListener('click', () => {
  game.requestHint();
});

const helperButton = document.getElementById('helper-toggle');
const updateHelperText = () => {
  if (!helperButton) return;
  helperButton.textContent = game.isHelperModeActive() ? 'Helper (on)' : 'Helper (off)';
};
helperButton?.addEventListener('click', () => {
  game.toggleHelperMode();
  updateHelperText();
});
updateHelperText();

function resetAutoSolveState(): void {
  if (!autoSolveButton) return;
  autoSolveButton.textContent = 'Solve';
  autoSolveButton.title = '';
  autoSolveButton.disabled = false;
  if (autoPlayButton) {
    autoPlayButton.disabled = !game.hasSolvePlan();
  }
}
resetAutoSolveState();
if (autoPlayButton) {
  autoPlayButton.setAttribute('title', 'Coming soon');
}

autoSolveButton?.addEventListener('click', () => {
  if (!autoSolveButton) return;
  if (game.isSolverActive()) {
    game.clearSolverVisualization();
    return;
  }
  autoSolveButton.textContent = 'Cancel';
  autoSolveButton.title = 'Click to abort the solve';
  autoSolveButton.textContent = 'Solving…';
  game.startAutoSolveVisualization().then(result => {
    if (result.solved) {
      autoSolveButton.textContent = 'Solved';
      autoSolveButton.title = `Time ${result.durationMs.toFixed(1)}ms · Nodes ${result.nodesExplored}`;
    } else {
      autoSolveButton.textContent = 'Solve';
      autoSolveButton.title = `No solution · ${result.nodesExplored} nodes · ${result.durationMs.toFixed(1)}ms`;
    }
    if (autoPlayButton) {
      autoPlayButton.disabled = !game.hasSolvePlan();
    }
  });
});
document.addEventListener('autoplay:started', () => {
  if (autoPlayButton) {
    autoPlayButton.textContent = 'Playing...';
  }
});
document.addEventListener('autoplay:stopped', () => {
  if (autoPlayButton) {
    autoPlayButton.textContent = 'Play';
  }
});
autoPlayButton?.addEventListener('click', () => {
  if (!autoPlayButton) return;
  if (!game.hasSolvePlan()) return;
  if (game.isAutoplayActive()) {
    game.stopAutoPlay();
    return;
  }
  game.startAutoPlayPlan();
});

const boardMenuButton = document.querySelector<HTMLButtonElement>(
  '.menu-panel.left .menu-item[data-menu="board"]',
);
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
  resetAutoSolveState();
  playStatsController?.refreshShapeFilter(shapes[currentShapeIndex].id);
  game.updateStatusMessage(`Shape: ${nextShape.name}`);
}

boardMenuButton?.addEventListener('click', cycleShape);
shapeButton?.addEventListener('click', cycleShape);
