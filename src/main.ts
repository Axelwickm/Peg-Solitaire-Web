import { KongmingGame } from './Game';
import { initPlayStats } from './playStats';
import { createMenuControls } from './menu/menuControls';
import { shapes } from './shapes';
import { initBuyView } from './buy/buyView';

const isMainSite = Boolean(MAIN_SITE);

const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');
const boardWrapper = document.querySelector('.board-wrapper');
const buyMenuButton = document.querySelector<HTMLButtonElement>('.menu-item[data-menu="buy"]');
const buyPanel = document.getElementById('buy-panel');
if (!isMainSite) {
  buyMenuButton?.remove();
  buyPanel?.remove();
}

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
const buyView = isMainSite ? initBuyView() : { show: () => {}, hide: () => {} };
const infoMenuButton = document.querySelector<HTMLButtonElement>('.menu-item[data-menu="info"]');
const playMenuButton = document.querySelector<HTMLButtonElement>('.menu-item[data-menu="play"]');
const boardMenuButton = document.querySelector<HTMLButtonElement>(
  '.menu-panel.left .menu-item[data-menu="board"]',
);
const rightMenuPanel = document.querySelector<HTMLElement>('.menu-panel.right');
let activeLeftMenu: string | null = 'board';
let activeRightMenu: string | null = 'play';
type PanelSide = 'left' | 'right';
type MenuActivateEvent = CustomEvent<{ menu?: string }>;
document.addEventListener('menu:activate', (event: Event) => {
  const menu = (event as MenuActivateEvent).detail.menu;
  const panel = event.target as HTMLElement | null;
  const isLeftPanel = panel?.classList.contains('left');
  const previousRightMenu = activeRightMenu;
  if (menu && isLeftPanel) {
    activeLeftMenu = menu;
  } else if (menu && panel?.classList.contains('right')) {
    activeRightMenu = menu;
    if (menu !== 'play' && activeLeftMenu !== 'board') {
      // Ensure board tools stay available when using non-play right panels.
      boardMenuButton?.click();
    }
  }
  if (menu === 'info') {
    infoPanel?.classList.add('visible');
  } else {
    infoPanel?.classList.remove('visible');
  }
  if (menu === 'buy' && isMainSite) {
    buyView.show();
  } else {
    buyView.hide();
  }
  const changedPanel: PanelSide | null = panel?.classList.contains('left')
    ? 'left'
    : panel?.classList.contains('right')
    ? 'right'
    : null;
  updateActionVisibility({
    changedPanel,
    previousRightMenu,
  });
});
document.addEventListener('solver:cleared', () => {
  resetAutoSolveState();
});
updateActionVisibility();
function updateActionVisibility(context?: {
  changedPanel: PanelSide | null;
  previousRightMenu?: string | null;
}): void {
  const boardInactive = activeRightMenu !== 'play';
  boardActions?.classList.toggle('hidden', boardInactive);
  helperActions?.classList.toggle('hidden', activeRightMenu !== 'helper');
  autoActions?.classList.toggle('hidden', activeRightMenu !== 'auto');
  (boardWrapper as HTMLElement | null)?.classList.toggle(
    'buy-mode',
    isMainSite && activeLeftMenu === 'buy',
  );
  const rightMenuChanged =
    context?.changedPanel === 'right' && context.previousRightMenu !== activeRightMenu;
  if (rightMenuChanged && activeRightMenu !== 'auto') {
    game.clearSolverVisualization();
    resetAutoSolveState();
  }
}

function handleDirectMenuQuery(): void {
  const params = new URLSearchParams(window.location.search);
  const show = params.get('show');
  const menuButtons: Record<string, HTMLButtonElement | null> = {
    buy: isMainSite ? buyMenuButton : null,
    info: infoMenuButton,
    play: playMenuButton,
    board: boardMenuButton,
  };
  const targetButton = show ? menuButtons[show] : null;
  if (!targetButton) return;
  params.delete('show');
  const search = params.toString();
  const newUrl = `${window.location.pathname}${search ? `?${search}` : ''}${window.location.hash}`;
  window.history.replaceState({}, document.title, newUrl);
  targetButton.click();
}
handleDirectMenuQuery();

const themes: Array<{ name: string; label: string; className: string }> = [
  { name: 'default', label: 'Theme', className: '' },
  { name: 'light', label: 'Light', className: 'theme-light' },
  { name: 'dark', label: 'Dark', className: 'theme-dark' },
  { name: 'rose', label: 'Rose Quartz', className: 'theme-rose' },
];
let themeIndex = 0;

function applyTheme(index: number, announce = false): void {
  themeIndex = (index + themes.length) % themes.length;
  document.body.classList.remove('theme-light', 'theme-dark', 'theme-rose');
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

let suppressBoardCycle = false;
const handleBoardMenuClick = (): void => {
  if (suppressBoardCycle) return;
  cycleShape();
};
boardMenuButton?.addEventListener('click', handleBoardMenuClick);
shapeButton?.addEventListener('click', cycleShape);

function forceBoardMode(): void {
  if (activeLeftMenu === 'board') return;
  suppressBoardCycle = true;
  boardMenuButton?.click();
  suppressBoardCycle = false;
}

const attachBoardModeGuard = (element: HTMLElement | null) => {
  if (!element) return;
  element.addEventListener('pointerdown', event => {
    if ((event.target as HTMLElement)?.id === 'theme') return;
    forceBoardMode();
  });
  element.addEventListener('click', event => {
    if ((event.target as HTMLElement)?.id === 'theme') return;
    forceBoardMode();
  });
};

attachBoardModeGuard(boardActions);
attachBoardModeGuard(helperActions);
attachBoardModeGuard(autoActions);
