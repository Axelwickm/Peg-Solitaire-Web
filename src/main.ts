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
resetButton?.addEventListener('click', () => {
  game.setup();
});
