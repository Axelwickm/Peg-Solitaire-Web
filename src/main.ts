import { KongmingGame } from './Game';
import { createMenuControls } from './menu/menuControls';

const boardEl = document.getElementById('board');
const statusEl = document.getElementById('status');

if (!boardEl || !statusEl) {
  throw new Error('Required DOM elements not found');
}

createMenuControls();
const game = new KongmingGame(boardEl, statusEl);

const resetButton = document.getElementById('reset');
resetButton?.addEventListener('click', () => {
  game.setup();
});
