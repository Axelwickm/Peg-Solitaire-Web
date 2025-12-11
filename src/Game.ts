import { BoardShape } from './shapes';
import {
  BidirectionalBfidaSolver,
  SolverResult,
  SolverMove,
  BidirectionalBfidaSession,
  SolverSessionProgress,
  SolverOptions,
} from './solver/BfidaSolver';

const SOLVER_STALL_TIMEOUT_MS = 120000;
const SOLVER_PERFECT_EXTRA_MS = 1000;

export type PlayLogEntry = {
  timestamp: string;
  durationMs: number;
  pegsLeft: number;
  perfect: boolean;
  shapeId: string;
};

type PegLighting = {
  rotation: number;
  shadowX: number;
  shadowY: number;
  shadowAngle: number;
};

export class KongmingGame {
  private boardEl: HTMLElement;
  private statusEl: HTMLElement;
  private boardWrapper: HTMLElement;
  private winOverlay: HTMLElement | null = null;
  private draggingPegKey: string | null = null;
  private dragTargetKey: string | null = null;
  private dragHoverKey: string | null = null;
  private dragMoved = false;
  private skipNextClick = false;
  private dragSelectableTargets: Set<string> | null = null;
  private selected: string | null = null;
  private pegs = new Set<string>();
  private validCells = new Set<string>();
  private allowedMoves = new Map<string, Map<string, string>>();
  private ghostPeg: HTMLDivElement | null = null;
  private draggingHole: HTMLElement | null = null;
  private solved = false;
  private moveMade = false;
  private currentShape!: BoardShape;
  private boundDragMove: (event: PointerEvent) => void;
  private boundDragEnd: () => void;
  private rowHoleCount = new Map<number, number>();
  private pickablePegs = new Set<string>();
  private pickupPulseTimeout: number | null = null;
  private history: string[][] = [];
  private helperMode = false;
  private helperBestMove: SolverMove | null = null;
  private helperResult: SolverResult | null = null;
  private helperStateDirty = true;
  private holePositions = new Map<string, { x: number; y: number }>();
  private solverOverlay: SVGSVGElement | null = null;
  private solverLines: SVGLineElement[] = [];
  private solverCircles: SVGCircleElement[] = [];
  private solverTexts: SVGTextElement[] = [];
  private solverSession: BidirectionalBfidaSession | null = null;
  private solverRunning = false;
  private solverTimer: number | null = null;
  private solverStartTime = 0;
  private solverResult: SolverResult | null = null;
  private solverPromiseResolve: ((result: SolverResult) => void) | null = null;
  private solverLastProgress: SolverSessionProgress | null = null;
  private solverLastImprovementTime = 0;
  private solverBestMovesLength = 0;
  private solverInitialPegCount = 0;
  private solverPerfectDetectedAt: number | null = null;
  private autoplayActive = false;
  private autoplayMoves: SolverMove[] = [];
  private autoplayIndex = 0;
  private autoplayTimer: number | null = null;
  private autoplayPlanVersion = 0;
  private autoplayKnownPlanVersion = 0;
  private pegLightTargetX = 0; // set from css
  private pegLightTargetY = 0;
  private pegSizePx = 0;
  private holeHitRadius = 0;
  private playStatsEl: HTMLElement | null = null;
  private playTimerEl: HTMLElement | null = null;
  private playTimerId: number | null = null;
  private playStartTime: number | null = null;
  private playElapsedMs = 0;
  private playResultLogged = false;
  private highScoreActive = false;
  private autoSolveUsed = false;
  private readonly playLogKey = 'kongming-play-log';
  private readonly bestPlayKey = 'kongming-play-best';
  private bestPlayEntry: PlayLogEntry | null = null;
  private dragWatchdogTimer: number | null = null;
  private static readonly DRAG_WATCHDOG_TIMEOUT_MS = 5000;
  constructor(
    boardEl: HTMLElement,
    statusEl: HTMLElement,
    boardWrapper: HTMLElement,
    defaultShape: BoardShape,
  ) {
    this.boardEl = boardEl;
    this.statusEl = statusEl;
    this.boardWrapper = boardWrapper;
    this.winOverlay = boardWrapper.querySelector('.win-overlay');
    this.boundDragMove = event => this.handleDragMove(event);
    this.boundDragEnd = () => this.handleDragEnd();
    this.prepareShape(defaultShape);
    this.setup();
  }

  public setup(): void {
    this.boardEl.innerHTML = '';
    this.clearSolverVisualization();
    this.pegs.clear();
    this.moveMade = false;
    this.validCells.forEach(cell => {
      if (cell !== this.currentShape.empty) {
        this.pegs.add(cell);
      }
    });
    this.history = [this.snapshotPegs()];
    this.selected = null;
    this.helperStateDirty = true;
    this.resetPlayStats();
    this.render();
    this.setStatus(`Remove pegs until one remains at the ${this.currentShape.finalTargetDescription}.`);
  }

  public hasProgress(): boolean {
    return this.moveMade;
  }

  public getCurrentShape(): BoardShape {
    return this.currentShape;
  }

  public changeShape(shape: BoardShape): void {
    this.prepareShape(shape);
    this.setup();
  }

  public forceWinState(): void {
    this.solved = true;
    this.boardWrapper.classList.add('solved');
  }

  public setPlayStatsElements(statsEl: HTMLElement | null, timerEl: HTMLElement | null): void {
    this.playStatsEl = statsEl;
    this.playTimerEl = timerEl;
    this.bestPlayEntry = this.loadBestPlayEntry();
    this.updatePlayStatsDisplay();
  }

  private render(): void {
    this.clearPickupPulse();
    if (this.helperMode) {
      this.refreshHelperState();
    } else {
      this.helperBestMove = null;
      this.helperResult = null;
    }
    this.boardEl.innerHTML = '';
    this.holePositions.clear();
    this.boardEl.style.setProperty('--hole-size', `${20}px`);
    this.holeHitRadius = 50;
    this.updatePegLightTarget();
    this.updatePickablePegs();
    const targetHoles = this.selected ? this.getSelectableTargets(this.selected) : new Set<string>();
    for (let r = 0; r < this.currentShape.height; r++) {
      for (let c = 0; c < this.currentShape.width; c++) {
        const key = `${r},${c}`;
        if (!this.validCells.has(key)) continue;
        const hole = document.createElement('div');
        hole.className = 'hole';
        if (key === this.currentShape.empty) {
          hole.classList.add('optimal-hole');
        }
        if (this.selected === key) {
          hole.classList.add('selected');
        }
        const position = this.getHolePosition(r, c);
        this.holePositions.set(key, position);
        hole.style.left = `${position.x * 100}%`;
        hole.style.top = `${position.y * 100}%`;
        hole.dataset.pos = key;
        const lighting = this.calculatePegLighting(position);
        this.applyPegLightingStyles(hole, lighting);
        if (this.pegs.has(key)) {
          const peg = document.createElement('div');
          peg.className = 'peg';
          hole.appendChild(peg);
          hole.classList.add('has-peg');
        }
        if (targetHoles.has(key)) {
          hole.classList.add('target');
        }
        if (this.helperMode && this.helperBestMove) {
          if (key === this.helperBestMove.from) {
            hole.classList.add('helper-hint-from');
          } else if (key === this.helperBestMove.to) {
            hole.classList.add('helper-hint-to');
          }
        }
        hole.addEventListener('pointerdown', event => this.startPegDrag(event, key));
        hole.addEventListener('click', event => this.handleHoleClick(key, event));
        this.boardEl.appendChild(hole);
      }
    }
    const gameOver = this.pickablePegs.size === 0;
    this.updateWinState(gameOver);
    this.updatePlayStatsDisplay();
    if (gameOver) {
      const pegsLeft = this.pegs.size;
      if (pegsLeft === 1) {
        const statusText = this.pegs.has(this.currentShape.empty)
          ? 'Game over · Solved.'
          : 'Game over · 1 left, not optimal.';
        this.setStatus(statusText);
      } else {
        this.setStatus(`Game over · ${pegsLeft} left.`);
      }
    }
  }

  private setStatus(text: string): void {
    this.statusEl.textContent = text;
  }

  private setHighScoreHighlight(active: boolean): void {
    this.highScoreActive = active;
    if (this.playStatsEl) {
      this.playStatsEl.classList.toggle('highscore', active);
    }
  }

  private loadBestPlayEntry(): PlayLogEntry | null {
    if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
      return null;
    }
    try {
      const raw = window.localStorage.getItem(this.bestPlayKey);
      if (!raw) return null;
      const entry: PlayLogEntry = JSON.parse(raw);
      if (
        typeof entry.timestamp !== 'string' ||
        typeof entry.durationMs !== 'number' ||
        typeof entry.pegsLeft !== 'number' ||
        typeof entry.perfect !== 'boolean' ||
        typeof entry.shapeId !== 'string'
      ) {
        return null;
      }
      return entry;
    } catch {
      return null;
    }
  }

  private saveBestPlayEntry(entry: PlayLogEntry): void {
    if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
      return;
    }
    try {
      window.localStorage.setItem(this.bestPlayKey, JSON.stringify(entry));
    } catch {
      // ignore storage failures
    }
  }

  private resetPlayStats(): void {
    this.stopPlayTimer();
    this.playElapsedMs = 0;
    this.playStartTime = null;
    this.playResultLogged = false;
    this.autoSolveUsed = false;
    this.setHighScoreHighlight(false);
    this.updatePlayStatsDisplay();
  }

  private startPlayTimer(): void {
    if (this.playTimerId) {
      window.clearInterval(this.playTimerId);
    }
    this.playElapsedMs = 0;
    this.playStartTime = Date.now();
    this.playTimerId = window.setInterval(() => this.updateTimerDisplay(), 500);
    this.updateTimerDisplay();
  }

  private stopPlayTimer(): void {
    if (this.playTimerId) {
      window.clearInterval(this.playTimerId);
      this.playTimerId = null;
    }
    if (this.playStartTime !== null) {
      this.playElapsedMs = Date.now() - this.playStartTime;
      this.playStartTime = null;
    }
    this.updateTimerDisplay();
  }

  private updateTimerDisplay(): void {
    const duration = this.getElapsedMs();
    this.playElapsedMs = duration;
    if (this.playTimerEl) {
      this.playTimerEl.textContent = this.formatDuration(duration);
    }
  }

  private getElapsedMs(): number {
    if (this.playStartTime !== null) {
      return Date.now() - this.playStartTime;
    }
    return this.playElapsedMs;
  }

  private formatDuration(ms: number): string {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }

  private updatePlayStatsDisplay(): void {
    this.updateTimerDisplay();
    const pegsLeft = this.pegs.size;
    const perfect = pegsLeft === 1 && this.pegs.has(this.currentShape.empty);
    const hasMoves = this.hasAvailableMoves();
    if (pegsLeft !== 1 && !hasMoves && !this.playResultLogged && this.playStartTime !== null) {
      this.stopPlayTimer();
      this.logPlayResult(this.playElapsedMs, pegsLeft, false);
      this.playResultLogged = true;
      return;
    }
    if (pegsLeft === 1 && !this.playResultLogged && this.playStartTime !== null) {
      this.stopPlayTimer();
      this.logPlayResult(this.playElapsedMs, pegsLeft, perfect);
      this.playResultLogged = true;
    }
  }

  private hasAvailableMoves(): boolean {
    for (const _ of this.validMoves()) {
      return true;
    }
    return false;
  }

  private logPlayResult(durationMs: number, pegsLeft: number, perfect: boolean): void {
    if (this.autoSolveUsed) {
      return;
    }
    const entry: PlayLogEntry = {
      timestamp: new Date().toISOString(),
      durationMs,
      pegsLeft,
      perfect,
      shapeId: this.currentShape.id,
    };
    const isHighScore = perfect && (!this.bestPlayEntry || durationMs < this.bestPlayEntry.durationMs);
    this.setHighScoreHighlight(isHighScore);
    if (isHighScore) {
      this.bestPlayEntry = entry;
      this.saveBestPlayEntry(entry);
    }
    this.appendLogEntry(entry);
    if (typeof document !== 'undefined') {
      document.dispatchEvent(new CustomEvent('playlog:updated'));
    }
  }

  private appendLogEntry(entry: PlayLogEntry): void {
    if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
      return;
    }
    try {
      const raw = window.localStorage.getItem(this.playLogKey);
      const entries: PlayLogEntry[] = raw ? JSON.parse(raw) : [];
      if (!Array.isArray(entries)) {
        throw new Error('Invalid log');
      }
      entries.push(entry);
      const capped = entries.slice(-20);
      window.localStorage.setItem(this.playLogKey, JSON.stringify(capped));
    } catch {
      // ignore storage failures
    }
  }

  private loadPlayLogEntries(): PlayLogEntry[] {
    if (typeof window === 'undefined' || typeof window.localStorage === 'undefined') {
      return [];
    }
    try {
      const raw = window.localStorage.getItem(this.playLogKey);
      const entries: PlayLogEntry[] = raw ? JSON.parse(raw) : [];
      if (!Array.isArray(entries)) {
        return [];
      }
      return entries.filter(
        entry =>
          typeof entry.timestamp === 'string' &&
          typeof entry.durationMs === 'number' &&
          typeof entry.pegsLeft === 'number' &&
          typeof entry.perfect === 'boolean' &&
          typeof entry.shapeId === 'string',
      );
    } catch {
      return [];
    }
  }

  public updateStatusMessage(text: string): void {
    this.setStatus(text);
  }

  public getPlayLogEntries(): PlayLogEntry[] {
    return this.loadPlayLogEntries();
  }

  private handleHoleClick(key: string, event: MouseEvent): void {
    this.ensureBoardInteractive();
    if (this.skipNextClick) {
      event.preventDefault();
      this.skipNextClick = false;
      return;
    }
    if (this.pegs.has(key) && !this.pickablePegs.has(key)) {
      this.logInvalidPick(key, 'click');
      this.pulsePickableHints();
      return;
    }
    this.onHoleClick(key);
  }

  private onHoleClick(key: string): void {
    if (!this.validCells.has(key)) return;
    if (this.pegs.has(key)) {
      const selectable = this.getSelectableTargets(key);
      if (selectable.size === 0) {
        if (this.selected !== null) {
          this.selected = null;
          this.render();
        }
        return;
      }
      if (this.selected === key) {
        this.selected = null;
        this.render();
        return;
      }
      this.selected = key;
      this.render();
      return;
    }
    if (!this.selected) return;
    const move = this.validateMove(this.selected, key);
    if (!move) {
      this.selected = null;
      this.render();
      return;
    }
    this.applyMove(this.selected, key, move.jump);
  }

  private validateMove(from: string, to: string): { jump: string } | null {
    if (this.pegs.has(to)) return null;
    const moves = this.allowedMoves.get(from);
    const jump = moves?.get(to);
    if (!jump) return null;
    if (!this.pegs.has(jump)) return null;
    return { jump };
  }

  private getSelectableTargets(from: string): Set<string> {
    const targets = new Set<string>();
    const moves = this.allowedMoves.get(from);
    if (!moves) return targets;
    for (const [to, jump] of moves.entries()) {
      if (this.pegs.has(to)) continue;
      if (!this.pegs.has(jump)) continue;
      targets.add(to);
    }
    return targets;
  }

  private *validMoves(): Generator<{ from: string; to: string }> {
    for (const peg of this.pegs) {
      const moves = this.allowedMoves.get(peg);
      if (!moves) continue;
      for (const [to, jump] of moves.entries()) {
        if (this.pegs.has(to)) continue;
        if (!this.pegs.has(jump)) continue;
        yield { from: peg, to };
      }
    }
  }

  private updatePickablePegs(): void {
    this.pickablePegs.clear();
    for (const peg of this.pegs) {
      const moves = this.allowedMoves.get(peg);
      if (!moves) continue;
      for (const [to, jump] of moves.entries()) {
        if (this.pegs.has(to)) continue;
        if (!this.pegs.has(jump)) continue;
        this.pickablePegs.add(peg);
        break;
      }
    }
  }

  private pulsePickableHints(): void {
    this.clearPickupPulse();
    if (this.pickablePegs.size === 0) return;
    this.pickablePegs.forEach(key => {
      const hole = this.getHoleElement(key);
      hole?.classList.add('pickup-pulse');
    });
    this.pickupPulseTimeout = window.setTimeout(() => this.clearPickupPulse(), 500);
  }

  private logInvalidPick(key: string, action: 'click' | 'drag'): void {
    console.log(`Invalid pick attempt (${action}) for peg ${key}.`);
  }

  private clearPickupPulse(): void {
    if (this.pickupPulseTimeout !== null) {
      window.clearTimeout(this.pickupPulseTimeout);
      this.pickupPulseTimeout = null;
    }
    this.boardEl.querySelectorAll('.hole.pickup-pulse').forEach(el => el.classList.remove('pickup-pulse'));
  }

  private startPegDrag(event: PointerEvent, key: string): void {
    this.ensureBoardInteractive();
    if (!this.pegs.has(key)) return;
    this.stopAutoPlay();
    if (!this.pickablePegs.has(key)) {
      this.logInvalidPick(key, 'drag');
      this.pulsePickableHints();
      return;
    }
    this.draggingPegKey = key;
    this.dragTargetKey = key;
    this.dragMoved = false;
    this.dragSelectableTargets = this.getSelectableTargets(key);
    this.dragSelectableTargets.add(key);
    this.draggingHole = this.getHoleElement(key);
    this.draggingHole?.classList.add('dragging');
    this.createGhostPeg(event);
    document.addEventListener('pointermove', this.boundDragMove);
    document.addEventListener('pointerup', this.boundDragEnd);
    document.addEventListener('pointercancel', this.boundDragEnd);
    event.preventDefault();
    this.scheduleDragWatchdog();
  }

  private handleDragMove(event: PointerEvent): void {
    if (!this.draggingPegKey) return;
    this.dragMoved = true;
    const targetKey = this.getHoleKeyFromPoint(event, this.dragSelectableTargets);
    this.dragTargetKey = targetKey;
    const canDrop =
      !!targetKey &&
      this.draggingPegKey !== null &&
      targetKey !== this.draggingPegKey &&
      !!this.validateMove(this.draggingPegKey, targetKey);
    if (canDrop) {
      this.setDragHover(targetKey);
    } else {
      this.clearDragHover();
    }
    this.updateGhostPosition(event);
    this.scheduleDragWatchdog();
  }

  private handleDragEnd(): void {
    const hadValidMove =
      !!(this.draggingPegKey && this.dragTargetKey && this.dragTargetKey !== this.draggingPegKey);
    if (hadValidMove) {
      this.attemptMove(this.draggingPegKey!, this.dragTargetKey!);
    }
    const moved = this.dragMoved;
    this.dragMoved = false;
    this.completeDragInteraction();
    if (hadValidMove && moved) {
      this.skipNextClick = true;
    }
  }

  private scheduleDragWatchdog(): void {
    this.clearDragWatchdog();
    this.dragWatchdogTimer = window.setTimeout(() => {
      if (this.draggingPegKey) {
        console.warn('[UI] Drag watchdog triggered');
        this.cancelDragDueToWatchdog();
      }
    }, KongmingGame.DRAG_WATCHDOG_TIMEOUT_MS);
  }

  private clearDragWatchdog(): void {
    if (this.dragWatchdogTimer !== null) {
      window.clearTimeout(this.dragWatchdogTimer);
      this.dragWatchdogTimer = null;
    }
  }

  private cancelDragDueToWatchdog(): void {
    if (!this.draggingPegKey) return;
    this.dragMoved = false;
    this.completeDragInteraction();
  }

  private completeDragInteraction(): void {
    this.draggingPegKey = null;
    this.dragTargetKey = null;
    this.dragSelectableTargets = null;
    this.draggingHole?.classList.remove('dragging');
    this.draggingHole = null;
    this.clearDragHover();
    this.removeGhostPeg();
    document.removeEventListener('pointermove', this.boundDragMove);
    document.removeEventListener('pointerup', this.boundDragEnd);
    document.removeEventListener('pointercancel', this.boundDragEnd);
    this.clearDragWatchdog();
  }

  private attemptMove(from: string, to: string): boolean {
    const move = this.validateMove(from, to);
    if (!move) return false;
    this.applyMove(from, to, move.jump);
    return true;
  }

  private applyMove(from: string, to: string, jump: string): void {
    if (!this.autoplayActive) {
      this.clearSolverVisualization();
    }
    if (!this.playStartTime) {
      this.startPlayTimer();
    }
    this.pegs.delete(from);
    this.pegs.delete(jump);
    this.pegs.add(to);
    this.selected = null;
    this.moveMade = true;
    this.pushHistoryState();
    this.helperStateDirty = true;
    this.render();
  }

  private setDragHover(key: string | null): void {
    if (this.dragHoverKey === key) return;
    if (this.dragHoverKey) {
      const prev = this.getHoleElement(this.dragHoverKey);
      if (prev) prev.classList.remove('drag-over');
    }
    this.dragHoverKey = key;
    if (key) {
      const next = this.getHoleElement(key);
      if (next) next.classList.add('drag-over');
    }
  }

  private clearDragHover(): void {
    this.setDragHover(null);
  }

  private getHoleElement(key: string | null): HTMLElement | null {
    if (!key) return null;
    return this.boardEl.querySelector(`[data-pos="${key}"]`);
  }

  private getHoleKeyFromPoint(event: PointerEvent, validTargets: Set<string> | null = null): string | null {
    const boardRect = this.boardEl.getBoundingClientRect();
    if (validTargets && validTargets.size === 0) {
      return null;
    }
    if (boardRect.width && boardRect.height && this.holePositions.size > 0) {
      const targetX = event.clientX - boardRect.left;
      const targetY = event.clientY - boardRect.top;
      let nearestKey: string | null = null;
      let nearestDist = Infinity;

      this.holePositions.forEach((position, key) => {
        if (validTargets && !validTargets.has(key)) {
          return;
        }
        const holeX = position.x * boardRect.width;
        const holeY = position.y * boardRect.height;
        const dx = holeX - targetX;
        const dy = holeY - targetY;
        const dist = Math.hypot(dx, dy);
        if (dist < nearestDist) {
          nearestDist = dist;
          nearestKey = key;
        }
      });

      const hitRadius = Math.max(this.holeHitRadius, 1);
      if (nearestKey && nearestDist <= hitRadius) {
        return nearestKey;
      }
    }

    return this.findHoleKeyFromElement(event, validTargets);
  }

  private findHoleKeyFromElement(event: PointerEvent, validTargets: Set<string> | null = null): string | null {
    let target: Element | null = document.elementFromPoint(event.clientX, event.clientY);
    while (target && target !== this.boardEl) {
      if (target instanceof HTMLElement && target.classList.contains('hole')) {
        break;
      }
      target = target.parentElement;
    }
    if (target instanceof HTMLElement && target.dataset.pos) {
      const pos = target.dataset.pos;
      if (validTargets && !validTargets.has(pos)) {
        return null;
      }
      return pos;
    }
    return null;
  }

  private createGhostPeg(event: PointerEvent): void {
    if (this.ghostPeg) {
      this.removeGhostPeg();
    }
    const peg = document.createElement('div');
    peg.className = 'peg ghost';
    document.body.appendChild(peg);
    this.ghostPeg = peg;
    const initialHolePosition = this.draggingPegKey ? this.holePositions.get(this.draggingPegKey) : null;
    if (initialHolePosition) {
      this.applyGhostOrientation(initialHolePosition);
    } else {
      const hoveredPosition = this.getBoardRelativePosition(event.clientX, event.clientY);
      if (hoveredPosition) {
        this.applyGhostOrientation(hoveredPosition);
      }
    }
    this.updateGhostPosition(event);
  }

  private updateGhostPosition(event: PointerEvent): void {
    if (!this.ghostPeg) return;
    const size = parseFloat(getComputedStyle(this.ghostPeg).width) || 40;
    this.ghostPeg.style.left = `${event.clientX - size / 2}px`;
    this.ghostPeg.style.top = `${event.clientY - size / 2}px`;
    const boardPosition = this.getBoardRelativePosition(event.clientX, event.clientY);
    if (boardPosition) {
      this.applyGhostOrientation(boardPosition);
    }
  }

  private removeGhostPeg(): void {
    if (!this.ghostPeg) return;
    this.ghostPeg.remove();
    this.ghostPeg = null;
  }

  private updateWinState(gameOver: boolean): void {
    const solved = this.pegs.size === 1 && this.pegs.has(this.currentShape.empty);
    this.solved = solved;
    this.boardWrapper.classList.toggle('solved', solved);
    const overlay = this.winOverlay;
    if (!overlay) {
      return;
    }
    if (!gameOver) {
      this.boardWrapper.dataset.gameOver = 'false';
      overlay.removeAttribute('data-overlay-text');
      this.boardWrapper.style.removeProperty('--game-overlay-border');
      this.boardWrapper.style.removeProperty('--game-overlay-badge-bg');
      this.boardWrapper.style.removeProperty('--game-overlay-badge-color');
      return;
    }
    const pegsLeft = this.pegs.size;
    let overlayText = 'Solved';
    let borderColor = '#000000';
    let badgeBg = '#000000';
    let badgeColor = '#ffffff';
    if (pegsLeft > 1) {
      overlayText = `${pegsLeft} left`;
      borderColor = '#3b0008';
      badgeBg = '#3b0008';
      badgeColor = '#ffffff';
      this.boardWrapper.classList.remove('solved');
    } else if (!solved) {
      overlayText = '1 left, not optimal';
      borderColor = '#c46200';
      badgeBg = '#c46200';
      badgeColor = '#1b1b1f';
      this.boardWrapper.classList.remove('solved');
    } else {
      overlayText = 'Solved';
      borderColor = '#000000';
      badgeBg = '#000000';
      badgeColor = '#ffffff';
    }
    overlay.setAttribute('data-overlay-text', overlayText);
    this.boardWrapper.dataset.gameOver = 'true';
    this.boardWrapper.style.setProperty('--game-overlay-border', borderColor);
    this.boardWrapper.style.setProperty('--game-overlay-badge-bg', badgeBg);
    this.boardWrapper.style.setProperty('--game-overlay-badge-color', badgeColor);
  }

  private prepareShape(shape: BoardShape): void {
    this.currentShape = shape;
    this.validCells = new Set(shape.holes);
    this.computeRowCounts();
    this.allowedMoves = shape.allowedMoves;
  }

  private computeRowCounts(): void {
    this.rowHoleCount.clear();
    this.validCells.forEach(cell => {
      const row = Number(cell.split(',')[0]);
      this.rowHoleCount.set(row, (this.rowHoleCount.get(row) ?? 0) + 1);
    });
  }

  private updatePegLightTarget(): void {
    const styles = getComputedStyle(document.documentElement);
    const parseOrDefault = (value: string, fallback: number): number => {
      const parsed = parseFloat(value);
      return Number.isFinite(parsed) ? parsed : fallback;
    };
    this.pegLightTargetX = parseOrDefault(styles.getPropertyValue('--peg-light-target-x'), 1);
    this.pegLightTargetY = parseOrDefault(styles.getPropertyValue('--peg-light-target-y'), -1);
  }

  private getLightPosition(): { x: number; y: number } {
    return {
      x: this.pegLightTargetX,
      y: this.pegLightTargetY
    };
  }

  private getHolePosition(row: number, col: number): { x: number; y: number } {
    if (this.currentShape.layout === 'triangle') {
      const rowCounts = [...this.rowHoleCount.values()];
      const maxRowHoles = rowCounts.length ? Math.max(...rowCounts) : 1;
      const horizontalSteps = Math.max(1, maxRowHoles - 1);
      const densityScale = 0.65;
      const dx = (1 / horizontalSteps) * densityScale;
      const dy = (Math.sqrt(3) / 2) * dx;
      const verticalSteps = Math.max(1, this.currentShape.height - 1);
      const triangleHeight = dy * verticalSteps;
      const verticalMargin = Math.max(0, (1 - triangleHeight) / 2);
      const rowHoleCount = this.rowHoleCount.get(row) ?? 0;
      const rowWidth = Math.max(0, (rowHoleCount - 1) * dx);
      const horizontalMargin = Math.max(0, (1 - rowWidth) / 2);
      const indexInRow = this.countRowHolesBefore(row, col);
      return {
        x: horizontalMargin + indexInRow * dx,
        y: verticalMargin + row * dy,
      };
    }
    const margin = 0.08;
    const span = Math.max(0.5, 1 - margin * 2);
    return {
      x: margin + ((col + 0.5) / this.currentShape.width) * span,
      y: margin + ((row + 0.5) / this.currentShape.height) * span,
    };
  }

  private countRowHolesBefore(row: number, col: number): number {
    let count = 0;
    for (let c = 0; c < col; c++) {
      if (this.validCells.has(`${row},${c}`)) {
        count += 1;
      }
    }
    return count;
  }

  private calculatePegLighting(pos: { x: number; y: number }): PegLighting {
    const light = this.getLightPosition();
    const dx = pos.x - light.x;
    const dy = pos.y - light.y;
    const distance = Math.hypot(dx, dy) || 1;
    const unitX = dx / distance;
    const unitY = dy / distance;
    let rotation = (Math.atan2(unitY, unitX) * 180) / Math.PI;
    rotation = (rotation + 360) % 360;
    rotation = (rotation + 180) % 360;
    const shadowLength = Math.max(6, (this.pegSizePx || 40) * 0.18);
    const shadowX = unitX * shadowLength;
    const shadowY = unitY * shadowLength;
    return {
      rotation,
      shadowX,
      shadowY,
      shadowAngle: rotation,
    };
  }

  private applyPegLightingStyles(target: HTMLElement, lighting: PegLighting): void {
    target.style.setProperty('--peg-rotation', `${lighting.rotation.toFixed(2)}deg`);
    target.style.setProperty('--peg-shadow-x', `${lighting.shadowX.toFixed(2)}px`);
    target.style.setProperty('--peg-shadow-y', `${lighting.shadowY.toFixed(2)}px`);
    target.style.setProperty('--peg-shadow-angle', `${lighting.shadowAngle.toFixed(2)}deg`);
  }

  private getBoardRelativePosition(clientX: number, clientY: number): { x: number; y: number } | null {
    const rect = this.boardEl.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {
      x: (clientX - rect.left) / rect.width,
      y: (clientY - rect.top) / rect.height,
    };
  }

  private applyGhostOrientation(position: { x: number; y: number }): void {
    if (!this.ghostPeg) return;
    const lighting = this.calculatePegLighting(position);
    this.applyPegLightingStyles(this.ghostPeg, lighting);
  }

  private snapshotPegs(): string[] {
    return [...this.pegs];
  }

  private buildSolverOptions(): SolverOptions {
    return {
      enforceTargetTypeInvariant: this.currentShape?.targetTypeInvariant !== false,
    };
  }

  public solvePuzzle(): SolverResult {
    console.log('[UI] solvePuzzle invoked', {
      shape: this.currentShape.id,
      pegCount: this.pegs.size,
    });
    const solver = new BidirectionalBfidaSolver(this.allowedMoves, this.buildSolverOptions());
    const result = solver.solve(new Set(this.pegs), this.currentShape.empty);
    console.log('[UI] solvePuzzle result', result);
    return result;
  }

  private pushHistoryState(): void {
    this.history.push(this.snapshotPegs());
  }

  public undo(): void {
    if (this.history.length <= 1) return;
    this.history.pop();
    const previous = this.history[this.history.length - 1];
    this.pegs = new Set(previous);
    this.selected = null;
    this.moveMade = this.history.length > 1;
    this.helperStateDirty = true;
    this.render();
  }

  public requestHint(): void {
    this.setStatus('Hint functionality is coming soon.');
  }

  public toggleHelperMode(): void {
    this.helperMode = !this.helperMode;
    this.setStatus(this.helperMode ? 'Helper mode on' : 'Helper mode off');
    this.helperStateDirty = true;
  }

  public isHelperModeActive(): boolean {
    return this.helperMode;
  }

  public isAutoplayActive(): boolean {
    return this.autoplayActive;
  }

  public isSolverActive(): boolean {
    return this.solverRunning;
  }

  public setSolverOverlay(overlay: SVGSVGElement): void {
    this.solverOverlay = overlay;
  }

  public async startAutoSolveVisualization(chunkMs = 50): Promise<SolverResult> {
    this.autoSolveUsed = true;
    if (this.solverRunning && this.solverResult) {
      console.log('[UI] Solver already running, returning cached result');
      return this.solverResult;
    }
    this.clearSolverVisualization();
    console.log('[UI] Starting auto-solve', {
      shape: this.currentShape.id,
      pegCount: this.pegs.size,
      chunkMs,
    });
    const solver = new BidirectionalBfidaSolver(this.allowedMoves, this.buildSolverOptions());
    const session = solver.createSession(new Set(this.pegs), this.currentShape.empty);
    if (!session) {
      const result: SolverResult = {
        solved: false,
        moves: [],
        bestMoves: [],
        nodesExplored: 0,
        durationMs: 0,
      };
      this.solverSession = null;
      this.solverResult = result;
      this.setStatus('No solution: start and goal are in different position classes.');
      return result;
    }
    this.solverSession = session;
    this.setStatus('Solving…');
    this.solverRunning = true;
    this.solverStartTime = performance.now();
    this.solverInitialPegCount = this.pegs.size;
    this.solverBestMovesLength = 0;
    this.solverLastImprovementTime = performance.now();
    this.solverPerfectDetectedAt = null;
    this.boardWrapper.classList.add('solver-busy');
    const promise = new Promise<SolverResult>(resolve => {
      this.solverPromiseResolve = resolve;
      const step = (): void => {
        if (!this.solverSession) return;
        const progress = this.solverSession.runChunk(chunkMs);
        this.solverLastProgress = progress;
        this.recordSolverImprovement(progress);
        const stalled =
          this.solverBestMovesLength > 0 &&
          performance.now() - this.solverLastImprovementTime > SOLVER_STALL_TIMEOUT_MS;
        const bestLeftNow = this.getBestPegsLeft();
        let shouldStopForPerfect = false;
        if (bestLeftNow === 1) {
          if (!this.solverPerfectDetectedAt) {
            this.solverPerfectDetectedAt = performance.now();
          } else if (performance.now() - this.solverPerfectDetectedAt > SOLVER_PERFECT_EXTRA_MS) {
            shouldStopForPerfect = true;
          }
        } else {
          this.solverPerfectDetectedAt = null;
        }
        console.log('[UI] Solver chunk', {
          done: progress.done,
          solved: progress.solved,
          nodes: progress.nodesExplored,
          pathLength: progress.bestMoves.length,
        });
        const color = progress.done && progress.solved ? 'lime' : 'red';
        this.updateSolverLines(progress.currentPath, color);
        if (!progress.done) {
          const bestSuffix = this.formatBestStatusSuffix();
          this.setStatus(
            `Solving… ${progress.nodesExplored} nodes · ${progress.bestMoves.length} moves ready${bestSuffix}`,
          );
        }
        if (progress.done) {
          this.finishSolverRun(progress, resolve, 'completed');
          return;
        }
        if (shouldStopForPerfect) {
          console.log('[UI] Perfect solution reached; finishing after grace period');
          this.finishSolverRun(progress, resolve, 'perfect');
          return;
        }
        if (stalled) {
          console.log('[UI] Solver stalled, returning best-so-far solution');
          this.finishSolverRun(progress, resolve, 'stalled');
          return;
        }
        this.solverTimer = window.setTimeout(step, 0);
      };
      step();
    });
    return promise;
  }

  private recordSolverImprovement(progress: SolverSessionProgress): void {
    const solutionLength = progress.bestMoves.length;
    const previewLength = progress.currentPath.length;
    const latestBest = Math.max(solutionLength, previewLength);
    const improved = latestBest > this.solverBestMovesLength;
    if (improved) {
      this.solverBestMovesLength = latestBest;
      this.solverLastImprovementTime = performance.now();
    }
  }

  private getBestPegsLeft(): number | null {
    if (!this.solverInitialPegCount) return null;
    const movesUsed = this.solverBestMovesLength ?? 0;
    const pegsLeft = Math.max(1, this.solverInitialPegCount - movesUsed);
    return pegsLeft;
  }

  private formatBestStatusSuffix(): string {
    const bestLeft = this.getBestPegsLeft();
    if (bestLeft === null) {
      return '';
    }
    return ` · Best: ${bestLeft} peg${bestLeft === 1 ? '' : 's'} left (${this.solverBestMovesLength} move${this.solverBestMovesLength === 1 ? '' : 's'})`;
  }

  private finishSolverRun(
    progress: SolverSessionProgress,
    resolve: (result: SolverResult) => void,
    reason: 'completed' | 'stalled' | 'perfect',
  ): void {
    const solved = reason === 'perfect' ? true : reason === 'completed' && progress.solved;
    this.solverRunning = false;
    this.boardWrapper.classList.remove('solver-busy');
    this.solverPerfectDetectedAt = null;
    const chosenPath = progress.bestMoves.length ? progress.bestMoves : progress.currentPath;
    const bestPath = [...chosenPath];
    if (bestPath.length > this.solverBestMovesLength) {
      this.solverBestMovesLength = bestPath.length;
    }
    const result: SolverResult = {
      solved,
      moves: bestPath,
      bestMoves: bestPath,
      nodesExplored: progress.nodesExplored,
      durationMs: Math.max(0, performance.now() - this.solverStartTime),
    };
    this.solverResult = result;
    if (result.bestMoves.length > 0) {
      this.autoplayPlanVersion += 1;
      this.autoplayKnownPlanVersion = this.autoplayPlanVersion;
      this.autoplayMoves = [...result.bestMoves];
      this.autoplayIndex = 0;
    }
    this.solverSession = null;
    if (this.solverTimer !== null) {
      window.clearTimeout(this.solverTimer);
      this.solverTimer = null;
    }
    this.solverPromiseResolve = null;
    this.updateSolverLines(bestPath, result.solved ? 'lime' : 'red');
    console.log('[UI] Solver finished', { ...result, reason });
    let status: string;
    if (reason === 'perfect') {
      status = `Perfect solution found · ${result.nodesExplored} nodes explored`;
    } else if (result.solved) {
      status = `Solved in ${result.durationMs.toFixed(1)}ms · ${result.nodesExplored} nodes`;
    } else if (reason === 'stalled') {
      status = `Timed out after 40s without improvement · ${result.nodesExplored} nodes explored`;
    } else {
      status = `No solution · ${result.nodesExplored} nodes explored`;
    }
    status += this.formatBestStatusSuffix();
    this.setStatus(status);
    resolve(result);
  }

  public clearSolverVisualization(): void {
    const wasRunning = this.solverRunning;
    const hadResult = !!this.solverResult;
    this.solverRunning = false;
    this.solverSession = null;
    if (this.solverTimer !== null) {
      window.clearTimeout(this.solverTimer);
      this.solverTimer = null;
    }
    this.boardWrapper.classList.remove('solver-busy');
    this.clearSolverLines();
    this.solverResult = null;
    this.autoplayPlanVersion += 1;
    this.autoplayMoves = [];
    this.autoplayIndex = 0;
    this.autoplayKnownPlanVersion = this.autoplayPlanVersion;
    if (this.solverPromiseResolve) {
      const nodes = this.solverLastProgress?.nodesExplored ?? 0;
      const duration = Math.max(0, performance.now() - this.solverStartTime);
      this.solverPromiseResolve({
        solved: false,
        moves: [],
        bestMoves: [],
        nodesExplored: nodes,
        durationMs: duration,
      });
      this.solverPromiseResolve = null;
    }
    this.solverLastProgress = null;
    this.solverInitialPegCount = 0;
    this.solverBestMovesLength = 0;
    this.solverLastImprovementTime = 0;
    this.solverPerfectDetectedAt = null;
    if (wasRunning && !hadResult) {
      this.setStatus('Solver stopped.');
    }
    const event = new CustomEvent('solver:cleared');
    document.dispatchEvent(event);
    this.stopAutoPlayInternal();
  }

  public hasSolvePlan(): boolean {
    return !!(this.solverResult?.bestMoves.length);
  }

  private stopAutoPlayInternal(): void {
    if (!this.autoplayActive) return;
    if (this.autoplayTimer !== null) {
      window.clearTimeout(this.autoplayTimer);
      this.autoplayTimer = null;
    }
    this.autoplayActive = false;
    document.dispatchEvent(new CustomEvent('autoplay:stopped'));
  }

  public stopAutoPlay(): void {
    this.stopAutoPlayInternal();
  }

  public startAutoPlayPlan(delayMs = 850): void {
    if (!this.hasSolvePlan()) return;
    if (this.autoplayKnownPlanVersion !== this.autoplayPlanVersion) {
      this.autoplayMoves = [...(this.solverResult?.moves ?? [])];
      this.autoplayIndex = 0;
      this.autoplayKnownPlanVersion = this.autoplayPlanVersion;
    }
    if (!this.autoplayMoves.length) return;
    this.autoplayActive = true;
    this.dropSolverOverlay();
    document.dispatchEvent(new CustomEvent('autoplay:started'));
    this.scheduleAutoMove(delayMs);
  }

  private scheduleAutoMove(delayMs: number): void {
    if (!this.autoplayActive) return;
    if (this.autoplayIndex >= this.autoplayMoves.length) {
      this.stopAutoPlayInternal();
      return;
    }
    this.autoplayTimer = window.setTimeout(() => {
      if (!this.autoplayActive) return;
      const move = this.autoplayMoves[this.autoplayIndex++];
      this.applyMove(move.from, move.to, move.jump);
      this.scheduleAutoMove(delayMs);
    }, delayMs);
  }

  private updateSolverLines(moves: SolverMove[], color: string): void {
    this.clearSolverLines();
    const overlay = this.solverOverlay;
    if (!overlay) return;
    const total = moves.length;
    moves.forEach((move, index) => {
      const from = this.holePositions.get(move.from);
      const to = this.holePositions.get(move.to);
      if (!from || !to) return;
      const ratio = total > 1 ? index / (total - 1) : 0;
      const baseHue = Math.round(120 * ratio);
      const strokeColor = `hsl(${baseHue}, 85%, ${color === 'lime' ? 65 : 48}%)`;

      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('stroke', strokeColor);
      line.setAttribute('stroke-width', '2');
      line.setAttribute('stroke-linecap', 'round');
      line.setAttribute('data-step', (index + 1).toString());
      line.setAttribute('data-move', `${move.from}->${move.to}`);
      line.setAttribute('x1', (from.x * 100).toString());
      line.setAttribute('y1', (from.y * 100).toString());
      line.setAttribute('x2', (to.x * 100).toString());
      line.setAttribute('y2', (to.y * 100).toString());
      overlay.appendChild(line);
      this.solverLines.push(line);

      const fromCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      fromCircle.setAttribute('cx', (from.x * 100).toString());
      fromCircle.setAttribute('cy', (from.y * 100).toString());
      fromCircle.setAttribute('r', '1.2');
      fromCircle.setAttribute('fill', strokeColor);
      fromCircle.setAttribute('stroke', '#0003');
      fromCircle.setAttribute('stroke-width', '0.2');
      fromCircle.setAttribute('data-node', move.from);
      overlay.appendChild(fromCircle);
      this.solverCircles.push(fromCircle);

      const toCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      toCircle.setAttribute('cx', (to.x * 100).toString());
      toCircle.setAttribute('cy', (to.y * 100).toString());
      toCircle.setAttribute('r', '1.5');
      toCircle.setAttribute('fill', color === 'lime' ? '#d4ffd4' : strokeColor);
      toCircle.setAttribute('stroke', '#000a');
      toCircle.setAttribute('stroke-width', '0.3');
      overlay.appendChild(toCircle);
      this.solverCircles.push(toCircle);

      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', ((from.x + to.x) * 50).toString());
      label.setAttribute('y', ((from.y + to.y) * 50).toString());
      label.setAttribute('fill', '#fff');
      label.setAttribute('font-size', '3');
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('dominant-baseline', 'central');
      label.setAttribute('pointer-events', 'none');
      label.textContent = `${index + 1}`;
      overlay.appendChild(label);
      this.solverTexts.push(label);
    });
  }

  private clearSolverLines(): void {
    this.solverLines.forEach(line => line.remove());
    this.solverCircles.forEach(circle => circle.remove());
    this.solverTexts.forEach(text => text.remove());
    this.solverLines = [];
    this.solverCircles = [];
    this.solverTexts = [];
  }

  private dropSolverOverlay(): void {
    this.clearSolverLines();
    this.boardWrapper.classList.remove('solver-busy');
  }

  private ensureBoardInteractive(): void {
    if (!this.solverRunning) {
      this.boardWrapper.classList.remove('solver-busy');
    }
  }

  private refreshHelperState(): void {
    if (!this.helperStateDirty && this.helperResult) return;
    this.helperResult = this.solvePuzzle();
    this.helperBestMove = this.helperResult.moves[0] ?? null;
    this.helperStateDirty = false;
  }

}
