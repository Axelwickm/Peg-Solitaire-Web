import { BoardShape } from './shapes';

export class KongmingGame {
  private boardEl: HTMLElement;
  private statusEl: HTMLElement;
  private boardWrapper: HTMLElement;
  private draggingPegKey: string | null = null;
  private dragTargetKey: string | null = null;
  private dragHoverKey: string | null = null;
  private dragMoved = false;
  private skipClickUntil = 0;
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

  constructor(
    boardEl: HTMLElement,
    statusEl: HTMLElement,
    boardWrapper: HTMLElement,
    defaultShape: BoardShape,
  ) {
    this.boardEl = boardEl;
    this.statusEl = statusEl;
    this.boardWrapper = boardWrapper;
    this.boundDragMove = event => this.handleDragMove(event);
    this.boundDragEnd = () => this.handleDragEnd();
    this.prepareShape(defaultShape);
    this.setup();
  }

  public setup(): void {
    this.boardEl.innerHTML = '';
    this.pegs.clear();
    this.moveMade = false;
    this.validCells.forEach(cell => {
      if (cell !== this.currentShape.empty) {
        this.pegs.add(cell);
      }
    });
    this.selected = null;
    this.render();
    this.setStatus('Remove pegs until one remains.');
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

  private render(): void {
    this.clearPickupPulse();
    this.boardEl.innerHTML = '';
    const holeSize = this.calculateHoleSize();
    this.boardEl.style.setProperty('--hole-size', `${holeSize}px`);
    this.updatePickablePegs();
    for (let r = 0; r < this.currentShape.height; r++) {
      for (let c = 0; c < this.currentShape.width; c++) {
        const key = `${r},${c}`;
        if (!this.validCells.has(key)) continue;
        const hole = document.createElement('div');
        hole.className = 'hole';
        if (this.selected === key) {
          hole.classList.add('selected');
        }
        const position = this.getHolePosition(r, c);
        hole.style.left = `${position.x * 100}%`;
        hole.style.top = `${position.y * 100}%`;
        hole.dataset.pos = key;
        if (this.pegs.has(key)) {
          const peg = document.createElement('div');
          peg.className = 'peg';
          hole.appendChild(peg);
        }
        hole.addEventListener('pointerdown', event => this.startPegDrag(event, key));
        hole.addEventListener('click', event => this.handleHoleClick(key, event));
        this.boardEl.appendChild(hole);
      }
    }
    this.updateWinState();
    if (this.pegs.size === 1) {
      this.setStatus(
        this.pegs.has(this.currentShape.empty) ? 'Perfect! Final peg in the center.' : 'Great! Only one peg left.',
      );
    } else if ([...this.validMoves()].length === 0) {
      this.setStatus('No moves left. Reset to try again.');
    }
  }

  private setStatus(text: string): void {
    this.statusEl.textContent = text;
  }

  private handleHoleClick(key: string, event: MouseEvent): void {
    if (performance.now() < this.skipClickUntil) {
      event.preventDefault();
      return;
    }
    this.onHoleClick(key);
  }

  private onHoleClick(key: string): void {
    if (!this.validCells.has(key)) return;
    if (this.pegs.has(key)) {
      this.selected = this.selected === key ? null : key;
      this.render();
      return;
    }
    if (!this.selected) return;
    const move = this.validateMove(this.selected, key);
    if (!move) return;
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

  private clearPickupPulse(): void {
    if (this.pickupPulseTimeout !== null) {
      window.clearTimeout(this.pickupPulseTimeout);
      this.pickupPulseTimeout = null;
    }
    this.boardEl.querySelectorAll('.hole.pickup-pulse').forEach(el => el.classList.remove('pickup-pulse'));
  }

  private startPegDrag(event: PointerEvent, key: string): void {
    if (!this.pegs.has(key)) return;
    if (!this.pickablePegs.has(key)) {
      this.pulsePickableHints();
      return;
    }
    this.draggingPegKey = key;
    this.dragTargetKey = key;
    this.dragMoved = false;
    this.draggingHole = this.getHoleElement(key);
    this.draggingHole?.classList.add('dragging');
    this.createGhostPeg(event);
    document.addEventListener('pointermove', this.boundDragMove);
    document.addEventListener('pointerup', this.boundDragEnd);
    event.preventDefault();
  }

  private handleDragMove(event: PointerEvent): void {
    if (!this.draggingPegKey) return;
    this.dragMoved = true;
    const targetKey = this.getHoleKeyFromPoint(event);
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
  }

  private handleDragEnd(): void {
    if (this.draggingPegKey && this.dragTargetKey && this.dragTargetKey !== this.draggingPegKey) {
      this.attemptMove(this.draggingPegKey, this.dragTargetKey);
    }
    this.draggingPegKey = null;
    this.dragTargetKey = null;
    this.draggingHole?.classList.remove('dragging');
    this.draggingHole = null;
    this.clearDragHover();
    this.removeGhostPeg();
    document.removeEventListener('pointermove', this.boundDragMove);
    document.removeEventListener('pointerup', this.boundDragEnd);
    if (this.dragMoved) {
      this.skipClickUntil = performance.now() + 200;
    }
  }

  private attemptMove(from: string, to: string): boolean {
    const move = this.validateMove(from, to);
    if (!move) return false;
    this.applyMove(from, to, move.jump);
    return true;
  }

  private applyMove(from: string, to: string, jump: string): void {
    this.pegs.delete(from);
    this.pegs.delete(jump);
    this.pegs.add(to);
    this.selected = null;
    this.moveMade = true;
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

  private getHoleKeyFromPoint(event: PointerEvent): string | null {
    let target: Element | null = document.elementFromPoint(event.clientX, event.clientY);
    while (target && target !== this.boardEl) {
      if (target instanceof HTMLElement && target.classList.contains('hole')) {
        break;
      }
      target = target.parentElement;
    }
    if (target instanceof HTMLElement && target.dataset.pos) {
      return target.dataset.pos;
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
    this.updateGhostPosition(event);
  }

  private updateGhostPosition(event: PointerEvent): void {
    if (!this.ghostPeg) return;
    const size = parseFloat(getComputedStyle(this.ghostPeg).width) || 40;
    this.ghostPeg.style.left = `${event.clientX - size / 2}px`;
    this.ghostPeg.style.top = `${event.clientY - size / 2}px`;
  }

  private removeGhostPeg(): void {
    if (!this.ghostPeg) return;
    this.ghostPeg.remove();
    this.ghostPeg = null;
  }

  private updateWinState(): void {
    const solved = this.pegs.size === 1 && this.pegs.has(this.currentShape.empty);
    this.solved = solved;
    this.boardWrapper.classList.toggle('solved', solved);
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

  private calculateHoleSize(): number {
    const rect = this.boardEl.getBoundingClientRect();
    const fallbackWidth = parseFloat(getComputedStyle(this.boardEl).width) || 320;
    const fallbackHeight = parseFloat(getComputedStyle(this.boardEl).height) || 320;
    const baseSize = Math.min(rect.width || fallbackWidth, rect.height || fallbackHeight);
    let effectiveWidth = this.currentShape.width;
    if (this.currentShape.layout === 'triangle') {
      const counts = [...this.rowHoleCount.values()];
      effectiveWidth = counts.length ? Math.max(...counts) : 1;
    }
    const maxDimension = Math.max(effectiveWidth, this.currentShape.height);
    const rawSize = baseSize / (maxDimension + 0.5);
    return Math.min(62, Math.max(32, rawSize));
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
    return {
      x: (col + 0.5) / this.currentShape.width,
      y: (row + 0.5) / this.currentShape.height,
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

}
