export class KongmingGame {
  private boardEl: HTMLElement;
  private statusEl: HTMLElement;
  private draggingPegKey: string | null = null;
  private dragTargetKey: string | null = null;
  private dragHoverKey: string | null = null;
  private dragMoved = false;
  private skipClickUntil = 0;
  private selected: string | null = null;
  private pegs = new Set<string>();
  private draggingHole: HTMLElement | null = null;
  private ghostPeg: HTMLDivElement | null = null;
  private validCells = new Set<string>();
  private boundDragMove: (event: PointerEvent) => void;
  private boundDragEnd: () => void;
  private boardWrapper: HTMLElement;
  private solved = false;

  constructor(boardEl: HTMLElement, statusEl: HTMLElement, boardWrapper: HTMLElement) {
    this.boardEl = boardEl;
    this.statusEl = statusEl;
    this.boardWrapper = boardWrapper;
    for (let r = 0; r < 7; r++) {
      for (let c = 0; c < 7; c++) {
        const cross = (r >= 2 && r <= 4) || (c >= 2 && c <= 4);
        if (cross) {
          this.validCells.add(`${r},${c}`);
        }
      }
    }
    this.boundDragMove = event => this.handleDragMove(event);
    this.boundDragEnd = () => this.handleDragEnd();
    this.setup();
  }

  public setup(): void {
    this.boardEl.innerHTML = '';
    this.pegs.clear();
    this.validCells.forEach(cell => {
      if (cell !== '3,3') {
        this.pegs.add(cell);
      }
    });
    this.selected = null;
    this.render();
    this.setStatus('Remove pegs until one remains.');
  }

  private render(): void {
    this.boardEl.innerHTML = '';
    for (let r = 0; r < 7; r++) {
      for (let c = 0; c < 7; c++) {
        const key = `${r},${c}`;
        const hole = document.createElement('div');
        hole.className = 'hole';
        if (!this.validCells.has(key)) {
          hole.classList.add('disabled');
        } else if (this.selected === key) {
          hole.classList.add('selected');
        }
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
      this.setStatus(this.pegs.has('3,3') ? 'Perfect! Final peg in the center.' : 'Great! Only one peg left.');
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
    this.pegs.delete(this.selected);
    this.pegs.delete(move.jump);
    this.pegs.add(key);
    this.selected = null;
    this.render();
  }

  private validateMove(from: string, to: string): { jump: string } | null {
    const [fr, fc] = from.split(',').map(Number);
    const [tr, tc] = to.split(',').map(Number);
    const dr = tr - fr;
    const dc = tc - fc;
    const isStraight = (dr === 0 && Math.abs(dc) === 2) || (dc === 0 && Math.abs(dr) === 2);
    if (!isStraight) return null;
    const mr = fr + dr / 2;
    const mc = fc + dc / 2;
    const jumpedKey = `${mr},${mc}`;
    if (!this.pegs.has(jumpedKey) || this.pegs.has(to)) return null;
    return { jump: jumpedKey };
  }

  private *validMoves(): Generator<{ from: string; to: string }> {
    for (const peg of this.pegs) {
      const [r, c] = peg.split(',').map(Number);
      const options: [number, number][] = [
        [r, c + 2],
        [r, c - 2],
        [r + 2, c],
        [r - 2, c],
      ];
      for (const [nr, nc] of options) {
        const key = `${nr},${nc}`;
        if (!this.validCells.has(key)) continue;
        if (this.validateMove(peg, key)) yield { from: peg, to: key };
      }
    }
  }

  private startPegDrag(event: PointerEvent, key: string): void {
    if (!this.pegs.has(key)) return;
    this.draggingPegKey = key;
    this.dragTargetKey = key;
    this.dragMoved = false;
    this.setDragHover(key);
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
    if (targetKey) {
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
    this.pegs.delete(from);
    this.pegs.delete(move.jump);
    this.pegs.add(to);
    this.selected = null;
    this.render();
    return true;
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
    let target = document.elementFromPoint(event.clientX, event.clientY) as HTMLElement | null;
    while (target && target !== this.boardEl && !target.classList.contains('hole')) {
      target = target.parentElement as HTMLElement | null;
    }
    if (target && target.dataset && target.dataset.pos) {
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
    const solved = this.pegs.size === 1 && this.pegs.has('3,3');
    this.solved = solved;
    this.boardWrapper.classList.toggle('solved', solved);
  }

  public forceWinState(): void {
    this.solved = true;
    this.boardWrapper.classList.add('solved');
  }
}
