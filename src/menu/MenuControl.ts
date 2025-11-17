export class MenuControl {
  private panel: HTMLElement;
  private line: HTMLElement | null;
  private handle: HTMLElement | null;
  private items: HTMLElement[];
  private menuPositions: number[] = [];
  private handleY = 0;
  private minHandleY = 0;
  private maxHandleY = 0;
  private velocity = 0;
  private dragging = false;
  private inertiaId: number | null = null;
  private lastPointerTime = 0;
  private lastFrameTime: number | null = null;
  private activeIndex = 0;
  private targetIndex: number | null = 0;
  private clickTargetIndex: number | null = null;
  private readonly moveHandler: (event: PointerEvent) => void;
  private readonly upHandler: () => void;

  constructor(panel: Element) {
    this.panel = panel as HTMLElement;
    this.line = this.panel.querySelector('.menu-line');
    this.handle = this.panel.querySelector('.menu-handle');
    this.items = Array.from(this.panel.querySelectorAll('.menu-item'));
    this.moveHandler = e => this.moveHandle(e);
    this.upHandler = () => this.endDrag();
    this.handle?.addEventListener('pointerdown', e => this.startDrag(e));
    this.targetIndex = 0;
    this.items.forEach((item, index) => {
      item.addEventListener('click', e => {
        e.preventDefault();
        this.stopInertia();
        this.clickTargetIndex = index;
        this.targetIndex = index;
        this.updateActiveMenu();
        this.velocity = 0;
        this.startInertia();
      });
    });
    this.updateGeometry();
  }

  public updateGeometry(): void {
    if (!this.panel || !this.line) return;
    const panelRect = this.panel.getBoundingClientRect();
    this.menuPositions = this.items.map(item => {
      const rect = item.getBoundingClientRect();
      return rect.top + rect.height / 2 - panelRect.top;
    });
    this.minHandleY = this.line.offsetTop;
    this.maxHandleY = this.line.offsetTop + this.line.offsetHeight;
    if (!this.handleY) {
      this.handleY = this.menuPositions[0] ?? (this.minHandleY + this.maxHandleY) / 2;
    }
    this.setHandlePosition(this.handleY);
  }

  private setHandlePosition(y: number): void {
    if (!this.handle) return;
    this.handleY = Math.max(this.minHandleY, Math.min(this.maxHandleY, y));
    this.handle.style.top = `${this.handleY}px`;
    this.updateActiveMenu();
  }

  private updateActiveMenu(): number {
    if (!this.menuPositions.length) return Infinity;
    let nearestIndex = 0;
    let nearestDist = Infinity;
    this.menuPositions.forEach((pos, index) => {
      const dist = Math.abs(this.handleY - pos);
      if (dist < nearestDist) {
        nearestDist = dist;
        nearestIndex = index;
      }
    });
    const highlightIndex = this.targetIndex ?? nearestIndex;
    this.items.forEach((item, index) => {
      item.classList.toggle('active', index === highlightIndex);
    });
    this.activeIndex = highlightIndex;
    return nearestDist;
  }

  private startDrag(event: PointerEvent): void {
    if (!this.panel || !this.handle) return;
    this.dragging = true;
    this.stopInertia();
    this.handle.classList.add('dragging');
    this.lastPointerTime = event.timeStamp || performance.now();
    this.targetIndex = null;
    this.clickTargetIndex = null;
    document.addEventListener('pointermove', this.moveHandler);
    document.addEventListener('pointerup', this.upHandler);
    this.moveHandle(event);
    event.preventDefault();
  }

  private moveHandle(event: PointerEvent): void {
    if (!this.dragging || !this.panel) return;
    const panelTop = this.panel.getBoundingClientRect().top;
    const targetY = event.clientY - panelTop;
    const now = event.timeStamp || performance.now();
    const dt = Math.max((now - this.lastPointerTime) / 1000, 1 / 120);
    this.velocity = (targetY - this.handleY) / dt;
    this.lastPointerTime = now;
    this.setHandlePosition(targetY);
    if (this.targetIndex !== null && this.clickTargetIndex === null) {
      const pos = this.menuPositions[this.targetIndex] ?? this.handleY;
      if (Math.abs(this.handleY - pos) > 20) {
        this.targetIndex = null;
      }
    }
  }

  private endDrag(): void {
    this.dragging = false;
    this.handle?.classList.remove('dragging');
    document.removeEventListener('pointermove', this.moveHandler);
    document.removeEventListener('pointerup', this.upHandler);
    this.targetIndex = this.activeIndex;
    this.startInertia();
  }

  private startInertia(): void {
    this.stopInertia();
    this.lastFrameTime = null;
    this.inertiaId = requestAnimationFrame(timestamp => this.animateHandle(timestamp));
  }

  private stopInertia(): void {
    if (this.inertiaId) cancelAnimationFrame(this.inertiaId);
    this.inertiaId = null;
    this.lastFrameTime = null;
  }

  private animateHandle(timestamp: number): void {
    if (!this.menuPositions.length) return;
    if (this.lastFrameTime === null) {
      this.lastFrameTime = timestamp;
    }
    const dt = (timestamp - this.lastFrameTime) / 1000;
    this.lastFrameTime = timestamp;
    this.handleY += this.velocity * dt;
    if (this.handleY < this.minHandleY) {
      this.handleY = this.minHandleY;
      this.velocity *= -0.3;
    } else if (this.handleY > this.maxHandleY) {
      this.handleY = this.maxHandleY;
      this.velocity *= -0.3;
    }
    this.setHandlePosition(this.handleY);
    const dist = Math.abs(this.handleY - (this.menuPositions[this.activeIndex] ?? this.handleY));
    if (
      this.targetIndex !== null &&
      this.clickTargetIndex === null &&
      Math.abs(this.handleY - (this.menuPositions[this.targetIndex] ?? this.handleY)) > 30
    ) {
      this.targetIndex = null;
      this.clickTargetIndex = null;
    }
    this.velocity *= Math.pow(0.92, dt * 60);
    const attractIndex = this.targetIndex ?? this.activeIndex;
    const attractTarget = this.menuPositions[attractIndex] ?? this.handleY;
    const attractionStrength = 4;
    const distToTarget = Math.abs(this.handleY - attractTarget);
    const magnetFactor = 1 + distToTarget / 15;
    this.velocity += (attractTarget - this.handleY) * attractionStrength * magnetFactor * dt;
    if (distToTarget < 4 && attractIndex === this.activeIndex) {
      this.velocity = 0;
      this.setHandlePosition(attractTarget);
      this.clickTargetIndex = null;
      return;
    }
    this.inertiaId = requestAnimationFrame(ts => this.animateHandle(ts));
  }
}
