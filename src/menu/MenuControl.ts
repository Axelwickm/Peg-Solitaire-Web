export class MenuControl {
  private panel: HTMLElement;
  private line: HTMLElement | null;
  private handle: HTMLElement | null;
  private items: HTMLElement[];
  private menuPositions: number[] = [];
  private handlePos = 0;
  private minHandlePos = 0;
  private maxHandlePos = 0;
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
  private readonly resizeHandler: () => void;
  private lastHighlightIndex = -1;
  private orientation: 'vertical' | 'horizontal' = 'vertical';
  private ticksContainer: HTMLElement | null = null;

  constructor(panel: Element) {
    this.panel = panel as HTMLElement;
    this.line = this.panel.querySelector('.menu-line');
    this.handle = this.panel.querySelector('.menu-handle');
    this.items = Array.from(this.panel.querySelectorAll('.menu-item'));
    if (this.line) {
      this.ticksContainer = document.createElement('div');
      this.ticksContainer.className = 'menu-line-ticks';
      this.ticksContainer.setAttribute('aria-hidden', 'true');
      this.line.appendChild(this.ticksContainer);
    }
    this.moveHandler = e => this.moveHandle(e);
    this.upHandler = () => this.endDrag();
    this.resizeHandler = () => this.updateGeometry();
    this.handle?.addEventListener('pointerdown', e => this.startDrag(e));
    window.addEventListener('resize', this.resizeHandler);
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
    const lineRect = this.line.getBoundingClientRect();
    this.orientation = lineRect.height >= lineRect.width ? 'vertical' : 'horizontal';
    const rawLineLength = this.orientation === 'vertical' ? lineRect.height : lineRect.width;
    const fallbackLineLength =
      this.orientation === 'vertical' ? this.line.offsetHeight : this.line.offsetWidth;
    const resolvedLineLength = rawLineLength || fallbackLineLength || 1;
    const lineStart =
      this.orientation === 'vertical'
        ? lineRect.top - panelRect.top
        : lineRect.left - panelRect.left;
    const lineEnd = lineStart + resolvedLineLength;
    if (!this.items.length) {
      this.menuPositions = [];
      return;
    }
    if (this.items.length === 1) {
      this.menuPositions = [lineStart + resolvedLineLength / 2];
    } else {
      const step = resolvedLineLength / (this.items.length - 1);
      this.menuPositions = this.items.map((_, index) => lineStart + step * index);
    }
    this.minHandlePos = lineStart;
    this.maxHandlePos = lineEnd;
    if (
      !this.handlePos ||
      this.handlePos < this.minHandlePos ||
      this.handlePos > this.maxHandlePos
    ) {
      this.handlePos = this.menuPositions[0] ?? (this.minHandlePos + this.maxHandlePos) / 2;
    }
    this.panel.classList.toggle('horizontal', this.orientation === 'horizontal');
    this.panel.classList.toggle('vertical', this.orientation === 'vertical');
    this.setHandlePosition(this.handlePos);
    this.renderTicks();
  }

  private setHandlePosition(pos: number): void {
    if (!this.handle) return;
    this.handlePos = Math.max(this.minHandlePos, Math.min(this.maxHandlePos, pos));
    if (this.orientation === 'vertical') {
      this.handle.style.top = `${this.handlePos}px`;
      this.handle.style.left = '';
    } else {
      this.handle.style.left = `${this.handlePos}px`;
      this.handle.style.top = '';
    }
    this.updateActiveMenu();
  }

  private updateActiveMenu(): number {
    if (!this.menuPositions.length) return Infinity;
    let nearestIndex = 0;
    let nearestDist = Infinity;
    this.menuPositions.forEach((pos, index) => {
      const dist = Math.abs(this.handlePos - pos);
      if (dist < nearestDist) {
        nearestDist = dist;
        nearestIndex = index;
      }
    });
    const highlightIndex = this.targetIndex ?? nearestIndex;
    this.items.forEach((item, index) => {
      item.classList.toggle('active', index === highlightIndex);
    });
    if (this.lastHighlightIndex !== highlightIndex) {
      this.lastHighlightIndex = highlightIndex;
      const menuName = this.items[highlightIndex]?.dataset.menu;
      const event = new CustomEvent('menu:activate', {
        bubbles: true,
        composed: true,
        detail: { menu: menuName },
      });
      this.panel.dispatchEvent(event);
    }
    this.activeIndex = highlightIndex;
    return nearestDist;
  }

  private renderTicks(): void {
    const container = this.ticksContainer;
    if (!container) return;
    container.innerHTML = '';
    if (this.orientation !== 'horizontal' || this.menuPositions.length === 0) {
      return;
    }
    const range = this.maxHandlePos - this.minHandlePos || 1;
    this.menuPositions.forEach(pos => {
      const tick = document.createElement('span');
      const ratio = range ? (pos - this.minHandlePos) / range : 0.5;
      tick.style.left = `${ratio * 100}%`;
      container.appendChild(tick);
    });
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
    const panelRect = this.panel.getBoundingClientRect();
    const targetPos =
      this.orientation === 'vertical'
        ? event.clientY - panelRect.top
        : event.clientX - panelRect.left;
    const now = event.timeStamp || performance.now();
    const dt = Math.max((now - this.lastPointerTime) / 1000, 1 / 120);
    this.velocity = (targetPos - this.handlePos) / dt;
    this.lastPointerTime = now;
    this.setHandlePosition(targetPos);
    if (this.targetIndex !== null && this.clickTargetIndex === null) {
      const pos = this.menuPositions[this.targetIndex] ?? this.handlePos;
      if (Math.abs(this.handlePos - pos) > 20) {
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
    this.handlePos += this.velocity * dt;
    if (this.handlePos < this.minHandlePos) {
      this.handlePos = this.minHandlePos;
      this.velocity *= -0.3;
    } else if (this.handlePos > this.maxHandlePos) {
      this.handlePos = this.maxHandlePos;
      this.velocity *= -0.3;
    }
    this.setHandlePosition(this.handlePos);
    const dist = Math.abs(this.handlePos - (this.menuPositions[this.activeIndex] ?? this.handlePos));
    if (
      this.targetIndex !== null &&
      this.clickTargetIndex === null &&
      Math.abs(this.handlePos - (this.menuPositions[this.targetIndex] ?? this.handlePos)) > 30
    ) {
      this.targetIndex = null;
      this.clickTargetIndex = null;
    }
    this.velocity *= Math.pow(0.92, dt * 60);
    const attractIndex = this.targetIndex ?? this.activeIndex;
    const attractTarget = this.menuPositions[attractIndex] ?? this.handlePos;
    const attractionStrength = 4;
    const distToTarget = Math.abs(this.handlePos - attractTarget);
    const magnetFactor = 1 + distToTarget / 15;
    this.velocity += (attractTarget - this.handlePos) * attractionStrength * magnetFactor * dt;
    if (distToTarget < 4 && attractIndex === this.activeIndex) {
      this.velocity = 0;
      this.setHandlePosition(attractTarget);
      this.clickTargetIndex = null;
      return;
    }
    this.inertiaId = requestAnimationFrame(ts => this.animateHandle(ts));
  }
}
