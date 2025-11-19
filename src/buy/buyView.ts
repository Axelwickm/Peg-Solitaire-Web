type Product = {
  title: string;
  link: string;
  image: string;
};

class ProductScroller {
  private container: HTMLElement;
  private track: HTMLElement;
  private products: Product[] = [];
  private cardMeta: Array<{ tileX: number; tileY: number }> = [];
  private cardPositions: Array<{ x: number; y: number }> = [];
  private cardElements: HTMLElement[] = [];
  private readonly tilesX = 5;
  private readonly tilesY = 5;
  private rafId: number | null = null;
  private lastFrame: number | null = null;
  private cameraX = 0;
  private cameraY = 0;
  private velocityX = 0;
  private velocityY = 0;
  private tileWidth = 1;
  private tileHeight = 1;
  private dragging = false;
  private lastPointerX = 0;
  private lastPointerY = 0;
  private lastPointerTime = 0;
  private clickStartX = 0;
  private clickStartY = 0;
  private activeClickCard: HTMLAnchorElement | null = null;

  private readonly onMove = (event: PointerEvent) => this.handlePointerMove(event);
  private readonly onUp = (event: PointerEvent) => this.handlePointerUp(event);

  constructor(container: HTMLElement, track: HTMLElement) {
    this.container = container;
    this.track = track;
    this.container.addEventListener('pointerdown', event => this.handlePointerDown(event));
    window.addEventListener('resize', () => this.layout());
  }

  public setProducts(products: Product[]): void {
    this.products = products;
    this.render();
    this.layout();
  }

  public start(): void {
    if (!this.products.length || this.rafId !== null) return;
    this.lastFrame = null;
    this.rafId = requestAnimationFrame(timestamp => this.tick(timestamp));
  }

  public stop(): void {
    if (this.rafId !== null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
  }

  private render(): void {
    this.track.innerHTML = '';
    this.cardMeta = [];
    this.cardPositions = [];
    this.cardElements = [];
    const base = this.products;
    for (let i = 0; i < this.tilesX * this.tilesY; i++) {
      const tileX = (i % this.tilesX) - 1;
      const tileY = Math.floor(i / this.tilesX) - 1;
      base.forEach(product => {
        const card = document.createElement('a');
        card.className = 'buy-card';
        card.href = product.link;
        card.target = '_blank';
        card.rel = 'noreferrer';

        const img = document.createElement('div');
        img.className = 'buy-image';
        img.style.backgroundImage = `url("${product.image}")`;

      const title = document.createElement('div');
      title.className = 'buy-title';
      title.textContent = product.title;

      card.appendChild(img);
      card.appendChild(title);
      card.style.setProperty('--card-scale', '1');
      this.track.appendChild(card);
      this.cardMeta.push({ tileX, tileY });
      this.cardElements.push(card);
    });
  }
  }

  private layout(): void {
    const cards = Array.from(this.track.children) as HTMLElement[];
    if (!cards.length || !this.products.length) return;
    const hexEdge = this.calculateHexEdge();
    this.container.style.setProperty('--hex-edge', `${hexEdge}px`);
    const hexWidth = hexEdge * 2;
    const hexHeight = hexEdge * Math.sqrt(3);
    const xSpacing = hexWidth * 0.75;
    const ySpacing = hexHeight;
    const cols = 4;
    const rows = Math.ceil(this.products.length / cols);
    this.tileWidth = cols * xSpacing;
    this.tileHeight = rows * ySpacing;
    this.track.style.width = `${this.tileWidth * this.tilesX}px`;
    this.track.style.height = `${this.tileHeight * this.tilesY}px`;
    this.cardPositions = [];
    cards.forEach((card, index) => {
      const baseIndex = index % this.products.length;
      const tile = this.cardMeta[index];
      const col = baseIndex % cols;
      const row = Math.floor(baseIndex / cols);
      const baseX = col * xSpacing;
      const baseY = row * ySpacing + (col % 2 === 1 ? ySpacing / 2 : 0);
      const x = baseX + (tile?.tileX ?? 0) * this.tileWidth;
      const y = baseY + (tile?.tileY ?? 0) * this.tileHeight;
      this.cardPositions.push({ x, y });
      card.style.transform = `translate(${x}px, ${y}px)`;
    });
    if (!this.cameraX && !this.cameraY) {
      this.cameraX = (this.tileWidth * this.tilesX) / 2;
      this.cameraY = (this.tileHeight * this.tilesY) / 2;
    }
    this.applyTransform();
  }

  private calculateHexEdge(): number {
    const minEdge = 120;
    const maxEdge = 170;
    const viewportEdge = window.innerWidth * 0.3;
    return Math.min(maxEdge, Math.max(minEdge, viewportEdge));
  }

  private tick(timestamp: number): void {
    if (this.lastFrame === null) {
      this.lastFrame = timestamp;
    }
    const dt = Math.min((timestamp - this.lastFrame) / 1000, 1 / 15);
    this.lastFrame = timestamp;

    this.cameraX += this.velocityX * dt;
    this.cameraY += this.velocityY * dt;
    this.velocityX *= Math.pow(0.9, dt * 60);
    this.velocityY *= Math.pow(0.9, dt * 60);
    this.applyTransform();

    if (Math.abs(this.velocityX) < 2 && Math.abs(this.velocityY) < 2 && !this.dragging) {
      this.stop();
      return;
    }

    this.rafId = requestAnimationFrame(ts => this.tick(ts));
  }

  private applyTransform(): void {
    if (!this.tileWidth || !this.tileHeight) return;
    const offsetX = ((this.cameraX % this.tileWidth) + this.tileWidth) % this.tileWidth;
    const offsetY = ((this.cameraY % this.tileHeight) + this.tileHeight) % this.tileHeight;
    this.track.style.transform = `translate(${-offsetX}px, ${-offsetY}px)`;
    if (!this.cardPositions.length || !this.cardElements.length) return;
    const centerX = this.container.clientWidth / 2;
    const centerY = this.container.clientHeight / 2;
    const wrapWidth = this.tileWidth * this.tilesX;
    const wrapHeight = this.tileHeight * this.tilesY;
    const maxDist = Math.hypot(centerX, centerY) || 1;
    this.cardPositions.forEach((pos, index) => {
      const card = this.cardElements[index];
      if (!card) return;
      let dx = pos.x - offsetX - centerX;
      let dy = pos.y - offsetY - centerY;
      if (dx > wrapWidth / 2) dx -= wrapWidth;
      if (dx < -wrapWidth / 2) dx += wrapWidth;
      if (dy > wrapHeight / 2) dy -= wrapHeight;
      if (dy < -wrapHeight / 2) dy += wrapHeight;
      const scale = 1;
      card.style.setProperty('--card-scale', `${scale}`);
      card.style.transform = `translate(${pos.x}px, ${pos.y}px) scale(${scale})`;
    });
  }

  private handlePointerDown(event: PointerEvent): void {
    if (!event.isPrimary) return;
    event.preventDefault();
    this.dragging = true;
    this.stop();
    this.lastPointerX = event.clientX;
    this.lastPointerY = event.clientY;
    this.lastPointerTime = event.timeStamp || performance.now();
    this.clickStartX = event.clientX;
    this.clickStartY = event.clientY;
    this.activeClickCard = (event.target as HTMLElement | null)?.closest('.buy-card') as
      | HTMLAnchorElement
      | null;
    this.container.setPointerCapture(event.pointerId);
    this.container.addEventListener('pointermove', this.onMove);
    this.container.addEventListener('pointerup', this.onUp);
    this.container.addEventListener('pointercancel', this.onUp);
  }

  private handlePointerMove(event: PointerEvent): void {
    if (!this.dragging) return;
    const now = event.timeStamp || performance.now();
    const deltaX = event.clientX - this.lastPointerX;
    const deltaY = event.clientY - this.lastPointerY;
    const dt = Math.max((now - this.lastPointerTime) / 1000, 1 / 120);
    this.cameraX -= deltaX;
    this.cameraY -= deltaY;
    this.velocityX = -(deltaX / dt);
    this.velocityY = -(deltaY / dt);
    this.lastPointerX = event.clientX;
    this.lastPointerY = event.clientY;
    this.lastPointerTime = now;
    this.applyTransform();
  }

  private handlePointerUp(event: PointerEvent): void {
    if (!this.dragging) return;
    this.dragging = false;
    this.container.releasePointerCapture(event.pointerId);
    this.container.removeEventListener('pointermove', this.onMove);
    this.container.removeEventListener('pointerup', this.onUp);
    this.container.removeEventListener('pointercancel', this.onUp);
    const dx = event.clientX - this.clickStartX;
    const dy = event.clientY - this.clickStartY;
    const dragDist = Math.hypot(dx, dy);
    const clickThreshold = 6;
    if (dragDist < clickThreshold && this.activeClickCard?.href) {
      window.open(this.activeClickCard.href, this.activeClickCard.target || '_blank', 'noopener');
    }
    this.activeClickCard = null;
    this.start();
  }
}

export function initBuyView(): { show: () => void; hide: () => void } {
  const panel = document.getElementById('buy-panel');
  const grid = document.getElementById('buy-grid');
  const track = document.getElementById('buy-track');
  const message = document.getElementById('buy-message');
  const status = document.getElementById('buy-status');

  if (!panel || !grid || !track || !message || !status) {
    return { show: () => {}, hide: () => {} };
  }

  const scroller = new ProductScroller(grid, track);
  let hasLoaded = false;
  let isLoading = false;

  const setMessage = (text: string | null): void => {
    if (!text) {
      message.classList.add('hidden');
      message.textContent = '';
      return;
    }
    message.classList.remove('hidden');
    message.textContent = text;
  };

  const setStatus = (text: string): void => {
    status.textContent = text;
  };

  const loadProducts = async (): Promise<void> => {
    if (hasLoaded || isLoading) return;
    isLoading = true;
    setMessage('Loading products…');
    setStatus('');
    try {
      const response = await fetch('./static/products.json', { cache: 'no-store' });
      if (response.status === 404) {
        setMessage('Store not available yet.');
        setStatus('');
        return;
      }
      if (!response.ok) {
        throw new Error(`Failed to load products: ${response.status}`);
      }
      const data = (await response.json()) as Product[] | unknown;
      const products = Array.isArray(data) ? data : [];
      const valid = products.filter(
        (item): item is Product =>
          typeof item === 'object' &&
          item !== null &&
          typeof (item as Product).title === 'string' &&
          typeof (item as Product).link === 'string' &&
          typeof (item as Product).image === 'string',
      );
      if (!valid.length) {
        setMessage('No products listed yet.');
        return;
      }
      scroller.setProducts(valid);
      hasLoaded = true;
      setMessage(null);
      setStatus('');
      scroller.start();
    } catch (error) {
      console.error(error);
      setMessage('Could not load the store.');
      setStatus('');
    } finally {
      isLoading = false;
    }
  };

  return {
    show: () => {
      panel.classList.add('visible');
      loadProducts().catch(() => {
        setMessage('Could not load the store.');
        setStatus('Offline');
      });
    },
    hide: () => {
      panel.classList.remove('visible');
      scroller.stop();
    },
  };
}
