type Product = {
  title: string;
  link: string;
  image: string;
};

class FeaturedProductCarousel {
  private panel: HTMLElement;
  private featureButton: HTMLButtonElement;
  private featureImage: HTMLElement;
  private featureTitle: HTMLElement;
  private carousel: HTMLElement;
  private track: HTMLElement;
  private statusEl: HTMLElement;
  private products: Product[] = [];
  private thumbnails: HTMLButtonElement[] = [];
  private activeIndex = 0;
  private swipeState:
    | { id: number; startX: number; startY: number; lastDx: number; swiping: boolean }
    | null = null;

  private readonly onSwipeMove = (event: PointerEvent) => this.handleSwipeMove(event);
  private readonly onSwipeEnd = (event: PointerEvent) => this.handleSwipeEnd(event);

  constructor(options: {
    panel: HTMLElement;
    featureButton: HTMLButtonElement;
    featureImage: HTMLElement;
    featureTitle: HTMLElement;
    carousel: HTMLElement;
    track: HTMLElement;
    status: HTMLElement;
  }) {
    this.panel = options.panel;
    this.featureButton = options.featureButton;
    this.featureImage = options.featureImage;
    this.featureTitle = options.featureTitle;
    this.carousel = options.carousel;
    this.track = options.track;
    this.statusEl = options.status;

    this.featureButton.addEventListener('click', () => this.openCurrentProduct());
    this.panel.addEventListener('pointerdown', event => this.handleSwipeStart(event));
  }

  public setProducts(products: Product[]): void {
    this.products = products;
    this.activeIndex = 0;
    this.renderThumbnails();
    this.updateFeature('auto');
  }

  private renderThumbnails(): void {
    this.track.innerHTML = '';
    this.thumbnails = [];
    this.products.forEach((product, index) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'buy-thumb';
      button.style.backgroundImage = `url("${product.image}")`;
      button.setAttribute('aria-label', product.title);
      button.addEventListener('click', event => {
        event.stopPropagation();
        this.setActiveIndex(index);
      });
      this.track.appendChild(button);
      this.thumbnails.push(button);
    });
  }

  private setActiveIndex(index: number, behavior: ScrollBehavior = 'smooth'): void {
    if (!this.products.length) return;
    const clamped = (index + this.products.length) % this.products.length;
    if (this.activeIndex === clamped) {
      this.ensureThumbInView(this.thumbnails[clamped], behavior);
      return;
    }
    this.activeIndex = clamped;
    this.updateFeature(behavior);
  }

  private updateFeature(behavior: ScrollBehavior): void {
    const product = this.products[this.activeIndex];
    if (!product) return;
    this.featureImage.style.backgroundImage = `url("${product.image}")`;
    this.featureTitle.textContent = product.title;
    this.thumbnails.forEach((thumb, thumbIndex) => {
      thumb.classList.toggle('active', thumbIndex === this.activeIndex);
    });
    this.ensureThumbInView(this.thumbnails[this.activeIndex], behavior);
  }

  private ensureThumbInView(button: HTMLButtonElement | undefined, behavior: ScrollBehavior): void {
    if (!button) return;
    const carouselRect = this.carousel.getBoundingClientRect();
    const buttonRect = button.getBoundingClientRect();
    const target =
      this.carousel.scrollLeft +
      buttonRect.left -
      carouselRect.left -
      carouselRect.width / 2 +
      buttonRect.width / 2;
    this.carousel.scrollTo({ left: target, behavior });
  }

  private openCurrentProduct(): void {
    const product = this.products[this.activeIndex];
    if (!product) return;
    window.open(product.link, '_blank', 'noopener');
  }

  private handleSwipeStart(event: PointerEvent): void {
    if (!event.isPrimary || this.products.length <= 1 || this.swipeState) return;
    this.swipeState = {
      id: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      lastDx: 0,
      swiping: false,
    };
    this.panel.addEventListener('pointermove', this.onSwipeMove);
    this.panel.addEventListener('pointerup', this.onSwipeEnd);
    this.panel.addEventListener('pointercancel', this.onSwipeEnd);
  }

  private handleSwipeMove(event: PointerEvent): void {
    if (!this.swipeState || event.pointerId !== this.swipeState.id) return;
    const dx = event.clientX - this.swipeState.startX;
    const dy = event.clientY - this.swipeState.startY;
    if (!this.swipeState.swiping) {
      if (Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 28) {
        this.swipeState.swiping = true;
        if (!this.panel.hasPointerCapture(event.pointerId)) {
          this.panel.setPointerCapture(event.pointerId);
        }
      } else {
        return;
      }
    }
    this.swipeState.lastDx = dx;
    event.preventDefault();
  }

  private handleSwipeEnd(event: PointerEvent): void {
    if (!this.swipeState || event.pointerId !== this.swipeState.id) return;
    if (this.panel.hasPointerCapture(event.pointerId)) {
      this.panel.releasePointerCapture(event.pointerId);
    }
    this.panel.removeEventListener('pointermove', this.onSwipeMove);
    this.panel.removeEventListener('pointerup', this.onSwipeEnd);
    this.panel.removeEventListener('pointercancel', this.onSwipeEnd);
    const state = this.swipeState;
    this.swipeState = null;
    if (!state.swiping || Math.abs(state.lastDx) < 10) return;
    if (state.lastDx > 0) {
      this.setActiveIndex(this.activeIndex - 1);
    } else if (state.lastDx < 0) {
      this.setActiveIndex(this.activeIndex + 1);
    }
    event.preventDefault();
  }
}

export function initBuyView(): { show: () => void; hide: () => void } {
  const panel = document.getElementById('buy-panel');
  const status = document.getElementById('buy-status');
  const featureButton = document.getElementById('buy-feature-button') as HTMLButtonElement | null;
  const featureImage = document.getElementById('buy-feature-image');
  const featureTitle = document.getElementById('buy-feature-title');
  const carousel = document.getElementById('buy-carousel');
  const track = document.getElementById('buy-carousel-track');
  const message = document.getElementById('buy-message');

  if (
    !panel ||
    !status ||
    !featureButton ||
    !featureImage ||
    !featureTitle ||
    !carousel ||
    !track ||
    !message
  ) {
    return { show: () => {}, hide: () => {} };
  }

  const carouselView = new FeaturedProductCarousel({
    panel,
    featureButton,
    featureImage,
    featureTitle,
    carousel,
    track,
    status,
  });

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

  const loadProducts = async (): Promise<void> => {
    if (hasLoaded || isLoading) return;
    isLoading = true;
    setMessage('Loading products…');
    status.textContent = 'Loading…';
    try {
      const response = await fetch('./static/products.json', { cache: 'no-store' });
      if (response.status === 404) {
        setMessage('Store not available yet.');
        status.textContent = '';
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
        status.textContent = '';
        featureTitle.textContent = 'Check back soon';
        featureImage.style.backgroundImage = '';
        return;
      }
      carouselView.setProducts(valid);
      hasLoaded = true;
      setMessage(null);
      status.textContent = '';
    } catch (error) {
      console.error(error);
      setMessage('Could not load the store.');
      status.textContent = 'Offline';
      featureTitle.textContent = 'Store unavailable';
      featureImage.style.backgroundImage = '';
    } finally {
      isLoading = false;
    }
  };

  return {
    show: () => {
      panel.classList.add('visible');
      loadProducts().catch(() => {
        setMessage('Could not load the store.');
        status.textContent = 'Offline';
      });
    },
    hide: () => {
      panel.classList.remove('visible');
    },
  };
}
