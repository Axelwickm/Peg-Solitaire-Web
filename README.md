# Kongming Chess (Peg Solitaire) Web

Classic Peg Solitaire (also known as Kongming Chess) brought to the browser with helper hints, timer tracking, solver visualization, and a tiny stats panel that logs your progress across shapes.

## Features

- Standard cross and triangle layouts with board-specific move validation.
- Timer with high-score highlight, perfect-run logging, and popup stats (graph, shape filter, CSV export).
- Helper hints, autoplay/auto-solve visualization, and installable PWA with service worker.

## Requirements

- Node.js 20+ with `npm` (or any compatible Node runtime)
- `esbuild` and `typescript` are bundled as dev dependencies.

## Run / Develop

```bash
npm install
npm run start
```

Runs a live dev server (esbuild `--serve`) at `http://localhost:8000` and watches your source files. Changes to `src/` and `index.html` auto-bundle in-memory.

## Build / Package

```bash
npm run build
```

Builds `src/main.ts` via esbuild, copies `index.html` into `dist/`, and runs `tsc --noEmit` as a sanity check. The output is ready for deployment (deploy `dist/` to Cloudflare Pages, GitHub Pages, etc.).

## Notes

- Assets like `icon.png`, `ball.webp`, and `board.webp` live in the project root and are referenced directly.
- The service worker caches nothing aggressively anymore but keeps installability; disable it during development via DevTools > Application if needed.
