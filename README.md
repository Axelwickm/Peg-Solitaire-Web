# Peg Solitaire Web

Classic Peg Solitaire (also known as Kongming Chess) brought to the browser, timer tracking, solver visualization, and a tiny stats panel that logs your progress across shapes.

[Play at pegsolitaire.fun](https://pegsolitaire.fun/)

![App preview](app.webp)

## Features

- Standard cross and triangle layouts with board-specific move validation.
- Timer with high-score highlight, perfect-run logging, and popup stats (graph, shape filter, CSV export).
- Autoplay/auto-solve visualization, and installable PWA with service worker.

## Requirements

- Node.js 20+ with `npm` (or any compatible Node runtime)
- `esbuild` and `typescript` are bundled as dev dependencies.

## Run / Develop

```bash
npm install
npm run start

```

Runs a live dev server (esbuild `--serve`) at `http://localhost:8000` and watches your source files.
