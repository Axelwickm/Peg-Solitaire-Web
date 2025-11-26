# Peg Solitaire Web

Classic Peg Solitaire (also known as Kongming Chess) brought to the browser, timer tracking, solver visualization, and a tiny stats panel that logs your progress across shapes.

[Play at pegsolitaire.fun](https://pegsolitaire.fun/)

![App preview](static/app.webp)

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

AI workflows live inside `ai/`; the training, inference, and export steps are detailed below.

## AI Training & Inference

1. (Optional) Create a Python virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Train a DQN checkpoint (defaults to the cross board but other shapes work via `--shape`; use `--load models/peg_solitaire_dqn.pt` to resume):
   ```bash
   python ai/train.py train --shape all --episodes 10000 --load models/peg_solitaire_dqn.pt
   ```
   This writes `peg_solitaire_dqn.pt` and shape-specific checkpoints to `models/`.
3. Run inference against a shape using the trained checkpoint (search is enabled unless you pass `--no-search`):
   ```bash
   python ai/train.py infer --shape german --load models/peg_solitaire_dqn.pt --episodes 100
   ```
   Use `--no-search` to skip the Markov search and take greedy actions.
4. Export browser-ready ONNX graphs for every board:
   ```bash
   python ai/train.py export --load models/peg_solitaire_dqn.pt --output static/models
   ```
   The solver now expects `static/models/peg_solitaire_dqn_<shape>.onnx`.
