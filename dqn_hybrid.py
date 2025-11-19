"""
Model-guided Kongming Chess solver/trainer.

Goals (reworked):
- Single, shape-agnostic attention model (no arch switches).
- Curriculum: start from easy endgames (few pegs) and expand.
- Hybrid inference: IDA* ordered by model Q-values.
- Hard-coded defaults live in one place (Defaults).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ida_solver import (
    ModelGuidedIdaStar,
    SearchMove,
    apply_action,
    legal_mask_for_state,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]

# --------------------------
# Shapes and board utilities
# --------------------------


def build_grid_axes(holes: List[str], width: int, height: int) -> List[List[str]]:
    hole_set = set(holes)
    axes: List[List[str]] = []
    for r in range(height):
        axis = [f"{r},{c}" for c in range(width) if f"{r},{c}" in hole_set]
        if axis:
            axes.append(axis)
    for c in range(width):
        axis = [f"{r},{c}" for r in range(height) if f"{r},{c}" in hole_set]
        if axis:
            axes.append(axis)
    return axes


def build_triangle_rows(width: int, height: int) -> List[List[str]]:
    rows: List[List[str]] = []
    for r in range(height):
        length = r + 1
        offset = (width - length) // 2
        rows.append([f"{r},{offset + c}" for c in range(length)])
    return rows


def collect_triangle_axes(rows: List[List[str]], dr: int, d_idx: int) -> List[List[str]]:
    axes: List[List[str]] = []

    def get(row: int, idx: int) -> str | None:
        if row < 0 or row >= len(rows):
            return None
        row_list = rows[row]
        if idx < 0 or idx >= len(row_list):
            return None
        return row_list[idx]

    for r, row in enumerate(rows):
        for i, _ in enumerate(row):
            if get(r - dr, i - d_idx):
                continue
            axis: List[str] = []
            rr, ii = r, i
            while True:
                key = get(rr, ii)
                if key is None:
                    break
                axis.append(key)
                rr += dr
                ii += d_idx
            if axis:
                axes.append(axis)
    return axes


def build_allowed_moves_from_axes(axes: List[List[str]]) -> Dict[str, Dict[str, str]]:
    moves: Dict[str, Dict[str, str]] = {}
    for axis in axes:
        for i in range(len(axis) - 2):
            a, b, c = axis[i], axis[i + 1], axis[i + 2]
            moves.setdefault(a, {})[c] = b
            moves.setdefault(c, {})[a] = b
    return moves


@dataclass
class ShapeDef:
    id: str
    holes: List[str]
    empty: str
    allowed: Dict[str, Dict[str, str]]
    width: int
    height: int


def create_shapes() -> Dict[str, ShapeDef]:
    # Cross
    cross_holes: List[str] = []
    for r in range(7):
        for c in range(7):
            if (2 <= r <= 4) or (2 <= c <= 4):
                cross_holes.append(f"{r},{c}")
    cross_axes = build_grid_axes(cross_holes, 7, 7)
    cross_moves = build_allowed_moves_from_axes(cross_axes)

    # Triangle
    tri_w, tri_h = 9, 5
    tri_rows = build_triangle_rows(tri_w, tri_h)
    tri_holes = [cell for row in tri_rows for cell in row]
    tri_axes = tri_rows + collect_triangle_axes(tri_rows, 1, 0) + collect_triangle_axes(
        tri_rows, 1, 1
    )
    tri_moves = build_allowed_moves_from_axes(tri_axes)

    return {
        "cross": ShapeDef(
            id="cross", holes=cross_holes, empty="3,3", allowed=cross_moves, width=7, height=7
        ),
        "triangle": ShapeDef(
            id="triangle",
            holes=tri_holes,
            empty="0,4",
            allowed=tri_moves,
            width=9,
            height=5,
        ),
    }


SHAPES = create_shapes()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# Environment
# --------------------------


class KongmingEnv:
    def __init__(
        self,
        shape: ShapeDef,
    ):
        self.shape = shape
        self.holes = shape.holes
        self.empty = shape.empty
        self.allowed = shape.allowed
        self.idx_map: Dict[str, int] = {h: i for i, h in enumerate(self.holes)}
        self.actions = self._build_actions()
        self.state = np.zeros(len(self.holes), dtype=np.bool_)
        self.reset_full()

    def _build_actions(self) -> np.ndarray:
        arr = []
        for frm, dests in self.allowed.items():
            for to, jump in dests.items():
                arr.append([self.idx_map[frm], self.idx_map[to], self.idx_map[jump]])
        return np.array(arr, dtype=np.int32)

    def reset_full(self):
        self.state[:] = False
        self.state[list(self.idx_map.values())] = True
        if self.empty in self.idx_map:
            self.state[self.idx_map[self.empty]] = False
        return self.obs()

    def reset_with_pegs(self, target_pegs: int):
        # Try to construct a start state by reversing legal moves from the solved position.
        generated = self._generate_backward_state(target_pegs)
        if generated is not None:
            self.state = generated
            return self.obs()

        # Fallback: random removals from a full board.
        self.reset_full()
        current_pegs = int(self.state.sum())
        to_remove = max(0, current_pegs - target_pegs)
        if to_remove > 0:
            removable = list(np.where(self.state)[0])
            drop = random.sample(removable, k=min(to_remove, len(removable)))
            self.state[drop] = False
        return self.obs()

    def _generate_backward_state(self, target_pegs: int, max_attempts: int = 64) -> np.ndarray | None:
        if target_pegs <= 1:
            # Solved state: single peg at the target.
            state = np.zeros_like(self.state)
            state[self.idx_map[self.empty]] = True
            return state

        for _ in range(max_attempts):
            state = np.zeros_like(self.state)
            state[self.idx_map[self.empty]] = True
            steps = 0
            # Reverse moves until we reach the desired peg count or run out of options.
            while state.sum() < target_pegs:
                reverse_moves = self._reverse_legal_moves(state)
                if not reverse_moves:
                    break
                move_idx = random.randrange(len(reverse_moves))
                frm, to, jump, action_idx = reverse_moves[move_idx]
                # Reverse application: place pegs on from and jump, clear to.
                state[frm] = True
                state[jump] = True
                state[to] = False
                steps += 1
                if steps > target_pegs * 4:  # avoid runaway loops
                    break
            if state.sum() == target_pegs:
                return state
        return None

    def _reverse_legal_moves(self, state: np.ndarray) -> List[Tuple[int, int, int, int]]:
        moves: List[Tuple[int, int, int, int]] = []
        for idx, (frm, to, jump) in enumerate(self.actions):
            if state[to] and (not state[frm]) and (not state[jump]):
                moves.append((frm, to, jump, idx))
        return moves

    def obs(self) -> torch.Tensor:
        return torch.from_numpy(self.state.astype(np.float32))

    def legal_mask_numpy(self) -> np.ndarray:
        return legal_mask_for_state(self.state, self.actions)

    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool]:
        frm, to, jump = self.actions[action_idx]
        if not (self.state[frm] and (not self.state[to]) and self.state[jump]):
            # illegal move: stop early
            return self.obs(), -1.0, True
        self.state[frm] = False
        self.state[jump] = False
        self.state[to] = True
        mask = self.legal_mask_numpy()
        done = (self.state.sum() == 1) or (mask.sum() == 0)
        reward = 0.0
        if done:
            reward = compute_final_reward(self, center_bonus=True)
        return self.obs(), reward, done


# --------------------------
# Model
# --------------------------


def normalize_coords(coords: List[Tuple[int, int]], width: int, height: int) -> torch.Tensor:
    if width <= 1:
        width = 2
    if height <= 1:
        height = 2
    normed = []
    for r, c in coords:
        r_n = (r / (height - 1)) * 2 - 1
        c_n = (c / (width - 1)) * 2 - 1
        normed.append([r_n, c_n])
    return torch.tensor(normed, dtype=torch.float32)


@dataclass
class ShapeContext:
    shape: ShapeDef
    idx_map: Dict[str, int]
    actions: torch.Tensor  # (A, 3)
    coords: torch.Tensor  # (H, 2) normalized
    shape_idx: int


def build_shape_contexts(shapes: Dict[str, ShapeDef]) -> Dict[str, ShapeContext]:
    ctxs: Dict[str, ShapeContext] = {}
    for idx, shape in enumerate(shapes.values()):
        idx_map = {h: i for i, h in enumerate(shape.holes)}
        actions = []
        for frm, dests in shape.allowed.items():
            for to, jump in dests.items():
                actions.append([idx_map[frm], idx_map[to], idx_map[jump]])
        coords = normalize_coords(
            [tuple(map(int, h.split(","))) for h in shape.holes], shape.width, shape.height
        )
        ctxs[shape.id] = ShapeContext(
            shape=shape,
            idx_map=idx_map,
            actions=torch.tensor(actions, dtype=torch.long),
            coords=coords,
            shape_idx=idx,
        )
    return ctxs


class PegAttentionQ(nn.Module):
    def __init__(
        self,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 192,
        dropout: float = 0.1,
        num_shapes: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.shape_embed = nn.Embedding(num_shapes, d_model)
        self.token_norm = nn.LayerNorm(d_model)
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, obs: torch.Tensor, ctx: ShapeContext) -> torch.Tensor:
        # obs: (B, H)
        batch = obs.shape[0]
        coords = ctx.coords.to(obs.device)  # (H, 2)
        coords = coords.unsqueeze(0).expand(batch, -1, -1)
        shape_emb = self.shape_embed(
            torch.tensor([ctx.shape_idx], device=obs.device, dtype=torch.long)
        ).view(1, 1, -1)
        shape_emb = shape_emb.expand(batch, coords.shape[1], -1)

        features = torch.cat([obs.unsqueeze(-1), coords], dim=-1)
        tokens = self.input_proj(features) + shape_emb
        tokens = self.encoder(tokens)
        tokens = self.token_norm(tokens)  # (B, H, D)

        actions = ctx.actions.to(obs.device)  # (A, 3)
        from_tok = tokens[:, actions[:, 0]]
        to_tok = tokens[:, actions[:, 1]]
        jump_tok = tokens[:, actions[:, 2]]
        act_feat = torch.cat([from_tok, to_tok, jump_tok], dim=-1)
        q_values = self.action_head(act_feat).squeeze(-1)  # (B, A)
        return q_values


# --------------------------
# Replay buffer with n-step
# --------------------------


@dataclass
class Transition:
    obs: torch.Tensor
    target: torch.Tensor
    legal_mask: torch.Tensor


class TeacherBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Transition | None] = [None] * capacity
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def push(self, transition: Transition):
        self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int) -> List[Transition]:
        n = len(self)
        if n == 0:
            return []
        indices = np.random.choice(n, size=min(batch_size, n), replace=False)
        result = []
        for i in indices:
            t = self.buffer[i]
            if t is not None:
                result.append(t)
        return result


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Transition | None] = [None] * capacity
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def push(self, transition: Transition):
        self.buffer[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int) -> List[Transition]:
        n = len(self)
        if n == 0:
            return []
        indices = np.random.choice(n, size=min(batch_size, n), replace=False)
        result = []
        for i in indices:
            t = self.buffer[i]
            if t is not None:
                result.append(t)
        return result


# --------------------------
# Utilities
# --------------------------


def compute_final_reward(env: KongmingEnv, center_bonus: bool = True) -> float:
    pegs = int(env.state.sum())
    reward = -float(pegs)
    center_idx = env.idx_map.get(env.empty)
    if center_bonus and center_idx is not None and env.state[center_idx]:
        reward += 5.0
    return reward


def build_targets_from_search(
    env: KongmingEnv,
    ctx: ShapeContext,
    device: torch.device,
    max_nodes: int,
    solved_bonus: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    legal_mask = legal_mask_for_state(env.state, env.actions)
    legal_indices = np.where(legal_mask)[0]
    act_dim = len(env.actions)
    target = torch.zeros(act_dim, dtype=torch.float32)
    if legal_indices.size == 0:
        return target, torch.zeros_like(target, dtype=torch.bool), {"best_pegs": float(env.state.sum())}

    best_pegs_overall = float(env.state.sum())
    for idx in legal_indices:
        next_state = apply_action(env.state, env.actions, int(idx))
        solver = ModelGuidedIdaStar(
            env, None, ctx, device, epsilon=0.0, max_nodes=max_nodes
        )
        res = solver.solve(next_state)
        score = -float(res["best_pegs"])
        if res["solved"]:
            score += solved_bonus
        target[idx] = score
        best_pegs_overall = min(best_pegs_overall, float(res["best_pegs"]))

    return target, torch.from_numpy(legal_mask), {"best_pegs": best_pegs_overall}


def linear_decay(start: float, end: float, t: int, total: int) -> float:
    if total <= 0:
        return end
    ratio = min(1.0, max(0.0, t / total))
    return start + (end - start) * ratio


# --------------------------
# Curriculum
# --------------------------


class Curriculum:
    def __init__(
        self,
        min_pegs: int,
        start_max: int,
        hard_cap: int,
        bump_epochs: int,
        threshold: float,
    ):
        self.min_pegs = min_pegs
        self.current_max = start_max
        self.hard_cap = hard_cap
        self.bump_epochs = bump_epochs
        self.threshold = threshold
        self.below_count = 0

    def sample_target(self) -> int:
        return random.randint(self.min_pegs, self.current_max)

    def update(self, mean_final_pegs: float) -> bool:
        if mean_final_pegs < self.threshold:
            self.below_count += 1
            if self.below_count >= self.bump_epochs and self.current_max < self.hard_cap:
                self.current_max = min(self.current_max + 1, self.hard_cap)
                self.below_count = 0
                return True
            return False
        self.below_count = 0
        return False


# --------------------------
# Training
# --------------------------


@dataclass
class Defaults:
    shape: str = "cross"
    epochs: int = 2000
    steps_per_epoch: int = 256
    batch_size: int = 512
    buffer_size: int = 200_000
    lr: float = 1e-4
    random_remove: int = 6
    start_pegs_min: int = 2
    start_pegs_max: int = 6
    hard_cap_pegs: int = 32
    curriculum_bump_epochs: int = 10
    curriculum_threshold: float = 1.2
    teacher_nodes: int = 50_000
    infer_games: int = 1
    search_nodes: int = 1_000_000
    search_epsilon: float = 0.0
    logdir: str = "runs"
    model_dir: str = "models"
    run_name: str | None = None


DEFAULTS = Defaults()


def compute_q_loss(
    batch: List[Transition],
    model: PegAttentionQ,
    target_model: PegAttentionQ,
    ctx: ShapeContext,
    gamma: float,
    n_step: int,
    device: torch.device,
) -> torch.Tensor:
    # Kept for reference; not used in teacher-guided training.
    obs_batch = torch.stack([t.obs for t in batch]).to(device)  # (B, H)
    act_batch = torch.tensor([0 for _ in batch], device=device, dtype=torch.long)
    rew_batch = torch.tensor([0.0 for _ in batch], device=device, dtype=torch.float32)
    next_obs_batch = torch.stack([t.obs for t in batch]).to(device)
    done_batch = torch.tensor([False for _ in batch], device=device, dtype=torch.bool)
    legal_next = torch.stack([t.legal_mask for t in batch]).to(device)
    q_values = model(obs_batch, ctx)
    q_taken = q_values.gather(1, act_batch.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        target_q = target_model(next_obs_batch, ctx)
        target_q = target_q.masked_fill(~legal_next, -1e9)
        max_next = target_q.max(dim=1).values
        target = rew_batch + ((gamma**n_step) * max_next * (~done_batch))
    return nn.functional.smooth_l1_loss(q_taken, target)


def compute_supervised_loss(
    batch: List[Transition],
    model: PegAttentionQ,
    ctx: ShapeContext,
    device: torch.device,
) -> torch.Tensor | None:
    if not batch:
        return None
    obs_batch = torch.stack([t.obs for t in batch]).to(device)
    target_batch = torch.stack([t.target for t in batch]).to(device)
    mask_batch = torch.stack([t.legal_mask for t in batch]).to(device).bool()
    preds = model(obs_batch, ctx)
    if not mask_batch.any():
        return None
    pred_sel = torch.masked_select(preds, mask_batch)
    target_sel = torch.masked_select(target_batch, mask_batch)
    return nn.functional.smooth_l1_loss(pred_sel, target_sel)


def train(
    defaults: Defaults,
    device: torch.device,
    writer: SummaryWriter | None,
    save_path: str,
):
    shape_contexts = build_shape_contexts(SHAPES)
    model = PegAttentionQ(num_shapes=len(SHAPES)).to(device)
    opt = optim.AdamW(model.parameters(), lr=defaults.lr)

    buffers = {sid: TeacherBuffer(defaults.buffer_size) for sid in SHAPES}
    curriculum = Curriculum(
        min_pegs=defaults.start_pegs_min,
        start_max=defaults.start_pegs_max,
        hard_cap=defaults.hard_cap_pegs,
        bump_epochs=defaults.curriculum_bump_epochs,
        threshold=defaults.curriculum_threshold,
    )

    global_step = 0
    for epoch in range(defaults.epochs):
        epoch_losses: List[float] = []
        epoch_final_pegs: List[int] = []
        epoch_start_pegs: List[int] = []
        for _ in range(defaults.steps_per_epoch):
            shape_id = random.choice(list(SHAPES.keys()))
            shape = SHAPES[shape_id]
            ctx = shape_contexts[shape_id]
            env = KongmingEnv(shape)

            target_pegs = curriculum.sample_target()
            obs = env.reset_with_pegs(target_pegs)
            epoch_start_pegs.append(int(env.state.sum()))
            target, legal_mask, stats = build_targets_from_search(
                env, ctx, device, defaults.teacher_nodes
            )
            if legal_mask.sum() == 0:
                continue
            buffers[shape_id].push(Transition(obs, target, legal_mask.bool()))
            epoch_final_pegs.append(int(stats["best_pegs"]))
            global_step += 1

            batch = buffers[shape_id].sample(defaults.batch_size)
            loss = compute_supervised_loss(batch, model, ctx, device)
            if loss is not None:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                epoch_losses.append(loss.item())

        mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        mean_final_pegs = (
            sum(epoch_final_pegs) / len(epoch_final_pegs) if epoch_final_pegs else 0.0
        )
        mean_start_pegs = (
            sum(epoch_start_pegs) / len(epoch_start_pegs) if epoch_start_pegs else 0.0
        )
        total_buffer = sum(len(buf) for buf in buffers.values())
        bumped = curriculum.update(mean_final_pegs)
        print(
            f"[epoch {epoch + 1}] loss={mean_loss:.4f} "
            f"start_pegs={mean_start_pegs:.2f} final_pegs={mean_final_pegs:.2f} "
            f"cap={curriculum.current_max} buffer={total_buffer}",
            flush=True,
        )
        if writer is not None:
            writer.add_scalar("loss", mean_loss, epoch)
            writer.add_scalar("final_pegs", mean_final_pegs, epoch)
            writer.add_scalar("start_pegs", mean_start_pegs, epoch)
            writer.add_scalar("buffer_size", total_buffer, epoch)
            writer.add_scalar("start_cap", curriculum.current_max, epoch)
        if bumped:
            print(f"🛠️  Increased start peg cap to {curriculum.current_max} after stable final pegs.")
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), save_path)
            print(f"[epoch {epoch+1}] checkpoint saved to {save_path}", flush=True)
    torch.save(model.state_dict(), save_path)
    print(f"Training complete, model saved to {save_path}", flush=True)
    return model


# --------------------------
# Model-guided IDA* search
# --------------------------


@dataclass
class SearchMove:
    frm: int
    to: int
    jump: int
    action_idx: int


@dataclass
class SearchFrame:
    state: np.ndarray
    key: str
    neighbors: List[SearchMove]
    idx: int
    g: int
    move: SearchMove | None


class ModelGuidedIdaStar:
    def __init__(
        self,
        env: KongmingEnv,
        model: PegAttentionQ | None,
        ctx: ShapeContext,
        device: torch.device,
        epsilon: float = 0.0,
        max_nodes: int | None = None,
    ):
        self.env = env
        self.model = model
        self.ctx = ctx
        self.device = device
        self.epsilon = epsilon
        self.max_nodes = max_nodes
        self.target_idx = env.idx_map[env.empty]
        self.legal_buf = np.empty(len(env.actions), dtype=np.int32)
        self.dist_to_target = self._compute_distances()

    def _compute_distances(self) -> Dict[int, int]:
        adjacency: Dict[int, List[int]] = {}
        for frm, dests in self.env.allowed.items():
            frm_idx = self.env.idx_map[frm]
            for to in dests:
                to_idx = self.env.idx_map[to]
                adjacency.setdefault(frm_idx, []).append(to_idx)
                adjacency.setdefault(to_idx, []).append(frm_idx)

        dist: Dict[int, int] = {self.target_idx: 0}
        queue = deque([self.target_idx])
        while queue:
            current = queue.popleft()
            base = dist[current]
            for neighbor in adjacency.get(current, []):
                if neighbor not in dist:
                    dist[neighbor] = base + 1
                    queue.append(neighbor)
        return dist

    def _serialize(self, state: np.ndarray) -> str:
        inds = np.where(state)[0]
        return ";".join(map(str, np.sort(inds)))

    def _heuristic(self, state: np.ndarray) -> int:
        peg_indices = np.where(state)[0]
        peg_count = len(peg_indices)
        max_dist = 0
        total_dist = 0
        reachable = True
        for idx in peg_indices:
            dist = self.dist_to_target.get(int(idx))
            if dist is None:
                reachable = False
                continue
            max_dist = max(max_dist, dist)
            total_dist += dist
        if not reachable:
            return peg_count + max_dist + 5
        span_penalty = math.ceil(max_dist / 2)
        spread_penalty = math.ceil(total_dist / max(1, peg_count * 2))
        return max(peg_count - 1, span_penalty, spread_penalty)

    def _legal_moves(self, state: np.ndarray) -> List[SearchMove]:
        legal = self.env.legal_mask_numpy()
        moves: List[SearchMove] = []
        for idx, ok in enumerate(legal):
            if not ok:
                continue
            frm, to, jump = self.env.actions[idx]
            moves.append(SearchMove(int(frm), int(to), int(jump), idx))
        return moves

    def _order_moves(self, state: np.ndarray, moves: List[SearchMove]) -> List[SearchMove]:
        if not moves:
            return moves
        if self.model is None or self.device is None:
            return moves
        if self.epsilon > 0 and random.random() < self.epsilon:
            random.shuffle(moves)
            return moves
        obs_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.model(obs_tensor, self.ctx).cpu().numpy().squeeze(0)
        return sorted(moves, key=lambda m: q_vals[m.action_idx], reverse=True)

    def solve(self, initial_state: np.ndarray) -> Dict[str, object]:
        start_state = initial_state.astype(np.bool_, copy=True)
        bound = self._heuristic(start_state)
        next_bound = math.inf
        best_path: List[SearchMove] = []
        best_pegs = int(start_state.sum())
        solved_path: List[SearchMove] | None = None
        nodes = 0
        path: List[SearchMove] = []
        seen: set[str] = set()
        stack: List[SearchFrame] = []
        start_time = time.time()

        def reset_stack():
            seen.clear()
            path.clear()
            stack.clear()
            root_state = start_state.copy()
            key = self._serialize(root_state)
            seen.add(key)
            neighbors = self._order_moves(root_state, self._legal_moves(root_state))
            stack.append(SearchFrame(root_state, key, neighbors, 0, 0, None))

        reset_stack()
        limit_hit = False
        while True:
            if not stack:
                if math.isinf(next_bound):
                    break
                bound = next_bound
                next_bound = math.inf
                reset_stack()
                continue

            frame = stack[-1]
            nodes += 1
            if self.max_nodes is not None and nodes >= self.max_nodes:
                limit_hit = True
                break

            h = self._heuristic(frame.state)
            f = frame.g + h
            peg_total = int(frame.state.sum())

            if f > bound:
                next_bound = min(next_bound, f)
                stack.pop()
                if frame.move:
                    path.pop()
                seen.discard(frame.key)
                continue

            if peg_total < best_pegs:
                best_pegs = peg_total
                best_path = list(path)

            if peg_total == 1 and frame.state[self.target_idx]:
                solved_path = list(path)
                best_path = list(path)
                break

            if frame.idx >= len(frame.neighbors):
                stack.pop()
                if frame.move:
                    path.pop()
                seen.discard(frame.key)
                continue

            move = frame.neighbors[frame.idx]
            frame.idx += 1
            next_state = frame.state.copy()
            frm, to, jump = self.env.actions[move.action_idx]
            next_state[frm] = False
            next_state[jump] = False
            next_state[to] = True
            key = self._serialize(next_state)
            if key in seen:
                continue
            seen.add(key)
            path.append(move)
            neighbors = self._order_moves(next_state, self._legal_moves(next_state))
            stack.append(SearchFrame(next_state, key, neighbors, 0, frame.g + 1, move))

        duration_ms = (time.time() - start_time) * 1000
        return {
            "solved": solved_path is not None,
            "moves": solved_path or [],
            "best_moves": solved_path or best_path,
            "nodes_explored": nodes,
            "duration_ms": duration_ms,
            "best_pegs": best_pegs,
            "limit_hit": limit_hit,
        }


def move_to_str(move: SearchMove, idx_to_hole: Dict[int, str]) -> str:
    return f"{idx_to_hole[move.frm]} -> {idx_to_hole[move.to]} (over {idx_to_hole[move.jump]})"


def render_state(env: KongmingEnv):
    if env.shape.id == "cross":
        size = env.shape.width
        grid = [[" " for _ in range(size)] for _ in range(size)]
        hole_set = set(env.holes)
        for hole in hole_set:
            r, c = map(int, hole.split(","))
            idx = env.idx_map[hole]
            symbol = "#" if env.state[idx] else "."
            grid[r][c] = symbol
        print("\n".join("".join(row) for row in grid))
    else:
        rows = build_triangle_rows(env.shape.width, env.shape.height)
        for row in rows:
            line = ""
            for hole in row:
                idx = env.idx_map[hole]
                line += "#" if env.state[idx] else "."
            print(line.center(15))
    print("-" * 25)


def inference_hybrid(
    model_path: str,
    defaults: Defaults,
    device: torch.device,
    start_pegs: int | None = None,
):
    shape_contexts = build_shape_contexts(SHAPES)
    model = PegAttentionQ(num_shapes=len(SHAPES)).to(device)
    if not os.path.exists(model_path):
        print(f"No model at {model_path}, train first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    shape = SHAPES[defaults.shape]
    ctx = shape_contexts[defaults.shape]
    env = KongmingEnv(shape)
    solver = ModelGuidedIdaStar(
        env,
        model,
        ctx,
        device,
        epsilon=defaults.search_epsilon,
        max_nodes=defaults.search_nodes if defaults.search_nodes > 0 else None,
    )

    start_pegs_val = start_pegs or defaults.hard_cap_pegs

    for game in range(defaults.infer_games):
        env.reset_with_pegs(start_pegs_val)
        start_state = env.state.copy()
        initial_pegs = int(start_state.sum())
        print(f"=== Hybrid game {game + 1} | start pegs: {initial_pegs} ===")
        result = solver.solve(start_state)
        best_moves: List[SearchMove] = result["best_moves"]
        print(
            f"nodes={result['nodes_explored']} "
            f"time={result['duration_ms']:.1f}ms best_pegs={result['best_pegs']}",
            flush=True,
        )
        if result["limit_hit"]:
            print("Stopped after hitting search node limit.")
        if not best_moves:
            print("No moves available from this state.\n")
            continue
        env.state = start_state.copy()
        idx_to_hole = {idx: hole for hole, idx in env.idx_map.items()}
        for step, move in enumerate(best_moves, 1):
            print(f"Step {step}: {move_to_str(move, idx_to_hole)}")
            env.step(move.action_idx)
        render_state(env)
        if result["solved"]:
            print("Solved with model-guided IDA*.\n")
        else:
            remaining = int(env.state.sum())
            print(f"Best found path leaves {remaining} pegs.\n")


# --------------------------
# CLI
# --------------------------


def add_common_args(subparser: argparse.ArgumentParser):
    subparser.add_argument("--shape", choices=list(SHAPES.keys()), default=DEFAULTS.shape)
    subparser.add_argument("--model-dir", default=DEFAULTS.model_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model-guided Kongming Chess")
    subparsers = parser.add_subparsers(dest="command")
    parser.set_defaults(command="train")

    # Train
    train_parser = subparsers.add_parser("train", help="Train the model")
    add_common_args(train_parser)
    train_parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    train_parser.add_argument("--steps-per-epoch", type=int, default=DEFAULTS.steps_per_epoch)
    train_parser.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    train_parser.add_argument("--buffer-size", type=int, default=DEFAULTS.buffer_size)
    train_parser.add_argument("--lr", type=float, default=DEFAULTS.lr)
    train_parser.add_argument("--start-pegs-min", type=int, default=DEFAULTS.start_pegs_min)
    train_parser.add_argument("--start-pegs-max", type=int, default=DEFAULTS.start_pegs_max)
    train_parser.add_argument("--hard-cap-pegs", type=int, default=DEFAULTS.hard_cap_pegs)
    train_parser.add_argument(
        "--bump-epochs",
        type=int,
        default=DEFAULTS.curriculum_bump_epochs,
        help="Consecutive epochs of low final pegs before raising start peg cap.",
    )
    train_parser.add_argument(
        "--teacher-nodes",
        type=int,
        default=DEFAULTS.teacher_nodes,
        help="Node budget for teacher search when building targets.",
    )
    train_parser.add_argument("--logdir", default=DEFAULTS.logdir)
    train_parser.add_argument("--run-name", default=None)

    # Inference
    infer_parser = subparsers.add_parser("infer", help="Run model-guided IDA* search")
    add_common_args(infer_parser)
    infer_parser.add_argument("--games", type=int, default=DEFAULTS.infer_games)
    infer_parser.add_argument("--search-nodes", type=int, default=DEFAULTS.search_nodes)
    infer_parser.add_argument("--search-epsilon", type=float, default=DEFAULTS.search_epsilon)
    infer_parser.add_argument(
        "--start-pegs",
        type=int,
        default=None,
        help="Override starting peg count for inference (defaults to full board).",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}", flush=True)
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "shared.pt")

    if args.command == "infer":
        defaults = Defaults(
            shape=args.shape,
            infer_games=args.games,
            search_nodes=args.search_nodes,
            search_epsilon=args.search_epsilon,
            hard_cap_pegs=DEFAULTS.hard_cap_pegs,
            model_dir=args.model_dir,
        )
        inference_hybrid(model_path, defaults, device, start_pegs=args.start_pegs)
        return

    defaults = Defaults(
        shape=args.shape,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        lr=args.lr,
        start_pegs_min=args.start_pegs_min,
        start_pegs_max=args.start_pegs_max,
        hard_cap_pegs=args.hard_cap_pegs,
        curriculum_bump_epochs=args.bump_epochs,
        teacher_nodes=args.teacher_nodes,
        logdir=args.logdir,
        model_dir=args.model_dir,
        run_name=args.run_name,
    )

    run_name = defaults.run_name or f"{defaults.shape}-{int(time.time())}"
    logdir_path = os.path.join(defaults.logdir, run_name)
    os.makedirs(logdir_path, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir_path) if SummaryWriter is not None else None
    print(f"TensorBoard logdir: {logdir_path}" if writer else "TensorBoard unavailable.")
    train(defaults, device, writer, model_path)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
