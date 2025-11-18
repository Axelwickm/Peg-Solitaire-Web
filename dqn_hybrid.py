"""
Hybrid-search–supervised DQN for Kongming Chess (cross/triangle) using PyTorch.

What it does:
- Recreates the board shapes and legal moves (cross + triangle).
- Samples randomized mid-game states.
- Uses a shallow hybrid search (limited-depth minimization of pegs) to label
  legal moves with target Q-values.
- Trains a masked DQN (supervised targets) in mini-batches.

Notes:
- This is a lightweight reference; hyperparameters are conservative to keep
  runtime modest. Increase depth/epochs/batch_size for better policies.
- Requires `torch`. Install with: pip install torch
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import argparse
import os
import time

import numpy as np
import numba as nb
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment]

# --------------------------
# Board utilities (mirrors TS logic)
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


def collect_triangle_axes(
    rows: List[List[str]], dr: int, d_idx: int
) -> List[List[str]]:
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


def create_shapes() -> Dict[str, Dict]:
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
    tri_axes = (
        tri_rows
        + collect_triangle_axes(tri_rows, 1, 0)
        + collect_triangle_axes(tri_rows, 1, 1)
    )
    tri_moves = build_allowed_moves_from_axes(tri_axes)

    return {
        "cross": {"holes": cross_holes, "empty": "3,3", "allowed": cross_moves},
        "triangle": {"holes": tri_holes, "empty": "0,4", "allowed": tri_moves},
    }


SHAPES = create_shapes()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# Env
# --------------------------


class KongmingEnv:
    def __init__(
        self,
        shape_id: str = "cross",
        random_remove: int = 6,
        start_pegs_min: int | None = None,
        start_pegs_max: int | None = None,
    ):
        cfg = SHAPES[shape_id]
        self.shape_id = shape_id
        self.holes = cfg["holes"]
        self.empty = cfg["empty"]
        self.allowed = cfg["allowed"]
        self.random_remove = random_remove
        self.start_pegs_min = start_pegs_min
        self.start_pegs_max = start_pegs_max
        self.idx_map: Dict[str, int] = {h: i for i, h in enumerate(self.holes)}
        self.actions = self._build_actions()
        self.state = np.zeros(len(self.holes), dtype=np.bool_)
        self.reset()

    def _build_actions(self) -> np.ndarray:
        arr = []
        for frm, dests in self.allowed.items():
            for to, jump in dests.items():
                arr.append(
                    [
                        self.idx_map[frm],
                        self.idx_map[to],
                        self.idx_map[jump],
                    ]
                )
        return np.array(arr, dtype=np.int32)

    def reset(self):
        self.state[:] = False
        self.state[list(self.idx_map.values())] = True
        if self.empty in self.idx_map:
            self.state[self.idx_map[self.empty]] = False
        if self.start_pegs_min is not None or self.start_pegs_max is not None:
            max_pegs = len(self.holes) - 1
            min_target = self.start_pegs_min if self.start_pegs_min is not None else 1
            min_target = max(1, min(min_target, max_pegs))
            max_target = (
                self.start_pegs_max if self.start_pegs_max is not None else max_pegs
            )
            max_target = max(min_target, min(max_target, max_pegs))
            target_pegs = random.randint(min_target, max_target)
            current_pegs = int(self.state.sum())
            to_remove = max(0, current_pegs - target_pegs)
            if to_remove > 0:
                removable = list(np.where(self.state)[0])
                drop = random.sample(removable, k=min(to_remove, len(removable)))
                self.state[drop] = False
        else:
            available = list(np.where(self.state)[0])
            drop = random.sample(available, k=min(self.random_remove, len(available)))
            self.state[drop] = False
        return self.obs()

    def obs(self) -> torch.Tensor:
        return torch.from_numpy(self.state.astype(np.float32))

    def legal_mask(self) -> torch.Tensor:
        mask = self.legal_mask_numpy()
        return torch.from_numpy(mask)

    def legal_mask_numpy(self) -> np.ndarray:
        mask = np.zeros(len(self.actions), dtype=np.bool_)
        for i, (frm, to, jump) in enumerate(self.actions):
            if self.state[frm] and (not self.state[to]) and self.state[jump]:
                mask[i] = True
        return mask

    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool]:
        frm, to, jump = self.actions[action_idx]
        if not (self.state[frm] and (not self.state[to]) and self.state[jump]):
            return self.obs(), -1.0, True
        self.state[frm] = False
        self.state[jump] = False
        self.state[to] = True
        mask = self.legal_mask_numpy()
        done = (self.state.sum() == 1) or (mask.sum() == 0)
        reward = 1.0 if done and self.state.sum() == 1 else -0.01
        return self.obs(), reward, done


# --------------------------
# Hybrid search for targets
# --------------------------


def min_pegs_after_move(
    env: KongmingEnv,
    depth: int | None = None,
    rollouts: int = 8,
    model: nn.Module | None = None,
    device: torch.device | None = None,
    epsilon: float = 0.05,
) -> Dict[int, float]:
    mask = env.legal_mask_numpy()
    scores: Dict[int, float] = {}
    if mask.sum() == 0:
        return scores
    legal_buf = np.empty(len(env.actions), dtype=np.int32)
    for action_idx, legal in enumerate(mask):
        if not legal:
            continue
        base_state = env.state.copy()
        apply_action_inplace(base_state, env.actions, action_idx)
        acc = 0
        for _ in range(rollouts):
            state_copy = base_state.copy()
            acc += run_rollout_guided(
                state_copy,
                env.actions,
                depth,
                model,
                device,
                epsilon,
                legal_buf,
            )
        scores[action_idx] = -(acc / rollouts) if rollouts else 0.0
    return scores


@nb.njit
def apply_action_inplace(state: np.ndarray, actions: np.ndarray, action_idx: int):
    frm = actions[action_idx, 0]
    to = actions[action_idx, 1]
    jump = actions[action_idx, 2]
    state[frm] = False
    state[jump] = False
    state[to] = True


@nb.njit
def compute_legal_indices(
    state: np.ndarray, actions: np.ndarray, legal: np.ndarray
) -> int:
    count = 0
    for i in range(actions.shape[0]):
        frm = actions[i, 0]
        to = actions[i, 1]
        jump = actions[i, 2]
        if state[frm] and (not state[to]) and state[jump]:
            legal[count] = i
            count += 1
    return count


@nb.njit
def count_pegs(state: np.ndarray) -> int:
    total = 0
    for i in range(state.shape[0]):
        if state[i]:
            total += 1
    return total


def run_rollout_guided(
    state: np.ndarray,
    actions: np.ndarray,
    depth: int | None,
    model: nn.Module | None,
    device: torch.device | None,
    epsilon: float,
    legal_buf: np.ndarray,
) -> int:
    steps = 0
    max_steps = depth if depth is not None and depth > 0 else None
    while True:
        legal_count = compute_legal_indices(state, actions, legal_buf)
        if legal_count == 0:
            break
        if model is not None and device is not None and random.random() > epsilon:
            obs_tensor = (
                torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                q_vals = model(obs_tensor).cpu().numpy().squeeze(0)
            legal_inds = legal_buf[:legal_count]
            q_slice = q_vals[legal_inds]
            best_idx = int(np.argmax(q_slice))
            action_idx = int(legal_inds[best_idx])
        else:
            action_idx = int(legal_buf[random.randrange(legal_count)])
        apply_action_inplace(state, actions, action_idx)
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return count_pegs(state)


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


class ModelGuidedIdaStarSolver:
    """Frontend-style IDA* search ordered by model Q-values."""

    def __init__(
        self,
        env: KongmingEnv,
        model: nn.Module | None,
        device: torch.device | None,
        epsilon: float = 0.0,
    ):
        if env.empty not in env.idx_map:
            raise ValueError("Target hole not found in index map")
        self.actions = env.actions
        self.allowed = env.allowed
        self.idx_map = env.idx_map
        self.idx_to_hole = {idx: hole for hole, idx in self.idx_map.items()}
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.target_idx = env.idx_map[env.empty]
        self.legal_buf = np.empty(len(self.actions), dtype=np.int32)
        self.adjacency = self._build_adjacency()
        self.dist_to_target = self._compute_distances()

    def _build_adjacency(self) -> Dict[int, List[int]]:
        adjacency: Dict[int, List[int]] = {}
        for frm, dests in self.allowed.items():
            frm_idx = self.idx_map[frm]
            for to in dests:
                to_idx = self.idx_map[to]
                adjacency.setdefault(frm_idx, []).append(to_idx)
                adjacency.setdefault(to_idx, []).append(frm_idx)
        return adjacency

    def _compute_distances(self) -> Dict[int, int]:
        dist: Dict[int, int] = {self.target_idx: 0}
        queue = deque([self.target_idx])
        while queue:
            current = queue.popleft()
            base = dist[current]
            for neighbor in self.adjacency.get(current, []):
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
        legal_count = compute_legal_indices(state, self.actions, self.legal_buf)
        legal_inds = self.legal_buf[:legal_count]
        return [
            SearchMove(
                int(self.actions[idx, 0]),
                int(self.actions[idx, 1]),
                int(self.actions[idx, 2]),
                int(idx),
            )
            for idx in legal_inds
        ]

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
            q_vals = self.model(obs_tensor).cpu().numpy().squeeze(0)
        return sorted(moves, key=lambda m: q_vals[m.action_idx], reverse=True)

    def _pop_frame(
        self, stack: List[SearchFrame], seen: set[str], path: List[SearchMove]
    ) -> None:
        frame = stack.pop()
        seen.discard(frame.key)
        if frame.move:
            path.pop()

    def solve(
        self, initial_state: np.ndarray, max_nodes: int | None = None
    ) -> Dict[str, object]:
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
            h = self._heuristic(frame.state)
            f = frame.g + h
            peg_total = int(frame.state.sum())

            if f > bound:
                next_bound = min(next_bound, f)
                self._pop_frame(stack, seen, path)
                continue

            if peg_total < best_pegs:
                best_pegs = peg_total
                best_path = list(path)

            if peg_total == 1 and frame.state[self.target_idx]:
                solved_path = list(path)
                best_path = list(path)
                break

            if max_nodes is not None and nodes >= max_nodes:
                break

            if frame.idx >= len(frame.neighbors):
                self._pop_frame(stack, seen, path)
                continue

            move = frame.neighbors[frame.idx]
            frame.idx += 1
            next_state = frame.state.copy()
            apply_action_inplace(next_state, self.actions, move.action_idx)
            key = self._serialize(next_state)
            if key in seen:
                continue
            seen.add(key)
            path.append(move)
            neighbors = self._order_moves(next_state, self._legal_moves(next_state))
            stack.append(
                SearchFrame(next_state, key, neighbors, 0, frame.g + 1, move)
            )

        duration_ms = (time.time() - start_time) * 1000
        return {
            "solved": solved_path is not None,
            "moves": solved_path or [],
            "best_moves": solved_path or best_path,
            "nodes_explored": nodes,
            "duration_ms": duration_ms,
            "best_pegs": best_pegs,
            "limit_hit": max_nodes is not None and nodes >= max_nodes,
        }


class OnPolicyBuffer:
    def __init__(self, maxlen: int):
        self.buffer: deque[Tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=maxlen)

    def append(self, obs: torch.Tensor, target: torch.Tensor) -> None:
        self.buffer.append((obs, target))

    def sample(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        length = len(self.buffer)
        if length == 0:
            return []
        count = min(batch_size, length)
        if count == length:
            return list(self.buffer)
        return random.sample(self.buffer, count)

    def all(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return list(self.buffer)


# --------------------------
# DQN
# --------------------------


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TransformerQNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.input_proj = nn.Linear(1, d_model)
        self.positional = nn.Parameter(torch.randn(obs_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Treat each board cell as a token; mean-pool transformer features for logits.
        x = obs.unsqueeze(-1)
        x = self.input_proj(x)
        x = x + self.positional.unsqueeze(0)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)


class Conv2DQNet(nn.Module):
    def __init__(
        self, obs_dim: int, act_dim: int, holes: List[str], shape_id: str, channels: int = 48
    ):
        super().__init__()
        if shape_id == "cross":
            self.height, self.width = 7, 7
        elif shape_id == "triangle":
            self.height, self.width = 5, 9
        else:
            side = int(math.ceil(math.sqrt(obs_dim)))
            self.height, self.width = side, side
        coords = [tuple(map(int, h.split(","))) for h in holes]
        self.register_buffer(
            "flat_indices",
            torch.tensor([r * self.width + c for r, c in coords], dtype=torch.long),
            persistent=False,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Map flat hole observations onto a padded 2D grid, keeping blanks zeroed.
        batch = obs.shape[0]
        grid = obs.new_zeros(batch, self.height * self.width)
        grid.scatter_(1, self.flat_indices.unsqueeze(0).expand(batch, -1), obs)
        grid = grid.view(batch, 1, self.height, self.width)
        feat = self.conv(grid)
        pooled = feat.mean(dim=[2, 3])
        return self.head(pooled)


def build_model(
    arch: str, obs_dim: int, act_dim: int, holes: List[str], shape_id: str
) -> nn.Module:
    if arch == "mlp":
        return QNet(obs_dim, act_dim)
    if arch == "transformer":
        return TransformerQNet(obs_dim, act_dim)
    if arch == "conv":
        return Conv2DQNet(obs_dim, act_dim, holes, shape_id)
    raise ValueError(f"Unknown arch '{arch}'")


def compute_final_reward(env: KongmingEnv) -> float:
    pegs = int(env.state.sum())
    reward = -float(pegs)
    center_idx = env.idx_map.get(env.empty)
    if center_idx is not None and env.state[center_idx]:
        reward += 1.0
    return reward


def select_action(
    obs: torch.Tensor,
    legal_indices: np.ndarray,
    model: nn.Module | None,
    device: torch.device | None,
    epsilon: float,
) -> int:
    if not legal_indices.size:
        raise RuntimeError("Attempted to select from zero legal actions")
    if model is not None and device is not None and random.random() > epsilon:
        obs_tensor = obs.unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = model(obs_tensor).cpu().numpy().squeeze(0)
        action_slice = q_vals[legal_indices]
        best = int(np.argmax(action_slice))
        return int(legal_indices[best])
    return int(np.random.choice(legal_indices))


def collect_episode_transitions(
    env: KongmingEnv,
    model: nn.Module | None,
    device: torch.device,
    epsilon: float,
) -> Tuple[List[Tuple[torch.Tensor, np.ndarray]], float]:
    transitions: List[Tuple[torch.Tensor, np.ndarray]] = []
    obs = env.obs()
    while True:
        mask = env.legal_mask_numpy()
        legal_indices = np.where(mask)[0]
        if legal_indices.size == 0:
            break
        transitions.append((obs.clone(), mask.copy()))
        action_idx = select_action(obs, legal_indices, model, device, epsilon)
        next_obs, _, done = env.step(action_idx)
        obs = next_obs
        if done:
            break
    final_reward = compute_final_reward(env)
    return transitions, final_reward


def train_dqn(
    shape_id: str,
    epochs: int,
    batch_size: int,
    random_remove: int,
    epsilon: float,
    start_pegs_min: int | None,
    start_pegs_max: int | None,
    buffer_size: int,
    collect_batch: int,
    device: torch.device | None,
    save_every: int,
    arch: str,
    model_path: str = "models/cross.pt",
    writer: SummaryWriter | None = None,
):
    env = KongmingEnv(
        shape_id=shape_id,
        random_remove=random_remove,
        start_pegs_min=start_pegs_min,
        start_pegs_max=start_pegs_max,
    )
    obs_dim = len(env.holes)
    act_dim = len(env.actions)
    device = device or get_device()
    model = build_model(arch, obs_dim, act_dim, env.holes, shape_id).to(device)
    print(
        f"[{shape_id}] initialized {arch} model (obs={obs_dim}, acts={act_dim})",
        flush=True,
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}", flush=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
    opt = optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = nn.MSELoss()

    replay_buffer = OnPolicyBuffer(buffer_size)

    last_loss = 0.0
    for epoch in range(epochs):
        print(f"[{shape_id}] starting epoch {epoch + 1}/{epochs}", flush=True)
        epoch_final_rewards: List[float] = []
        transition_count = 0

        collected = 0
        while collected < collect_batch:
            env.reset()
            transitions, final_reward = collect_episode_transitions(
                env, model, device, epsilon
            )
            if transitions:
                epoch_final_rewards.append(final_reward)
                transition_count += len(transitions)
                for obs_tensor, mask in transitions:
                    target = torch.zeros(act_dim, dtype=torch.float32)
                    mask_tensor = torch.from_numpy(mask)
                    target[mask_tensor] = final_reward
                    replay_buffer.append(obs_tensor, target)
            collected += 1
        print(
            f"[{shape_id}] epoch {epoch + 1} collected {collected} episodes "
            f"({transition_count} transitions), replay buffer size: {len(replay_buffer.buffer)}",
            flush=True,
        )

        training_data = replay_buffer.all()
        if not training_data:
            print(
                f"[{shape_id}] epoch {epoch + 1} buffer empty, skipping update",
                flush=True,
            )
            continue
        random.shuffle(training_data)
        total_loss = 0.0
        total_avg_target = 0.0
        updates = 0
        for start in range(0, len(training_data), batch_size):
            batch = training_data[start : start + batch_size]
            obs_tensor = torch.stack([sample[0] for sample in batch]).to(device)
            target_tensor = torch.stack([sample[1] for sample in batch]).to(device)
            pred_q = model(obs_tensor)
            loss = loss_fn(pred_q, target_tensor)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_avg_target += target_tensor.mean().item()
            updates += 1

        last_loss = total_loss / updates
        avg_targets = total_avg_target / updates
        avg_final_reward = (
            sum(epoch_final_rewards) / len(epoch_final_rewards)
            if epoch_final_rewards
            else 0.0
        )
        print(
            f"[{shape_id}] epoch {epoch + 1}/{epochs} loss={last_loss:.4f} "
            f"avg_target={avg_targets:.3f} avg_final_reward={avg_final_reward:.3f}",
            flush=True,
        )
        if writer is not None:
            writer.add_scalar(f"{shape_id}/loss", last_loss, epoch + 1)
            writer.add_scalar(f"{shape_id}/avg_target", avg_targets, epoch + 1)
            writer.add_scalar(
                f"{shape_id}/avg_final_reward", avg_final_reward, epoch + 1
            )
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), model_path)
            print(f"[{shape_id}] saved model to {model_path}", flush=True)

    return model


def render_state(env: KongmingEnv):
    if env.shape_id == "cross":
        size = 7
        grid = [[" " for _ in range(size)] for _ in range(size)]
        hole_set = set(env.holes)
        for hole in hole_set:
            r, c = map(int, hole.split(","))
            idx = env.idx_map[hole]
            symbol = "#" if env.state[idx] else "."
            grid[r][c] = symbol
        print("\n".join("".join(row) for row in grid))
    else:
        rows = build_triangle_rows(9, 5)
        for row in rows:
            line = ""
            for hole in row:
                idx = env.idx_map[hole]
                line += "#" if env.state[idx] else "."
            print(line.center(15))
    print("-" * 25)


def move_to_text(move: SearchMove, idx_to_hole: Dict[int, str]) -> str:
    return (
        f"{idx_to_hole[move.frm]} -> {idx_to_hole[move.to]} "
        f"(over {idx_to_hole[move.jump]})"
    )


def inference_hybrid(
    shape_id: str,
    model_path: str,
    device: torch.device,
    games: int = 1,
    start_pegs_min: int | None = 3,
    start_pegs_max: int | None = None,
    arch: str = "mlp",
    epsilon: float = 0.0,
    random_remove: int = 6,
    max_nodes: int | None = None,
):
    env = KongmingEnv(
        shape_id=shape_id,
        random_remove=random_remove,
        start_pegs_min=start_pegs_min,
        start_pegs_max=start_pegs_max,
    )
    obs_dim = len(env.holes)
    act_dim = len(env.actions)
    model = build_model(arch, obs_dim, act_dim, env.holes, shape_id).to(device)
    if not os.path.exists(model_path):
        print(f"No model at {model_path}, run training first.")
        return
    print(
        f"[{shape_id}] loading {arch} model from {model_path} for hybrid search",
        flush=True,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    solver = ModelGuidedIdaStarSolver(env, model, device, epsilon=epsilon)

    for game in range(games):
        env.reset()
        start_state = env.state.copy()
        initial_pegs = int(start_state.sum())
        print(f"=== Hybrid game {game + 1} | start pegs: {initial_pegs} ===")
        result = solver.solve(start_state, max_nodes=max_nodes)
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
        for step, move in enumerate(best_moves, 1):
            print(f"Step {step}: {move_to_text(move, solver.idx_to_hole)}")
            env.step(move.action_idx)
        render_state(env)
        if result["solved"]:
            print("Solved with model-guided IDA*.\n")
        else:
            remaining = int(env.state.sum())
            print(f"Best found path leaves {remaining} pegs.\n")


def inference(
    shape_id: str,
    model_path: str,
    device: torch.device,
    games: int = 3,
    delay: float = 0.4,
    start_pegs_min: int | None = 3,
    start_pegs_max: int | None = None,
    arch: str = "mlp",
):
    env = KongmingEnv(
        shape_id=shape_id,
        random_remove=6,
        start_pegs_min=start_pegs_min,
        start_pegs_max=start_pegs_max,
    )
    obs_dim = len(env.holes)
    act_dim = len(env.actions)
    model = build_model(arch, obs_dim, act_dim, env.holes, shape_id).to(device)
    if not os.path.exists(model_path):
        print(f"No model at {model_path}, run training first.")
        return
    print(
        f"[{shape_id}] loading {arch} model from {model_path}",
        flush=True,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    for game in range(games):
        obs = env.reset()
        done = False
        steps = 0
        print(f"=== Game {game + 1} ===")
        while not done:
            mask = env.legal_mask_numpy()
            legal = np.where(mask)[0]
            render_state(env)
            if len(legal) == 0:
                print("No moves left.")
                break
            with torch.no_grad():
                q_vals = model(obs.unsqueeze(0).to(device)).cpu().squeeze(0)
            q_masked = q_vals[legal]
            action_idx = int(legal[int(q_masked.argmax().item())])
            obs, _, done = env.step(action_idx)
            if done:
                render_state(env)
                print("done")
            steps += 1
            time.sleep(delay)
        print(f"Game {game + 1} finished after {steps} moves.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid DQN trainer")
    parser.add_argument("--shape", choices=["cross", "triangle"], default="cross")
    parser.add_argument("--epochs", type=int, default=100_000_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="Max steps per rollout (0 runs until no legal moves remain).",
    )
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="ε for on-policy rollouts (lower = greedier policy, some randomness).",
    )
    parser.add_argument("--random-remove", type=int, default=6)
    parser.add_argument(
        "--start-pegs-min",
        type=int,
        default=3,
        help="Minimum number of pegs to leave when the episode starts.",
    )
    parser.add_argument(
        "--start-pegs-max",
        type=int,
        default=None,
        help="Maximum number of pegs to leave when the episode starts.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1_000_000,
        help="Max number of on-policy samples to keep for training.",
    )
    parser.add_argument(
        "--collect-batch",
        type=int,
        default=64,
        help="Number of inference episodes to collect before training on the buffer.",
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--logdir", default="runs")
    parser.add_argument("--run-name", default=None, help="Name for this training run")
    parser.add_argument(
        "--infer", action="store_true", help="Run inference instead of training"
    )
    parser.add_argument(
        "--infer-hybrid",
        action="store_true",
        help="Run frontend-style IDA* search guided by the model",
    )
    parser.add_argument(
        "--arch",
        choices=["mlp", "transformer", "conv"],
        default="mlp",
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--games", type=int, default=3, help="Games to play in inference mode"
    )
    parser.add_argument(
        "--delay", type=float, default=0.4, help="Delay between inference steps"
    )
    parser.add_argument(
        "--search-epsilon",
        type=float,
        default=0.0,
        help="Random move ordering probability for hybrid inference.",
    )
    parser.add_argument(
        "--max-search-nodes",
        type=int,
        default=1_000_000,
        help="Maximum nodes to expand in hybrid inference (0 disables the cap).",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}", flush=True)
    os.makedirs(args.model_dir, exist_ok=True)
    model_filename = (
        f"{args.shape}.pt" if args.arch == "mlp" else f"{args.shape}-{args.arch}.pt"
    )
    model_path = os.path.join(args.model_dir, model_filename)
    writer = None
    if not args.infer and not args.infer_hybrid:
        run_name = args.run_name or f"{args.shape}-{int(time.time())}"
        logdir_path = os.path.join(args.logdir, run_name)
        os.makedirs(logdir_path, exist_ok=True)
        if SummaryWriter is not None:
            print(f"TensorBoard logdir: {logdir_path}", flush=True)
            writer = SummaryWriter(log_dir=logdir_path)
        else:
            print(
                "TensorBoard summary writer unavailable; skipping logging.",
                flush=True,
            )

    if args.infer_hybrid:
        inference_hybrid(
            args.shape,
            model_path,
            device,
            games=args.games,
            start_pegs_min=args.start_pegs_min,
            start_pegs_max=args.start_pegs_max,
            arch=args.arch,
            epsilon=args.search_epsilon,
            random_remove=args.random_remove,
            max_nodes=args.max_search_nodes if args.max_search_nodes > 0 else None,
        )
    elif args.infer:
        inference(
            args.shape,
            model_path,
            device,
            games=args.games,
            delay=args.delay,
            start_pegs_min=args.start_pegs_min,
            start_pegs_max=args.start_pegs_max,
            arch=args.arch,
        )
    else:
        train_dqn(
            args.shape,
            epochs=args.epochs,
            batch_size=args.batch_size,
            random_remove=args.random_remove,
            epsilon=args.epsilon,
            start_pegs_min=args.start_pegs_min,
            start_pegs_max=args.start_pegs_max,
            buffer_size=args.buffer_size,
            collect_batch=args.collect_batch,
            device=device,
            save_every=args.save_every,
            arch=args.arch,
            model_path=model_path,
            writer=writer,
        )
    if writer is not None:
        writer.close()
