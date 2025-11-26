"""Train/inference runner for the Kongming Chess DQN agent."""

import argparse
import math
import random
import time
from collections import deque, namedtuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

from environment import (
    KongmingEnv,
    SHAPES,
    ShapeContext,
    build_shape_contexts,
    compute_final_reward,
    render_cli_state,
)
from dqn import TokenAttentionQNetwork
from numba_kernels import (
    legal_mask_kernel,
    mask_any_kernel,
    peg_count_kernel,
    simulate_action_kernel,
    terminal_reward_kernel,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_SHAPE_ID = "cross"
ALL_SHAPES_ID = "all"
DEFAULT_REPLAY_CAPACITY = 2_000_000
DEFAULT_BATCH_SIZE = 2048
DEFAULT_LR = 1e-4
GAMMA = 0.99

NUM_EPISODES = 100_000_000
MAX_STEPS_PER_EPISODE = 100
COLLECT_STEPS_PER_EPOCH = 4096
TRAIN_UPDATES_PER_EPOCH = 1024

TARGET_UPDATE_INTERVAL = 1000  # episodes
CURRICULUM_EVAL_INTERVAL = 1000
EVAL_EPISODES_PER_CHECK = 150

MODELS_DIR = REPO_ROOT / "models"
RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_INFER_EPISODES = 1000
STEP_RENDER_DELAY = 0.5

CURRICULUM_MIN_PEGS = 3
CURRICULUM_START_MAX_PEGS = 3
CURRICULUM_MAX_LIMIT = 32
CURRICULUM_WINDOW = 500
AVG_FINAL_PEGS_THRESHOLD = 1.3

DEFAULT_INFER_MIN_PEGS = CURRICULUM_MIN_PEGS
DEFAULT_INFER_MAX_PEGS = CURRICULUM_START_MAX_PEGS

EPSILON_START = 0.01
EPSILON_END = 0.01
EPSILON_DECAY = 10_000

BRANCHING_FACTOR = 500
SEARCH_MAX_DEPTH = 100

MODEL_NAME_BASE = "peg_solitaire_dqn"

Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done", "next_legal_mask"),
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def get_default_model_path(shape_id: str) -> Path:
    filename = f"{MODEL_NAME_BASE}.pt" if shape_id == ALL_SHAPES_ID else f"{MODEL_NAME_BASE}_{shape_id}.pt"
    return MODELS_DIR / filename


def ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


class DummyWriter:
    def __init__(self) -> None:
        self.log_dir = ""

    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def _default_run_name(shape_id: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{MODEL_NAME_BASE}_{shape_id}_{timestamp}"


def _normalize_center_idx(center_idx: int | None) -> int:
    if center_idx is None or center_idx < 0:
        return -1
    return center_idx


def create_summary_writer(
    shape_id: str, run_name: str | None = None
) -> tuple[SummaryWriter | DummyWriter, str, str]:
    if SummaryWriter is None:
        return DummyWriter(), run_name or "tb_disabled", ""
    runs_dir = RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_name or _default_run_name(shape_id)
    log_dir = runs_dir / run_name
    writer = SummaryWriter(log_dir=log_dir)
    return writer, run_name, str(log_dir)


def save_checkpoint(
    path: Path,
    q_net: TokenAttentionQNetwork,
    optimizer: optim.Optimizer,
    episode: int,
    steps_done: int,
    shape_id: str,
    tb_run_name: str,
    tb_log_dir: str,
    replay_capacity: int,
    batch_size: int,
    learning_rate: float,
    max_initial_pegs: int | dict[str, int],
    shape_ids: list[str] | None = None,
) -> None:
    checkpoint = {
        "model_state": q_net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "episode": episode,
        "steps_done": steps_done,
        "shape_id": shape_id,
        "tb_run_name": tb_run_name,
        "tb_log_dir": tb_log_dir,
        "replay_capacity": replay_capacity,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_initial_pegs": max_initial_pegs,
    }
    if isinstance(max_initial_pegs, dict):
        checkpoint["shape_curriculum"] = max_initial_pegs
    if shape_ids is not None:
        checkpoint["shape_ids"] = shape_ids
    torch.save(checkpoint, path)


def select_action(
    q_net: TokenAttentionQNetwork,
    state: torch.Tensor,
    legal_mask: np.ndarray,
    epsilon: float,
    shape_ctx: ShapeContext,
) -> int:
    if random.random() < epsilon:
        legal_indices = np.nonzero(legal_mask)[0]
        if len(legal_indices) == 0:
            return -1
        return int(random.choice(legal_indices))

    with torch.no_grad():
        s = state.unsqueeze(0).to(DEVICE)
        q_values = q_net(s, shape_ctx).squeeze(0)

        legal = torch.from_numpy(legal_mask).to(DEVICE)
        if not legal.any():
            return -1

        q_values_masked = q_values.clone()
        q_values_masked[~legal] = -1e9
        return int(torch.argmax(q_values_masked).item())


def _state_tensor_from_np(state_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(state_np.astype(np.float32))


def _legal_mask_from_state(state_np: np.ndarray, actions: np.ndarray) -> np.ndarray:
    state_np = state_np.astype(np.bool_, copy=False)
    return legal_mask_kernel(state_np, actions)


def _terminal_reward_from_state(state_np: np.ndarray, center_idx: int | None) -> float:
    center_value = _normalize_center_idx(center_idx)
    return float(terminal_reward_kernel(state_np, center_value))


def _simulate_action_from_state(
    state_np: np.ndarray,
    action_idx: int,
    actions: np.ndarray,
    center_idx: int | None,
) -> tuple[np.ndarray, float, bool, np.ndarray]:
    state_np = state_np.astype(np.bool_, copy=False)
    next_state, is_valid = simulate_action_kernel(state_np, action_idx, actions)
    if not is_valid:
        dummy_mask = np.zeros(actions.shape[0], dtype=np.bool_)
        return next_state, -1.0, True, dummy_mask
    next_legal = legal_mask_kernel(next_state, actions)
    peg_total = peg_count_kernel(next_state)
    done = bool((peg_total == 1) or (not mask_any_kernel(next_legal)))
    reward = _terminal_reward_from_state(next_state, center_idx) if done else 0.0
    return next_state, reward, done, next_legal


def _eval_q_values(
    q_net: TokenAttentionQNetwork,
    state_tensor: torch.Tensor,
    shape_ctx: ShapeContext,
) -> torch.Tensor:
    with torch.no_grad():
        return (
            q_net(state_tensor.unsqueeze(0).to(DEVICE), shape_ctx)
            .squeeze(0)
            .detach()
            .cpu()
        )


def _fallback_legal_action(legal_mask: np.ndarray) -> int:
    legal_indices = np.flatnonzero(legal_mask)
    if len(legal_indices) == 0:
        return -1
    return int(random.choice(legal_indices))


@dataclass
class _ActionEval:
    action_idx: int
    q_value: float
    reward: float
    done: bool
    next_state: np.ndarray
    next_legal: np.ndarray
    next_state_key: bytes
    peg_count: int
    used: bool = False
    queued: bool = False


@dataclass
class _PlanCandidate:
    action: _ActionEval
    path: list[int]
    history_keys: set[bytes]


class _PlanQueue:
    def __init__(self) -> None:
        self._entries: list[_PlanCandidate] = []

    def push(self, entry: _PlanCandidate) -> None:
        self._entries.append(entry)

    def pop(self) -> _PlanCandidate | None:
        if not self._entries:
            return None
        best_idx = 0
        best_val = self._entries[0].action.q_value
        for idx in range(1, len(self._entries)):
            value = self._entries[idx].action.q_value
            if value > best_val:
                best_val = value
                best_idx = idx
        return self._entries.pop(best_idx)

    def clear(self) -> None:
        self._entries.clear()


def _state_cache_key(state_np: np.ndarray) -> bytes:
    state_np = state_np.astype(np.bool_, copy=False)
    return state_np.tobytes()


def _evaluate_state_actions(
    q_net: TokenAttentionQNetwork,
    state_np: np.ndarray,
    actions: np.ndarray,
    center_idx: int | None,
    shape_ctx: ShapeContext,
    state_cache: dict[bytes, list[_ActionEval]],
    replay: ReplayBuffer | None,
    collect_transitions: bool,
) -> list[_ActionEval]:
    state_np = state_np.astype(np.bool_, copy=False)
    cache_key = _state_cache_key(state_np)
    cached = state_cache.get(cache_key)
    if cached is not None:
        return cached

    legal_mask = _legal_mask_from_state(state_np, actions)
    if not legal_mask.any():
        state_cache[cache_key] = []
        return []

    state_tensor = _state_tensor_from_np(state_np)
    q_values = _eval_q_values(q_net, state_tensor, shape_ctx).numpy()
    legal_indices = np.flatnonzero(legal_mask)
    entries: list[_ActionEval] = []

    for action_idx in legal_indices:
        next_state_np, reward, done, next_legal = _simulate_action_from_state(
            state_np, action_idx, actions, center_idx
        )
        next_state_np = next_state_np.astype(np.bool_, copy=False)
        next_key = _state_cache_key(next_state_np)
        peg_total = int(peg_count_kernel(next_state_np))
        entries.append(
            _ActionEval(
                action_idx=action_idx,
                q_value=float(q_values[action_idx]),
                reward=float(reward),
                done=bool(done),
                next_state=next_state_np,
                next_legal=next_legal,
                next_state_key=next_key,
                peg_count=peg_total,
            )
        )

        if collect_transitions and replay is not None:
            replay.push(
                state_tensor.clone(),
                int(action_idx),
                float(reward),
                _state_tensor_from_np(next_state_np),
                bool(done),
                next_legal.copy(),
            )

    entries.sort(key=lambda entry: entry.q_value, reverse=True)
    state_cache[cache_key] = entries
    return entries


def _enqueue_alternatives(
    queue: _PlanQueue,
    alternatives: list[_ActionEval],
    path: list[int],
    history_keys: set[bytes],
) -> None:
    for alt in alternatives:
        if alt.used or alt.queued:
            continue
        alt.queued = True
        queue.push(
            _PlanCandidate(
                action=alt,
                path=list(path),
                history_keys=set(history_keys),
            )
        )


def _update_best_tracker(
    tracker: dict[str, list[int] | int], peg_count: int, path: list[int]
) -> None:
    current_best = tracker.get("path")
    best_pegs = tracker.get("peg_count", math.inf)
    if not path:
        return
    if not current_best or peg_count < best_pegs:
        tracker["peg_count"] = peg_count
        tracker["path"] = list(path)


def _pursue_greedy_plan(
    start_state: np.ndarray,
    path_prefix: list[int],
    history_keys: set[bytes],
    max_depth: int,
    q_net: TokenAttentionQNetwork,
    actions: np.ndarray,
    center_idx: int | None,
    shape_ctx: ShapeContext,
    queue: _PlanQueue,
    tracker: dict[str, list[int] | int],
    state_cache: dict[bytes, list[_ActionEval]],
    replay: ReplayBuffer | None,
    collect_transitions: bool,
) -> dict[str, object]:
    current_state = start_state
    current_path = list(path_prefix)
    visited = set(history_keys)

    while len(current_path) < max_depth:
        options = _evaluate_state_actions(
            q_net,
            current_state,
            actions,
            center_idx,
            shape_ctx,
            state_cache,
            replay,
            collect_transitions,
        )
        available = [op for op in options if op.next_state_key not in visited]
        if not available:
            break
        best_action = available[0]
        best_action.used = True
        _enqueue_alternatives(queue, available[1:], current_path, visited)
        current_path.append(best_action.action_idx)
        visited.add(best_action.next_state_key)
        _update_best_tracker(tracker, best_action.peg_count, current_path)
        if best_action.done and best_action.peg_count == 1:
            return {
                "solved": True,
                "path": list(current_path),
                "history": set(visited),
                "state": best_action.next_state,
            }
        current_state = best_action.next_state

    return {
        "solved": False,
        "path": list(current_path),
        "history": set(visited),
        "state": current_state,
    }


def markov_search_action(
    q_net: TokenAttentionQNetwork,
    state_np: np.ndarray,
    actions: np.ndarray,
    center_idx: int | None,
    shape_ctx: ShapeContext,
    epsilon: float,
    branching_factor: int = BRANCHING_FACTOR,
    max_depth: int = SEARCH_MAX_DEPTH,
    replay: ReplayBuffer | None = None,
    collect_transitions: bool = False,
) -> tuple[int, float]:
    state_np = state_np.astype(np.bool_, copy=False)
    legal_mask = _legal_mask_from_state(state_np, actions)
    if not legal_mask.any():
        terminal_reward = _terminal_reward_from_state(state_np, center_idx)
        return -1, terminal_reward

    if epsilon > 0.0 and random.random() < epsilon:
        random_action = _fallback_legal_action(legal_mask)
        if random_action < 0:
            terminal_reward = _terminal_reward_from_state(state_np, center_idx)
            return -1, terminal_reward
        root_values = _eval_q_values(
            q_net, _state_tensor_from_np(state_np), shape_ctx
        ).numpy()
        return random_action, float(root_values[random_action])

    queue = _PlanQueue()
    state_cache: dict[bytes, list[_ActionEval]] = {}
    initial_pegs = int(peg_count_kernel(state_np))
    tracker: dict[str, list[int] | int] = {
        "peg_count": initial_pegs,
        "path": [],
    }

    root_key = _state_cache_key(state_np)
    history_keys: set[bytes] = {root_key}

    greedy_result = _pursue_greedy_plan(
        state_np,
        [],
        history_keys,
        max_depth,
        q_net,
        actions,
        center_idx,
        shape_ctx,
        queue,
        tracker,
        state_cache,
        replay,
        collect_transitions,
    )

    best_path: list[int] | None = None
    if greedy_result["solved"]:
        best_path = greedy_result["path"]  # type: ignore[assignment]
    else:
        max_expansions = max(0, branching_factor - 1)
        expansions = 0
        while expansions < max_expansions:
            candidate = queue.pop()
            if candidate is None:
                break
            action = candidate.action
            if action.used:
                continue
            action.used = True
            new_path = candidate.path + [action.action_idx]
            new_history = set(candidate.history_keys)
            new_history.add(action.next_state_key)
            _update_best_tracker(tracker, action.peg_count, new_path)
            if action.done and action.peg_count == 1:
                best_path = list(new_path)
                tracker["path"] = list(new_path)
                break
            greedy_result = _pursue_greedy_plan(
                action.next_state,
                new_path,
                new_history,
                max_depth,
                q_net,
                actions,
                center_idx,
                shape_ctx,
                queue,
                tracker,
                state_cache,
                replay,
                collect_transitions,
            )
            if greedy_result["solved"]:
                best_path = greedy_result["path"]  # type: ignore[assignment]
                break
            expansions += 1

        if not collect_transitions and max_expansions > 0:
            print(
                f"[Search] expanded {expansions} alternative branches "
                f"(limit {max_expansions})"
            )

    if not best_path:
        best_path = tracker.get("path")
    if not best_path:
        fallback_action = _fallback_legal_action(legal_mask)
        if fallback_action < 0:
            terminal_reward = _terminal_reward_from_state(state_np, center_idx)
            return -1, terminal_reward
        root_values = _eval_q_values(
            q_net, _state_tensor_from_np(state_np), shape_ctx
        ).numpy()
        return fallback_action, float(root_values[fallback_action])

    chosen_action = int(best_path[0])
    root_actions = state_cache.get(root_key)
    chosen_value = 0.0
    if root_actions is not None:
        for entry in root_actions:
            if entry.action_idx == chosen_action:
                chosen_value = entry.q_value
                break
    else:
        root_values = _eval_q_values(
            q_net, _state_tensor_from_np(state_np), shape_ctx
        ).numpy()
        chosen_value = float(root_values[chosen_action])

    return chosen_action, chosen_value


def compute_td_loss(
    q_net: TokenAttentionQNetwork,
    target_net: TokenAttentionQNetwork,
    batch: Transition,
    shape_ctx: ShapeContext,
) -> torch.Tensor:
    states = torch.stack(batch.state).to(DEVICE)
    actions = torch.tensor(batch.action, dtype=torch.long, device=DEVICE)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
    next_states = torch.stack(batch.next_state).to(DEVICE)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)
    next_masks = torch.stack(
        [torch.from_numpy(m.astype(np.bool_)) for m in batch.next_legal_mask]
    ).to(DEVICE)

    q_values = q_net(states, shape_ctx)
    state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states, shape_ctx)
        next_q_values[~next_masks] = -1e9
        next_state_values, _ = next_q_values.max(dim=1)
        no_legal = ~next_masks.any(dim=1)
        next_state_values[no_legal] = 0.0

        target_values = rewards + GAMMA * (1.0 - dones) * next_state_values

    loss = nn.MSELoss()(state_action_values, target_values)
    return loss


def evaluate_policy(
    q_net: TokenAttentionQNetwork,
    shape_ctx: ShapeContext,
    start_pegs: int,
    episodes: int = EVAL_EPISODES_PER_CHECK,
) -> tuple[float, float]:
    eval_env = KongmingEnv(shape_ctx.shape)
    rewards: list[float] = []
    final_pegs: list[int] = []
    was_training = q_net.training
    q_net.eval()

    try:
        for _ in range(episodes):
            state = eval_env.reset_with_pegs(start_pegs)
            done = False
            episode_reward = 0.0
            steps = 0

            while not done and steps < MAX_STEPS_PER_EPISODE:
                legal_mask_np = eval_env.legal_mask_numpy()
                if legal_mask_np.sum() == 0:
                    reward = compute_final_reward(eval_env, center_bonus=True)
                    episode_reward += reward
                    done = True
                    break

                action_idx = select_action(
                    q_net, state, legal_mask_np, epsilon=0.0, shape_ctx=shape_ctx
                )
                if action_idx < 0:
                    break

                state, reward, done = eval_env.step(action_idx)
                episode_reward += reward
                steps += 1

            if not done:
                reward = compute_final_reward(eval_env, center_bonus=True)
                episode_reward += reward

            rewards.append(episode_reward)
            final_pegs.append(int(eval_env.state.sum()))
    finally:
        if was_training:
            q_net.train()

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    avg_final_pegs = sum(final_pegs) / len(final_pegs) if final_pegs else 0.0
    return avg_reward, avg_final_pegs


CHECKPOINT_SAVE_INTERVAL = TARGET_UPDATE_INTERVAL


def train(
    shape_id: str = DEFAULT_SHAPE_ID,
    load_path: Path | str | None = None,
    max_episodes: int = NUM_EPISODES,
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LR,
) -> None:
    shape_ctxs = build_shape_contexts(SHAPES)
    if shape_id == ALL_SHAPES_ID:
        active_shapes = sorted(shape_ctxs.keys())
    else:
        if shape_id not in shape_ctxs:
            raise ValueError(f"Unknown shape '{shape_id}'")
        active_shapes = [shape_id]

    envs: dict[str, KongmingEnv] = {}
    full_pegs_map: dict[str, int] = {}
    curriculum_limits: dict[str, int] = {}
    episode_final_pegs: dict[str, deque[int]] = {}
    for sid in active_shapes:
        env = KongmingEnv(shape_ctxs[sid].shape)
        env.reset_full()
        envs[sid] = env
        full_pegs = int(env.state.sum())
        full_pegs_map[sid] = full_pegs
        curriculum_limits[sid] = min(CURRICULUM_START_MAX_PEGS, full_pegs)
        episode_final_pegs[sid] = deque(maxlen=CURRICULUM_WINDOW)

    q_net = TokenAttentionQNetwork().to(DEVICE)
    target_net = TokenAttentionQNetwork().to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    replay_buffers = {sid: ReplayBuffer(replay_capacity) for sid in active_shapes}

    start_episode = 1
    steps_done = 0
    tb_run_name = None
    tb_log_dir = ""

    if load_path is not None:
        load_path = Path(load_path)
        if not load_path.exists():
            print(
                f"[Warning] Requested checkpoint '{load_path}' not found, "
                "starting from scratch."
            )
        else:
            checkpoint = torch.load(load_path, map_location=DEVICE)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                q_net.load_state_dict(checkpoint["model_state"], strict=False)
                if "optimizer_state" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
                start_episode = checkpoint.get("episode", 0) + 1
                steps_done = checkpoint.get("steps_done", 0)
                tb_run_name = checkpoint.get("tb_run_name")
                tb_log_dir = checkpoint.get("tb_log_dir", "")
                saved_curriculum = checkpoint.get("shape_curriculum")
                if isinstance(saved_curriculum, dict):
                    for sid in active_shapes:
                        if sid in saved_curriculum:
                            curriculum_limits[sid] = int(saved_curriculum[sid])
                else:
                    legacy = checkpoint.get("max_initial_pegs")
                    if isinstance(legacy, dict):
                        for sid in active_shapes:
                            if sid in legacy:
                                curriculum_limits[sid] = int(legacy[sid])
                    elif isinstance(legacy, int):
                        for sid in active_shapes:
                            curriculum_limits[sid] = legacy
            else:
                q_net.load_state_dict(checkpoint, strict=False)
                start_episode = 1
            target_net.load_state_dict(q_net.state_dict())
            target_net.eval()
            print(f"Loaded pretrained weights from {load_path}")

    writer, tb_run_name, tb_log_dir = create_summary_writer(
        shape_id, run_name=tb_run_name
    )
    multi_shape_mode = len(active_shapes) > 1

    ensure_models_dir()
    model_path = get_default_model_path(shape_id)

    end_episode = start_episode + max_episodes - 1
    last_episode = start_episode - 1
    episode = start_episode
    epoch_idx = 0
    train_steps_taken = 0
    shape_order = active_shapes.copy()
    if len(shape_order) > 1:
        random.shuffle(shape_order)
    shape_order_idx = 0

    def _next_shape_id() -> str:
        nonlocal shape_order_idx, shape_order
        if len(shape_order) == 1:
            return shape_order[0]
        sid = shape_order[shape_order_idx]
        shape_order_idx += 1
        if shape_order_idx >= len(shape_order):
            shape_order_idx = 0
            random.shuffle(shape_order)
        return sid

    try:
        while episode <= end_episode:
            epoch_idx += 1
            collected_steps = 0
            epoch_episode_count = 0
            epoch_reward_total = 0.0

            while episode <= end_episode and collected_steps < COLLECT_STEPS_PER_EPOCH:
                current_episode = episode
                last_episode = current_episode
                epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(
                    -1.0 * steps_done / EPSILON_DECAY
                )

                current_shape_id = _next_shape_id()
                env = envs[current_shape_id]
                shape_ctx = shape_ctxs[current_shape_id]
                max_pegs = curriculum_limits[current_shape_id]
                min_pegs = CURRICULUM_MIN_PEGS
                if max_pegs < min_pegs:
                    max_pegs = min_pegs
                target_pegs = random.randint(min_pegs, max_pegs)

                state = env.reset_with_pegs(target_pegs)
                done = False
                total_reward = 0.0
                step_count = 0

                while not done and step_count < MAX_STEPS_PER_EPISODE:
                    legal_mask_np = env.legal_mask_numpy()
                    if legal_mask_np.sum() == 0:
                        reward = compute_final_reward(env, center_bonus=True)
                        total_reward += reward
                        done = True

                        next_state = state.clone()
                        next_legal_mask = np.zeros_like(legal_mask_np, dtype=np.bool_)
                        replay_buffers[current_shape_id].push(
                            state.cpu(),
                            -1,
                            reward,
                            next_state.cpu(),
                            True,
                            next_legal_mask,
                        )
                        collected_steps += 1
                        break

                    search_state = env.state.astype(np.bool_, copy=True)
                    action_idx, _ = markov_search_action(
                        q_net,
                        search_state,
                        env.actions,
                        env.idx_map.get(env.empty),
                        shape_ctx,
                        epsilon,
                        replay=replay_buffers[current_shape_id],
                        collect_transitions=True,
                    )
                    if action_idx < 0 or not legal_mask_np[action_idx]:
                        action_idx = _fallback_legal_action(legal_mask_np)
                        if action_idx < 0:
                            break

                    next_state, reward, done = env.step(action_idx)
                    total_reward += reward

                    state = next_state
                    step_count += 1
                    steps_done += 1
                    collected_steps += 1

                final_pegs = int(env.state.sum())
                final_window = episode_final_pegs[current_shape_id]
                final_window.append(final_pegs)
                avg_final_pegs = (
                    sum(final_window) / len(final_window)
                    if final_window
                    else final_pegs
                )
                tag_prefix = f"{current_shape_id}/" if multi_shape_mode else ""
                writer.add_scalar(
                    f"{tag_prefix}episode/reward", total_reward, current_episode
                )
                writer.add_scalar(
                    f"{tag_prefix}episode/avg_final_pegs",
                    avg_final_pegs,
                    current_episode,
                )
                writer.add_scalar(
                    f"{tag_prefix}episode/max_initial_pegs",
                    curriculum_limits[current_shape_id],
                    current_episode,
                )

                if current_episode % TARGET_UPDATE_INTERVAL == 0:
                    target_net.load_state_dict(q_net.state_dict())
                    print(
                        f"[Episode {current_episode}] Target network updated "
                        f"(shape {current_shape_id}). Last reward {total_reward:.2f}, "
                        f"final pegs {final_pegs}"
                    )

                if current_episode % CURRICULUM_EVAL_INTERVAL == 0:
                    eval_reward, eval_final_pegs = evaluate_policy(
                        q_net,
                        shape_ctx,
                        curriculum_limits[current_shape_id],
                        episodes=EVAL_EPISODES_PER_CHECK,
                    )
                    writer.add_scalar(
                        f"{tag_prefix}eval/reward", eval_reward, current_episode
                    )
                    writer.add_scalar(
                        f"{tag_prefix}eval/final_pegs", eval_final_pegs, current_episode
                    )
                    print(
                        f"[Eval {current_episode}] shape {current_shape_id} "
                        f"reward {eval_reward:.2f} avg_final_pegs {eval_final_pegs:.2f} "
                        f"max_start_pegs {curriculum_limits[current_shape_id]}"
                    )
                    if (
                        eval_final_pegs <= AVG_FINAL_PEGS_THRESHOLD
                        and curriculum_limits[current_shape_id] < CURRICULUM_MAX_LIMIT
                    ):
                        curriculum_limits[current_shape_id] = min(
                            curriculum_limits[current_shape_id] + 1,
                            full_pegs_map[current_shape_id],
                            CURRICULUM_MAX_LIMIT,
                        )
                        print(
                            f"[Episode {current_episode}] shape {current_shape_id} "
                            f"curriculum level up, max starting pegs "
                            f"{curriculum_limits[current_shape_id]}"
                        )

                if current_episode % CHECKPOINT_SAVE_INTERVAL == 0:
                    save_checkpoint(
                        model_path,
                        q_net,
                        optimizer,
                        current_episode,
                        steps_done,
                        shape_id,
                        tb_run_name,
                        tb_log_dir,
                        replay_capacity,
                        batch_size,
                        learning_rate,
                        curriculum_limits,
                        active_shapes,
                    )

                if current_episode % 100 == 0:
                    avg_last = (
                        sum(final_window) / max(len(final_window), 1)
                        if final_window
                        else final_pegs
                    )
                    print(
                        f"Episode {current_episode} shape {current_shape_id} "
                        f"reward {total_reward:.2f} "
                        f"final_pegs {final_pegs} "
                        f"avg_final_pegs_window {avg_last:.3f} "
                        f"max_start_pegs {curriculum_limits[current_shape_id]} "
                        f"buffer_size {sum(len(buf) for buf in replay_buffers.values())} "
                        f"epoch {epoch_idx}"
                    )

                epoch_episode_count += 1
                epoch_reward_total += total_reward
                episode += 1

            if epoch_episode_count > 0:
                avg_epoch_reward = epoch_reward_total / epoch_episode_count
                print(
                    f"[Epoch {epoch_idx}] collected {collected_steps} steps over "
                    f"{epoch_episode_count} episodes, avg reward {avg_epoch_reward:.2f}; "
                    f"buffer_size {sum(len(buf) for buf in replay_buffers.values())}"
                )

            available_shapes = [
                sid for sid, buf in replay_buffers.items() if len(buf) >= batch_size
            ]
            updates_to_run = TRAIN_UPDATES_PER_EPOCH if available_shapes else 0
            epoch_training_loss = 0.0
            log_interval = 1024  # max(1, updates_to_run // 10) if updates_to_run else 0
            if updates_to_run > 0:
                print(
                    f"[Epoch {epoch_idx}] starting {updates_to_run} training updates "
                    f"(batch {batch_size})",
                    flush=True,
                )
            for update_idx in range(1, updates_to_run + 1):
                chosen_shape = random.choice(available_shapes)
                batch = Transition(
                    *zip(*replay_buffers[chosen_shape].sample(batch_size))
                )
                loss = compute_td_loss(
                    q_net, target_net, batch, shape_ctxs[chosen_shape]
                )
                writer.add_scalar("train/loss", loss.item(), train_steps_taken)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
                optimizer.step()
                train_steps_taken += 1
                epoch_training_loss += loss.item()
                if log_interval and update_idx % log_interval == 0:
                    pct = int(update_idx / updates_to_run * 100)
                    print(
                        f"[Epoch {epoch_idx}] training updates "
                        f"{update_idx}/{updates_to_run} ({pct}%)",
                        flush=True,
                    )

            if updates_to_run == 0 and not available_shapes:
                print(
                    f"[Epoch {epoch_idx}] Skipped training updates "
                    f"(buffer size "
                    f"{sum(len(buf) for buf in replay_buffers.values())}/{batch_size})"
                )
            elif updates_to_run > 0:
                avg_train_loss = epoch_training_loss / updates_to_run
                print(
                    f"[Epoch {epoch_idx}] ran {updates_to_run} training updates "
                    f"(avg loss {avg_train_loss:.3f})"
                )
    finally:
        writer.close()

    ensure_models_dir()
    model_path = get_default_model_path(shape_id)
    save_checkpoint(
        model_path,
        q_net,
        optimizer,
        last_episode,
        steps_done,
        shape_id,
        tb_run_name,
        tb_log_dir,
        replay_capacity,
        batch_size,
        learning_rate,
        curriculum_limits,
        active_shapes,
    )
    print(f"Training finished, model saved to {model_path}")


def infer(
    shape_id: str = DEFAULT_SHAPE_ID,
    load_path: Path | str | None = None,
    episodes: int = DEFAULT_INFER_EPISODES,
    min_pegs: int = DEFAULT_INFER_MIN_PEGS,
    max_pegs: int | None = None,
    render: bool = False,
    use_search: bool = False,
) -> None:
    shape_ctxs = build_shape_contexts(SHAPES)
    if shape_id == ALL_SHAPES_ID:
        shared_load = (
            Path(load_path)
            if load_path is not None
            else get_default_model_path(ALL_SHAPES_ID)
        )
        _infer_multi(
            shape_ctxs,
            shared_load,
            episodes,
            min_pegs,
            max_pegs,
            render,
            use_search,
        )
        return
    if shape_id not in shape_ctxs:
        raise ValueError(f"Unknown shape '{shape_id}'")
    _infer_single(
        shape_ctxs[shape_id],
        load_path,
        episodes,
        min_pegs,
        max_pegs,
        render,
        use_search,
    )


def _load_q_network(model_path: Path) -> tuple[TokenAttentionQNetwork, str | None]:
    q_net = TokenAttentionQNetwork().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    tb_run_name: str | None = None
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        q_net.load_state_dict(checkpoint["model_state"], strict=False)
        tb_run_name = checkpoint.get("tb_run_name")
    else:
        q_net.load_state_dict(checkpoint, strict=False)
    q_net.eval()
    return q_net, tb_run_name


def _run_inference_episode(
    env: KongmingEnv,
    q_net: TokenAttentionQNetwork,
    shape_ctx: ShapeContext,
    target_pegs: int,
    render: bool,
    use_search: bool,
    episode_number: int,
) -> tuple[float, int, int]:
    state = env.reset_with_pegs(target_pegs)
    done = False
    episode_reward = 0.0
    steps = 0

    print(
        f"[Infer {episode_number}] shape={shape_ctx.shape.id} "
        f"start target_pegs={target_pegs}"
    )
    if render:
        render_cli_state(env)
        time.sleep(STEP_RENDER_DELAY)

    while not done and steps < MAX_STEPS_PER_EPISODE:
        legal_mask_np = env.legal_mask_numpy()
        if legal_mask_np.sum() == 0:
            reward = compute_final_reward(env, center_bonus=True)
            episode_reward += reward
            done = True
            print(
                f"[Infer {episode_number} step {steps}] shape={shape_ctx.shape.id} "
                f"no legal moves, added terminal reward {reward:.2f}"
            )
            if render:
                render_cli_state(env)
            time.sleep(STEP_RENDER_DELAY)
            break

        if use_search:
            search_state = env.state.astype(np.bool_, copy=True)
            action_idx, _ = markov_search_action(
                q_net,
                search_state,
                env.actions,
                env.idx_map.get(env.empty),
                shape_ctx,
                epsilon=0.0,
                collect_transitions=False,
            )
            if action_idx < 0 or not legal_mask_np[action_idx]:
                action_idx = _fallback_legal_action(legal_mask_np)
        else:
            action_idx = select_action(
                q_net, state, legal_mask_np, epsilon=0.0, shape_ctx=shape_ctx
            )
        if action_idx < 0:
            print(
                f"[Infer {episode_number} step {steps}] shape={shape_ctx.shape.id} "
                "no legal action, aborting"
            )
            time.sleep(STEP_RENDER_DELAY)
            break

        frm, to, jump = env.actions[action_idx]
        action_desc = (
            f"{env.holes[int(frm)]} -> {env.holes[int(to)]} via {env.holes[int(jump)]}"
        )

        state, reward, done = env.step(action_idx)
        steps += 1
        episode_reward += reward

        print(
            f"[Infer {episode_number} step {steps}] shape={shape_ctx.shape.id} "
            f"action {action_idx}: {action_desc} reward {reward:.2f} done {done}"
        )
        if render:
            render_cli_state(env)
        time.sleep(STEP_RENDER_DELAY)

    if not done:
        reward = compute_final_reward(env, center_bonus=True)
        episode_reward += reward

    final_pegs = int(env.state.sum())
    return episode_reward, final_pegs, steps


def _infer_single(
    shape_ctx: ShapeContext,
    load_path: Path | str | None,
    episodes: int,
    min_pegs: int,
    max_pegs: int | None,
    render: bool,
    use_search: bool,
) -> None:
    model_path = (
        Path(load_path) if load_path else get_default_model_path(shape_ctx.shape.id)
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    env = KongmingEnv(shape_ctx.shape)
    env.reset_full()
    q_net, tb_run_name = _load_q_network(model_path)
    if tb_run_name:
        print(f"Inference uses TensorBoard run '{tb_run_name}'")

    full_pegs = int(env.state.sum())
    max_pegs_val = DEFAULT_INFER_MAX_PEGS if max_pegs is None else max_pegs
    max_pegs_val = max(1, min(max_pegs_val, full_pegs))
    min_pegs_val = max(1, min(min_pegs, full_pegs))
    if min_pegs_val > max_pegs_val:
        raise ValueError("min_pegs must be <= max_pegs and within valid range")

    rewards: list[float] = []
    for episode in range(1, episodes + 1):
        target_pegs = random.randint(min_pegs_val, max_pegs_val)
        episode_reward, final_pegs, _ = _run_inference_episode(
            env,
            q_net,
            shape_ctx,
            target_pegs,
            render,
            use_search,
            episode,
        )
        rewards.append(episode_reward)
        print(
            f"[Infer {episode}] shape={shape_ctx.shape.id} reward {episode_reward:.2f} "
            f"final_pegs {final_pegs}"
        )
        print("-" * 40)

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(
        f"Inference finished for shape '{shape_ctx.shape.id}' "
        f"({episodes} episodes), avg reward {avg_reward:.2f}"
    )


def _infer_multi(
    shape_ctxs: dict[str, ShapeContext],
    load_path: Path,
    episodes: int,
    min_pegs: int,
    max_pegs: int | None,
    render: bool,
    use_search: bool,
) -> None:
    if not load_path.exists():
        raise FileNotFoundError(
            f"Model file not found for multi-shape inference: {load_path}"
        )
    q_net, tb_run_name = _load_q_network(load_path)
    if tb_run_name:
        print(f"Inference uses TensorBoard run '{tb_run_name}'")
    shape_ids = sorted(shape_ctxs.keys())
    if not shape_ids:
        print("No shapes available for inference.")
        return
    envs = {sid: KongmingEnv(shape_ctxs[sid].shape) for sid in shape_ids}
    for env in envs.values():
        env.reset_full()
    full_pegs = {sid: int(envs[sid].state.sum()) for sid in shape_ids}
    rewards: dict[str, list[float]] = {sid: [] for sid in shape_ids}

    print(f"Running inference across shapes {shape_ids} for {episodes} total episodes.")

    for episode_idx in range(1, episodes + 1):
        sid = shape_ids[(episode_idx - 1) % len(shape_ids)]
        env = envs[sid]
        shape_ctx = shape_ctxs[sid]
        full_pegs_val = full_pegs[sid]
        max_pegs_val = max(
            1,
            min(
                max_pegs if max_pegs is not None else DEFAULT_INFER_MAX_PEGS,
                full_pegs_val,
            ),
        )
        min_pegs_val = max(1, min(min_pegs, full_pegs_val))
        if min_pegs_val > max_pegs_val:
            raise ValueError(
                f"min_pegs must be <= max_pegs for shape '{sid}' (available up to {full_pegs_val})"
            )
        target_pegs = random.randint(min_pegs_val, max_pegs_val)
        episode_reward, final_pegs, _ = _run_inference_episode(
            env,
            q_net,
            shape_ctx,
            target_pegs,
            render,
            use_search,
            episode_idx,
        )
        rewards[sid].append(episode_reward)
        print(
            f"[Infer {episode_idx}] shape={sid} reward {episode_reward:.2f} "
            f"final_pegs {final_pegs}"
        )
        print("-" * 40)

    for sid in shape_ids:
        if not rewards[sid]:
            continue
        avg_reward = sum(rewards[sid]) / len(rewards[sid])
        print(
            f"[Summary] shape={sid} episodes={len(rewards[sid])} "
            f"avg reward {avg_reward:.2f}"
        )


def export_model(
    shape_id: str = ALL_SHAPES_ID,
    load_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> None:
    """Export a trained model to ONNX for browser execution."""
    shape_ctxs = build_shape_contexts(SHAPES)
    if shape_id == ALL_SHAPES_ID:
        shared_load = (
            Path(load_path)
            if load_path is not None
            else get_default_model_path(ALL_SHAPES_ID)
        )
        for sid in sorted(shape_ctxs.keys()):
            print(f"=== Exporting shape '{sid}' ===")
            _export_single_shape(shape_ctxs[sid], shared_load, output_path)
        return
    if shape_id not in shape_ctxs:
        raise ValueError(f"Unknown shape '{shape_id}'")
    _export_single_shape(shape_ctxs[shape_id], load_path, output_path)


def _export_single_shape(
    shape_ctx: ShapeContext,
    load_path: Path | str | None,
    output_path: Path | str | None,
) -> None:
    shape_id = shape_ctx.shape.id
    model_path = (
        get_default_model_path(shape_id) if load_path is None else Path(load_path)
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    q_net = TokenAttentionQNetwork().to("cpu")
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        q_net.load_state_dict(checkpoint["model_state"], strict=False)
    else:
        q_net.load_state_dict(checkpoint, strict=False)
    q_net.eval()

    if output_path is None:
        resolved_output = model_path.with_name(f"{MODEL_NAME_BASE}_{shape_id}.onnx")
    else:
        resolved_output = Path(output_path)
        if resolved_output.is_dir():
            resolved_output = resolved_output / f"{MODEL_NAME_BASE}_{shape_id}.onnx"

    class _ShapeBoundModel(nn.Module):
        def __init__(self, net: TokenAttentionQNetwork, ctx: ShapeContext):
            super().__init__()
            self.net = net
            self.shape_ctx = ctx

        def forward(self, state: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(state, self.shape_ctx)

    dummy_state = torch.zeros(
        1, len(shape_ctx.shape.holes), dtype=torch.float32, device="cpu"
    )
    export_model_module = _ShapeBoundModel(q_net, shape_ctx)
    torch.onnx.export(
        export_model_module,
        dummy_state,
        resolved_output,
        input_names=["state"],
        output_names=["q_values"],
        dynamic_axes={"state": {0: "batch"}, "q_values": {0: "batch"}},
        opset_version=17,
        export_params=True,
        external_data=False,
    )
    print(f"Exported ONNX model for '{shape_id}' to {resolved_output}")


def main() -> None:
    _TRAIN_ARGS = argparse.ArgumentParser(add_help=False)
    _TRAIN_ARGS.add_argument(
        "--shape",
        "-s",
        choices=sorted(list(SHAPES.keys()) + [ALL_SHAPES_ID]),
        default=DEFAULT_SHAPE_ID,
        help="Board shape to operate on.",
    )
    _TRAIN_ARGS.add_argument(
        "--episodes",
        "-n",
        type=int,
        default=None,
        help="Number of episodes to train.",
    )
    _TRAIN_ARGS.add_argument(
        "--replay",
        "-R",
        type=int,
        default=DEFAULT_REPLAY_CAPACITY,
        help="Replay buffer capacity during training.",
    )
    _TRAIN_ARGS.add_argument(
        "--batch",
        "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for DQN updates.",
    )
    _TRAIN_ARGS.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate for the optimizer during training.",
    )
    _TRAIN_ARGS.add_argument(
        "--load",
        "-l",
        type=Path,
        help="Path to a saved model to load before running.",
    )

    _INFER_ARGS = argparse.ArgumentParser(add_help=False)
    _INFER_ARGS.add_argument(
        "--shape",
        "-s",
        choices=sorted(list(SHAPES.keys()) + [ALL_SHAPES_ID]),
        default=DEFAULT_SHAPE_ID,
        help="Board shape to operate on.",
    )
    _INFER_ARGS.add_argument(
        "--episodes",
        "-n",
        type=int,
        default=None,
        help="Number of episodes to run inference for.",
    )
    _INFER_ARGS.add_argument(
        "--load",
        "-l",
        type=Path,
        help="Path to a saved model to load before running.",
    )
    _INFER_ARGS.add_argument(
        "--min-pegs",
        type=int,
        default=DEFAULT_INFER_MIN_PEGS,
        help="Minimum number of pegs to start an inference episode with.",
    )
    _INFER_ARGS.add_argument(
        "--max-pegs",
        type=int,
        default=None,
        help="Maximum number of pegs to start an inference episode with.",
    )
    _INFER_ARGS.add_argument(
        "--render",
        "-r",
        action="store_true",
        help="Render the board at every inference step.",
    )
    _INFER_ARGS.add_argument(
        "--no-search",
        dest="search",
        action="store_false",
        help="Skip the Markov search and act greedily.",
    )
    _INFER_ARGS.set_defaults(search=True)

    _EXPORT_ARGS = argparse.ArgumentParser(add_help=False)
    _EXPORT_ARGS.add_argument(
        "--shape",
        "-s",
        choices=sorted(list(SHAPES.keys()) + [ALL_SHAPES_ID]),
        default=ALL_SHAPES_ID,
        help="Board shape to operate on.",
    )
    _EXPORT_ARGS.add_argument(
        "--load",
        "-l",
        type=Path,
        help="Path to a saved model checkpoint to export.",
    )
    _EXPORT_ARGS.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Destination path for the exported ONNX file.",
    )
    parser = argparse.ArgumentParser(
        description="Train or run the Kongming DQN agent.", parents=[_TRAIN_ARGS]
    )
    parser.set_defaults(mode="train")
    subparsers = parser.add_subparsers(dest="mode")

    train_parser = subparsers.add_parser(
        "train", help="Train a DQN agent.", parents=[_TRAIN_ARGS]
    )
    train_parser.set_defaults(mode="train")

    infer_parser = subparsers.add_parser(
        "infer", help="Run inference using a trained agent.", parents=[_INFER_ARGS]
    )
    infer_parser.set_defaults(mode="infer")

    export_parser = subparsers.add_parser(
        "export",
        help="Export a trained agent to ONNX for browser inference.",
        parents=[_EXPORT_ARGS],
    )
    export_parser.set_defaults(mode="export")

    args = parser.parse_args()

    if args.mode == "train":
        train(
            shape_id=args.shape,
            load_path=args.load,
            max_episodes=args.episodes or NUM_EPISODES,
            replay_capacity=args.replay,
            batch_size=args.batch,
            learning_rate=args.lr,
        )
    elif args.mode == "infer":
        infer(
            shape_id=args.shape,
            load_path=args.load,
            episodes=args.episodes or DEFAULT_INFER_EPISODES,
            min_pegs=args.min_pegs,
            max_pegs=args.max_pegs,
            render=args.render,
            use_search=args.search,
        )
    else:
        export_model(
            shape_id=args.shape,
            load_path=args.load,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
