"""Train/inference runner for the Kongming Chess DQN agent."""

import argparse
import math
import random
import time
from collections import deque, namedtuple
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_SHAPE_ID = "cross"
DEFAULT_REPLAY_CAPACITY = 10_000
DEFAULT_BATCH_SIZE = 2048
DEFAULT_LR = 1e-3
GAMMA = 0.99

NUM_EPISODES = 100_000_000
MAX_STEPS_PER_EPISODE = 200

TARGET_UPDATE_INTERVAL = 500  # episodes
CURRICULUM_EVAL_INTERVAL = 1000
EVAL_EPISODES_PER_CHECK = 150

MODELS_DIR = Path("models")
DEFAULT_INFER_EPISODES = 1000
STEP_RENDER_DELAY = 0.5

CURRICULUM_MIN_PEGS = 3
CURRICULUM_START_MAX_PEGS = 3
CURRICULUM_MAX_LIMIT = 32
CURRICULUM_WINDOW = 500
AVG_FINAL_PEGS_THRESHOLD = 1.1

DEFAULT_INFER_MIN_PEGS = CURRICULUM_MIN_PEGS
DEFAULT_INFER_MAX_PEGS = CURRICULUM_START_MAX_PEGS

EPSILON_START = 0.05
EPSILON_END = 0.05
EPSILON_DECAY = 10_000

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
    return MODELS_DIR / f"dqn_kongming_{shape_id}.pt"


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
    return f"dqn_{shape_id}_{timestamp}"


def create_summary_writer(
    shape_id: str, run_name: str | None = None
) -> tuple[SummaryWriter | DummyWriter, str, str]:
    if SummaryWriter is None:
        return DummyWriter(), run_name or "tb_disabled", ""
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
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
    max_initial_pegs: int,
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
    torch.save(checkpoint, path)


def select_action(
    q_net: TokenAttentionQNetwork,
    state: torch.Tensor,
    legal_mask: np.ndarray,
    epsilon: float,
) -> int:
    if random.random() < epsilon:
        legal_indices = np.nonzero(legal_mask)[0]
        if len(legal_indices) == 0:
            return -1
        return int(random.choice(legal_indices))

    with torch.no_grad():
        s = state.unsqueeze(0).to(DEVICE)
        q_values = q_net(s).squeeze(0)

        legal = torch.from_numpy(legal_mask).to(DEVICE)
        if not legal.any():
            return -1

        q_values_masked = q_values.clone()
        q_values_masked[~legal] = -1e9
        return int(torch.argmax(q_values_masked).item())


def compute_td_loss(
    q_net: TokenAttentionQNetwork,
    target_net: TokenAttentionQNetwork,
    batch: Transition,
) -> torch.Tensor:
    states = torch.stack(batch.state).to(DEVICE)
    actions = torch.tensor(batch.action, dtype=torch.long, device=DEVICE)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
    next_states = torch.stack(batch.next_state).to(DEVICE)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)
    next_masks = torch.stack(
        [torch.from_numpy(m.astype(np.bool_)) for m in batch.next_legal_mask]
    ).to(DEVICE)

    q_values = q_net(states)
    state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states)
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

                action_idx = select_action(q_net, state, legal_mask_np, epsilon=0.0)
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
    if shape_id not in shape_ctxs:
        raise ValueError(f"Unknown shape '{shape_id}'")
    shape_ctx = shape_ctxs[shape_id]

    env = KongmingEnv(shape_ctx.shape)
    env.reset_full()
    full_pegs = int(env.state.sum())
    max_curriculum_pegs = min(CURRICULUM_START_MAX_PEGS, full_pegs)

    q_net = TokenAttentionQNetwork(shape_ctx).to(DEVICE)
    target_net = TokenAttentionQNetwork(shape_ctx).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    replay = ReplayBuffer(replay_capacity)

    start_episode = 1
    steps_done = 0
    tb_run_name = None
    tb_log_dir = ""

    if load_path is not None:
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model to load not found: {load_path}")
        checkpoint = torch.load(load_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            q_net.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_episode = checkpoint.get("episode", 0) + 1
            steps_done = checkpoint.get("steps_done", 0)
            tb_run_name = checkpoint.get("tb_run_name")
            tb_log_dir = checkpoint.get("tb_log_dir", "")
            max_curriculum_pegs = checkpoint.get(
                "max_initial_pegs", max_curriculum_pegs
            )
        else:
            q_net.load_state_dict(checkpoint)
            start_episode = 1
        target_net.load_state_dict(q_net.state_dict())
        target_net.eval()
        print(f"Loaded pretrained weights from {load_path}")

    writer, tb_run_name, tb_log_dir = create_summary_writer(
        shape_id, run_name=tb_run_name
    )
    episode_final_pegs = deque(maxlen=CURRICULUM_WINDOW)

    ensure_models_dir()
    model_path = get_default_model_path(shape_id)

    end_episode = start_episode + max_episodes - 1
    last_episode = start_episode - 1

    try:
        for episode in range(start_episode, end_episode + 1):
            last_episode = episode
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(
                -1.0 * steps_done / EPSILON_DECAY
            )

            min_pegs = CURRICULUM_MIN_PEGS
            max_pegs = max_curriculum_pegs
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
                    replay.push(
                        state.cpu(),
                        -1,
                        reward,
                        next_state.cpu(),
                        True,
                        next_legal_mask,
                    )
                    break

                action_idx = select_action(q_net, state, legal_mask_np, epsilon)
                if action_idx < 0:
                    break

                next_state, reward, done = env.step(action_idx)
                total_reward += reward

                next_legal_mask = (
                    env.legal_mask_numpy()
                    if not done
                    else np.zeros_like(legal_mask_np, dtype=np.bool_)
                )

                replay.push(
                    state.cpu(),
                    action_idx,
                    reward,
                    next_state.cpu(),
                    done,
                    next_legal_mask,
                )

                state = next_state
                step_count += 1
                steps_done += 1

                if len(replay) >= batch_size:
                    batch = Transition(*zip(*replay.sample(batch_size)))
                    loss = compute_td_loss(q_net, target_net, batch)
                    writer.add_scalar("train/loss", loss.item(), steps_done)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
                    optimizer.step()

            final_pegs = int(env.state.sum())
            episode_final_pegs.append(final_pegs)
            avg_final_pegs = (
                sum(episode_final_pegs) / len(episode_final_pegs)
                if episode_final_pegs
                else final_pegs
            )
            writer.add_scalar("episode/reward", total_reward, episode)
            writer.add_scalar("episode/avg_final_pegs", avg_final_pegs, episode)
            writer.add_scalar("episode/max_initial_pegs", max_curriculum_pegs, episode)

            if episode % TARGET_UPDATE_INTERVAL == 0:
                target_net.load_state_dict(q_net.state_dict())
                print(
                    f"[Episode {episode}] Target network updated. "
                    f"Last episode reward {total_reward:.2f}, final pegs {final_pegs}"
                )

            if episode % CURRICULUM_EVAL_INTERVAL == 0:
                eval_reward, eval_final_pegs = evaluate_policy(
                    q_net,
                    shape_ctx,
                    max_curriculum_pegs,
                    episodes=EVAL_EPISODES_PER_CHECK,
                )
                writer.add_scalar("eval/reward", eval_reward, episode)
                writer.add_scalar("eval/final_pegs", eval_final_pegs, episode)
                print(
                    f"[Eval {episode}] reward {eval_reward:.2f} avg_final_pegs {eval_final_pegs:.2f} "
                    f"max_start_pegs {max_curriculum_pegs}"
                )
                if (
                    eval_final_pegs <= AVG_FINAL_PEGS_THRESHOLD
                    and max_curriculum_pegs < CURRICULUM_MAX_LIMIT
                ):
                    max_curriculum_pegs = min(
                        max_curriculum_pegs + 1, full_pegs, CURRICULUM_MAX_LIMIT
                    )
                    print(
                        f"[Episode {episode}] Curriculum level up via eval, "
                        f"max starting pegs is now {max_curriculum_pegs}"
                    )

            if episode % CHECKPOINT_SAVE_INTERVAL == 0:
                save_checkpoint(
                    model_path,
                    q_net,
                    optimizer,
                    episode,
                    steps_done,
                    shape_id,
                    tb_run_name,
                    tb_log_dir,
                    replay_capacity,
                    batch_size,
                    learning_rate,
                    max_curriculum_pegs,
                )

            if episode % 100 == 0:
                avg_last = sum(episode_final_pegs) / max(len(episode_final_pegs), 1)
                print(
                    f"Episode {episode} "
                    f"reward {total_reward:.2f} "
                    f"final_pegs {final_pegs} "
                    f"avg_final_pegs_window {avg_last:.3f} "
                    f"max_start_pegs {max_curriculum_pegs} "
                    f"buffer_size {len(replay)}"
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
        max_curriculum_pegs,
    )
    print(f"Training finished, model saved to {model_path}")


def infer(
    shape_id: str = DEFAULT_SHAPE_ID,
    load_path: Path | str | None = None,
    episodes: int = DEFAULT_INFER_EPISODES,
    min_pegs: int = DEFAULT_INFER_MIN_PEGS,
    max_pegs: int | None = None,
    render: bool = False,
) -> None:
    shape_ctxs = build_shape_contexts(SHAPES)
    if shape_id not in shape_ctxs:
        raise ValueError(f"Unknown shape '{shape_id}'")
    shape_ctx = shape_ctxs[shape_id]

    env = KongmingEnv(shape_ctx.shape)
    q_net = TokenAttentionQNetwork(shape_ctx).to(DEVICE)
    model_path = Path(load_path) if load_path else get_default_model_path(shape_id)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    tb_run_name = None
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        q_net.load_state_dict(checkpoint["model_state"])
        tb_run_name = checkpoint.get("tb_run_name")
    else:
        q_net.load_state_dict(checkpoint)
    q_net.eval()
    if tb_run_name:
        print(f"Inference uses TensorBoard run '{tb_run_name}'")

    full_pegs = int(env.state.sum())
    max_pegs = DEFAULT_INFER_MAX_PEGS if max_pegs is None else max_pegs
    max_pegs = max(1, min(max_pegs, full_pegs))
    min_pegs = max(1, min(min_pegs, full_pegs))
    if min_pegs > max_pegs:
        raise ValueError("min_pegs must be <= max_pegs and within valid range")

    rewards: list[float] = []
    for episode in range(1, episodes + 1):
        target_pegs = random.randint(min_pegs, max_pegs)
        state = env.reset_with_pegs(target_pegs)
        done = False
        episode_reward = 0.0
        steps = 0

        print(f"[Infer {episode}] start target_pegs={target_pegs}")
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
                    f"[Infer {episode} step {steps}] no legal moves, "
                    f"added terminal reward {reward:.2f}"
                )
                if render:
                    render_cli_state(env)
                time.sleep(STEP_RENDER_DELAY)
                break

            action_idx = select_action(q_net, state, legal_mask_np, epsilon=0.0)
            if action_idx < 0:
                print(f"[Infer {episode} step {steps}] no legal action, aborting")
                time.sleep(STEP_RENDER_DELAY)
                break

            frm, to, jump = env.actions[action_idx]
            action_desc = f"{env.holes[int(frm)]} -> {env.holes[int(to)]} via {env.holes[int(jump)]}"

            state, reward, done = env.step(action_idx)
            steps += 1
            episode_reward += reward

            print(
                f"[Infer {episode} step {steps}] action {action_idx}: {action_desc} "
                f"reward {reward:.2f} done {done}"
            )
            if render:
                render_cli_state(env)
            time.sleep(STEP_RENDER_DELAY)

        if not done:
            reward = compute_final_reward(env, center_bonus=True)
            episode_reward += reward

        final_pegs = int(env.state.sum())
        rewards.append(episode_reward)
        print(
            f"[Infer {episode}] reward {episode_reward:.2f} "
            f"final_pegs {final_pegs} steps {steps}"
        )
        if render:
            render_cli_state(env)
        print("-" * 40)

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"Inference finished ({episodes} episodes), avg reward {avg_reward:.2f}")


def main() -> None:
    _TRAIN_ARGS = argparse.ArgumentParser(add_help=False)
    _TRAIN_ARGS.add_argument(
        "--shape",
        "-s",
        choices=sorted(SHAPES.keys()),
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
        choices=sorted(SHAPES.keys()),
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
    else:
        infer(
            shape_id=args.shape,
            load_path=args.load,
            episodes=args.episodes or DEFAULT_INFER_EPISODES,
            min_pegs=args.min_pegs,
            max_pegs=args.max_pegs,
            render=args.render,
        )


if __name__ == "__main__":
    main()
