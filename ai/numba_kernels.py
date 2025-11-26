"""Numba-accelerated kernels for Kongming Chess helpers."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def legal_mask_kernel(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Compute legal moves for a state/actions pair."""
    num_actions = actions.shape[0]
    mask = np.empty(num_actions, dtype=np.bool_)
    for idx in range(num_actions):
        frm = actions[idx, 0]
        to = actions[idx, 1]
        jump = actions[idx, 2]
        mask[idx] = state[frm] and (not state[to]) and state[jump]
    return mask


@njit(cache=True)
def simulate_action_kernel(
    state: np.ndarray, action_idx: int, actions: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Apply an action and return (next_state, is_valid)."""
    next_state = state.copy()
    frm = actions[action_idx, 0]
    to = actions[action_idx, 1]
    jump = actions[action_idx, 2]
    if not (next_state[frm] and (not next_state[to]) and next_state[jump]):
        return next_state, False
    next_state[frm] = False
    next_state[jump] = False
    next_state[to] = True
    return next_state, True


@njit(cache=True)
def peg_count_kernel(state: np.ndarray) -> int:
    total = 0
    for i in range(state.shape[0]):
        if state[i]:
            total += 1
    return total


@njit(cache=True)
def mask_any_kernel(mask: np.ndarray) -> bool:
    for i in range(mask.shape[0]):
        if mask[i]:
            return True
    return False


@njit(cache=True)
def terminal_reward_kernel(state: np.ndarray, center_idx: int) -> float:
    reward = -float(peg_count_kernel(state))
    if center_idx >= 0 and center_idx < state.shape[0] and state[center_idx]:
        reward += 5.0
    return reward


@njit(cache=True)
def reverse_legal_moves_kernel(
    state: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, int]:
    num_actions = actions.shape[0]
    reverse_moves = np.empty((num_actions, 3), dtype=np.int32)
    count = 0
    for idx in range(num_actions):
        frm = actions[idx, 0]
        to = actions[idx, 1]
        jump = actions[idx, 2]
        if state[to] and (not state[frm]) and (not state[jump]):
            reverse_moves[count, 0] = frm
            reverse_moves[count, 1] = to
            reverse_moves[count, 2] = jump
            count += 1
    return reverse_moves, count


@njit(cache=True)
def generate_backward_state_kernel(
    target_pegs: int,
    empty_idx: int,
    actions: np.ndarray,
    max_attempts: int,
    num_holes: int,
) -> tuple[np.ndarray, bool]:
    if target_pegs <= 1:
        state = np.zeros(num_holes, dtype=np.bool_)
        if 0 <= empty_idx < num_holes:
            state[empty_idx] = True
        return state, True

    for _ in range(max_attempts):
        state = np.zeros(num_holes, dtype=np.bool_)
        if 0 <= empty_idx < num_holes:
            state[empty_idx] = True
        steps = 0
        while peg_count_kernel(state) < target_pegs:
            moves, count = reverse_legal_moves_kernel(state, actions)
            if count == 0:
                break
            move_idx = np.random.randint(count)
            frm = moves[move_idx, 0]
            to = moves[move_idx, 1]
            jump = moves[move_idx, 2]
            state[frm] = True
            state[jump] = True
            state[to] = False
            steps += 1
            if steps > target_pegs * 4:
                break
        if peg_count_kernel(state) == target_pegs:
            return state, True
    return np.zeros(num_holes, dtype=np.bool_), False


@njit(cache=True)
def select_branch_actions_kernel(
    q_values: np.ndarray,
    legal_mask: np.ndarray,
    branching_factor: int,
    epsilon: float,
) -> np.ndarray:
    num_actions = q_values.shape[0]
    legal_indices = np.empty(num_actions, dtype=np.int32)
    count = 0
    for idx in range(num_actions):
        if legal_mask[idx]:
            legal_indices[count] = idx
            count += 1
    if count == 0:
        return np.empty(0, dtype=np.int32)

    if branching_factor >= count:
        result = legal_indices[:count].copy()
        for i in range(count - 1, 0, -1):
            j = np.random.randint(i + 1)
            tmp = result[i]
            result[i] = result[j]
            result[j] = tmp
        return result

    top_k = branching_factor if branching_factor < count else count
    scores = np.empty(count, dtype=np.float32)
    for i in range(count):
        scores[i] = q_values[legal_indices[i]]
    order = np.argsort(-scores)
    result = np.empty(top_k, dtype=np.int32)
    for i in range(top_k):
        result[i] = legal_indices[order[i]]

    if epsilon > 0.0 and count > top_k:
        random_slots = int(epsilon * top_k)
        if random_slots < 1:
            random_slots = 1
        replaced = 0
        attempts = 0
        max_attempts = 64 * random_slots
        while replaced < random_slots and attempts < max_attempts:
            candidate = legal_indices[np.random.randint(count)]
            duplicate = False
            for j in range(top_k):
                if result[j] == candidate:
                    duplicate = True
                    break
            if duplicate:
                attempts += 1
                continue
            replace_idx = np.random.randint(top_k)
            result[replace_idx] = candidate
            replaced += 1
        for i in range(top_k - 1, 0, -1):
            j = np.random.randint(i + 1)
            tmp = result[i]
            result[i] = result[j]
            result[j] = tmp
    else:
        for i in range(top_k - 1, 0, -1):
            j = np.random.randint(i + 1)
            tmp = result[i]
            result[i] = result[j]
            result[j] = tmp

    return result
