"""
Board rules, shape utilities, and environment helpers for Kongming Chess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import random
import numpy as np
import torch

from numba_kernels import generate_backward_state_kernel, legal_mask_kernel


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


@dataclass
class ShapeDef:
    id: str
    holes: List[str]
    empty: str
    allowed: Dict[str, Dict[str, str]]
    width: int
    height: int


def _sorted_coords(coords: List[str]) -> List[str]:
    return sorted(coords, key=lambda cell: tuple(map(int, cell.split(","))))


def _generate_european_holes(cross_holes: List[str]) -> List[str]:
    base = set(cross_holes)
    for cell in ["1,1", "1,5", "5,1", "5,5"]:
        base.add(cell)
    return _sorted_coords(list(base))


def _generate_german_holes() -> List[str]:
    base: List[str] = []
    for r in range(9):
        for c in range(9):
            if (3 <= r <= 5) or (3 <= c <= 5):
                base.append(f"{r},{c}")
    return _sorted_coords(base)


def _generate_asym_cross_holes() -> List[str]:
    base: List[str] = []
    for r in range(8):
        for c in range(9):
            vertical_arm = 3 <= c <= 5
            horizontal_arm = 3 <= r <= 5 and 1 <= c <= 8
            if vertical_arm or horizontal_arm:
                base.append(f"{r},{c}")
    return _sorted_coords(base)


def _generate_diamond_holes(size: int = 9, radius: int = 4) -> List[str]:
    base: List[str] = []
    center = size // 2
    for r in range(size):
        for c in range(size):
            if abs(r - center) + abs(c - center) <= radius:
                base.append(f"{r},{c}")
    return _sorted_coords(base)


def create_shapes() -> Dict[str, ShapeDef]:
    cross_holes: List[str] = []
    for r in range(7):
        for c in range(7):
            if (2 <= r <= 4) or (2 <= c <= 4):
                cross_holes.append(f"{r},{c}")
    cross_axes = build_grid_axes(cross_holes, 7, 7)
    cross_moves = build_allowed_moves_from_axes(cross_axes)

    tri_w, tri_h = 9, 5
    tri_rows = build_triangle_rows(tri_w, tri_h)
    tri_holes = [cell for row in tri_rows for cell in row]
    tri_axes = (
        tri_rows
        + collect_triangle_axes(tri_rows, 1, 0)
        + collect_triangle_axes(tri_rows, 1, 1)
    )
    tri_moves = build_allowed_moves_from_axes(tri_axes)

    european_holes = _generate_european_holes(cross_holes)
    european_axes = build_grid_axes(european_holes, 7, 7)
    european_moves = build_allowed_moves_from_axes(european_axes)

    german_holes = _generate_german_holes()
    german_axes = build_grid_axes(german_holes, 9, 9)
    german_moves = build_allowed_moves_from_axes(german_axes)

    asym_cross_holes = _generate_asym_cross_holes()
    asym_cross_axes = build_grid_axes(asym_cross_holes, 9, 8)
    asym_cross_moves = build_allowed_moves_from_axes(asym_cross_axes)

    diamond_holes = _generate_diamond_holes()
    diamond_axes = build_grid_axes(diamond_holes, 9, 9)
    diamond_moves = build_allowed_moves_from_axes(diamond_axes)

    return {
        "cross": ShapeDef(
            id="cross",
            holes=cross_holes,
            empty="3,3",
            allowed=cross_moves,
            width=7,
            height=7,
        ),
        "triangle": ShapeDef(
            id="triangle",
            holes=tri_holes,
            empty="0,4",
            allowed=tri_moves,
            width=9,
            height=5,
        ),
        "european": ShapeDef(
            id="european",
            holes=european_holes,
            empty="2,3",
            allowed=european_moves,
            width=7,
            height=7,
        ),
        "diamond": ShapeDef(
            id="diamond",
            holes=diamond_holes,
            empty="4,4",
            allowed=diamond_moves,
            width=9,
            height=9,
        ),
        "german": ShapeDef(
            id="german",
            holes=german_holes,
            empty="4,4",
            allowed=german_moves,
            width=9,
            height=9,
        ),
        "asym-cross": ShapeDef(
            id="asym-cross",
            holes=asym_cross_holes,
            empty="4,4",
            allowed=asym_cross_moves,
            width=9,
            height=8,
        ),
    }


SHAPES = create_shapes()


def normalize_coords(
    coords: List[Tuple[int, int]], width: int, height: int
) -> torch.Tensor:
    if width <= 1:
        width = 2
    if height <= 1:
        height = 2
    normed: List[List[float]] = []
    for r, c in coords:
        r_n = (r / (height - 1)) * 2 - 1
        c_n = (c / (width - 1)) * 2 - 1
        normed.append([r_n, c_n])
    return torch.tensor(normed, dtype=torch.float32)


@dataclass
class ShapeContext:
    shape: ShapeDef
    idx_map: Dict[str, int]
    actions: torch.Tensor
    coords: torch.Tensor
    shape_idx: int


def build_shape_contexts(shapes: Dict[str, ShapeDef]) -> Dict[str, ShapeContext]:
    ctxs: Dict[str, ShapeContext] = {}
    for idx, shape in enumerate(shapes.values()):
        idx_map = {h: i for i, h in enumerate(shape.holes)}
        actions: List[List[int]] = []
        for frm, dests in shape.allowed.items():
            for to, jump in dests.items():
                actions.append([idx_map[frm], idx_map[to], idx_map[jump]])
        coords = normalize_coords(
            [tuple(map(int, h.split(","))) for h in shape.holes],
            shape.width,
            shape.height,
        )
        ctxs[shape.id] = ShapeContext(
            shape=shape,
            idx_map=idx_map,
            actions=torch.tensor(actions, dtype=torch.long),
            coords=coords,
            shape_idx=idx,
        )
    return ctxs


class KongmingEnv:
    def __init__(self, shape: ShapeDef):
        self.shape = shape
        self.holes = shape.holes
        self.empty = shape.empty
        self.allowed = shape.allowed
        self.idx_map: Dict[str, int] = {h: i for i, h in enumerate(self.holes)}
        self.actions = self._build_actions()
        self.state = np.zeros(len(self.holes), dtype=np.bool_)
        self.reset_full()

    def _build_actions(self) -> np.ndarray:
        arr: List[List[int]] = []
        for frm, dests in self.allowed.items():
            for to, jump in dests.items():
                arr.append([self.idx_map[frm], self.idx_map[to], self.idx_map[jump]])
        return np.array(arr, dtype=np.int32)

    def reset_full(self) -> torch.Tensor:
        self.state[:] = False
        self.state[list(self.idx_map.values())] = True
        if self.empty in self.idx_map:
            self.state[self.idx_map[self.empty]] = False
        return self.obs()

    def reset_with_pegs(self, target_pegs: int) -> torch.Tensor:
        generated = self._generate_backward_state(target_pegs)
        if generated is not None:
            self.state = generated
            return self.obs()

        self.reset_full()
        current_pegs = int(self.state.sum())
        to_remove = max(0, current_pegs - target_pegs)
        if to_remove > 0:
            removable = list(np.where(self.state)[0])
            drop = random.sample(removable, k=min(to_remove, len(removable)))
            self.state[drop] = False
        return self.obs()

    def _generate_backward_state(
        self, target_pegs: int, max_attempts: int = 64
    ) -> np.ndarray | None:
        empty_idx = self.idx_map.get(self.empty, -1)
        state, success = generate_backward_state_kernel(
            target_pegs,
            empty_idx,
            self.actions,
            max_attempts,
            self.state.shape[0],
        )
        if success:
            return state
        return None

    def obs(self) -> torch.Tensor:
        return torch.from_numpy(self.state.astype(np.float32))

    def legal_mask(self) -> np.ndarray:
        return legal_mask_kernel(self.state, self.actions)

    def legal_mask_numpy(self) -> np.ndarray:
        return self.legal_mask()

    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool]:
        frm, to, jump = self.actions[action_idx]
        if not (self.state[frm] and (not self.state[to]) and self.state[jump]):
            return self.obs(), -1.0, True
        self.state[frm] = False
        self.state[jump] = False
        self.state[to] = True
        mask = self.legal_mask()
        done = (self.state.sum() == 1) or (mask.sum() == 0)
        reward = 0.0
        if done:
            reward = compute_final_reward(self, center_bonus=True)
        return self.obs(), reward, done


def compute_final_reward(env: KongmingEnv, center_bonus: bool = True) -> float:
    pegs = int(env.state.sum())
    reward = -float(pegs)
    center_idx = env.idx_map.get(env.empty)
    if center_bonus and center_idx is not None and env.state[center_idx]:
        reward += 5.0
    return reward


def render_cli_state(env: KongmingEnv) -> None:
    peg_sym = "O"
    empty_sym = "."
    lines: List[str] = []
    idx_map = env.idx_map

    if env.shape.id in {"cross", "european", "german", "asym-cross", "diamond"}:
        size = env.shape.width
        for r in range(size):
            row_cells: List[str] = []
            for c in range(size):
                key = f"{r},{c}"
                if key not in idx_map:
                    row_cells.append(" ")
                    continue
                idx = idx_map[key]
                row_cells.append(peg_sym if env.state[idx] else empty_sym)
            lines.append("".join(row_cells))
    else:
        triangle_rows = build_triangle_rows(env.shape.width, env.shape.height)
        for row in triangle_rows:
            row_cells = [
                peg_sym if env.state[idx_map[hole]] else empty_sym for hole in row
            ]
            lines.append(" ".join(row_cells).center(env.shape.width * 2))

    board_width = max((len(line) for line in lines), default=0)
    pegs_left = int(env.state.sum())
    legal_moves = int(env.legal_mask().sum())
    center_idx = idx_map.get(env.empty)
    center_status = "unknown"
    if center_idx is not None:
        center_status = "filled" if env.state[center_idx] else "empty"

    stats = (
        f"Pegs left: {pegs_left} | Legal moves: {legal_moves} | Center: {center_status}"
    )
    separator = "-" * max(board_width, len(stats))

    print("\n".join(lines))
    print(separator)
    print(stats)
