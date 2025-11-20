"""Token-based Q-Network for the Kongming Chess environment."""

import torch
import torch.nn as nn

from environment import ShapeContext


class TokenAttentionQNetwork(nn.Module):
    """Token-level attention on holes feeding into per-action Q heads."""

    def __init__(
        self,
        shape_ctx: ShapeContext,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        action_hidden: int = 64,
    ):
        super().__init__()

        self.num_holes = len(shape_ctx.shape.holes)
        self.num_actions = shape_ctx.actions.shape[0]

        self.register_buffer("actions_idx", shape_ctx.actions[:, :3])
        self.register_buffer("coords", shape_ctx.coords)

        self.coord_embed = nn.Linear(2, d_model, bias=False)
        self.occ_embed = nn.Linear(1, d_model)
        self.token_proj = nn.Linear(d_model * 2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.action_mlp = nn.Sequential(
            nn.Linear(3 * d_model, action_hidden),
            nn.ReLU(),
            nn.Linear(action_hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B, N = state.shape
        assert N == self.num_holes

        occ_emb = self.occ_embed(state.unsqueeze(-1))
        coord_emb = self.coord_embed(self.coords.unsqueeze(0).expand(B, -1, -1))

        tokens = self.token_proj(torch.cat([occ_emb, coord_emb], dim=-1))
        encoded = self.encoder(tokens)

        frm_idx = self.actions_idx[:, 0]
        to_idx = self.actions_idx[:, 1]
        jump_idx = self.actions_idx[:, 2]

        frm_emb = encoded[:, frm_idx, :]
        to_emb = encoded[:, to_idx, :]
        jump_emb = encoded[:, jump_idx, :]

        act_feat = torch.cat([frm_emb, to_emb, jump_emb], dim=-1)
        return self.action_mlp(act_feat).squeeze(-1)
