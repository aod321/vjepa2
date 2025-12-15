import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    Temporal head with shared projection, value regression, and optional policy logits.

    Args:
        encoder_dim: Dimension of encoder tokens.
        tokens_per_frame: Number of spatial tokens per frame (grid_size^2).
        hidden_dim: Hidden size for the shared projection.
        num_layers: Number of projection layers.
        dropout: Dropout applied after each projection layer.
        pooling: Spatial pooling type ("mean" or "max").
        policy_num_actions: Optional number of discrete actions for policy logits.
    """

    def __init__(
        self,
        encoder_dim: int,
        tokens_per_frame: int,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
        pooling: str = "mean",
        policy_num_actions: int | None = None,
    ) -> None:
        super().__init__()
        if tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive.")
        self.tokens_per_frame = tokens_per_frame
        pooling = pooling.lower()
        if pooling not in {"mean", "max"}:
            raise ValueError("pooling must be 'mean' or 'max'.")
        self.pooling = pooling

        layers = []
        in_dim = encoder_dim
        for _ in range(max(num_layers, 1)):
            layers.append(nn.LayerNorm(in_dim))
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.proj = nn.Sequential(*layers)
        self.value_head = nn.Linear(in_dim, 1)

        if policy_num_actions is not None:
            if policy_num_actions <= 0:
                raise ValueError("policy_num_actions must be positive.")
            self.policy_head = nn.Linear(in_dim, policy_num_actions)
        else:
            self.policy_head = None

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B, N, D], got shape {tokens.shape}.")
        b, n, d = tokens.shape
        if n % self.tokens_per_frame != 0:
            raise ValueError(
                f"Token count {n} is not divisible by tokens_per_frame {self.tokens_per_frame}."
            )
        t = n // self.tokens_per_frame
        frames = tokens.view(b, t, self.tokens_per_frame, d)
        if self.pooling == "mean":
            frames = frames.mean(dim=2)
        else:
            frames = frames.max(dim=2).values

        feats = self.proj(frames)
        outputs: dict[str, torch.Tensor] = {}
        outputs["value"] = self.value_head(feats).squeeze(-1)
        if self.policy_head is not None:
            outputs["policy"] = self.policy_head(feats)
        return outputs
