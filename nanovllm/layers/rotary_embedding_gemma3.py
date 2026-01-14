import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class Gemma3RotaryEmbedding(nn.Module):
    """
    Gemma-3 RoPE with linear scaling for extended context.
    Applies uniform scaling factor to all frequency components.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        scaling_factor: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        
        # Compute inverse frequencies with linear scaling
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # Apply linear scaling uniformly to all frequencies
        inv_freq = inv_freq / scaling_factor
        
        # Precompute cos/sin cache
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    Create RoPE embedding with optional Gemma-3 linear scaling.
    
    For Gemma-3, rope_scaling should contain:
    - rope_type: "linear"
    - factor: Scaling factor (e.g., 8.0)
    """
    if rope_scaling is not None and rope_scaling.get("rope_type") == "linear":
        rotary_emb = Gemma3RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
            base=base,
            scaling_factor=rope_scaling.get("factor", 1.0),
        )
    else:
        # Fallback to standard RoPE
        from .rotary_embedding import RotaryEmbedding
        rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    
    return rotary_emb

