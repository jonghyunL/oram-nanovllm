import math
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding using neox-style (rotate pairs)."""
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class Phi3RotaryEmbedding(nn.Module):
    """
    Phi-3 RoPE with SU (Short/Long) scaling for extended context.
    Uses per-dimension scaling factors for short and long contexts.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        original_max_position_embeddings: int,
        short_factor: list[float],
        long_factor: list[float],
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        
        # Compute mscale (magnitude scaling factor)
        scale = max_position_embeddings / original_max_position_embeddings
        if scale <= 1.0:
            short_mscale = 1.0
            long_mscale = 1.0
        else:
            scaling_factor = math.sqrt(
                1 + math.log(scale) / math.log(original_max_position_embeddings)
            )
            short_mscale = scaling_factor
            long_mscale = scaling_factor
        
        # Compute short context cache (0 to original_max_position)
        short_cache = self._compute_cos_sin_cache(
            original_max_position_embeddings,
            short_factor,
            short_mscale,
        )
        
        # Compute long context cache (original_max_position to max_position)
        long_cache = self._compute_cos_sin_cache(
            max_position_embeddings,
            long_factor,
            long_mscale,
        )
        
        # Concatenate: [short_cache, long_cache]
        # Positions 0-4095 use short_cache, 4096+ use long_cache offset
        combined_cache = torch.cat([short_cache, long_cache], dim=0)
        self.register_buffer("cos_sin_cache", combined_cache, persistent=False)

    def _compute_inv_freq(self, rescale_factors: list[float]) -> torch.Tensor:
        """Compute inverse frequencies with per-dimension rescaling."""
        rescale_factors = torch.tensor(rescale_factors, dtype=torch.float32)
        inv_freq = 1.0 / (
            rescale_factors * (
                self.base ** (
                    torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
                )
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(
        self,
        max_positions: int,
        rescale_factors: list[float],
        mscale: float,
    ) -> torch.Tensor:
        """Precompute cos/sin cache with rescaling and mscale."""
        inv_freq = self._compute_inv_freq(rescale_factors)
        t = torch.arange(max_positions, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * mscale).unsqueeze_(1)
        sin = (freqs.sin() * mscale).unsqueeze_(1)
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings with position-dependent cache selection.
        For positions < original_max_position: use short cache
        For positions >= original_max_position: use long cache with offset
        """
        # Compute offset for long context positions
        k = self.original_max_position_embeddings
        long_prompt_offset = (positions > k).float() * k
        idx = (positions + long_prompt_offset.long()).long()
        
        # Fetch cos/sin from cache
        cos_sin = self.cos_sin_cache[idx]
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        # Apply rotary embedding
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
    Create RoPE embedding with optional Phi-3 SU scaling.
    
    For Phi-3, rope_scaling should contain:
    - type: "su" (short/long scaling)
    - short_factor: List of 64 per-dimension scaling factors for short context
    - long_factor: List of 64 per-dimension scaling factors for long context
    - original_max_position_embeddings: Original context length (e.g., 4096)
    """
    if rope_scaling is not None and rope_scaling.get("type") == "su":
        rotary_emb = Phi3RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
            base=base,
            original_max_position_embeddings=rope_scaling["original_max_position_embeddings"],
            short_factor=rope_scaling["short_factor"],
            long_factor=rope_scaling["long_factor"],
        )
    else:
        # Fallback to standard RoPE
        from .rotary_embedding import RotaryEmbedding
        rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    
    return rotary_emb

