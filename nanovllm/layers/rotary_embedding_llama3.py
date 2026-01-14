from functools import lru_cache
import math
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


class Llama3RotaryEmbedding(nn.Module):
    """
    Llama 3.1 RoPE with frequency-dependent scaling for extended context.
    Applies different scaling factors based on wavelength ranges.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        
        # Compute scaled inverse frequencies
        inv_freq = self._compute_inv_freq(base, rotary_dim)
        
        # Precompute cos/sin cache for all positions
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: float, rotary_dim: int) -> torch.Tensor:
        """Compute inverse frequencies with Llama 3.1 scaling."""
        # Standard RoPE inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # Compute wavelength thresholds
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor
        
        # Compute wavelengths from frequencies
        wave_len = 2 * math.pi / inv_freq
        
        # Compute smooth interpolation factor
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
        else:
            smooth = torch.zeros_like(wave_len)
        
        # Apply frequency-dependent scaling
        new_freq = torch.where(
            wave_len < high_freq_wavelen,
            inv_freq,  # High freq: no scaling
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freq / self.scaling_factor,  # Low freq: full scaling
                (1 - smooth) * inv_freq / self.scaling_factor + smooth * inv_freq,  # Mid freq: interpolate
            ),
        )
        
        return new_freq

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


# @lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    Create RoPE embedding with optional Llama 3.1 scaling.
    
    For Llama 3.1, rope_scaling should contain:
    - factor: Overall scaling factor (e.g., 8.0)
    - low_freq_factor: Low frequency threshold (e.g., 1.0)
    - high_freq_factor: High frequency threshold (e.g., 4.0)
    - original_max_position_embeddings: Original context length (e.g., 8192)
    - rope_type: "llama3"
    """
    if rope_scaling is not None and rope_scaling.get("rope_type") == "llama3":
        rotary_emb = Llama3RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
            base=base,
            scaling_factor=rope_scaling["factor"],
            low_freq_factor=rope_scaling["low_freq_factor"],
            high_freq_factor=rope_scaling["high_freq_factor"],
            orig_max_position=rope_scaling["original_max_position_embeddings"],
        )
    else:
        # Fallback to standard RoPE (import from original module)
        from .rotary_embedding import RotaryEmbedding
        rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    
    return rotary_emb

