import torch
from torch import nn
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class GptOssAttention(nn.Module):
    """GPT-OSS attention with GQA (Grouped Query Attention) support."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 131072,
        head_dim: int | None = None,
        rope_theta: float = 150000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=True,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class GptOssMoEExperts(nn.Module):
    """Sparse MoE with num_local_experts and experts_per_token."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_local_experts: int,
        experts_per_token: int,
        hidden_act: str = "silu",
    ) -> None:
        super().__init__()
        self.num_local_experts = num_local_experts
        self.experts_per_token = experts_per_token
        
        # Router: maps hidden_size -> num_local_experts
        self.router = nn.Linear(hidden_size, num_local_experts, bias=True)
        
        # Shared experts submodule to match checkpoint names:
        # model.layers.<i>.mlp.experts.(gate_up_proj|down_proj).(weight|bias|blocks|scales)
        class ExpertsShared(nn.Module):
            def __init__(self, hidden_size: int, intermediate_size: int):
                super().__init__()
                self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=True)
                self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

                # Optional quantization tensors may appear in checkpoint; register as buffers if present later
                # We do not know shapes here; loader will skip if not found.
        
        self.experts = ExpertsShared(hidden_size, intermediate_size)
        
        # Activation function
        if hidden_act == "silu":
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Dispatch tokens to top-k experts.
        Args:
            hidden_states: (seq_len, hidden_size)
        Returns:
            output: (seq_len, hidden_size)
        """
        # Get routing scores
        router_logits = self.router(hidden_states)  # (seq_len, num_local_experts)
        
        # Select top-k experts per token
        topk_scores, topk_indices = torch.topk(
            router_logits, 
            k=self.experts_per_token, 
            dim=-1
        )  # (seq_len, experts_per_token)
        
        # Normalize expert scores (softmax over selected experts)
        topk_scores = torch.softmax(topk_scores, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # Process tokens through shared expert weights with routing
        for token_idx in range(hidden_states.size(0)):
            token_output = torch.zeros_like(hidden_states[token_idx])
            for expert_pos in range(self.experts_per_token):
                expert_idx = topk_indices[token_idx, expert_pos].item()
                expert_score = topk_scores[token_idx, expert_pos]
                
                # Apply shared gate_up_proj
                gate_up = self.experts.gate_up_proj(hidden_states[token_idx:token_idx+1])
                gate, up = gate_up.chunk(2, dim=-1)
                
                # Apply activation to up projection
                up = self.act(up)
                
                # Gate the up projection (element-wise multiply)
                x = gate * up
                
                # Apply shared down_proj
                expert_output = self.experts.down_proj(x)
                
                token_output = token_output + expert_score * expert_output.squeeze(0)
            output[token_idx] = token_output
        
        return output


class GptOssDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        
        # Determine if this is sliding or full attention based on config
        layer_types = getattr(config, "layer_types", None)
        is_sliding = False
        if layer_types and layer_idx < len(layer_types):
            is_sliding = layer_types[layer_idx] == "sliding_attention"
        
        self.self_attn = GptOssAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        
        # MoE layer
        self.mlp = GptOssMoEExperts(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_local_experts=config.num_local_experts,
            experts_per_token=config.experts_per_token,
            hidden_act=config.hidden_act,
        )
        
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class GptOssModel(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            GptOssDecoderLayer(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class GptOssForCausalLM(nn.Module):
    """GPT-OSS model for causal language modeling."""
    
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = GptOssModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # GPT-OSS doesn't tie embeddings by default
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

