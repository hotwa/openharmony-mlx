import math
from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
    
    def __call__(self, x: mx.array) -> mx.array:
        return self._forward(x)
    
    def _forward(self, x: mx.array) -> mx.array:
        # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
        mean_sq = mx.mean(x * x, axis=-1, keepdims=True)
        x_normed = x * mx.rsqrt(mean_sq + self.eps)
        return x_normed * self.weight


def compute_rope_embeddings(
    positions: mx.array, 
    head_dim: int, 
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    ntk_alpha: float = 1.0,
    context_length: int = 4096
) -> Tuple[mx.array, mx.array]:
    """Compute Rotary Position Embeddings with YaRN scaling."""
    
    # Apply NTK-aware scaling
    if scaling_factor > 1.0:
        # YaRN scaling: mix linear and NTK interpolation
        theta = theta * (scaling_factor ** (head_dim / (head_dim - 2)))
    
    # Create frequency bands
    freqs = mx.arange(0, head_dim, 2, dtype=mx.float32)
    freqs = theta ** (-freqs / head_dim)
    
    # Apply position-dependent frequencies
    angles = positions[:, None] * freqs[None, :]
    cos = mx.cos(angles)
    sin = mx.sin(angles)
    
    return cos, sin


def apply_rope(
    x: mx.array,
    cos: mx.array,
    sin: mx.array
) -> mx.array:
    """Apply rotary position embeddings to input tensor."""
    # x shape: [batch, seq_len, num_heads, head_dim]
    # cos/sin shape: [seq_len, head_dim//2]
    # Need to expand for broadcasting
    cos = cos[None, :, None, :]  # [1, seq_len, 1, head_dim//2]
    sin = sin[None, :, None, :]  # [1, seq_len, 1, head_dim//2]
    
    # Split into pairs for rotation
    x1, x2 = mx.split(x, 2, axis=-1)
    
    # Rotate pairs with proper broadcasting
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    
    # Concatenate back
    return mx.concatenate([rx1, rx2], axis=-1)


class Attention(nn.Module):
    """Multi-head attention with optional sliding window."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: Optional[int] = None,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        rope_ntk_alpha: float = 1.0,
        initial_context_length: int = 4096
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        
        # RoPE parameters
        self.rope_theta = rope_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_ntk_alpha = rope_ntk_alpha
        self.initial_context_length = initial_context_length
        
        # Grouped query attention ratio
        self.num_groups = num_heads // num_kv_heads
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.scale = head_dim ** -0.5
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        # Reshape for multi-head attention
        queries = queries.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        values = values.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        positions = mx.arange(seq_len)
        cos, sin = compute_rope_embeddings(
            positions, 
            self.head_dim, 
            self.rope_theta,
            self.rope_scaling_factor,
            self.rope_ntk_alpha,
            self.initial_context_length
        )
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)
        
        # Handle KV cache
        if cache is not None:
            k_cache, v_cache = cache
            keys = mx.concatenate([k_cache, keys], axis=1)
            values = mx.concatenate([v_cache, values], axis=1)
        
        new_cache = (keys, values) if cache is not None else None
        
        # Grouped query attention: repeat KV heads
        if self.num_groups > 1:
            keys = mx.repeat(keys, self.num_groups, axis=2)
            values = mx.repeat(values, self.num_groups, axis=2)
        
        # Transpose for attention computation: [batch, heads, seq, head_dim]
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)  
        values = values.transpose(0, 2, 1, 3)
        
        # Compute attention scores
        # keys need to be transposed on last two dims: [batch, heads, seq, head_dim] -> [batch, heads, head_dim, seq]
        scores = mx.matmul(queries, keys.swapaxes(-2, -1)) * self.scale
        
        # Apply sliding window mask if specified
        if self.sliding_window is not None and seq_len > 1:
            window_mask = self._create_sliding_window_mask(seq_len)
            scores = scores + window_mask
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax and apply attention
        attn_weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(attn_weights, values)
        
        # Reshape and project output: [batch, seq, heads, head_dim] -> [batch, seq, hidden]
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        return output, new_cache
    
    def _create_sliding_window_mask(self, seq_len: int) -> mx.array:
        """Create sliding window attention mask."""
        mask = mx.ones((seq_len, seq_len))
        for i in range(seq_len):
            start = max(0, i - self.sliding_window + 1)
            mask[i, :start] = 0
            mask[i, i+1:] = 0
        return mx.where(mask == 0, -1e9, 0.0)


class FeedForward(nn.Module):
    """Standard MLP with SwiGLU activation."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: float = 7.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit
        
        # SwiGLU uses 3 linear layers
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def __call__(self, x: mx.array) -> mx.array:
        # SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Clamped SiLU activation
        gate = gate * mx.sigmoid(gate)
        gate = mx.clip(gate, -self.swiglu_limit, self.swiglu_limit)
        
        return self.down_proj(gate * up)