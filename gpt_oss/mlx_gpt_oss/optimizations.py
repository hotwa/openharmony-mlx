import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Any, Tuple


def quantize_model(model: nn.Module, bits: int = 4) -> nn.Module:
    """Quantize model weights for memory efficiency."""
    # MLX provides quantization utilities
    from mlx.nn.utils import quantize
    
    # Quantize the model
    quantized_model = quantize(model, group_size=64, bits=bits)
    return quantized_model


def optimize_attention_memory(config):
    """Configure attention for memory efficiency."""
    # Use smaller sliding window for memory efficiency
    if config.sliding_window and config.sliding_window > 256:
        config.sliding_window = 256
    
    return config


def enable_kv_cache_compression(cache: Tuple[mx.array, mx.array]) -> Tuple[mx.array, mx.array]:
    """Compress KV cache to save memory."""
    if cache is None:
        return None
    
    k_cache, v_cache = cache
    
    # Simple compression: keep only recent tokens beyond sliding window
    max_cache_length = 2048
    
    if k_cache.shape[1] > max_cache_length:
        k_cache = k_cache[:, -max_cache_length:, :, :]
        v_cache = v_cache[:, -max_cache_length:, :, :]
    
    return k_cache, v_cache


class OptimizedTransformerBlock(nn.Module):
    """Memory-optimized transformer block."""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        from .model import TransformerBlock
        
        # Use gradient checkpointing for memory efficiency
        self.block = TransformerBlock(config, layer_idx)
        self.gradient_checkpointing = True
    
    def __call__(self, x, mask=None, cache=None):
        if self.gradient_checkpointing and self.training:
            # Use MLX's checkpoint functionality if available
            return mx.checkpoint(self.block, x, mask, cache)
        else:
            return self.block(x, mask, cache)


class MemoryEfficientMoE(nn.Module):
    """Memory-efficient MoE implementation."""
    
    def __init__(self, hidden_size, intermediate_size, num_experts, experts_per_token, swiglu_limit=7.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.swiglu_limit = swiglu_limit
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Shared expert computation to save memory
        self.shared_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.shared_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        
        # Expert-specific weights (smaller footprint)
        self.expert_gates = mx.random.normal((num_experts, intermediate_size, intermediate_size)) * 0.02
        self.expert_ups = mx.random.normal((num_experts, intermediate_size, intermediate_size)) * 0.02
        self.expert_downs = mx.random.normal((num_experts, intermediate_size, hidden_size)) * 0.02
    
    def __call__(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Router
        router_logits = self.router(x)
        top_k_logits, top_k_indices = mx.topk(router_logits, k=self.experts_per_token, axis=-1)
        top_k_weights = mx.softmax(top_k_logits, axis=-1)
        
        # Shared computation
        base_gate = self.shared_gate(x)
        base_up = self.shared_up(x)
        
        # Apply SwiGLU
        base_gate = base_gate * mx.sigmoid(base_gate)
        base_gate = mx.clip(base_gate, -self.swiglu_limit, self.swiglu_limit)
        base_hidden = base_gate * base_up
        
        # Expert-specific processing
        output = mx.zeros_like(x)
        
        for k in range(self.experts_per_token):
            expert_idx = top_k_indices[:, :, k]
            expert_weight = top_k_weights[:, :, k:k+1]
            
            # Get unique expert indices to minimize computation
            unique_experts = mx.unique(expert_idx.flatten())
            
            for expert_id in unique_experts:
                mask = (expert_idx == expert_id)
                if not mx.any(mask):
                    continue
                
                # Apply expert-specific transformations
                expert_hidden = mx.matmul(base_hidden, self.expert_gates[expert_id])
                expert_out = mx.matmul(expert_hidden, self.expert_downs[expert_id])
                
                # Add weighted output where this expert is selected
                output = mx.where(mask[:, :, None], 
                                output + expert_weight * expert_out, 
                                output)
        
        return output


def apply_memory_optimizations(model, config):
    """Apply various memory optimizations to the model."""
    # Enable memory mapping for weights
    mx.metal.set_memory_limit(8 * 1024 * 1024 * 1024)  # 8GB limit
    
    # Configure for memory efficiency
    config = optimize_attention_memory(config)
    
    return model, config