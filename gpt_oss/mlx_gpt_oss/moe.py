import mlx.core as mx
import mlx.nn as nn
from typing import Tuple


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer for GPT-OSS."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        experts_per_token: int,
        swiglu_limit: float = 7.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.swiglu_limit = swiglu_limit
        
        # Router to compute expert scores
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Expert layers - separate layers for each expert
        self.gate_projs = [nn.Linear(hidden_size, intermediate_size, bias=False) for _ in range(num_experts)]
        self.up_projs = [nn.Linear(hidden_size, intermediate_size, bias=False) for _ in range(num_experts)]
        self.down_projs = [nn.Linear(intermediate_size, hidden_size, bias=False) for _ in range(num_experts)]
    
    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute router scores
        router_logits = self.router(x)  # [batch, seq_len, num_experts]
        
        # Select top-k experts
        top_k_indices = mx.argpartition(router_logits, -self.experts_per_token, axis=-1)[..., -self.experts_per_token:]
        # Get the corresponding logits
        batch_indices = mx.arange(router_logits.shape[0])[:, None, None]
        seq_indices = mx.arange(router_logits.shape[1])[None, :, None]
        top_k_logits = router_logits[batch_indices, seq_indices, top_k_indices]
        
        # Compute softmax weights for selected experts only
        top_k_weights = mx.softmax(top_k_logits, axis=-1)  # [batch, seq_len, experts_per_token]
        
        # Initialize output
        output = mx.zeros_like(x)
        
        # Process each selected expert
        for k in range(self.experts_per_token):
            # Get expert index for this position
            expert_idx = top_k_indices[:, :, k]  # [batch, seq_len]
            expert_weight = top_k_weights[:, :, k:k+1]  # [batch, seq_len, 1]
            
            # Compute expert output
            expert_out = self._compute_expert(x, expert_idx)
            
            # Add weighted expert output
            output = output + expert_weight * expert_out
        
        return output
    
    def _compute_expert(self, x: mx.array, expert_idx: mx.array) -> mx.array:
        """Compute output for selected experts."""
        batch_size, seq_len, hidden_size = x.shape
        
        # For simplicity, compute one expert at a time
        output = mx.zeros_like(x)
        
        for expert_id in range(self.num_experts):
            # Find positions where this expert is selected
            mask = (expert_idx == expert_id)
            if not mx.any(mask):
                continue
            
            # Get tokens for this expert
            expert_x = mx.where(mask[..., None], x, 0.0)
            
            # Compute expert gate and up projections using individual layers
            gates = self.gate_projs[expert_id](expert_x)
            ups = self.up_projs[expert_id](expert_x)
            
            # SwiGLU activation
            gates = gates * mx.sigmoid(gates)
            gates = mx.clip(gates, -self.swiglu_limit, self.swiglu_limit)
            hidden_states = gates * ups
            
            # Down projection
            expert_out = self.down_projs[expert_id](hidden_states)
            
            # Add to output where this expert is selected
            output = mx.where(mask[..., None], expert_out, output)
        
        return output


class OptimizedMixtureOfExperts(nn.Module):
    """Optimized MoE implementation with better batching."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        experts_per_token: int,
        swiglu_limit: float = 7.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.swiglu_limit = swiglu_limit
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Expert weights stored as single tensors
        self.w_gate = mx.zeros((num_experts, hidden_size, intermediate_size))
        self.w_up = mx.zeros((num_experts, hidden_size, intermediate_size))
        self.w_down = mx.zeros((num_experts, intermediate_size, hidden_size))
    
    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute router scores and select experts
        router_logits = self.router(x)
        top_k_logits, top_k_indices = mx.topk(router_logits, k=self.experts_per_token, axis=-1)
        top_k_weights = mx.softmax(top_k_logits, axis=-1)
        
        # Reshape for expert computation
        x_reshaped = x.reshape(-1, hidden_size)
        
        # Initialize output
        output = mx.zeros((batch_size * seq_len, hidden_size))
        
        # Process each expert
        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = mx.any(top_k_indices == expert_id, axis=-1)
            expert_mask_flat = expert_mask.reshape(-1)
            
            if not mx.any(expert_mask_flat):
                continue
            
            # Get tokens for this expert
            expert_tokens = x_reshaped[expert_mask_flat]
            
            # Compute expert output
            gate = mx.matmul(expert_tokens, self.w_gate[expert_id])
            up = mx.matmul(expert_tokens, self.w_up[expert_id])
            
            # SwiGLU activation
            gate = gate * mx.sigmoid(gate)
            gate = mx.clip(gate, -self.swiglu_limit, self.swiglu_limit)
            expert_out = mx.matmul(gate * up, self.w_down[expert_id])
            
            # Add weighted output
            # Find weights for tokens assigned to this expert
            expert_positions = mx.where(expert_mask_flat)[0]
            for k in range(self.experts_per_token):
                mask_k = top_k_indices.reshape(-1, self.experts_per_token)[:, k] == expert_id
                mask_k = mask_k[expert_mask_flat]
                if mx.any(mask_k):
                    weights = top_k_weights.reshape(-1, self.experts_per_token)[expert_mask_flat, k]
                    output[expert_positions[mask_k]] += weights[mask_k, None] * expert_out[mask_k]
        
        return output.reshape(batch_size, seq_len, hidden_size)