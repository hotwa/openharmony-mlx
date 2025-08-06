import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import mlx.core as mx
import mlx.nn as nn

from .config import GPTOSSConfig
from .modules import RMSNorm, Attention, FeedForward
from .moe import MixtureOfExperts


class TransformerBlock(nn.Module):
    """Transformer block with attention and MoE/FFN."""
    
    def __init__(self, config: GPTOSSConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-normalization
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        
        # Attention (alternating sliding window and dense)
        use_sliding_window = (layer_idx % 2 == 0) and config.sliding_window is not None
        self.attention = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            sliding_window=config.sliding_window if use_sliding_window else None,
            rope_theta=config.rope_theta,
            rope_scaling_factor=config.rope_scaling_factor,
            rope_ntk_alpha=config.rope_ntk_alpha,
            initial_context_length=config.initial_context_length
        )
        
        # MLP or MoE
        if config.num_experts > 1:
            self.mlp = MixtureOfExperts(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                experts_per_token=config.experts_per_token,
                swiglu_limit=config.swiglu_limit
            )
        else:
            self.mlp = FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                swiglu_limit=config.swiglu_limit
            )
    
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        attn_out, new_cache = self.attention(x, mask, cache)
        x = residual + attn_out
        
        # MLP/MoE with residual
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_out = self.mlp(x)
        x = residual + mlp_out
        
        return x, new_cache


class GPTOSSModel(nn.Module):
    """GPT-OSS model implementation in MLX."""
    
    def __init__(self, config: GPTOSSConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = [
            TransformerBlock(config, i) 
            for i in range(config.num_hidden_layers)
        ]
        
        # Output normalization
        self.norm = RMSNorm(config.hidden_size)
        
        # LM head (can be tied with embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def __call__(
        self,
        input_ids: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        # Token embeddings
        h = self.embed_tokens(input_ids)
        
        # Create causal mask if not provided
        if mask is None:
            seq_len = input_ids.shape[1]
            mask = mx.triu(mx.ones((seq_len, seq_len)) * -1e9, k=1)
        
        # Process through transformer layers
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, layer_new_cache = layer(h, mask, layer_cache)
            if cache is not None:
                new_cache.append(layer_new_cache)
        
        # Final normalization and output projection
        h = self.norm(h)
        logits = self.lm_head(h)
        
        return logits, new_cache if cache is not None else None
    
    def generate(
        self,
        prompt: mx.array,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> mx.array:
        """Generate tokens autoregressively."""
        generated = prompt
        cache = None
        
        for _ in range(max_tokens):
            # Forward pass
            if cache is None:
                logits, cache = self(generated)
                next_token_logits = logits[:, -1, :]
            else:
                # Use only the last token for generation with cache
                logits, cache = self(generated[:, -1:], cache=cache)
                next_token_logits = logits[:, 0, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                # Use a simpler approach: find threshold and mask
                top_k_threshold = mx.sort(next_token_logits, axis=-1)[:, -top_k:top_k+1][:, 0:1]
                next_token_logits = mx.where(next_token_logits >= top_k_threshold, next_token_logits, -float('inf'))
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits = mx.sort(next_token_logits, axis=-1)[:, ::-1]
                cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
                
                # Find cutoff
                cutoff_idx = mx.sum(cumulative_probs < top_p, axis=-1) + 1
                cutoff_idx = mx.minimum(cutoff_idx, mx.array(sorted_logits.shape[-1] - 1))
                
                # Apply cutoff
                cutoff_value = sorted_logits[mx.arange(sorted_logits.shape[0]), cutoff_idx]
                next_token_logits = mx.where(next_token_logits < cutoff_value[:, None], -float('inf'), next_token_logits)
            
            # Sample next token
            probs = mx.softmax(next_token_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs))
            
            # Append to generated sequence
            generated = mx.concatenate([generated, next_token[:, None]], axis=1)
            
            # Check for EOS token (assuming 0 is EOS)
            if mx.all(next_token == 0):
                break
        
        return generated
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> "GPTOSSModel":
        """Load model from pretrained weights."""
        model_path = Path(model_path)
        
        # Load config
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise ValueError(f"Config file not found at {config_path}")
        
        with open(config_path) as f:
            config_dict = json.load(f)
        
        config = GPTOSSConfig.from_dict(config_dict)
        
        # Initialize model
        model = cls(config)
        
        # Load weights
        from .weights import load_gpt_oss_weights
        load_gpt_oss_weights(model, model_path)
        
        return model
    
    def save_pretrained(self, save_path: str):
        """Save model weights and config."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {
            k: getattr(self.config, k) 
            for k in self.config.__dataclass_fields__
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        
        # Save weights
        from .weights import save_mlx_weights
        save_mlx_weights(self, save_path)