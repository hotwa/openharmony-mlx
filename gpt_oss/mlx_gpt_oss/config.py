from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTOSSConfig:
    """Configuration for GPT-OSS models in MLX."""
    
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    
    # MLX specific parameters
    dtype: str = "float32"  # MLX supports float16, float32, bfloat16
    use_quantization: bool = False
    quantization_bits: int = 4
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "GPTOSSConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def gpt_oss_120b(cls) -> "GPTOSSConfig":
        """GPT-OSS 120B configuration."""
        return cls(
            num_hidden_layers=36,
            num_experts=128,
            experts_per_token=4,
            vocab_size=201088,
            hidden_size=2880,
            intermediate_size=2880,
            swiglu_limit=7.0,
            head_dim=64,
            num_attention_heads=64,
            num_key_value_heads=8,
            sliding_window=128,
            initial_context_length=4096,
            rope_theta=150000.0,
            rope_scaling_factor=32.0,
            rope_ntk_alpha=1.0,
            rope_ntk_beta=32.0
        )
    
    @classmethod
    def gpt_oss_20b(cls) -> "GPTOSSConfig":
        """GPT-OSS 20B configuration."""
        return cls(
            num_hidden_layers=24,
            num_experts=32,
            experts_per_token=4,
            vocab_size=201088,
            hidden_size=2048,
            intermediate_size=2048,
            swiglu_limit=7.0,
            head_dim=64,
            num_attention_heads=48,
            num_key_value_heads=6,
            sliding_window=128,
            initial_context_length=4096,
            rope_theta=150000.0,
            rope_scaling_factor=32.0,
            rope_ntk_alpha=1.0,
            rope_ntk_beta=32.0
        )