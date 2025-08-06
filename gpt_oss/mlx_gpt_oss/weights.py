import json
from pathlib import Path
from typing import Dict, Any, Optional
import mlx.core as mx
import mlx.nn as nn


def load_safetensors(path: Path) -> Dict[str, mx.array]:
    """Load weights from safetensors format with MXFP4 support."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("safetensors library is required for loading weights. Install with: pip install safetensors")
    
    from .mxfp4 import load_mxfp4_weights
    
    # First pass: load all tensors as-is
    raw_weights = {}
    with safe_open(path, framework="np") as f:
        for key in f.keys():
            raw_weights[key] = f.get_tensor(key)
    
    print(f"Loaded {len(raw_weights)} raw tensors from safetensors")
    
    # Second pass: handle MXFP4 conversion
    weights = load_mxfp4_weights(raw_weights)
    
    print(f"Converted to {len(weights)} final weight tensors")
    return weights


def convert_torch_to_mlx_weights(torch_weights: Dict[str, Any]) -> Dict[str, mx.array]:
    """Convert PyTorch weights to MLX format."""
    mlx_weights = {}
    
    for name, weight in torch_weights.items():
        # Convert naming conventions
        mlx_name = name
        
        # Handle specific conversions
        if "mlp.experts" in name:
            # MoE expert weights need special handling
            parts = name.split(".")
            if "gate_proj" in name:
                mlx_name = name.replace("mlp.experts", "mlp.gate_projs")
            elif "up_proj" in name:
                mlx_name = name.replace("mlp.experts", "mlp.up_projs")
            elif "down_proj" in name:
                mlx_name = name.replace("mlp.experts", "mlp.down_projs")
        
        # Convert to MLX array
        mlx_weights[mlx_name] = mx.array(weight.numpy() if hasattr(weight, 'numpy') else weight)
    
    return mlx_weights


def load_gpt_oss_weights(model: nn.Module, checkpoint_path: str):
    """Load GPT-OSS weights into MLX model."""
    checkpoint_path = Path(checkpoint_path)
    
    # Check for different weight formats
    safetensor_path = checkpoint_path / "model.safetensors"
    pytorch_path = checkpoint_path / "pytorch_model.bin"
    
    if safetensor_path.exists():
        weights = load_safetensors(safetensor_path)
    elif pytorch_path.exists():
        # Load PyTorch weights
        import torch
        torch_weights = torch.load(pytorch_path, map_location="cpu")
        weights = convert_torch_to_mlx_weights(torch_weights)
    else:
        raise ValueError(f"No weights found at {checkpoint_path}")
    
    # Map weights to model
    model_dict = model.parameters()
    
    # Handle weight mapping
    for name, param in model_dict.items():
        if name in weights:
            param.data = weights[name]
        else:
            # Try alternative names
            alt_names = get_alternative_names(name)
            for alt_name in alt_names:
                if alt_name in weights:
                    param.data = weights[alt_name]
                    break
            else:
                print(f"Warning: No weights found for {name}")


def get_alternative_names(param_name: str) -> list:
    """Get alternative names for parameter mapping."""
    alternatives = []
    
    # Common transformations
    if "mlp.gate_projs" in param_name:
        alternatives.append(param_name.replace("mlp.gate_projs", "mlp.experts.gate_proj"))
    if "mlp.up_projs" in param_name:
        alternatives.append(param_name.replace("mlp.up_projs", "mlp.experts.up_proj"))
    if "mlp.down_projs" in param_name:
        alternatives.append(param_name.replace("mlp.down_projs", "mlp.experts.down_proj"))
    
    # Layer normalization alternatives
    if "input_layernorm" in param_name:
        alternatives.append(param_name.replace("input_layernorm", "ln_1"))
    if "post_attention_layernorm" in param_name:
        alternatives.append(param_name.replace("post_attention_layernorm", "ln_2"))
    
    return alternatives


def save_mlx_weights(model: nn.Module, save_path: str):
    """Save MLX model weights."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Get all parameters
    weights = {}
    for name, param in model.parameters().items():
        weights[name] = param.data
    
    # Save weights (in production, convert to safetensors)
    # For now, we'll save as numpy arrays
    import numpy as np
    
    np_weights = {}
    for name, weight in weights.items():
        np_weights[name] = np.array(weight)
    
    # Save as .npz file
    np.savez(save_path / "mlx_weights.npz", **np_weights)
    
    # Save metadata
    metadata = {
        "format": "mlx",
        "version": "1.0",
        "num_parameters": sum(w.size for w in weights.values())
    }
    
    with open(save_path / "mlx_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)