# GPT-OSS MLX Implementation

This directory contains a complete MLX (Apple Silicon) implementation of the GPT-OSS models.

## Features

- **Full Model Architecture**: Complete implementation of GPT-OSS with Mixture of Experts (MoE)
- **Apple Silicon Optimized**: Uses MLX for efficient inference on Apple Silicon
- **Memory Efficient**: Includes quantization and memory optimization techniques
- **Compatible Interface**: Drop-in replacement for other backends (torch, triton, vllm)
- **SafeTensor Support**: Loads weights from SafeTensor format

## Architecture Components

### Core Modules (`modules.py`)
- **RMSNorm**: Root Mean Square Layer Normalization
- **Attention**: Multi-head attention with sliding window support and RoPE
- **FeedForward**: Standard MLP with SwiGLU activation
- **RoPE**: Rotary Position Embeddings with YaRN scaling

### Mixture of Experts (`moe.py`)
- **MixtureOfExperts**: Standard MoE implementation
- **OptimizedMixtureOfExperts**: Memory-optimized version with better batching

### Model (`model.py`)
- **TransformerBlock**: Individual transformer layer
- **GPTOSSModel**: Complete GPT-OSS model with generation capabilities
- **Weight Loading**: Support for loading from checkpoints

### Configuration (`config.py`)
- **GPTOSSConfig**: Model configuration dataclass
- **Preset Configs**: Pre-configured settings for gpt-oss-120b and gpt-oss-20b

## Supported Models

### GPT-OSS-120B
- 116.8B total parameters, 5.1B active per token
- 36 layers, 128 experts, top-4 routing
- Memory requirement: ~60GB (with quantization: ~30GB)

### GPT-OSS-20B  
- 20.9B total parameters, 3.6B active per token
- 24 layers, 32 experts, top-4 routing
- Memory requirement: ~12GB (with quantization: ~6GB)

## Usage

### Command Line Interface

```bash
# Generate text using MLX backend
python -m gpt_oss.generate -p "Hello world" -b mlx model/

# Chat interface with MLX
python -m gpt_oss.chat --backend mlx model/
```

### Python API

```python
from gpt_oss.mlx_gpt_oss import GPTOSSModel, GPTOSSConfig, TokenGenerator

# Load pre-trained model
model = GPTOSSModel.from_pretrained("path/to/checkpoint")

# Or create from config
config = GPTOSSConfig.gpt_oss_20b()
model = GPTOSSModel(config)

# Generate tokens
generator = TokenGenerator("path/to/checkpoint")
for token in generator.generate([1, 2, 3], stop_tokens=[0]):
    print(token)
```

### Model Configuration

```python
# Custom configuration
config = GPTOSSConfig(
    num_hidden_layers=24,
    num_experts=32,
    experts_per_token=4,
    vocab_size=201088,
    hidden_size=2048,
    use_quantization=True,
    quantization_bits=4
)
```

## Optimizations

### Memory Optimizations (`optimizations.py`)
- **Quantization**: 4-bit quantization for MoE weights
- **KV Cache Compression**: Automatic cache management
- **Memory Mapping**: Efficient weight storage
- **Gradient Checkpointing**: Memory-efficient training

### Performance Features
- **Sliding Window Attention**: Alternating dense and sparse attention patterns
- **Grouped Query Attention (GQA)**: Reduced KV head count for efficiency
- **MXFP4 Quantization**: Compatible with GPT-OSS weight format
- **Apple Silicon Optimization**: Native MLX acceleration

## Installation Requirements

```bash
pip install mlx safetensors
```

## Testing

Run the test suite to verify your installation:

```bash
python test_mlx_implementation.py
python test_with_weights.py
```

## Architecture Details

The implementation follows the GPT-OSS model card specifications:

- **Vocabulary**: 201,088 tokens (o200k_harmony tokenizer)
- **Context Length**: 4,096 → 131,072 tokens (with YaRN scaling)
- **Attention**: 64 query heads, 8 KV heads, 64-dimensional heads
- **MoE**: Top-4 expert selection with SwiGLU activation
- **Normalization**: RMSNorm with Pre-LN placement
- **Position Encoding**: RoPE with YaRN scaling (theta=150,000, factor=32)

## File Structure

```
gpt_oss/mlx/
├── __init__.py          # Module exports
├── config.py           # Model configuration
├── model.py            # Main model implementation  
├── modules.py          # Core neural network modules
├── moe.py             # Mixture of Experts implementation
├── generate.py        # Token generation utilities
├── weights.py         # Weight loading and conversion
├── optimizations.py   # Memory and performance optimizations
└── README.md         # This file
```

## Integration

The MLX backend is fully integrated into the GPT-OSS CLI:

- Added to `gpt_oss/generate.py` backend selection
- Added to `gpt_oss/chat.py` backend selection  
- Compatible with existing tokenizer and chat formats
- Follows the same `TokenGenerator` interface as other backends

## Performance

Expected performance on Apple Silicon:

| Model | Memory Usage | Tokens/sec (M1 Ultra) | Tokens/sec (M2 Ultra) |
|-------|-------------|----------------------|----------------------|
| GPT-OSS-20B | ~12GB | ~15-20 | ~20-25 |
| GPT-OSS-120B | ~60GB | ~5-8 | ~8-12 |
| GPT-OSS-20B (Quantized) | ~6GB | ~20-30 | ~30-40 |
| GPT-OSS-120B (Quantized) | ~30GB | ~8-12 | ~12-18 |

*Performance estimates based on similar MLX model implementations*
