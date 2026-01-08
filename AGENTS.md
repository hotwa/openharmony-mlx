# AGENTS.md - AI Assistant Guidelines for OpenHarmony MLX

This file provides guidelines for AI assistants (like Claude Code) when working with this project.

## Project Overview

**OpenHarmony MLX** is a high-performance Metal implementation of OpenAI's GPT-OSS models for Apple Silicon (M-series chips). The project provides:

- **Metal Backend**: Native GPU acceleration via Apple's Metal framework
- **Transformers Backend**: Hugging Face compatible inference
- **REST API**: OpenAI-compatible `/v1/responses` endpoint
- **MoE Support**: Mixture-of-Experts model architecture

## Key Technologies

- **Language**: Python 3.12, C/C++, Metal Shading Language
- **Build System**: CMake, pybind11
- **ML Framework**: PyTorch, safetensors
- **API Framework**: FastAPI, uvicorn
- **Tokenization**: tiktoken (o200k_base)

## Project Structure

```
openharmony-mlx/
├── gpt_oss/
│   ├── responses_api/       # REST API server
│   │   ├── inference/       # Backend implementations
│   │   │   ├── metal.py     # Metal backend
│   │   │   ├── transformers.py  # Transformers backend
│   │   │   ├── triton.py    # Triton backend
│   │   │   └── ...
│   │   ├── api_server.py    # FastAPI server
│   │   ├── events.py        # SSE events
│   │   ├── types.py         # Type definitions
│   │   └── serve.py         # CLI entry point
│   ├── metal/               # Metal backend (C extension)
│   │   ├── source/          # C/Metal source files
│   │   ├── python/          # Python bindings
│   │   ├── scripts/         # Utility scripts
│   │   │   └── create-local-model.py  # Weight conversion
│   │   └── CMakeLists.txt   # Build configuration
│   ├── tools/               # Tool implementations
│   │   ├── simple_browser/  # Web search tool
│   │   └── python_docker/   # Python execution tool
│   └── ...
├── models/                  # Model weights
├── pyproject.toml           # Project configuration
└── usage.md                 # Deployment guide
```

## Development Commands

### Setup Development Environment

```bash
# Create and activate virtual environment
uv venv --managed-python -p 3.12 --seed .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[metal]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install Metal Toolchain
xcodebuild -downloadComponent MetalToolchain
```

### Build Metal Extension

```bash
cd gpt_oss/metal
mkdir -p build && cd build

# Get pybind11 path
PYBIND11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$PYBIND11_DIR
make -j$(sysctl -n hw.ncpu)

# Install artifacts
cp _metal.cpython-312-darwin.so ../gpt_oss/metal/
cp default.metallib ../gpt_oss/metal/
```

### Run API Server

```bash
# Transformers backend (bf16 models)
python -m gpt_oss.responses_api.serve \
  --inference-backend transformers \
  --checkpoint /path/to/model \
  --host 0.0.0.0 \
  --port 18080

# Metal backend (MXFP4 models)
python -m gpt_oss.responses_api.serve \
  --inference-backend metal \
  --checkpoint /path/to/model.bin \
  --host 0.0.0.0 \
  --port 18080
```

### Test API

```bash
# Basic test
curl -X POST "http://localhost:18080/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer empty" \
  -d '{"model":"gpt-oss-120b","input":[{"role":"user","content":"Hello!"}]}'
```

## Model Weight Conversion

### Converting Safetensors to Metal Format

The `create-local-model.py` script converts Hugging Face safetensors to Metal format:

```bash
python gpt_oss/metal/scripts/create-local-model.py \
  -s /path/to/huggingface/model \
  -d /path/to/output/model.bin
```

**Note**: Metal backend requires MXFP4 quantization format. The original openai/gpt-oss-20b model uses MXFP4. Third-party models like ArliAI Derestricted use bf16 and require the Transformers backend.

## Code Conventions

### Python

- Use type hints for function signatures
- Follow PEP 8 style guide
- Use `pydantic` for data models (see `types.py`)
- Import ordering: stdlib, third-party, local

### C/Metal

- C99 standard for C code
- C++20 for C++ code
- Metal Shading Language for GPU kernels
- Use `pybind11` for Python bindings

### File Naming

- Python modules: `snake_case.py`
- C sources: `lowercase.c` / `lowercase.metal`
- Header files: `snake_case.h`

## Common Tasks

### Adding a New Inference Backend

1. Create `inference/<backend_name>.py`
2. Implement `setup_model(checkpoint: str) -> Callable`
3. Register backend in `serve.py`
4. Add CLI argument parsing

### Modifying Model Configuration

Model identifier is fixed in `gpt_oss/responses_api/types.py`:
```python
MODEL_IDENTIFIER = "gpt-oss-120b"
```

This is intentional - the API always returns this identifier regardless of the actual model loaded.

### Debugging

```bash
# Enable verbose output
export TRANSFORMERS_VERBOSITY=info

# Check logs
tail -f /tmp/gptoss.log
```

## Testing

### API Testing

```bash
# Run basic functionality tests
python -c "
import requests

url = 'http://localhost:18080/v1/responses'
data = {
    'model': 'gpt-oss-120b',
    'input': [{'role': 'user', 'content': 'Hello!'}]
}

response = requests.post(url, json=data, headers={'Authorization': 'Bearer empty'})
print(f'Status: {response.status_code}')
print(f'Response: {response.text[:200]}...')
"
```

### Backend Selection

| Model Format | Backend | Installation |
|--------------|---------|--------------|
| MXFP4 | Metal | Requires CMake + Metal Toolchain |
| bf16/fp16 | Transformers | Requires `pip install transformers accelerate` |

## Dependencies

### Core Dependencies

- `openai-harmony` - Response format handling
- `tiktoken` - Tokenization
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation

### Optional Dependencies

- `torch` - PyTorch (for Transformers backend)
- `transformers` - Hugging Face models
- `accelerate` - Model parallelism
- `pybind11` - C++/Python bindings
- `cmake` - Build system

## Known Issues & Workarounds

1. **Metal compiler not found**: Install full Xcode from App Store
2. **Network timeouts**: Use `-i https://pypi.tuna.tsinghua.edu.cn/simple`
3. **Memory issues**: 20B models require 32GB+ RAM
4. **Transformers one-token-at-a-time**: Expected behavior for this implementation

## Configuration

### Environment Variables

- `OPENAI_HARMONY_CACHE_DIR` - Tokenizer cache directory
- `TRANSFORMERS_VERBOSITY` - Debug output level

### CLI Arguments

```bash
python -m gpt_oss.responses_api.serve --help

# Options:
#   --checkpoint PATH      Model checkpoint directory/file
#   --port PORT            Server port (default: 8000)
#   --host HOST            Bind host (default: 127.0.0.1)
#   --inference-backend    metal | transformers | triton | stub | ollama | vllm
```
