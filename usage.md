# OpenHarmony MLX - GPT-OSS 模型部署指南

本项目是在 Apple Silicon (M系列芯片) 上使用 Metal 后端运行 OpenAI GPT-OSS 模型的实现。

## 目录

- [系统要求](#系统要求)
- [环境准备](#环境准备)
- [模型权重准备](#模型权重准备)
- [安装步骤](#安装步骤)
- [启动 API 服务](#启动-api-服务)
- [测试 API](#测试-api)
- [客户端配置](#客户端配置)
- [常见问题](#常见问题)

## 系统要求

- **macOS**: Apple Silicon (M1/M2/M3/M4 系列)
- **Python**: 3.12
- **内存**: 建议 32GB+ (20B 模型较大)
- **Xcode**: 必须从 App Store 安装完整版（非命令行工具）

## 环境准备

### 1. 安装 Xcode（关键步骤）

**⚠️ 常见问题：仅安装 `xcode-select` 命令行工具无法编译 Metal！**

Metal 编译器 `metal` 不包含在命令行工具包中，必须安装完整的 Xcode 应用：

```bash
# 1. 下载地址: https://developer.apple.com/xcode/resources/
# 2. 登录 Apple Developer 账号
# 3. 下载 Xcode（需要 Apple ID 登录）
# 4. 将 Xcode.app 拖动到 /Applications/ 目录
# 5. 首次启动 Xcode，等待完成组件安装
```

设置 Xcode 路径：

```bash
# 设置命令行工具使用完整 Xcode
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer

# 验证 metal 编译器可用
xcrun -find metal
# 应该输出: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/metal
```

### 2. 下载 Metal Toolchain

```bash
xcodebuild -downloadComponent MetalToolchain
```

### 3. 创建 Python 环境

```bash
# 使用 uv (推荐)
uv venv --managed-python -p 3.12 --seed .venv

# 或使用 micromamba
micromamba create -n gptoss python=3.12 -y
```

## 模型权重准备

### 支持的模型格式

| 后端 | 模型格式 | 说明 |
|------|----------|------|
| Transformers | bf16 / fp16 | Hugging Face 标准格式 |
| Metal | MXFP4 | 量化格式，需要转换 |

### 推荐模型

- **ArliAI/gpt-oss-20b-Derestricted**: 去审查版本的 20B 模型（bf16 格式）
- **openai/gpt-oss-20b**: 官方 20B 模型（MXFP4 格式）

### 下载模型

```bash
# 激活环境
source .venv/bin/activate

# 安装 huggingface_hub
pip install huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载 ArliAI Derestricted 模型
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ArliAI/gpt-oss-20b-Derestricted',
    local_dir='/path/to/models/ArliAI/gpt-oss-20b-Derestricted'
)
"
```

## 安装步骤

### 1. 安装基础依赖

```bash
source .venv/bin/activate

# 安装项目依赖
pip install -e ".[metal]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 pybind11 (CMake 构建需要)
pip install pybind11
```

### 2. 编译 Metal 后端

```bash
cd gpt_oss/metal
mkdir -p build
cd build

# 获取 pybind11 cmake 路径
PYBIND11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")

# 配置并编译
cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$PYBIND11_DIR
make -j$(sysctl -n hw.ncpu)
```

### 3. 安装编译产物

```bash
# 复制 Python 扩展模块
cp _metal.cpython-312-darwin.so ../gpt_oss/metal/

# 复制 Metal 库文件
cp default.metallib ../gpt_oss/metal/
```

### 4. 验证安装

```bash
source .venv/bin/activate
python -c "import gpt_oss.metal._metal; print('Metal module loaded successfully')"
```

### 5. 安装 Transformers 后端依赖（可选）

如果使用 bf16 模型：

```bash
pip install transformers accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 启动 API 服务

### 使用 Transformers 后端（推荐）

适用于 bf16 格式模型（如 ArliAI Derestricted）：

```bash
source .venv/bin/activate
cd /path/to/openharmony-mlx

python -m gpt_oss.responses_api.serve \
  --inference-backend transformers \
  --checkpoint /path/to/models/ArliAI/gpt-oss-20b-Derestricted \
  --host 0.0.0.0 \
  --port 18080
```

### 使用 Metal 后端

适用于 MXFP4 格式模型：

```bash
source .venv/bin/activate
cd /path/to/openharmony-mlx

python -m gpt_oss.responses_api.serve \
  --inference-backend metal \
  --checkpoint /path/to/model/gpt-oss-20b.bin \
  --host 0.0.0.0 \
  --port 18080
```

### 后台运行

```bash
# 使用 nohup
nohup python -m gpt_oss.responses_api.serve ... > /tmp/gptoss.log 2>&1 &

# 或使用 tmux/screen
tmux new-session -d -s gptoss 'source .venv/bin/activate && python -m gpt_oss.responses_api.serve ...'
```

## 测试 API

### 本地测试

```bash
curl -s -X POST "http://localhost:18080/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer empty" \
  -d '{"model":"gpt-oss-120b","input":[{"role":"user","content":"Hello!"}]}'
```

### 内网测试（Tailscale）

```bash
curl -s -X POST "http://100.64.0.19:18080/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer empty" \
  -d '{"model":"gpt-oss-120b","input":[{"role":"user","content":"Hello!"}]}'
```

### 完整测试脚本

```bash
cat > /tmp/test_api.sh << 'EOF'
#!/bin/bash
API_URL="${API_URL:-http://localhost:18080}"
curl -s -X POST "$API_URL/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer empty" \
  -d '{"model":"gpt-oss-120b","input":[{"role":"user","content":"Hello!"}]}'
EOF
chmod +x /tmp/test_api.sh
/tmp/test_api.sh
```

## 客户端配置

### CherryStudio

添加提供商选择 `OpenAI-Response`

| 参数 | 值 |
|------|-----|
| 模型ID | `gpt-oss-120b` |
| 模型名称 | `gpt-oss-120b` |
| 分组名称 | `gpt-oss` |
| API地址 | `http://localhost:18080` 或 `http://100.64.0.19:18080` |
| 密钥 | 无 |

### cURL 测试

```bash
# 基础对话
curl -X POST "http://localhost:18080/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer empty" \
  -d '{"model":"gpt-oss-120b","input":[{"role":"user","content":"What is 1+1?"}]}'

# 带推理
curl -X POST "http://localhost:18080/v1/responses" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer empty" \
  -d '{"model":"gpt-oss-120b","input":[{"role":"user","content":"Explain quantum mechanics"}],"reasoning":{"effort":"high"}}'
```

## 常见问题

### Q: `xcrun: error: unable to find utility "metal"`

**A**: 必须安装完整的 Xcode 应用，不能只装命令行工具。参见[步骤 1](#1-安装-xcode关键步骤)。

### Q: `Could not find pybind11`

**A**: 运行 `pip install pybind11` 安装。

### Q: `ValueError: Using a device_map requires accelerate`

**A**: 运行 `pip install accelerate`。

### Q: 模型权重格式不兼容

**A**: Metal 后端只支持 MXFP4 格式，Transformers 后端支持 bf16/fp16。根据模型格式选择后端。

### Q: 推理速度慢

**A**: Transformers 后端一次生成 1 个 token，速度较慢。如需更快速度，使用 Metal 后端 + MXFP4 格式模型。

### Q: 内存不足

**A**: 20B 模型需要约 16-32GB 内存。确保有足够可用内存。

## 注意事项

1. **模型名称固定**: API 请求必须使用 `gpt-oss-120b`，无论实际加载的是哪个模型
2. **网络问题**: 安装依赖时如遇超时，使用国内镜像源
3. **首次加载**: 模型首次加载需要较长时间（25+ 秒）
4. **端口**: 默认使用 18080 端口，可通过 `--port` 参数修改
