## 模型权重准备

Metal后端需要特定格式的权重文件。你有两个选择：

### 转换现有权重：

```bash
python gpt_oss/metal/scripts/create-local-model.py -s <model_dir> -d <output_file>
```

### 下载预转换权重：

```bash
huggingface-cli download openai/gpt-oss-120b --include "metal/*" --local-dir gpt-oss-120b/metal/  
huggingface-cli download openai/gpt-oss-20b --include "metal/*" --local-dir gpt-oss-20b/metal/
```

这里的"Metal版本"指的是GPT-OSS模型的Metal后端实现。


## 环境准备

macOS系统（Apple Silicon）

1. 准备环境

```bash
xcode-select --install
micromamba create -n gptoss python=3.12 -y
```

2. 手动运行CMake构建

```bash
git clone https://github.com/hotwa/openharmony-mlx.git
cd openharmony-mlx
# 自动安装cmake安装
GPTOSS_BUILD_METAL=1 pip install -e ".[metal]"
# 手动编译cmake安装
cd gpt_oss/metal  
mkdir build  
cd build  
cmake .. -DCMAKE_BUILD_TYPE=Release -DGPTOSS_BUILD_PYTHON=ON  
make -j$(nproc)
```

3. Metal着色器编译
CMake会自动编译Metal源文件： CMakeLists.txt:16-28

这些.metal文件会被编译成.air中间文件，然后链接成default.metallib：

4. Python扩展模块构建
CMake会创建名为_metal的Python扩展模块：

```bash
# 安装扩展模块  
cp _metal.so /path/to/your/python/site-packages/gpt_oss/metal/  
  
# 安装Metal库文件    
cp default.metallib /path/to/your/python/site-packages/gpt_oss/metal/
```

5. 验证metal模块是否正确安装

```python
python -c "import gpt_oss.metal._metal; print('Metal module loaded successfully')"
```

## 启动服务

缓存下载并启动服务

```bash
mkdir -p ~/.cache/openai_harmony/
cd ~/.cache/openai_harmony/
wget https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
export OPENAI_HARMONY_CACHE_DIR=~/.cache/openai_harmony/
chmod 755 ~/.cache/openai_harmony/
python -m gpt_oss.responses_api.serve --inference-backend metal --checkpoint /Volumes/long990max/gpustack_data/openai/gpt-oss-20b/metal/model.bin --host 0.0.0.0 --port 8080
```

## cherrystudio 配置

模型ID：gpt-oss-120b
模型名称：gpt-oss-120b
分组名称：gpt-oss

请求虽然是gpt-oss-120b，但是实际使用的是gpt-oss-20b。由于后台写死的是120b，所以请求使用gpt-oss-120b