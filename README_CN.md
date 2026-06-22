# boogu_image_edit_model_parallel_server

**Boogu-Image-0.1-Edit-fp8** 的双卡（模型并行）推理脚本和 OpenAI 兼容 HTTP 服务器。

FP8 流水线如果全部放在一张卡上需要约 21 GB 显存。本仓库把它拆到两张卡上：

| GPU 角色 | 组件 | 稳态显存（1024×1024） |
|---|---|---|
| `--device`（如 `cuda:0`） | transformer + VAE + 隐变量 | 约 16 GB |
| `--mllm_device`（如 `cuda:1`） | MLLM 指令编码器 | 约 10 GB |

通过给 MLLM 挂一个 `accelerate.AlignDevicesHook`，流水线其余部分仍按单一逻辑设备运作 —— 每次推理只让 instruction-feature 张量跨一次 PCIe，50 步扩散循环全部在 `--device` 上跑。

24 GB + 16 GB 这种异构组合（例如 RTX 4090 + RTX 5060 Ti）即可在 1024 量级分辨率下无 offload 跑 FP8 编辑。

---

## 1. 环境搭建

### Conda 环境

```bash
conda create -n boogu python=3.10 -y
conda activate boogu

# PyTorch（按你的 CUDA 选 index）
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
pip install "torchao>=0.15,<0.18"

# 项目依赖
pip install "diffusers[torch]>=0.35.2,<0.39" "transformers[torch]>=4.57.3,<6" \
            "accelerate>=1.0" "kernels>=0.14,<0.15" "cache-dit>=1.3,<2" \
            "einops>=0.7" "scipy>=1.11" "webdataset>=1.0,<2" \
            "python-dotenv>=1.0,<2" "omegaconf>=2.3,<3"

# Triton（FP8 fallback kernel）
pip install triton              # Linux
pip install triton-windows      # Windows

# API 服务器依赖
pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" "python-multipart>=0.0.9"

# 可编辑安装 boogu 包
pip install -e .
```

### 下载 FP8 模型

```bash
mkdir -p models
git lfs install
git clone https://huggingface.co/Boogu/Boogu-Image-0.1-Edit-fp8 models/Boogu-Image-0.1-Edit-fp8
```

模型目录应包含 `model_index.json`、`transformer/`、`mllm/`、`vae/`、`scheduler/`、`processor/`，权重总量约 21 GB（transformer 是 .bin，mllm/vae 是 safetensors）。

---

## 2. 双卡推理

### 命令行单次调用

`inference.py` 提供了 `--device`（transformer + VAE）和 `--mllm_device`（MLLM）两个参数。当两者不同时，脚本会跳过统一的 `pipeline.to(device)`，改为手动放置各组件，并在 MLLM 上挂上设备对齐 hook。`--mllm_device` 不允许和任何 CPU/group offload 标志共存 —— 模型并行本身就是显存拆分方案。

```bash
python inference.py \
    --pretrained_pipeline_name_or_path "models/Boogu-Image-0.1-Edit-fp8" \
    --use_fp8_weights True \
    --input_image_paths "input_image_examples/03.jpg" \
    --instruction "把背景替换到沙滩." \
    --num_inference_steps 50 \
    --height 1024 --width 1024 \
    --text_guidance_scale 4.0 --image_guidance_scale 1.0 \
    --output_image_path "outputs/edit/out.png" \
    --device "cuda:0" \
    --mllm_device "cuda:1"
```

启动后会看到：

```
[Pipeline Loader]: Model-parallel placement — transformer/VAE on cuda:0, MLLM on cuda:1.
```

### 自带的烟囱测试

`test_ti2i_fp8.sh` 用模型并行单次跑完现有的 TI2I batch yml：

```bash
bash test_ti2i_fp8.sh
```

输出落在 `outputs/test_ti2i_fp8/`。

### 数据并行分片（可选）

如果你有 N 张同型号大显存卡，`--num_shards N --shard_index i` 可以把一个 batch yml 切片，让多个进程分摊任务。每个分片输出的文件名带原始 global index，避免冲突。`test_ti2i_fp8.sh` 没用到这个能力，需要的话自己写 launcher。

---

## 3. OpenAI 兼容 API 服务器

### 启动

```bash
bash run_api_server.sh
```

默认配置：`0.0.0.0:8000`，`cuda:0` 跑 transformer/VAE，`cuda:1` 跑 MLLM，model id 为 `boogu-image-edit-fp8`。可通过环境变量覆盖：

```bash
HOST=127.0.0.1 PORT=18000 \
DEVICE=cuda:0 MLLM_DEVICE=cuda:1 \
PRETRAINED_PATH=models/Boogu-Image-0.1-Edit-fp8 \
bash run_api_server.sh
```

服务器在启动时一次性加载流水线，使用和 CLI 相同的模型并行拆分。推理通过 `asyncio.Lock` 串行化，并派到 worker 线程执行，所以即使有耗时编辑在跑，`/health` 和 `/v1/models` 仍能即时响应。并发的编辑请求会排队 —— 同一时刻只跑一个（扩散流水线不可重入）。

### 接口列表

| 路径 | 说明 |
|---|---|
| `GET /health` | 权重加载完成后返回 `{"status": "ok", "ready": true, "model": "..."}`；启动期间为 `"loading"` |
| `GET /v1/models` | OpenAI 风格的模型列表，包含本服务的 model id |
| `POST /v1/images/edits` | OpenAI multipart 协议 —— 字段见下 |

### `POST /v1/images/edits` 表单字段

OpenAI 字段：
- `image`（必填，文件）
- `prompt`（必填，字符串）—— 编辑指令
- `model`（可选）—— 如果传值，必须等于服务的 model id
- `n`（可选，默认 1，范围 1–8）—— 单次请求生成几张
- `size`（可选，默认 `auto`）—— `auto` 沿用参考图尺寸；否则 `WIDTHxHEIGHT`
- `response_format`（可选，默认 `b64_json`）—— `b64_json` 或 `url`
- `mask`、`quality`、`user` —— 接收但忽略，仅为客户端兼容（Boogu 是指令条件的，不是 mask 条件的）

Boogu 扩展字段（命名和默认值与 `inference.py` CLI 对齐）：
- `negative_instruction`（不传则用 Boogu 标准负向提示）
- `num_inference_steps`（默认 50）
- `text_guidance_scale`（默认 4.0）
- `image_guidance_scale`（默认 1.0；身份保留要求高时调到 1.5 左右）
- `seed`（默认 0）

### 快速测试

**API key**：任意非空字符串即可。服务器并不校验，OpenAI SDK 只是要求该字段非空。

```bash
curl -X POST http://127.0.0.1:8000/v1/images/edits \
    -F "model=boogu-image-edit-fp8" \
    -F "image=@input_image_examples/03.jpg" \
    -F "prompt=把背景替换到沙滩." \
    -F "response_format=url"
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-no-key")

resp = client.images.edit(
    model="boogu-image-edit-fp8",
    image=open("input_image_examples/03.jpg", "rb"),
    prompt="把背景替换到沙滩.",
    response_format="b64_json",
    extra_body={
        "num_inference_steps": 50,
        "text_guidance_scale": 4.0,
        "image_guidance_scale": 1.5,
        "seed": 0,
    },
)
```

---

## 协议与署名

Apache 2.0（见 `LICENSE`）。基于上游 Boogu-Image 项目构建：https://github.com/boogu-project/Boogu-Image
