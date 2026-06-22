# boogu_image_edit_model_parallel_server

Dual-GPU (model-parallel) inference and an OpenAI-compatible HTTP server for **Boogu-Image-0.1-Edit-fp8**.

The FP8 pipeline needs ~21 GB of VRAM if it lives on one card. This fork splits it across two GPUs:

| GPU role | Components | VRAM (steady state, 1024x1024) |
|---|---|---|
| `--device` (e.g. `cuda:0`) | transformer + VAE + latents | ~16 GB |
| `--mllm_device` (e.g. `cuda:1`) | MLLM instruction encoder | ~10 GB |

An `accelerate.AlignDevicesHook` is attached to the MLLM so the rest of the pipeline keeps operating with a single logical device — instruction-feature tensors cross PCIe once per inference and the 50-step diffusion loop runs entirely on `--device`.

A 24 GB + 16 GB pair (e.g. RTX 4090 + RTX 5060 Ti) is enough to run FP8 edits at 1024-class resolutions without offload.

---

## 1. Setup

### Conda env

```bash
conda create -n boogu python=3.10 -y
conda activate boogu

# PyTorch (pick the index that matches your CUDA)
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
pip install "torchao>=0.15,<0.18"

# Project deps
pip install "diffusers[torch]>=0.35.2,<0.39" "transformers[torch]>=4.57.3,<6" \
            "accelerate>=1.0" "kernels>=0.14,<0.15" "cache-dit>=1.3,<2" \
            "einops>=0.7" "scipy>=1.11" "webdataset>=1.0,<2" \
            "python-dotenv>=1.0,<2" "omegaconf>=2.3,<3"

# Triton (FP8 fallback kernel)
pip install triton              # Linux
pip install triton-windows      # Windows

# API server deps
pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" "python-multipart>=0.0.9"

# Editable install of the boogu package
pip install -e .
```

### Download the FP8 model

```bash
mkdir -p models
git lfs install
git clone https://huggingface.co/Boogu/Boogu-Image-0.1-Edit-fp8 models/Boogu-Image-0.1-Edit-fp8
```

The model directory must contain `model_index.json`, `transformer/`, `mllm/`, `vae/`, `scheduler/`, `processor/`. Total ~21 GB of weights (.bin transformer + safetensors mllm/vae).

---

## 2. Dual-GPU inference

### One-shot CLI

`inference.py` exposes `--device` (transformer + VAE) and `--mllm_device` (MLLM). When the two differ, the script skips the unified `pipeline.to(device)` and places each component manually, then attaches the device-aligning hook to the MLLM. It refuses to combine `--mllm_device` with any CPU/group offload flag — model parallelism is the memory split.

```bash
python inference.py \
    --pretrained_pipeline_name_or_path "models/Boogu-Image-0.1-Edit-fp8" \
    --use_fp8_weights True \
    --input_image_paths "input_image_examples/03.jpg" \
    --instruction "Replace the background with a sandy beach." \
    --num_inference_steps 50 \
    --height 1024 --width 1024 \
    --text_guidance_scale 4.0 --image_guidance_scale 1.0 \
    --output_image_path "outputs/edit/out.png" \
    --device "cuda:0" \
    --mllm_device "cuda:1"
```

You will see:

```
[Pipeline Loader]: Model-parallel placement — transformer/VAE on cuda:0, MLLM on cuda:1.
```

### Bundled smoke test

`test_ti2i_fp8.sh` runs the existing TI2I batch yml in one model-parallel pass:

```bash
bash test_ti2i_fp8.sh
```

Outputs land in `outputs/test_ti2i_fp8/`.

### Data-parallel sharding (optional)

If you have N identical large GPUs, `--num_shards N --shard_index i` slices a batch yml so several processes can split the work. Per-shard outputs use the original global index in their filename to avoid collisions. Not used in `test_ti2i_fp8.sh`; bring your own launcher.

---

## 3. OpenAI-compatible API server

### Start it

```bash
bash run_api_server.sh
```

Defaults: `0.0.0.0:8000`, `cuda:0` for transformer/VAE, `cuda:1` for MLLM, model id `boogu-image-edit-fp8`. Override via env vars:

```bash
HOST=127.0.0.1 PORT=18000 \
DEVICE=cuda:0 MLLM_DEVICE=cuda:1 \
PRETRAINED_PATH=models/Boogu-Image-0.1-Edit-fp8 \
bash run_api_server.sh
```

The server loads the pipeline once at startup with the same model-parallel split as the CLI. Inference is serialized through an `asyncio.Lock` and dispatched to a worker thread, so `/health` and `/v1/models` stay responsive while a long edit is running. Concurrent edits queue up — only one runs at a time (the diffusion pipeline isn't reentrant).

### Endpoints

| Route | Description |
|---|---|
| `GET /health` | `{"status": "ok", "ready": true, "model": "..."}` once weights are loaded; `"loading"` during startup |
| `GET /v1/models` | OpenAI-shape list with the served model id |
| `POST /v1/images/edits` | OpenAI multipart contract — see below |

### `POST /v1/images/edits` form fields

OpenAI fields:
- `image` (required, file)
- `prompt` (required, string) — the edit instruction
- `model` (optional) — must equal the served model id if provided
- `n` (optional, default 1, range 1–8) — images per request
- `size` (optional, default `auto`) — `auto` follows the reference image; otherwise `WIDTHxHEIGHT`
- `response_format` (optional, default `b64_json`) — `b64_json` or `url`
- `mask`, `quality`, `user` — accepted for client compatibility, ignored. (Boogu is instruction-conditioned, not mask-conditioned.)

Boogu extras (names match `inference.py` CLI flags, defaults match argparse):
- `negative_instruction` — defaults to the standard Boogu negative-prompt template
- `num_inference_steps` (default 50)
- `text_guidance_scale` (default 4.0)
- `image_guidance_scale` (default 1.0) — raise to ~1.5 when identity preservation matters
- `seed` (default 0)

### Quick test

**API key:** any non-empty string. The server doesn't validate it; the OpenAI SDK only requires the field to be present.

```bash
curl -X POST http://127.0.0.1:8000/v1/images/edits \
    -F "model=boogu-image-edit-fp8" \
    -F "image=@input_image_examples/03.jpg" \
    -F "prompt=Replace the background with a sandy beach." \
    -F "response_format=url"
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-no-key")

resp = client.images.edit(
    model="boogu-image-edit-fp8",
    image=open("input_image_examples/03.jpg", "rb"),
    prompt="Replace the background with a sandy beach.",
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

## License & attribution

Apache 2.0 (see `LICENSE`). Built on top of upstream Boogu-Image at https://github.com/boogu-project/Boogu-Image.
