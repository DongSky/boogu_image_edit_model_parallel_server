"""
OpenAI-compatible image-edit API server for Boogu-Image-0.1-Edit-fp8.

Endpoints:
  POST /v1/images/edits   — OpenAI-shaped multipart edit endpoint (mask field accepted but ignored)
  GET  /v1/models         — model list
  GET  /health            — readiness probe

The pipeline is loaded once at startup with model-parallel placement
(transformer/VAE on --device, MLLM on --mllm_device). Inference runs are
serialized through an asyncio.Lock and executed in a worker thread so the
event loop stays responsive for /health and /v1/models.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import List, Optional, Tuple

import dotenv

dotenv.load_dotenv(override=True)

# Make sure the repo root is on sys.path so `from inference import load_pipeline` works
# whether the server is started from the repo root or anywhere else.
_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from inference import load_pipeline  # noqa: E402  (sys.path tweak above)

PIPELINE = None
PIPELINE_CFG: Optional[SimpleNamespace] = None
PIPELINE_LOCK: Optional[asyncio.Lock] = None
IMAGES_DIR: str = ""
PUBLIC_BASE_URL: str = ""
SERVED_MODEL_NAME: str = "boogu-image-edit-fp8"
def _build_pipeline_cfg(args: argparse.Namespace) -> SimpleNamespace:
    """Build a SimpleNamespace with every field load_pipeline() reads.

    Most fields are pinned to inert defaults; we only honor the knobs that
    matter for FP8 dual-GPU edit inference. This avoids dragging the CLI's
    full argparse into the server while staying compatible with load_pipeline.
    """
    return SimpleNamespace(
        # Required core
        pretrained_pipeline_name_or_path=args.pretrained_pipeline_name_or_path,
        use_fp8_weights=args.use_fp8_weights,
        device=args.device,
        mllm_device=args.mllm_device,
        # Custom-component overrides (all disabled)
        custom_diffusion_transformer_path=None,
        custom_pretrained_instruction_encoder_model_name_or_path=None,
        custom_prompt_tuning_model_path=None,
        custom_prompt_tuning_model_lora_weights_path=None,
        custom_transformer_lora_path=None,
        custom_local_instruction_rewriter_model=None,
        # Mode flags
        use_prompt_tuning=False,
        scheduler="euler",
        mask_vision_tokens_feature=False,
        vision_token_ids=[],
        # Caching strategies (all off — FP8 transformer is already fast enough)
        enable_teacache=False,
        teacache_rel_l1_thresh=0.05,
        enable_teacache_for_all_layers=False,
        enable_taylorseer=False,
        enable_taylorseer_for_all_layers=False,
        enable_cache_dit_caching=False,
        enable_cache_dit_caching_for_all_layers=False,
        # Offload strategies (all off — model parallelism is doing the memory split)
        enable_sequential_cpu_offload_flag=False,
        enable_model_cpu_offload_flag=False,
        enable_group_offload_flag=False,
        enable_inner_devices_manager=False,
        # Rewriter (off; relevant fields kept as placeholders)
        use_rewrite_text_instruction=False,
        use_dashscope_remote_rewriting=False,
        rewriter_device=None,
        unload_rewriter_level="destroy",
        # Torch compile (off; can flip on via --enable_torch_compile)
        enable_torch_compile=args.enable_torch_compile,
        torch_compile_mode="default",
    )
def _parse_size(size: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Parse OpenAI 'size' string. 'auto'/None -> (None, None) (follow reference image)."""
    if not size or size.lower() == "auto":
        return None, None
    try:
        w_s, h_s = size.lower().split("x", 1)
        w, h = int(w_s), int(h_s)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid 'size' value: {size!r}. Expected 'WIDTHxHEIGHT' or 'auto'.",
        )
    if not (64 <= w <= 4096 and 64 <= h <= 4096):
        raise HTTPException(status_code=400, detail=f"'size' out of range (64-4096): {size!r}.")
    return w, h


def _load_image_from_upload(upload: UploadFile, raw: bytes) -> Image.Image:
    if not raw:
        raise HTTPException(status_code=400, detail=f"Empty image file: {upload.filename!r}.")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image {upload.filename!r}: {e}")
    return ImageOps.exif_transpose(img)


def _encode_png_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _save_png_and_url(image: Image.Image) -> str:
    name = f"{uuid.uuid4().hex}.png"
    path = os.path.join(IMAGES_DIR, name)
    image.save(path, format="PNG")
    # PUBLIC_BASE_URL is "" by default, in which case clients should resolve relative paths
    # against the request's host. We always return an absolute-from-root path under /images.
    if PUBLIC_BASE_URL:
        return f"{PUBLIC_BASE_URL.rstrip('/')}/images/{name}"
    return f"/images/{name}"
# Defaults are taken verbatim from inference.py argparse so the API behaves
# identically to running the CLI with no overrides.
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_SEED = 0
DEFAULT_TEXT_GUIDANCE_SCALE = 4.0
DEFAULT_IMAGE_GUIDANCE_SCALE = 1.0
DEFAULT_NEGATIVE_INSTRUCTION = (
    "(((deformed))), blurry, over saturation, bad anatomy, disfigured, "
    "poorly drawn face, mutation, mutated, (extra_limb), (ugly), "
    "(poorly drawn hands), fused fingers, messy drawing, broken legs censor, "
    "censored, censor_bar"
)


def _run_pipeline_sync(
    *,
    instruction: str,
    negative_instruction: str,
    input_image: Image.Image,
    width: Optional[int],
    height: Optional[int],
    num_inference_steps: int,
    text_guidance_scale: float,
    image_guidance_scale: float,
    num_images_per_instruction: int,
    seed: int,
) -> List[Image.Image]:
    """Synchronous pipeline call. Runs in a worker thread under PIPELINE_LOCK."""
    if PIPELINE is None or PIPELINE_CFG is None:
        raise RuntimeError("Pipeline not loaded.")

    device = PIPELINE_CFG.device
    generator = torch.Generator(device=device).manual_seed(seed)

    # When the user did not pin a size, fall back to align_res so the output follows
    # the reference image's aspect/size. Only valid for batch_size==1 — true here
    # because the edits endpoint serves one edit per request.
    align_res = width is None and height is None

    result = PIPELINE(
        instruction=[instruction],
        input_images=[[input_image]],
        negative_instruction=negative_instruction,
        height=height,
        width=width,
        align_res=align_res,
        max_input_image_pixels=2048 * 2048,
        max_input_image_side_length=2048 * 2,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        image_guidance_scale=image_guidance_scale,
        num_images_per_instruction=num_images_per_instruction,
        generator=generator,
        output_type="pil",
        device=device,
    )
    return list(result.images)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global PIPELINE, PIPELINE_LOCK
    if PIPELINE_CFG is None:
        raise RuntimeError("PIPELINE_CFG must be initialized before app startup.")

    weight_dtype = torch.bfloat16  # FP8 weights with bf16 activations — matches test_ti2i_fp8.sh
    print(
        f"[api] loading pipeline: {PIPELINE_CFG.pretrained_pipeline_name_or_path} "
        f"(device={PIPELINE_CFG.device}, mllm_device={PIPELINE_CFG.mllm_device}, "
        f"fp8={PIPELINE_CFG.use_fp8_weights})"
    )
    PIPELINE = load_pipeline(PIPELINE_CFG, weight_dtype)
    PIPELINE_LOCK = asyncio.Lock()
    print("[api] pipeline ready.")
    try:
        yield
    finally:
        PIPELINE = None


app = FastAPI(title="Boogu-Image Edit API", lifespan=lifespan)


@app.get("/health")
async def health():
    ready = PIPELINE is not None
    return {
        "status": "ok" if ready else "loading",
        "ready": ready,
        "model": SERVED_MODEL_NAME,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "boogu",
            }
        ],
    }
@app.post("/v1/images/edits")
async def images_edits(
    request: Request,
    prompt: str = Form(..., description="Edit instruction."),
    image: UploadFile = File(..., description="Reference image to edit (PNG/JPEG)."),
    # OpenAI fields. mask/quality/user are accepted for client compatibility and ignored.
    model: Optional[str] = Form(None),
    mask: Optional[UploadFile] = File(None),
    n: Optional[int] = Form(1),
    size: Optional[str] = Form("auto"),
    response_format: Optional[str] = Form("b64_json"),
    quality: Optional[str] = Form(None),
    user: Optional[str] = Form(None),
    # Boogu extras (names mirror inference.py CLI flags).
    negative_instruction: Optional[str] = Form(None),
    num_inference_steps: int = Form(DEFAULT_NUM_INFERENCE_STEPS),
    text_guidance_scale: float = Form(DEFAULT_TEXT_GUIDANCE_SCALE),
    image_guidance_scale: float = Form(DEFAULT_IMAGE_GUIDANCE_SCALE),
    seed: int = Form(DEFAULT_SEED),
):
    if PIPELINE is None or PIPELINE_LOCK is None:
        raise HTTPException(status_code=503, detail="Pipeline is still loading. Try again shortly.")
    if model is not None and model != SERVED_MODEL_NAME:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model {model!r}. This server only serves {SERVED_MODEL_NAME!r}.",
        )
    if response_format not in ("b64_json", "url"):
        raise HTTPException(status_code=400, detail=f"Invalid response_format: {response_format!r}.")
    if n is None or not (1 <= int(n) <= 8):
        raise HTTPException(status_code=400, detail="'n' must be in [1, 8].")
    if not (1 <= num_inference_steps <= 200):
        raise HTTPException(status_code=400, detail="'num_inference_steps' must be in [1, 200].")
    if mask is not None:
        # Boogu's edit model is instruction-conditioned, not mask-conditioned. We accept
        # the field for OpenAI client compatibility and ignore it.
        print(f"[api] note: 'mask' field provided ({mask.filename!r}) but ignored by Boogu.")

    width, height = _parse_size(size)
    raw = await image.read()
    pil_image = _load_image_from_upload(image, raw)

    neg = negative_instruction if negative_instruction is not None else DEFAULT_NEGATIVE_INSTRUCTION

    request_id = uuid.uuid4().hex[:8]
    started = time.time()
    print(
        f"[api {request_id}] edit start: n={n} size={size} steps={num_inference_steps} "
        f"tg={text_guidance_scale} ig={image_guidance_scale} seed={seed} prompt={prompt[:80]!r}"
    )
    loop = asyncio.get_running_loop()
    try:
        async with PIPELINE_LOCK:
            images = await loop.run_in_executor(
                None,
                lambda: _run_pipeline_sync(
                    instruction=prompt,
                    negative_instruction=neg,
                    input_image=pil_image,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    text_guidance_scale=text_guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    num_images_per_instruction=int(n),
                    seed=seed,
                ),
            )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {type(e).__name__}: {e}")

    elapsed = time.time() - started
    print(f"[api {request_id}] edit done in {elapsed:.1f}s ({len(images)} image(s)).")

    data = []
    if response_format == "b64_json":
        for im in images:
            data.append({"b64_json": _encode_png_b64(im)})
    else:
        for im in images:
            data.append({"url": _save_png_and_url(im)})

    return JSONResponse({"created": int(time.time()), "data": data})


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": {"message": f"{type(exc).__name__}: {exc}", "type": "server_error"}},
    )
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible API for Boogu-Image edit inference.")
    p.add_argument(
        "--pretrained_pipeline_name_or_path",
        type=str,
        default="models/Boogu-Image-0.1-Edit-fp8",
        help="Path or HF name of the pretrained pipeline. Default matches test_ti2i_fp8.sh.",
    )
    p.add_argument("--use_fp8_weights", action="store_true", default=True,
                   help="Load FP8 quantized weights (default: True).")
    p.add_argument("--no_fp8", dest="use_fp8_weights", action="store_false",
                   help="Disable FP8 (load bf16 weights instead).")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Main device (transformer + VAE).")
    p.add_argument("--mllm_device", type=str, default="cuda:1",
                   help="MLLM device. Set equal to --device to disable model-parallel split.")
    p.add_argument("--enable_torch_compile", action="store_true", default=False,
                   help="Enable torch.compile on the transformer.")
    p.add_argument("--host", type=str, default="0.0.0.0",
                   help="Bind address.")
    p.add_argument("--port", type=int, default=8000,
                   help="Bind port.")
    p.add_argument("--images_dir", type=str, default="outputs/api_images",
                   help="Where to save images when response_format=url.")
    p.add_argument("--public_base_url", type=str, default="",
                   help="Public base URL prefix for response_format=url (e.g. https://host:8000). "
                        "If empty, returns root-relative paths under /images/.")
    p.add_argument("--served_model_name", type=str, default="boogu-image-edit-fp8",
                   help="The model id reported by /v1/models and required in /v1/images/edits.")
    return p.parse_args()


def main() -> None:
    global PIPELINE_CFG, IMAGES_DIR, PUBLIC_BASE_URL, SERVED_MODEL_NAME

    args = _parse_cli()
    PIPELINE_CFG = _build_pipeline_cfg(args)
    IMAGES_DIR = os.path.abspath(args.images_dir)
    PUBLIC_BASE_URL = args.public_base_url
    SERVED_MODEL_NAME = args.served_model_name
    os.makedirs(IMAGES_DIR, exist_ok=True)

    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

