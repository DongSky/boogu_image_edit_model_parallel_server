#!/bin/bash
# Launch the OpenAI-compatible image-edit API server.
# Mirrors test_ti2i_fp8.sh: model-parallel placement on cuda:0 (transformer+VAE)
# and cuda:1 (MLLM). Customize via env vars before invoking, e.g.:
#   HOST=127.0.0.1 PORT=18000 bash run_api_server.sh
set -u

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_MODULES_CACHE="$PWD/.hf_modules_cache"

PRETRAINED_PATH="${PRETRAINED_PATH:-models/Boogu-Image-0.1-Edit-fp8}"
DEVICE="${DEVICE:-cuda:0}"
MLLM_DEVICE="${MLLM_DEVICE:-cuda:1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
IMAGES_DIR="${IMAGES_DIR:-outputs/api_images}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-boogu-image-edit-fp8}"

mkdir -p "$IMAGES_DIR"

echo "[launcher] device=$DEVICE  mllm_device=$MLLM_DEVICE  http=$HOST:$PORT"
echo "[launcher] images_dir=$IMAGES_DIR  served_model_name=$SERVED_MODEL_NAME"

exec python api_server.py \
    --pretrained_pipeline_name_or_path "$PRETRAINED_PATH" \
    --device "$DEVICE" \
    --mllm_device "$MLLM_DEVICE" \
    --host "$HOST" \
    --port "$PORT" \
    --images_dir "$IMAGES_DIR" \
    --public_base_url "$PUBLIC_BASE_URL" \
    --served_model_name "$SERVED_MODEL_NAME"
