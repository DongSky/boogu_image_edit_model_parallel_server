#!/bin/bash
# Dual-GPU TI2I FP8 test (model-parallel).
# Single inference process, MLLM on cuda:1 (~10.5 GB), transformer + VAE on cuda:0 (~10.3 GB).
# An accelerate AlignDevicesHook on the MLLM transparently moves the small instruction-
# encoder I/O across the PCIe boundary; the 50-step diffusion loop runs entirely on cuda:0.
# Both GPUs hold weights and contribute compute within each end-to-end inference.
set -u

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_MODULES_CACHE="$PWD/.hf_modules_cache"

PRETRAINED_PATH="models/Boogu-Image-0.1-Edit-fp8"
BATCH_CONFIG="batch_data_samples/ti2i_batch_data_sample.yml"
OUTPUT_DIR="outputs/test_ti2i_fp8"
DEVICE="cuda:0"
MLLM_DEVICE="cuda:1"
LOG_PATH="$OUTPUT_DIR/run.log"

mkdir -p "$OUTPUT_DIR"

echo "[launcher] device=$DEVICE  mllm_device=$MLLM_DEVICE"
echo "[launcher] log: $LOG_PATH"

python inference.py \
    --pretrained_pipeline_name_or_path "$PRETRAINED_PATH" \
    --use_fp8_weights True \
    --use_batch_inference True \
    --batch_data_config_path "$BATCH_CONFIG" \
    --num_inference_steps 50 \
    --height 1024 --width 1024 \
    --text_guidance_scale 4.0 --image_guidance_scale 1.0 \
    --output_image_path "$OUTPUT_DIR/out.png" \
    --device "$DEVICE" \
    --mllm_device "$MLLM_DEVICE" \
    2>&1 | tee "$LOG_PATH"

rc="${PIPESTATUS[0]}"
echo "[launcher] inference exit code: $rc"
echo "[launcher] outputs in: $OUTPUT_DIR"
exit "$rc"
