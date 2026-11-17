#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_remaining_a6000.sh — the models that do not yet have a full run.
#
# RUNS ON: the KNUST GPU server, from the repo root, inside tmux.
#
#   tmux new -s train 'bash scripts/train_remaining_a6000.sh'
#
# Order:
#   1. Faster R-CNN v2 baseline          (30 epochs)
#   2. SE-FPN final Faster R-CNN         (30 epochs)
#   3. ViTDet, ViT-B/16                  (40 epochs)
#   4. Swin-V2-T                         (40 epochs)
#
# Classifier, YOLO26 and RT-DETR already have finished runs and are not repeated.
# The 7-config Faster R-CNN ablation is not in this queue. See
# docs/11_knust_a6000_run.md.
#
# GPU 0 on ghid-lambda02 was already taken when this was written. The default
# pins training to GPU 1 and caps this process at 30% of that card so the
# resident vLLM server is not pushed out of memory. Override both when the
# card is dedicated:
#
#   CUDA_VISIBLE_DEVICES=0 TRAIN_MEM_FRACTION= bash scripts/train_remaining_a6000.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 0.30 of a 48 GB A6000 is about 14.4 GB, which is the free slice next to the
# vLLM reservation measured on 2026-10-17. Empty string disables the cap.
export TRAIN_MEM_FRACTION="${TRAIN_MEM_FRACTION:-0.30}"

PY="${PY:-./.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
mkdir -p logs

# Measured on this card with TRAIN_MEM_FRACTION=0.30 (about 14.4 GB),
# AMP, backbone unfrozen, synthetic 640px batches, 2026-10-17:
#   frcnn batch 12 = 9.41 GB    final batch 12 = 9.72 GB
#   vit   batch  8 = 5.47 GB    swin  batch  8 = 7.29 GB  (batch 12 = 11.66, too close to the cap)
# ViT throughput plateaus by batch 8, so the smaller ViT batch is not lost speed.
BATCH_FRCNN="${BATCH_FRCNN:-12}"
BATCH_FINAL="${BATCH_FINAL:-12}"
BATCH_VIT="${BATCH_VIT:-8}"
BATCH_SWIN="${BATCH_SWIN:-8}"

SUMMARY="logs/remaining_a6000_summary.log"
echo "queue start $(date '+%F %T')  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  fraction=${TRAIN_MEM_FRACTION:-none}" | tee "$SUMMARY"

run() {
  local label="$1"; shift
  echo "════════════════════════════════════════════════════════════════"
  echo "  ▶  ${label}   ($(date '+%F %H:%M:%S'))"
  echo "     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  TRAIN_MEM_FRACTION=${TRAIN_MEM_FRACTION:-none}"
  echo "     $*"
  echo "════════════════════════════════════════════════════════════════"
  local start rc mins
  start=$(date +%s)
  "$@" 2>&1 | tee "logs/${label}.log"
  rc=${PIPESTATUS[0]}
  mins=$(( ($(date +%s) - start) / 60 ))
  if [ "$rc" -eq 0 ]; then
    echo "  ✓ ${label} finished in ${mins}m" | tee -a "$SUMMARY"
  else
    echo "  ✗ ${label} FAILED rc=${rc} after ${mins}m — next model will still start" | tee -a "$SUMMARY"
  fi
}

run fasterrcnn "$PY" -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline \
  --batch-size "$BATCH_FRCNN" --skip-negatives
run final "$PY" -m src.fasterrcnn.faster_rcnn_final \
  --batch-size "$BATCH_FINAL" --skip-negatives
run vit "$PY" -m src.vit.train_vit \
  --batch-size "$BATCH_VIT" --skip-negatives
run swin "$PY" -m src.swin.train_swin \
  --batch-size "$BATCH_SWIN" --skip-negatives

echo "queue finished $(date '+%F %T')" | tee -a "$SUMMARY"
