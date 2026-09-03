#!/usr/bin/env bash
set -euo pipefail
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES=5 llamafactory-cli train \
  --stage sft --do_predict --predict_with_generate true \
  --model_name_or_path openai/gpt-oss-20b \
  --adapter_name_or_path "$TRAIN_DIR/saves/gpt-20b/full/sft_ez_uga" \
  --eval_dataset ez_uga_test --dataset_dir "$TRAIN_DIR/data" \
  --template gpt \
  --per_device_eval_batch_size 1 \
  --cutoff_len 512 --max_new_tokens 128 \
  --temperature 0 --top_p 1 --do_sample false \
  --output_dir "$TRAIN_DIR/saves/gpt-20b/full/sft_ez_uga"
