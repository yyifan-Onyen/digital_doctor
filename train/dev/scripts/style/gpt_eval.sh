#!/usr/bin/env bash
set -euo pipefail
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES=7 llamafactory-cli train \
  --model_name_or_path "$TRAIN_DIR/saves/mannu/sft_mannu_style" \
  --stage sft --do_predict --predict_with_generate true \
  --eval_dataset mannu_test_all --dataset_dir "$TRAIN_DIR/data" \
  --template gpt \
  --per_device_eval_batch_size 1 \
  --cutoff_len 512 --max_new_tokens 128 \
  --temperature 0.7 --top_p 1 --do_sample false \
  --output_dir "$TRAIN_DIR/saves/mannu/sft_mannu_style"
