#!/usr/bin/env bash
set -euo pipefail
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$TRAIN_DIR"

CUDA_VISIBLE_DEVICES=1,2,3 llamafactory-cli train \
  --model_name_or_path Qwen/Qwen3-32B \
  --trust_remote_code \
  --stage sft --do_train \
  --finetuning_type lora \
  --lora_rank 32 \
  --lora_target all \
  --deepspeed examples/deepspeed/ds_z3_offload_config.json \
  --dataset ocd_train \
  --eval_dataset ocd_test \
  --dataset_dir "$TRAIN_DIR/data" \
  --template qwen \
  --cutoff_len 512 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 --num_train_epochs 1 \
  --bf16 --eval_strategy steps --eval_steps 500 \
  --logging_steps 10 --save_steps 500 --overwrite_output_dir \
  --output_dir "$TRAIN_DIR/saves/qwen3-32b/lora/sft_ocd_32"
