CUDA_VISIBLE_DEVICES=1 llamafactory-cli train \
  --model_name_or_path /workspace/saves/qwen2_5-7b/full/sft_ocd \
  --stage sft --do_predict --predict_with_generate true \
  --eval_dataset ocd_test --dataset_dir /workspace/data \
  --template qwen \
  --per_device_eval_batch_size 1 \
  --cutoff_len 512 --max_new_tokens 128 \
  --temperature 0 --top_p 1 --do_sample false \
  --output_dir /workspace/saves/qwen2_5-7b/full/sft_ocd