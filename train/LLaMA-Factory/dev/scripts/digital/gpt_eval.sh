CUDA_VISIBLE_DEVICES=7 llamafactory-cli train \
  --model_name_or_path /workspace/saves/gpt-20b/full/sft_ocd_v2 \
  --stage sft --do_predict --predict_with_generate true \
  --eval_dataset ocd_test_v2 --dataset_dir /workspace/data \
  --template gpt \
  --per_device_eval_batch_size 1 \
  --cutoff_len 512 --max_new_tokens 128 \
  --temperature 0 --top_p 1 --do_sample false \
  --output_dir /workspace/saves/gpt-20b/full/sft_ocd_v2