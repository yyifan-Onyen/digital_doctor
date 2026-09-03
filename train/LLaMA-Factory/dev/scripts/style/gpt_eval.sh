# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
#   --model_name_or_path /workspace/saves/gpt-20b/full/sft_ocd \
#   --stage sft --do_predict --predict_with_generate true \
#   --eval_dataset ocd_test --dataset_dir /workspace/data \
#   --template gpt \
#   --per_device_eval_batch_size 1 \
#   --cutoff_len 512 --max_new_tokens 128 \
#   --temperature 0 --top_p 1 --do_sample false \
#   --output_dir /workspace/saves/gpt-20b/full/sft_ocd


CUDA_VISIBLE_DEVICES=7 llamafactory-cli train \
  --model_name_or_path /workspace/LLaMA-Factory/saves/mannu/sft_mannu_style \
  --stage sft --do_predict --predict_with_generate true \
  --eval_dataset mannu_test_all --dataset_dir /workspace/LLaMA-Factory/data \
  --template gpt \
  --per_device_eval_batch_size 1 \
  --cutoff_len 512 --max_new_tokens 128 \
  --temperature 0.7 --top_p 1 --do_sample false \
  --output_dir /workspace/LLaMA-Factory/saves/mannu/sft_mannu_style