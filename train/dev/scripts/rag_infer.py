#!/usr/bin/env python3
"""
Batch inference with vLLM over an augmented test JSON (array), rotating models.

Input JSON schema (each item): {instruction: str, input: str, output: str}

It will:
  1) Load the JSON array
  2) For each listed model, run vLLM generation on prompts "instruction\ninput"
  3) Save generated JSONL per-model
  4) Compute BLEU-4 and ROUGE-1/2/L and print averages

Usage:
python dev/scripts/rag_infer.py \
    --data dev/test_rag.json \
    --save_dir saves/rag_Qwen2_5-7b \
    --models Qwen/Qwen2.5-7B-Instruct \
    --max_new_tokens 128 --temperature 0 --top_p 1 --tp 2
    
python dev/scripts/rag_infer.py \
    --data dev/test_rag.json \
    --save_dir saves/rag_gpt-oss-20b \
    --models openai/gpt-oss-20b \
    --max_new_tokens 128 --temperature 0 --top_p 1 --tp 2
    
python dev/scripts/rag_infer.py \
    --data dev/test_rag.json \
    --save_dir saves/rag_Qwen3-32B \
    --models Qwen/Qwen3-32B \
    --backend tf
    --max_new_tokens 128 --temperature 0 --top_p 1 --tp 2
"""

import argparse
import json
import os
import re
import time
from typing import List

try:
    from vllm import LLM, SamplingParams
except Exception as e:
    raise RuntimeError("vLLM is required. Please install vllm compatible with your torch CUDA.") from e

# transformers fallback for models vLLM cannot load (e.g., unsupported quantization or vocab resize needed)
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    # reuse project metrics to avoid extra deps
    from scripts.eval_bleu_rouge import compute_metrics as _compute_metrics
except Exception as e:
    _compute_metrics = None


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def run_infer_one(
    model_name: str,
    prompts: List[str],
    labels: List[str],
    save_dir: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tp: int,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=max(tp, 1),
        max_model_len=2048 + max_new_tokens,
    )

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p if top_p and top_p > 0 else 1.0,
        max_tokens=max_new_tokens,
        seed=1234,
    )

    results = llm.generate(prompts, params)
    preds = [r.outputs[0].text for r in results]

    out_path = os.path.join(save_dir, f"{_sanitize(model_name)}_generated.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for p, y, ref in zip(prompts, preds, labels):
            f.write(json.dumps({"prompt": p, "predict": y, "label": ref}, ensure_ascii=False) + "\n")

    return out_path


def evaluate_file(pred_jsonl: str) -> dict:
    if _compute_metrics is None:
        raise RuntimeError(
            "metrics dependencies missing. Install extras or use scripts/eval_bleu_rouge.py manually."
        )

    scores = {"bleu-4": [], "rouge-1": [], "rouge-2": [], "rouge-l": []}
    with open(pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            m = _compute_metrics(obj)
            # map keys
            scores["bleu-4"].append(m.get("bleu-4", 0.0))
            scores["rouge-1"].append(m.get("rouge-1", 0.0))
            scores["rouge-2"].append(m.get("rouge-2", 0.0))
            scores["rouge-l"].append(m.get("rouge-l", 0.0))

    avg = {k: (sum(v) / len(v) if v else 0.0) for k, v in scores.items()}
    return avg


def run_infer_one_tf(
    model_name: str,
    prompts: List[str],
    labels: List[str],
    save_dir: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tf_batch_size: int,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # decoder-only models require left padding for correct generation alignment
    try:
        tok.padding_side = "left"
    except Exception:
        pass
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype="bfloat16", device_map="auto"
    )
    preds: List[str] = []
    do_sample = False if temperature == 0 else True
    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p)
    for i in tqdm(range(0, len(prompts), max(1, tf_batch_size)), total=(len(prompts)+max(1,tf_batch_size)-1)//max(1,tf_batch_size), desc=f"TF generate: {_sanitize(model_name)}"):
        batch_prompts = prompts[i : i + max(1, tf_batch_size)]
        inputs = tok(batch_prompts, return_tensors="pt", padding=True, truncation=False)
        # If not using device_map, move to model device; otherwise let HF handle sharded placement.
        has_device_map = getattr(model, "hf_device_map", None)
        if not has_device_map:
            inputs = {k: v.to(getattr(model, "device", "cuda")) for k, v in inputs.items()}
        out = model.generate(**inputs, **gen_kwargs)
        batch_text = tok.batch_decode(out, skip_special_tokens=True)
        preds.extend(batch_text)

    out_path = os.path.join(save_dir, f"{_sanitize(model_name)}_generated.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for pr, y, ref in zip(prompts, preds, labels):
            f.write(json.dumps({"prompt": pr, "predict": y, "label": ref}, ensure_ascii=False) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Augmented test JSON (array) path")
    ap.add_argument("--save_dir", required=True, help="Directory to save per-model generations")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen2.5-7B-Instruct", "openai/gpt-oss-20b", "Qwen/Qwen3-32B"],
        help="List of HF model ids to evaluate.",
    )
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--tf_batch_size", type=int, default=16, help="batch size for Transformers fallback generation")
    ap.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "vllm", "tf"],
        help="Generation backend: auto (try vLLM then TF), vllm only, or tf only.",
    )
    args = ap.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = [f"{d.get('instruction','')}\n{d.get('input','')}" for d in data]
    labels = [d.get("output", "") for d in data]

    for m in tqdm(args.models, total=len(args.models), desc="Models"):
        print(f"\n=== Evaluating model: {m} ===")
        if args.backend == "tf":
            t0 = time.perf_counter()
            print("[backend=tf] using Transformers for generation")
            pred_path = run_infer_one_tf(
                model_name=m,
                prompts=prompts,
                labels=labels,
                save_dir=args.save_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                tf_batch_size=args.tf_batch_size,
            )
            print(f"TF time: {time.perf_counter()-t0:.2f}s, prompts: {len(prompts)}")
        elif args.backend == "vllm":
            t0 = time.perf_counter()
            print("[backend=vllm] using vLLM for generation")
            pred_path = run_infer_one(
                model_name=m,
                prompts=prompts,
                labels=labels,
                save_dir=args.save_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                tp=args.tp,
            )
            print(f"vLLM time: {time.perf_counter()-t0:.2f}s, prompts: {len(prompts)}")
        else:  # auto
            try:
                t0 = time.perf_counter()
                print("[backend=auto] trying vLLM first…")
                pred_path = run_infer_one(
                    model_name=m,
                    prompts=prompts,
                    labels=labels,
                    save_dir=args.save_dir,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    tp=args.tp,
                )
                print(f"vLLM time: {time.perf_counter()-t0:.2f}s, prompts: {len(prompts)}")
            except BaseException as e:
                print("vLLM 生成失败，切换到 Transformers 推理……", e)
                t0 = time.perf_counter()
                pred_path = run_infer_one_tf(
                    model_name=m,
                    prompts=prompts,
                    labels=labels,
                    save_dir=args.save_dir,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    tf_batch_size=args.tf_batch_size,
                )
                print(f"TF time: {time.perf_counter()-t0:.2f}s, prompts: {len(prompts)}")
        try:
            print("You can run: python scripts/eval_bleu_rouge.py", pred_path)
            # avg = evaluate_file(pred_path)
            # print(
            #     f"predict_bleu-4: {avg['bleu-4']:.4f}\n"
            #     f"predict_rouge-1: {avg['rouge-1']:.4f}\n"
            #     f"predict_rouge-2: {avg['rouge-2']:.4f}\n"
            #     f"predict_rouge-l: {avg['rouge-l']:.4f}"
            # )
        except Exception as e:
            print("Metric evaluation failed:", e)
            print("You can run: python scripts/eval_bleu_rouge.py", pred_path)


if __name__ == "__main__":
    main()
