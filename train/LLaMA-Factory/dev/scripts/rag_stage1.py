#!/usr/bin/env python3
"""
Simple retrieval helper (concise):
Given training and test (Alpaca-style JSON/JSONL), for each test example
find top-K most similar training inputs (token Jaccard). Output a JSON array
that mirrors test.json (instruction/input/output), where input is appended with
few-shot examples in the following format:

  here are examples
  input: <train_input_1>
  output: <train_output_1>

  input: <train_input_2>
  output: <train_output_2>

  input: <train_input_3>
  output: <train_output_3>

Usage:
python dev/scripts/rag_stage1.py \
    --train /workspace/data/ocd_train.json \
    --test  /workspace/data/ocd_test.json \
    --k 3 \
    --out /workspace/dev/test_rag.json
"""

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any, Iterable, List, Tuple


def _load_records(path: str) -> List[dict]:
    """Load a dataset file that is either JSON array or JSONL."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    # default: JSON array
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
        if isinstance(obj, list):
            return obj
        raise ValueError(f"Unsupported JSON structure in {path}, expected a list.")


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _to_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _tokenize(s: str) -> List[str]:
    s = s.lower()
    return _TOKEN_RE.findall(s)


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def build_inverted_index(tokenized_train: List[List[str]]) -> dict:
    inv: dict[str, List[int]] = defaultdict(list)
    for idx, toks in enumerate(tokenized_train):
        for t in set(toks):  # set to avoid duplicate postings
            inv[t].append(idx)
    return inv


def retrieve_topk(
    train_inputs: List[str],
    train_outputs: List[str],
    test_inputs: List[str],
    k: int = 3,
) -> List[List[Tuple[int, float]]]:
    tokenized_train = [_tokenize(x) for x in train_inputs]
    inv = build_inverted_index(tokenized_train)

    results: List[List[Tuple[int, float]]] = []
    for q in test_inputs:
        q_tokens = _tokenize(q)
        candidates: set[int] = set()
        for t in set(q_tokens):
            if t in inv:
                candidates.update(inv[t])
        # Fallback: if no token overlap, compare against all (rare but safe)
        if not candidates:
            candidates = set(range(len(train_inputs)))

        scored: List[Tuple[int, float]] = []
        for idx in candidates:
            score = _jaccard(q_tokens, tokenized_train[idx])
            if score > 0.0:
                scored.append((idx, score))

        if not scored:  # still empty, take first k with score 0
            scored = [(i, 0.0) for i in range(min(k, len(train_inputs)))]

        scored.sort(key=lambda x: x[1], reverse=True)
        results.append(scored[:k])

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to training JSON/JSONL file")
    ap.add_argument("--test", required=True, help="Path to test JSON/JSONL file")
    ap.add_argument("--k", type=int, default=3, help="Top-K neighbors to retrieve")
    ap.add_argument("--out", required=True, help="Output JSON (array) path with input appended by examples")
    args = ap.parse_args()

    train_recs = _load_records(args.train)
    test_recs = _load_records(args.test)

    train_inputs = [_to_str(rec.get("input", "")) for rec in train_recs]
    train_outputs = [_to_str(rec.get("output", "")) for rec in train_recs]
    test_inputs = [_to_str(rec.get("input", "")) for rec in test_recs]

    topk = retrieve_topk(train_inputs, train_outputs, test_inputs, k=args.k)

    # Build augmented JSON array that mirrors test.json
    aug_list = []
    k = args.k  # local alias to avoid NameError below
    for rec, neigh in zip(test_recs, topk):
        orig_input = _to_str(rec.get("input", ""))
        test_out_str = _to_str(rec.get("output", ""))
        # first take filtered top candidates
        selected: List[int] = []
        for idx, _ in neigh:
            if train_outputs[idx] == test_out_str:
                continue
            selected.append(idx)
            if len(selected) == k:
                break
        # if not enough, backfill from the rest of training set (simple, deterministic)
        if len(selected) < k:
            for cand in range(len(train_inputs)):
                if cand in selected:
                    continue
                if train_outputs[cand] == test_out_str:
                    continue
                selected.append(cand)
                if len(selected) == k:
                    break
        # build display blocks
        blocks: List[str] = []
        for idx in selected:
            blocks.append(f"input: {train_inputs[idx]}")
            blocks.append(f"output: {train_outputs[idx]}")
            blocks.append("")  # blank line between examples
        new_input = orig_input
        if blocks:
            examples_block = "here are examples\n" + "\n".join(blocks).rstrip()
            new_input = orig_input + "\n\n" + examples_block
        aug_list.append(
            {
                "instruction": _to_str(rec.get("instruction", "")),
                "input": new_input,
                "output": _to_str(rec.get("output", "")),
            }
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(aug_list, f, ensure_ascii=False, indent=2)
    print(f"Saved augmented test JSON to {args.out}")


if __name__ == "__main__":
    main()

#/workspace/data/ocd_train.json
#/workspace/data/ocd_test.json

#先找到每个test example的类似 然后放进去做 这样vllm生成快 