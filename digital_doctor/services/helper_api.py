from __future__ import annotations

import os
from typing import Optional

import torch
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..paths import resolve_repo_path

APP = FastAPI(title="Digital Doctor Helper API")

MODEL_DIR = os.getenv(
    "HELPER_MODEL_DIR",
    str(resolve_repo_path("train/saves/gpt-20b/full/sft_ocd_v2")),
)
API_KEY = os.getenv("HELPER_API_KEY")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
eos_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|end|>")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)


class HelperRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7


def _verify_api_key(authorization: Optional[str]) -> None:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )


@APP.post("/helper/generate")
def generate(req: HelperRequest, authorization: Optional[str] = Header(default=None)) -> dict[str, str]:
    _verify_api_key(authorization)
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )
    new_tokens = outputs[0][inputs.input_ids.shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    return {"text": text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(APP, host="0.0.0.0", port=8001)
