import os
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
CUDA_VISIBLE_DEVICES=7 uvicorn dev.expose_api:app --host 0.0.0.0 --port 8000

curl -X POST "http://10.162.9.148:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello"}
  
  <|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.<|end|><|start|>user<|message|>You are Digital OCD Therapeutic Agent.\nWhat are intrusive thoughts?<|end|><|start|>assistant", 
"""


# --------------------
# Configuration
# --------------------
TRAIN_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = os.getenv(
    "HELPER_MODEL_DIR",
    str(TRAIN_DIR / "saves" / "gpt-20b" / "full" / "sft_ocd_v2"),
)



# --------------------
# App
# --------------------
app = FastAPI(title="GPT-20B SFT API", version="1.0.0")


def _resolve_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # Prefer bfloat16 on modern GPUs; fallback to float16
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


# Load tokenizer & model once at startup
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False,
)

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=_resolve_dtype(),
    device_map="auto",  # spread across available GPUs automatically
)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    text: str


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    # Move inputs to first device in model's device map
    device = next(iter(model.hf_device_map.values())) if hasattr(model, "hf_device_map") else model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    input_length = int(input_ids.shape[-1])

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Only decode newly generated tokens (avoid echoing the prompt)
    full_seq = output_ids[0]
    gen_only = full_seq[input_length:] if full_seq.shape[0] > input_length else full_seq
    text = tokenizer.decode(gen_only, skip_special_tokens=True)

    # optional stop-word truncation
    if req.stop:
        for s in req.stop:
            if s and s in text:
                text = text.split(s)[0]
                break

    return GenerateResponse(text=text)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("dev.expose_api:app", host=host, port=port, reload=False)
