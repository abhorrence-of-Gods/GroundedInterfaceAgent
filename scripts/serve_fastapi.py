#!/usr/bin/env python3
"""FastAPI server exposing GIA inference.

Run:
    uvicorn scripts.serve_fastapi:app --host 0.0.0.0 --port 8000

Optional environment variables:
    CKPT_PATH   path to a .pt or .safetensors checkpoint to load.
    HYDRA_OPTS  extra Hydra overrides, e.g. "model.language_tower.use_8bit=true".
"""
from __future__ import annotations

import os
import base64
import io
from typing import Optional, List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from hydra import initialize, compose

from engine.inference import InferenceEngine
from models.gia_agent import GiaAgent

# ------------------------------------------------------------
# One-time model initialisation (done at module import)        
# ------------------------------------------------------------
print("[serve_fastapi] ⏳ Loading config and model …")

with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name="default", overrides=os.environ.get("HYDRA_OPTS", "").split())
model = GiaAgent(cfg.model)

ckpt_path = os.environ.get("CKPT_PATH")
if ckpt_path and os.path.isfile(ckpt_path):
    print(f"[serve_fastapi] Loading checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state.get("model_state", state), strict=False)
else:
    print("[serve_fastapi] ⚠️  No checkpoint provided – using random weights.")

# Inference engine (does not execute OS actions)
engine = InferenceEngine(cfg, model)

# ------------------------------------------------------------
# FastAPI app definitions                                      
# ------------------------------------------------------------
app = FastAPI(title="Grounded Interface Agent API", version="0.1")

class PredictRequest(BaseModel):
    instruction: str
    image_b64: Optional[str] = None  # Screenshot; PNG/JPEG base64 string

class PredictResponse(BaseModel):
    x_norm: float  # Normalised X in [0,1] (or pixels if screenshot absent)
    y_norm: float  # Normalised Y
    click: int     # 1 if click, 0 if move
    raw_commands: List


def _decode_image(b64_str: str) -> Image.Image:
    try:
        data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode image_b64: {exc}") from exc

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    screenshot: Optional[Image.Image] = _decode_image(req.image_b64) if req.image_b64 else None
    commands = engine.run(req.instruction, execute_env=False, screenshot=screenshot)
    if not commands:
        raise HTTPException(status_code=500, detail="Model returned no commands")

    cmd, args = commands[0]
    if cmd not in {"mouse_move", "mouse_click"}:
        raise HTTPException(status_code=500, detail=f"Unexpected command {cmd}")

    if screenshot:
        w, h = screenshot.size
        x_norm = args[0] / w
        y_norm = args[1] / h
    else:
        # Cannot normalise without resolution; return raw pixels (>1)
        x_norm, y_norm = float(args[0]), float(args[1])

    click_flag = 1 if cmd == "mouse_click" else 0
    return PredictResponse(x_norm=x_norm, y_norm=y_norm, click=click_flag, raw_commands=commands)

@app.get("/healthz")
def healthz():
    return {"status": "ok"} 