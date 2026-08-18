"""
Script to set the training environment in modal.

Usage:
    modal run setup_env_training.py
"""

import modal
from pathlib import Path

URL_TRAIN_SCR = ""
MODEL = "Aquiles-ai/Kairos-Proj-80k" # or Aquiles-ai/Kairos-Initialized
DW_MODEL = True

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "curl", "build-essential",)
    .entrypoint([])
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --upgrade setuptools wheel"
    )
    .uv_pip_install(
        "torch==2.11",
        "datasets",
        "transformers==5.14.1",
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl",
        "kernels",
        "torchvision",
        "wandb",
        "accelerate",
        "peft"
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data = modal.Volume.from_name("data", create_if_missing=True)
app = modal.App("dw-train-kairos")

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.local/share": data,
    },
    timeout=3600
)
def dw():
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoProcessor
    import requests

    script_dir = Path("/root/.local/share")

    scr = script_dir / "train.py"

    scr.parent.mkdir(exist_ok=True)

    response = requests.get(URL_TRAIN_SCR)
    response.raise_for_status()
    scr.write_bytes(response.content)

    load_dataset("Aquiles-ai/Kairos-Multimodal-Reasoning")
    load_dataset("Aquiles-ai/LLaVA-CC3M-Pretrain-595K-Embedded")

    if DW_MODEL:
        processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)
        print(processor)
        print(model)

@app.local_entrypoint()
def main():
    print("Starting download")

    dw.remote()