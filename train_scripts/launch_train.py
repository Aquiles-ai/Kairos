"""
Script to launch training on Modal.

Usage:
    modal run launch_train.py
"""

import modal
import subprocess
import threading
import time
from pathlib import Path

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
    .env({"HF_XET_HIGH_PERFORMANCE": "1",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "TORCH_CUDA_ARCH_LIST": "9.0a;10.0a"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data = modal.Volume.from_name("data", create_if_missing=True)
app = modal.App("train-projector-kairos")

N_GPU = 1
SCR_TR = "train.py"
MAX_USE_MODAL = True
MINUTES = 15

def get_time_for_gpu_use(max_use_modal: bool, minutes: int) -> int:
    if max_use_modal:
        return 86400
    return minutes * 60

class VRAMMonitor:
    def __init__(self, interval=0.5, gpu_index=0):
        self.interval = interval
        self.gpu_index = gpu_index
        self.monitoring = False
        self.thread = None
 
    def _query_nvidia_smi(self):
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={self.gpu_index}",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
        )
        used_mb, total_mb, util_pct = out.decode().strip().split(", ")
        return float(used_mb) / 1024, float(total_mb) / 1024, float(util_pct)
 
    def _monitor_loop(self):
        while self.monitoring:
            try:
                used_gb, total_gb, util_pct = self._query_nvidia_smi()
                print(
                    f"[VRAM Monitor] used={used_gb:.2f}GB total={total_gb:.2f}GB "
                    f"gpu_util={util_pct:.0f}%"
                )
            except Exception as e:
                print(f"[VRAM Monitor Error] {e}")
 
            time.sleep(self.interval)
 
    def start(self):
        if not self.monitoring:
            self.monitoring = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            print("[VRAM Monitor] Started")
 
    def stop(self):
        if self.monitoring:
            self.monitoring = False
            if self.thread:
                self.thread.join(timeout=1)
            print("[VRAM Monitor] Stopped")

TIME = get_time_for_gpu_use(MAX_USE_MODAL, MINUTES)

@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu=f"RTX-PRO-6000:{N_GPU}",
    timeout=TIME,
    scaledown_window=3600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.local/share": data,
    },
)
class LaunchTrain:
    @modal.enter()
    def check(self):
        self.vram_monitor = VRAMMonitor(interval=0.5)
        self.vram_monitor.start()
        self.file = Path(f"/root/.local/share/{SCR_TR}")

        if self.file.is_file():
            print("Exists")
        else:
            raise Exception("File does not exist")

    @modal.method()
    def launch_train(self):
        import subprocess

        
        cmd = [
            "python",
            str(self.file)
        ]

        print(f"Command: {cmd}")

        subprocess.run(
            cmd
        )

@app.local_entrypoint()
def main():
    print("Starting training")

    tr = LaunchTrain()

    tr.launch_train.remote()