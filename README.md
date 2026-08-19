# Kairos

**Experimental research** on building a small multimodal model from scratch: a frozen vision tower ([Aquiles-ai/MoonViT-3D](https://huggingface.co/Aquiles-ai/MoonViT-3D), extracted from [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6)) + a Kimi-style 2-layer MLP projector + [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) as the LLM. Image ↔ text only (no video).

> **⚠️ Warning:** every checkpoint published for this project is an **experiment, not a competent model**. None of them are usable for real-world image understanding. Do not use them in production or for automated decisions. See [Model checkpoints](#model-checkpoints) and the [blog post](https://aquiles-ai.vercel.app/blog/kairos-a-multimodal-model) for what was actually measured.

## Architecture

```
MoonViT-3D (frozen) ──> 2-layer MLP projector ──> LFM2.5-2.6B
                      (Kimi-style, L2-capped)
```

- **Vision tower:** MoonViT-3D, frozen in every training stage. Native-resolution image processing (no resize required).
- **Projector:** `LayerNorm → Linear → GELU → Linear`, mapping merged vision features (4608) to the LLM hidden size (2048), with an optional hard L2 output cap (`projector_output_scale = 0.89`, ≈ mean norm of LFM2.5 text embeddings).
- **LLM:** LFM2.5-2.6B, LoRA-trainable in the final stage (adapters merged at the end, no PEFT runtime needed).
- Model code (modeling, config, processor) follows the Kimi-K2.5/Kimi-K2.6 implementation on `transformers` 5.x and ships with the checkpoints (`trust_remote_code=True`).

## Model checkpoints

| Checkpoint | What changed | Status |
|---|---|---|
| [Kairos-Initialized](https://huggingface.co/Aquiles-ai/Kairos-Initialized) | Assembled architecture, **randomly initialized** projector | Scaffold only, no training |
| [Kairos-Proj-80k](https://huggingface.co/Aquiles-ai/Kairos-Proj-80k) | Stage-1: projector aligned on 80k image-caption pairs (LLM frozen) | Not a usable VLM |
| [Kairos-Alig-30k](https://huggingface.co/Aquiles-ai/Kairos-Alig-30k) | Single-phase: projector + LoRA(LLM) trained together on 30k reasoning samples | Early experiments, still not competent |

## Quick start (inference)

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.image_utils import load_image

model_id = "Aquiles-ai/Kairos-Alig-30k"  # or Kairos-Proj-80k / Kairos-Initialized

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

image = load_image("https://example.com/image.jpg")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "What's in this image?"},
]}]

enc = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)
out = model.generate(**enc, max_new_tokens=1024)
print(processor.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
```

A complete walkthrough with example outputs is in [`inference/Kairos_Inference.ipynb`](inference/Kairos_Inference.ipynb).

## Repository layout

```
kairos/            Model code: modeling_kairos.py, configuration_kairos.py, kairos_processor.py
train_scripts/     Stage-1 projector training, single-phase training, Modal launch + env setup
inference/         Inference notebook
vlm_ablations/     Ablation studies: grounding behavior of the final checkpoint
```

## Training

Two recipes are implemented, mirroring the experiment progression:

1. **Stage-1 projector alignment** ([`train_projector.py`](train_scripts/train_projector.py)): projector-only training on image-caption pairs from [Aquiles-ai/LLaVA-CC3M-Pretrain-595K-Embedded](https://huggingface.co/datasets/Aquiles-ai/LLaVA-CC3M-Pretrain-595K-Embedded), with the vision tower and LLM frozen. Zero-init `out_proj`, fp32 trainable projector, L2 output cap, empty-think supervision.
2. **Single-phase alignment** ([`train_sft.py`](train_scripts/train_sft.py)): projector + LoRA (r=16, alpha=32) on all LFM2.5 attention projections, trained together from the start, on [Aquiles-ai/Kairos-Multimodal-Reasoning](https://huggingface.co/datasets/Aquiles-ai/Kairos-Multimodal-Reasoning). LoRA is merged back into the base weights at the end.

Training runs on [Modal](https://modal.com/) with a single RTX-PRO-6000 (see [`launch_train.py`](train_scripts/launch_train.py) and [`setup_env_training.py`](train_scripts/setup_env_training.py)).

### Requirements

```
uv pip install torch==2.11 transformers==5.14.1 https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl datasets kernels torchvision wandb peft accelerate
```

## Findings and limitations

The short version: **stage-1 alignment against a frozen LLM is not enough**, and the final single-phase checkpoint only grounds on the image under narrow conditions. The details are documented in two places, so we don't repeat them here:

- The [blog post](https://aquiles-ai.vercel.app/blog/kairos-a-multimodal-model) walks through the full experiment, including why stage-1 was abandoned, the distillation pipeline behind the dataset, and the costs involved.
- [`vlm_ablations/`](vlm_ablations/README.md) contains the measured grounding behavior of Kairos-Alig-30k (prompt-conditioned grounding, text-prior fallback on generic prompts, limited generalization outside the training distribution).

## Datasets

- [Aquiles-ai/LLaVA-CC3M-Pretrain-595K-Embedded](https://huggingface.co/datasets/Aquiles-ai/LLaVA-CC3M-Pretrain-595K-Embedded): embedded version of LLaVA-CC3M-Pretrain-595K (image bytes stored inside the dataset, no zip extraction at train time).
- [Aquiles-ai/Kairos-Multimodal-Reasoning](https://huggingface.co/datasets/Aquiles-ai/Kairos-Multimodal-Reasoning): 116k multimodal reasoning samples built via distillation from frontier teacher models.

## License

The repository code is Apache-2.0. The model checkpoints combine third-party components (MoonViT-3D from Kimi-K2.6, LFM2.5-2.6B) and training data distilled from proprietary frontier models: check each component's license and dataset terms before any use.

## References

- Blog post: [*Kairos: Building a Multimodal Model with LFM2.5 and Kimi-K2.6*](https://aquiles-ai.vercel.app/blog/kairos-a-multimodal-model)
- Papers (as covered in the blog post):
  - [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)
  - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929)
  - [Kimi-VL Technical Report](https://arxiv.org/abs/2504.07491)
  - [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276)
  - [From Unimodal to Multimodal: Scaling up Projectors](https://arxiv.org/abs/2409.19425v1)