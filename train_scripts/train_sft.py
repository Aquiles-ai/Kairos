"""Fused single-phase training for Kairos: projector + LoRA adapters on LFM2.5.

Why this exists (the evidence from the frozen stage-1 experiments):
- A trainable projector against a 100%-frozen LLM learns to shift the logits
  enough for teacher-forced CE gains (ablation +3.7 nats) but never enough
  to flip greedy argmax — 4 generation probes across formats and a hard L2
  output cap all produced image-INDEPENDENT decoding. Content alignment
  without behavioral readability = not an aligned model in any usable sense.
- The LLM-side circuit that turns image features into conversational
  behavior lives in the LLM's weights, so this script trains projector +
  LoRA(LLM) TOGETHER in one phase (INIT_FROM_STAGE1=False starts from the
  kairos-init base; True warm-starts the projector from the stage-1 run).
- LoRA (r=16) keeps the instruct behavior intact; the projector trains in
  fp32; at the end the LoRA deltas are MERGED back (`merge_and_unload`), so
  the saved checkpoint is still a plain KairosForConditionalGeneration — no
  PEFT runtime needed downstream.

"""

import random
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import wandb

wandb.login(key="YOUR_KEY")

# Single-phase recipe: projector + LoRA(LLM) trained together from the base
# checkpoint. The frozen stage-1 probe showed alignment under teacher
# forcing but zero behavioral grounding; with adapters on the LLM the
# "reads image features -> talks about them" circuit is learned directly.
INIT_FROM_STAGE1 = True  # True = warm-start projector from ./kairos-stage1/final
CKPT = "Aquiles-ai/Kairos-Proj-80k" if INIT_FROM_STAGE1 else "Aquiles-ai/Kairos-Initialized"
DATASET_ID = "Aquiles-ai/Kairos-Multimodal-Reasoning"
OUTPUT_DIR = "/root/.local/share/kairos_early_fusion_with_proj"

IM_START_ID = 124899      # <|im_start|>
IM_END_ID = 124900        # <|im_end|> / eos
PAD_ID = 124893           # <|pad|>

LORA_R = 16
LORA_ALPHA = 32          # standard alpha = 2x rank
LORA_DROPOUT = 0.05

ZERO_INIT_PROJECTOR = not INIT_FROM_STAGE1  # fresh projector: neutral start (validated in stage-1)
PROJECTOR_OUTPUT_SCALE = 0.89  # hard L2 cap: costs nothing in CE, keeps LoRA inputs stable
PROJECTOR_LR = 1e-4 if INIT_FROM_STAGE1 else 1e-3  # scratch projector needs the stage-1 lr
LORA_LR = 2e-4           # LoRA does the heavy behavioral learning
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 16           # effective batch 64 (LoRA fwd/bwd is heavier than projector-only)
EPOCHS = 1.0             # small sets (10-50k): multiple passes; big sets (200k+): 1 epoch
WARMUP_STEPS = 100        # small sets: 100-step warmup would eat half the run
MAX_SAMPLES = 30000  # validation under the adjusted GPU budget: 30k (~1.6h at 1.25 epochs); raise to 50-117k if the ablation delta improves

EVAL_SIZE = 512
LOGGING_STEPS = 20
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2
NUM_WORKERS = 8
SEED = 42
BF16 = True

# ----------------------------------------------------------------------------

def build_user_message(image, question: str):
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }

def assert_special_tokens(processor):
    """Fail fast if the hardcoded special-token ids drift from the
    checkpoint's actual vocabulary, instead of silently corrupting
    label masking or generation stopping."""
    tok = processor.tokenizer
    checks = {
        "<|im_start|>": IM_START_ID,
        "<|im_end|>": IM_END_ID,
        "<|pad|>": PAD_ID,
        # The whole image-merge pipeline keys on this id; if the tokenizer
        # ever maps "<image>" elsewhere, the scatter would target nothing.
        processor.image_token: processor.image_token_id,
    }
    for token_str, expected_id in checks.items():
        actual_id = tok.convert_tokens_to_ids(token_str)
        if actual_id != expected_id:
            raise ValueError(
                f"special token id mismatch for {token_str}: "
                f"hardcoded={expected_id} actual={actual_id}. "
                f"Update IM_START_ID/IM_END_ID/PAD_ID to match this checkpoint."
            )

def collate(features):
    max_len = max(f["input_ids"].shape[0] for f in features)
    bsz = len(features)
    input_ids = torch.full((bsz, max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    pixel_values, grids = [], []
    for i, f in enumerate(features):
        n = f["input_ids"].shape[0]
        input_ids[i, :n] = f["input_ids"]
        labels[i, :n] = f["labels"]
        attention_mask[i, :n] = 1
        pixel_values.append(f["pixel_values"])      # (n_tokens_img, dim) NaViT-packed
        grids.append(f["image_grid_thw"])           # (1, 3)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": torch.cat(pixel_values, dim=0),
        "image_grid_thw": torch.cat(grids, dim=0),  # (bsz, 3)
    }
 
 
@torch.no_grad()
def embedding_scale_check(model, dataset, n=8, tag=""):
    """L2-norm scale of image embeds (post-projector) vs the text embeds
    actually present in the prompts (embedding-table rows).

    Why this matters: the frozen LFM2.5 expects token vectors of a certain
    scale. Image embeds 10-100x larger saturate attention over the image
    slots, which strangles the projector gradient (loss stuck at ~ln(128k)
    ~= 11.8 — the flat 12.2 plateau from the previous run). Image embeds
    at ~0 are neutral (the zero-init start = text-only prior). The healthy
    end state is a ratio of order ~1, not 0.01 nor 100.
    """
    was_training = model.training
    model.eval()
    feats = [dataset[random.randrange(len(dataset))] for _ in range(n)]
    batch = collate(feats)
    batch = {k: v.to(model.device) for k, v in batch.items()}

    image_embeds = torch.cat(
        model.model.get_image_features(batch["pixel_values"], batch["image_grid_thw"]),
        dim=0,
    )
    text_mask = (batch["attention_mask"] == 1) & (batch["input_ids"] != model.config.image_token_id)
    text_embeds = model.get_input_embeddings()(batch["input_ids"][text_mask])

    img_norms = image_embeds.float().norm(dim=-1)
    txt_norms = text_embeds.float().norm(dim=-1)
    ratio = img_norms.mean() / txt_norms.mean().clamp_min(1e-12)
    print(f"\n===== embedding scale check ({tag}) =====")
    print(f"  image_embeds (n={img_norms.numel()}): norm mean {img_norms.mean():.3f}  "
          f"std {img_norms.std():.3f}  max {img_norms.max():.3f}")
    print(f"  text_embeds  (n={txt_norms.numel()}): norm mean {txt_norms.mean():.3f}  "
          f"std {txt_norms.std():.3f}  max {txt_norms.max():.3f}")
    print(f"  ratio image/text    : {ratio:.3f}  (healthy: order ~1; 0.0 expected after zero-init)")
    if was_training:
        model.train()


@torch.no_grad()
def qualitative_check(model, processor, ds, n=3, tag="after", max_tokens: int = 1024):
    """Greedy-decode a few held-out samples: the visible before/after signal.

        LFM2.5's chat template always opens a `<think>` block with
        add_generation_prompt=True. Since the model was trained with
        "thinking{reasoning} response{answer}" as the assistant turn
        content, the opener must stay in the generation prompt: the model
        learned to see that token as the start of its turn and to close it
        itself. It is never trimmed.
    """
    was_training = model.training
    model.eval()
    print(f"\n===== qualitative check ({tag}) =====")
    for k in range(min(n, len(ds))):
        item = ds[random.randrange(len(ds))]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt = (item["prompt"] or "").strip()
        ref_reasoning = (item["reasoning"] or "").strip()
        ref_answer = (item["answer"] or "").strip()

        enc = processor.apply_chat_template(
            [build_user_message(image, prompt)],
            tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        out = model.generate(
            **enc, max_new_tokens=max_tokens, do_sample=False,
            eos_token_id=IM_END_ID, pad_token_id=PAD_ID,
        )
        gen = out[0, enc["input_ids"].shape[1]:]
        gen_text = processor.decode(gen, skip_special_tokens=True)
        if "</think>" in gen_text:
            gen_reasoning, gen_answer = gen_text.split("</think>", 1)
            gen_reasoning = gen_reasoning.strip()
            gen_answer = gen_answer.strip()
        else:
            gen_reasoning = gen_text.strip()
            gen_answer = "[TRUNCATED: I don't close </think> inside max_new_tokens]"

        print(f"  [{k}] prompt          : {prompt[:160]}")
        print(f"      prompt tokens   : {enc['input_ids'].shape[1]}")
        print(f"      model reasoning : {gen_reasoning[:300]}")
        print(f"      ref   reasoning : {ref_reasoning[:300]}")
        print(f"      model answer    : {gen_answer[:160]}")
        print(f"      ref   answer    : {ref_answer[:160]}")
    if was_training:
        model.train()
 
 
@torch.no_grad()
def image_ablation_check(trainer, eval_ds, n_batches=4):
    """Compares eval loss WITH real image embeddings vs WITH pixel_values
    dropped entirely (frozen LLM falls back to its own language prior).
    A small gap here means the projector isn't contributing useful signal
    yet -- distinguishes "needs more training" from "just needs more scale".
    """
    model = trainer.model
    # train_sft.py wraps the LLM in a PeftModel: unwrap to the Kairos base for
    # direct forwards (KairosForConditionalGeneration.base_model would return
    # the lm_head-less KairosModel, so only unwrap actual PEFT wrappers).
    if hasattr(model, "peft_config"):
        model = model.base_model
    was_training = model.training
    model.eval()

    loader = trainer.get_eval_dataloader(eval_ds)
    with_image_losses, no_image_losses = [], []
 
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
 
        loss_with = trainer.compute_loss(model, dict(batch))
        with_image_losses.append(loss_with.item())
 
        batch_no_image = dict(batch)
        batch_no_image["pixel_values"] = None
        batch_no_image["image_grid_thw"] = None
        loss_without = trainer.compute_loss(model, batch_no_image)
        no_image_losses.append(loss_without.item())
 
    avg_with = sum(with_image_losses) / len(with_image_losses)
    avg_without = sum(no_image_losses) / len(no_image_losses)
    print(f"\n===== image ablation =====")
    print(f"  eval loss with real image : {avg_with:.4f}")
    print(f"  eval loss without image   : {avg_without:.4f}")
    print(f"  delta                     : {avg_without - avg_with:.4f}")

    if was_training:
        model.train()
    return {
        "eval/image_loss": avg_with,
        "eval/no_image_loss": avg_without,
        "eval/ablation_delta": avg_without - avg_with,
    }

class AblationCallback(TrainerCallback):
    """Runs the image ablation after every evaluation and logs it
    to wandb at the same step as the eval loss, to see whether the delta
    crosses positive while the loss drops (an image-text alignment signal)."""

    def __init__(self, trainer, eval_ds):
        self.trainer = trainer
        self.eval_ds = eval_ds

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        result = image_ablation_check(self.trainer, self.eval_ds)
        wandb.log(result, step=state.global_step)

def _find_last_subsequence(haystack: torch.Tensor, needle: list) -> int:
        n, m = haystack.shape[0], len(needle)
        for start in range(n - m, -1, -1):
            if haystack[start:start + m].tolist() == needle:
                return start + m
        return -1

def _tokenize_item(item, processor, assistant_header_ids):
    empty = {"input_ids": [], "labels": [], "pixel_values": [], "image_grid_thw": []}
    try:
        prompt = (item["prompt"] or "").strip()
        reasoning = (item["reasoning"] or "").strip()
        answer = (item["answer"] or "").strip()
        if not prompt or not reasoning or not answer:
            return empty

        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = [
            build_user_message(image, prompt),
            {"role": "assistant", "reasoning": reasoning, "content": answer},
        ]
        full_enc = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
        )
        input_ids = full_enc["input_ids"][0]

        prompt_len = _find_last_subsequence(input_ids, assistant_header_ids)
        if prompt_len == -1 or prompt_len >= input_ids.shape[0]:
            return empty

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids.tolist(),
            "labels": labels.tolist(),
            "pixel_values": full_enc["pixel_values"].tolist(),
            "image_grid_thw": full_enc["image_grid_thw"].tolist(),
        }
    except (OSError, ValueError) as e:
        print(f"[warn] sample discarded during tokenization: {e}")
        return empty

class KairosDataset(torch.utils.data.Dataset):
    """Wraps the RAW HF dataset and tokenizes each item on the fly in
    __getitem__, during training (no pre-tokenized cache on disk: instant
    startup, zero GB of cache; in exchange the chat template, image
    patchify and label masking run on every epoch).

    Items that fail tokenization are not filtered a priori: __getitem__
    scans forward for the next valid item (with wrap-around), so the
    dataloader never receives an empty sample and no upfront scan is needed.
    """

    def __init__(self, raw_ds, processor, desc=""):
        self.ds = raw_ds
        self.processor = processor
        self.desc = desc
        self.assistant_header_ids = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    def __len__(self):
        return len(self.ds)

    @staticmethod
    def _to_tensor(item):
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "pixel_values": torch.tensor(item["pixel_values"], dtype=torch.float32),
            "image_grid_thw": torch.tensor(item["image_grid_thw"], dtype=torch.long),
        }

    def __getitem__(self, idx):
        item = _tokenize_item(self.ds[idx], self.processor, self.assistant_header_ids)
        if item["input_ids"]:
            return self._to_tensor(item)
        probe = (idx + 1) % len(self.ds)
        while probe != idx:
            item = _tokenize_item(self.ds[probe], self.processor, self.assistant_header_ids)
            if item["input_ids"]:
                return self._to_tensor(item)
            probe = (probe + 1) % len(self.ds)
        raise RuntimeError(f"dataset {self.desc}: no valid sample found")

def wrap_with_lora(model):
    """Adds LoRA adapters to every nn.Linear inside the frozen LFM2.5.

    Target selection is done by RELATIVE full names under language_model —
    never bare class names — because the projector shares suffix names
    ("in_proj"/"out_proj") and must NOT receive adapters.
    """
    from peft import LoraConfig, get_peft_model

    lm = model.model.language_model
    target_modules = [
        name
        for name, module in lm.named_modules()
        if isinstance(module, torch.nn.Linear)
        and ".self_attn." in name
        and any(name.endswith(s) for s in ("q_proj", "k_proj", "v_proj", "out_proj"))
    ]
    assert target_modules, "no LoRA targets found under language_model"
    print(f"LoRA target_modules ({len(target_modules)}): {target_modules}")

    cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    peft_model = get_peft_model(model, cfg)

    # LoRA adapters inherit the base dtype (bf16) — Adam on raw bf16 is the
    # silent-stall trap from stage-1. Trainable params go fp32.
    for name, p in peft_model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return peft_model


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"loading {CKPT} ...")
    processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    assert_special_tokens(processor)

    model = AutoModelForCausalLM.from_pretrained(
        CKPT, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda")
    model.model.freeze_for_pretraining()          # vision + llm frozen, projector trainable
    model.model.mm_projector.to(torch.float32)
    if ZERO_INIT_PROJECTOR:
        torch.nn.init.zeros_(model.model.mm_projector.out_proj.weight)
        torch.nn.init.zeros_(model.model.mm_projector.out_proj.bias)
        print("out_proj: zero-init applied")
    # NB: the projector reads output_scale at __init__ from the config — set
    # the module attr directly (kairos-init's config.json predates the cap).
    model.model.mm_projector.output_scale = PROJECTOR_OUTPUT_SCALE
    model.config.projector_output_scale = PROJECTOR_OUTPUT_SCALE
    print(f"projector output cap: L2-normalize to {PROJECTOR_OUTPUT_SCALE}")

    # get_peft_model wraps modules IN-PLACE: `model` (the Kairos base) keeps
    # working for generate()/probes WITH LoRA active.
    peft_model = wrap_with_lora(model)
    # PEFT freezes ALL base-model params when wrapping (only leaves the
    # adapters and modules_to_save trainable). The projector is not an
    # adapter: its gradient must be re-activated and it must go back to fp32.
    for name, p in peft_model.named_parameters():
        if "mm_projector" in name:
            p.requires_grad = True
            p.data = p.data.float()
    peft_model.accepts_loss_kwargs = False  # keep grad-accum normalization correct
    peft_model.print_trainable_parameters()

    print(f"loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID)["train"].shuffle(seed=SEED)
    eval_ds_raw = ds.select(range(EVAL_SIZE))
    train_ds_raw = ds.select(range(EVAL_SIZE, len(ds)))

    if MAX_SAMPLES:
        train_ds_raw = train_ds_raw.select(range(min(MAX_SAMPLES, len(train_ds_raw))))

    train_ds = KairosDataset(train_ds_raw, processor, "train")
    eval_ds = KairosDataset(eval_ds_raw, processor, "eval")
    print(f"train: {len(train_ds)}  eval: {len(eval_ds)}  "
          f"effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")

    proj_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    print(f"trainable: projector {sum(p.numel() for p in proj_params)/1e6:.1f}M, "
          f"lora {sum(p.numel() for p in lora_params)/1e6:.1f}M")
    optimizer = torch.optim.AdamW(
        [
            {"params": proj_params, "lr": PROJECTOR_LR},
            {"params": lora_params, "lr": LORA_LR},
        ],
        weight_decay=0.01,
    )

    targs = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LORA_LR,           # groups carry their own; this is the scheduler's base
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=1.0,
        bf16=BF16,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        remove_unused_columns=False,
        dataloader_num_workers=NUM_WORKERS,
        report_to="wandb",
        seed=SEED,
    )
    trainer = Trainer(
        model=peft_model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
        optimizers=(optimizer, None),    # Trainer builds the scheduler over the 2 groups
    )
    trainer.add_callback(AblationCallback(trainer, eval_ds))

    qualitative_check(model, processor, eval_ds_raw, n=2, tag="before stage-2")
    image_ablation_check(trainer, eval_ds)

    trainer.train()

    final_dir = f"{OUTPUT_DIR}/final"
    merged = peft_model.merge_and_unload()   # plain KairosForConditionalGeneration again
    merged.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"saved (LoRA merged into LLM) -> {final_dir}")

    image_ablation_check(trainer, eval_ds)
    embedding_scale_check(model, eval_ds, tag="post-stage2")
    qualitative_check(model, processor, eval_ds_raw, n=3, tag="after stage-2")
    print("\now run:  python probe_stage1.py", final_dir)


if __name__ == "__main__":
    main()
