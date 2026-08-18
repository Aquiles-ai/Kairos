"""Stage-1 projector alignment training for Kairos (LLaVA-style).

Trains ONLY `KairosMultimodalProjection` on image-caption pairs from
the embedded version of LLaVA-CC3M-Pretrain-595K (image column already
contains decoded bytes, no zip extraction needed at train time).
Vision tower (MoonViT-3D) and the LLM (LFM2.5) stay frozen the whole run.

Notes:
- The embedded dataset already stores "conversations" as clean
  role/content turns, with every literal `<image>` placeholder removed. 
  This script only attaches the image to the first
  user turn; the processor is the ONLY owner of image-token expansion,
  exactly like at inference time.
- Labels supervise ONLY the assistant's answer: everything up to and
  including the assistant header (`<|im_start|>assistant\n`) is masked
  with -100.
- Loss is computed by the MODEL from `labels`. In transformers 5.x,
  `PreTrainedModel.loss_function` resolves any `*ForConditionalGeneration`
  class to `ForCausalLMLoss` (modeling_utils.py loss_type lookup +
  loss/loss_utils.py LOSS_MAPPING), which already does the next-token
  shift and upcasts logits to fp32. `KairosForConditionalGeneration`
  sets `accepts_loss_kwargs = False` (like the Kimi_K25 reference) so
  the Trainer applies its standard grad-accum normalization.
- Mixed precision: the backbone (vision tower + LFM2.5) loads in bf16
  to save memory, but the trainable projector is upcast to fp32. Adam
  over raw bf16 weights silently stops updating: with |w| ~ 0.02 one
  bf16 ULP is ~8e-5, so any update smaller than that (lr <~ 1e-4,
  i.e. most of a cosine schedule) rounds to exactly the old value.
- `ZERO_INIT_PROJECTOR` zeroes `out_proj` so training starts from the
  LLM's text-only prior (image slots ~neutral) instead of injecting
  random, disproportionately large embeddings into a frozen LM. The
  ablation check before/after tells you whether the projector learned.
- `EMPTY_THINK_SUPERVISION` renders each answer as `<think></think><CAPTION>`:
  LFM2.5's template (and its instruct habits) always enter a think block,
  so bare-caption stage-1 models generate garbage in think-mode ("no image
  attached"). Empty-think teaches the model to close the block and answer,
  no LLM-rewritten data needed. Reasoning-style answers are a stage-2
  (LoRA) job, only there can the LLM itself be taught new behavior.

"""

import random

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
)
import wandb

# Set WANDB_API_KEY as an environment variable before running this script
# instead of hardcoding a key here (e.g. `export WANDB_API_KEY=...`).
# wandb picks it up automatically; no explicit login() call needed.
# If a previous run leaked a key in a shared chat/log, revoke it at
# https://wandb.ai/authorize and generate a new one.
wandb.login(key="YOUR_KEY")

MODEL_ID = "Aquiles-ai/Kairos-Initialized"

# IMPORTANT: this must be the EMBEDDED dataset (image column holds bytes,
# decodes to PIL.Image automatically), not the original liuhaotian repo
# where "image" is just a filename string relative to images.zip.
DATASET_ID = "Aquiles-ai/LLaVA-CC3M-Pretrain-595K-Embedded"

OUTPUT_DIR = "/root/.local/share/kairos"

LR = 1e-3          # stage-1 projector-only alignment (LLaVA pretrain scale)
PER_DEVICE_BATCH = 16
GRAD_ACCUM = 4      # effective batch = PER_DEVICE_BATCH * GRAD_ACCUM (= 64)
                    # grad-accum keeps the fp32 logits memory spike at 1/4
                    # (ForCausalLMLoss upcasts logits to float32: ~vocab 128k)
EPOCHS = 1.0
WARMUP_STEPS = 100
MAX_SAMPLES = 80000  # scale test (80k); 40000 for a shorter one; None = full 595k
ZERO_INIT_PROJECTOR = True  # start from the LLM's text-only prior (see docstring)
# Hard L2 cap on projector output norms (~mean norm of LFM2.5's text
# embeddings, measured 0.88-0.89). Disables the magnitude-cheat channel so
# image info must be expressed in directions the frozen LLM understands.
# None = uncapped (previous behavior).
PROJECTOR_OUTPUT_SCALE = 0.89
# LFM2.5's template ALWAYS opens <think> on the assistant turn and the
# instruct model re-opens it at generation time even when trimmed from the
# prompt. Stage-1 captions are bare, so think-mode generation is OOD (the
# model defends "no image attached"). Empty-think supervision renders each
# answer as `<think></think><CAPTION>`: the model learns to close the
# block immediately and answer — matching its native generation structure.
EMPTY_THINK_SUPERVISION = True

EVAL_SIZE = 512     # held-out samples: eval loss + qualitative check
LOGGING_STEPS = 20
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = 2
NUM_WORKERS = 8     # raise if your CPU allows: processing 80k images on-the-fly is a bottleneck
SEED = 42
BF16 = True         # bf16 autocast; trainable params (projector) stay fp32

IM_START_ID = 124899      # <|im_start|>
IM_END_ID = 124900        # <|im_end|> / eos
PAD_ID = 124893           # <|pad|>


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
 
 
def build_user_message(image, question: str):
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }
 
 
class CaptionDataset(torch.utils.data.Dataset):
    """On-the-fly tokenization + label masking for image-caption pairs.
 
    The image is encoded exactly once per sample. prompt_len (where the
    assistant's supervised answer starts) is found by locating the fixed
    "<|im_start|>assistant\n" header inside the already-encoded sequence,
    instead of re-running apply_chat_template (and re-encoding the image)
    a second time just to measure a length.
    """
 
    def __init__(self, hf_ds, processor, desc=""):
        self.ds = hf_ds
        self.processor = processor
        self.desc = desc
        self._assistant_header_ids = processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
 
    def __len__(self):
        return len(self.ds)
 
    @staticmethod
    def _find_last_subsequence(haystack: torch.Tensor, needle: list) -> int:
        """Index right after the last occurrence of `needle` in `haystack`.
        Returns -1 if not found."""
        n, m = haystack.shape[0], len(needle)
        for start in range(n - m, -1, -1):
            if haystack[start:start + m].tolist() == needle:
                return start + m
        return -1
 
    def _build(self, idx: int):
        item = self.ds[idx]
        conv = item["conversations"]
        if len(conv) < 2:
            raise ValueError("conversation too short")
 
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
 
        # The dataset already stores conversations as clean role/content
        # turns. The only transform needed here is attaching the image to
        # the first user turn, since the dataset stores that turn's
        # content as a plain string but the processor expects a
        # multimodal content list for it.
        messages = []
        image_attached = False
        for turn in conv:
            if turn["role"] == "user" and not image_attached:
                messages.append(build_user_message(image, turn["content"]))
                image_attached = True
            else:
                messages.append({"role": turn["role"], "content": turn["content"]})
        if not image_attached:
            raise ValueError("no user turn found to attach the image")

        if EMPTY_THINK_SUPERVISION:
            # Teach the native `"<think></think>"` sentinel before the answer;
            # see the config flag's comment.
            messages[-1]["content"] = "<think></think>" + messages[-1]["content"]

        caption = conv[-1]["content"].strip()
        if not caption:
            raise ValueError("empty caption")
 
        full_enc = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
        )
        input_ids = full_enc["input_ids"][0]
 
        prompt_len = self._find_last_subsequence(input_ids, self._assistant_header_ids)
        if prompt_len == -1:
            raise ValueError("assistant header not found in encoded sequence")
        if prompt_len >= input_ids.shape[0]:
            raise ValueError("no supervised tokens left")
 
        labels = input_ids.clone()
        labels[:prompt_len] = -100
 
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": full_enc["pixel_values"],
            "image_grid_thw": full_enc["image_grid_thw"],
        }
 
    def __getitem__(self, idx: int):
        # Only retry on expected, sample-level problems (corrupt image,
        # empty caption, header not found). Anything else (AttributeError,
        # KeyError, TypeError) is a programming/config bug and must surface
        # immediately instead of being masked as "skipping sample".
        expected_errors = (OSError, ValueError)
        for _ in range(4):
            try:
                return self._build(idx)
            except expected_errors as e:
                print(f"[warn] skipping sample {idx} ({self.desc}): {e}")
                idx = random.randrange(len(self))
        raise RuntimeError("too many consecutive bad samples")

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
def qualitative_check(model, processor, ds, n=3, tag="after", trim_think=None):
    """Greedy-decode a few held-out samples: the visible before/after signal.

    The LFM2.5 chat template always opens a `<think>` block with
    `add_generation_prompt=True`. `trim_think=True` strips that opener so
    decoding starts at the bare assistant turn (the format bare-caption
    stage-1 runs supervise); with EMPTY_THINK_SUPERVISION the model itself
    was trained on `<think></think><CAPTION>` and must be probed with the
    opener left in place so it can close the block and answer.
    """
    if trim_think is None:
        trim_think = not EMPTY_THINK_SUPERVISION
    was_training = model.training
    model.eval()
    think_ids = processor.tokenizer.encode("<think>", add_special_tokens=False)
    print(f"\n===== qualitative check ({tag}, trim_think={trim_think}) =====")
    for k in range(min(n, len(ds))):
        item = ds[random.randrange(len(ds))]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        ref = item["conversations"][-1]["content"].strip()
        enc = processor.apply_chat_template(
            [build_user_message(image, "Describe this image.")],
            tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        # Optionally trim the trailing "<think>" the template opens (no-op
        # when probing models trained with EMPTY_THINK_SUPERVISION).
        seq_len = enc["input_ids"].shape[1]
        if trim_think and enc["input_ids"][0, -len(think_ids):].tolist() == think_ids:
            cut = seq_len - len(think_ids)
            enc = BatchFeature({
                k: (v[..., :cut] if torch.is_tensor(v) and v.dim() >= 2 and v.shape[-1] == seq_len else v)
                for k, v in enc.items()
            })

        out = model.generate(
            **enc, max_new_tokens=128, do_sample=False,
            eos_token_id=IM_END_ID, pad_token_id=PAD_ID,
        )
        gen = out[0, enc["input_ids"].shape[1]:]
        print(f"  [{k}] prompt tokens : {enc['input_ids'].shape[1]}")
        print(f"      model         : {processor.decode(gen, skip_special_tokens=True)}")
        print(f"      reference     : {ref[:160]}")
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
 
 
def main():
    random.seed(SEED)
 
    print(f"loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    assert_special_tokens(processor)
 
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16,
    ).to("cuda")
    model.model.freeze_for_pretraining()             # vision + llm frozen, projector trainable
    # The projector is the ONLY trainable module: give it fp32 params so Adam
    # updates are representable. `from_pretrained(dtype=bf16)` would cast the
    # (deliberately fp32-stored) projector weights down to bf16, killing most
    # updates below ~1 bf16 ULP throughout the cosine schedule.
    model.model.mm_projector.to(torch.float32)
 
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train / 1e6:.1f}M / {n_total / 1e9:.2f}B "
          f"(only the projector should be trainable)")
 
    print(f"loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID)["train"].shuffle(seed=SEED)
    eval_ds_raw = ds.select(range(EVAL_SIZE))
    train_ds_raw = ds.select(range(EVAL_SIZE, len(ds)))
    if MAX_SAMPLES:
        train_ds_raw = train_ds_raw.select(range(min(MAX_SAMPLES, len(train_ds_raw))))
    train_ds = CaptionDataset(train_ds_raw, processor, "train")
    eval_ds = CaptionDataset(eval_ds_raw, processor, "eval")
    print(f"train: {len(train_ds)}  eval: {len(eval_ds)}  "
          f"effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")

    # Scale of the RANDOM projector output vs real text embeddings. If the
    # image/text norm ratio here is >>1, that alone can stall a frozen-LLM
    # run (this was the poisoning regime of the previous run).
    embedding_scale_check(model, eval_ds, tag="random init")
    if ZERO_INIT_PROJECTOR:
        torch.nn.init.zeros_(model.model.mm_projector.out_proj.weight)
        torch.nn.init.zeros_(model.model.mm_projector.out_proj.bias)
        print("out_proj: zero-init applied (image embeds = 0 until gradient lifts them)")
    # NB: KairosMultimodalProjection reads the scale at __init__ (during
    # from_pretrained) - setting ONLY the config attr here is too late and
    # silently runs uncapped (that's what happened in the previous run: its
    # metrics/bit-identical loss curves correlated with a no-op cap). Set
    # BOTH: the module attr for THIS run, the config attr so it persists
    # into the saved checkpoint for later loads.
    model.model.mm_projector.output_scale = PROJECTOR_OUTPUT_SCALE
    model.config.projector_output_scale = PROJECTOR_OUTPUT_SCALE
    if PROJECTOR_OUTPUT_SCALE is not None:
        print(f"projector output cap: L2-normalize to {PROJECTOR_OUTPUT_SCALE} "
              "(persisted to the checkpoint config for inference)")

    targs = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=8,  # fp32-upcast logits over 128k vocab; keep eval batches small
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        # Only the projector is trainable, so this hits only the projector:
        # bounds the norm-explosion channel a frozen-LLM stage-1 otherwise
        # exploits (image embeds reached norm 143/1235 in the 80k test run).
        weight_decay=0.01,
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
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
    )

    # Baselines BEFORE training: with/zero projectors, "with image" should
    # roughly equal "without image" here (especially with ZERO_INIT_PROJECTOR).
    qualitative_check(model, processor, eval_ds_raw, n=2, tag="before training")
    image_ablation_check(trainer, eval_ds)

    trainer.train()
 
    final_dir = f"{OUTPUT_DIR}/final"
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    print(f"saved -> {final_dir}")

    # The success criterion: "with image" should end clearly below "without image".
    image_ablation_check(trainer, eval_ds)
    # Post-train: ratio should be order ~1 — far from both the poisoned-init
    # regime (>>1) and the frozen-at-zero failure (==0 means nothing learned).
    embedding_scale_check(model, eval_ds, tag="post-train")
    qualitative_check(model, processor, eval_ds_raw, n=3, tag="after training")

if __name__ == "__main__":
    main()
