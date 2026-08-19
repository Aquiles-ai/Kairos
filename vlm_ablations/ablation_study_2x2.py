"""2x2 ablation study: visual anchor x task type, with SEM.

Measures how much a sample's loss depends on the image (paired delta:
with vs without pixel values, same item) as a function of two prompt
dimensions:

- visual anchor: does the prompt explicitly reference the image
  ("the image", "shown in", "displayed", ...) or not?
- task type: is the question computational (objective, mechanically
  derivable answer: counting, summing, comparing numbers, graph paths)
  or descriptive (interpretation, atmosphere, relations between
  elements, aesthetics)?

The two dimensions control each other: anchored questions tend to be
direct lookups and unanchored ones interpretive, so the keyword effect
must be measured WITHIN each task type, not only between types. Each
sample contributes its own paired delta (same item, with/without image),
so results are reported as mean +/- SEM instead of a bare average --
with n=25 per cell, SEM separates signal from noise.
"""

import random
import re
import statistics
import gc
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

CKPT = "Aquiles-ai/Kairos-Alig-30k"
DATASET_ID = "Aquiles-ai/Kairos-Multimodal-Reasoning"

# Replicate the same eval split used during training, so the model never
# sees examples it was trained on.
EVAL_SIZE = 512
SEED = 42
N_PER_CELL = 25  # per cell of the 2x2 table (4 cells)

GENERIC_PROMPT = "Describe this."

IMAGE_PHRASES = [
    "this image", "the image", "shown in", "displayed", "in the picture",
    "the picture", "pictured", "photo", "photograph",
]
IMAGE_WORDS = [
    "provided", "given", "depicted", "shown", "illustrated", "pictured",
    "photographed", "graph", "diagram", "chart", "figure", "photo",
    "picture", "image", "scene", "infographic",
]
ANCHOR_RE = re.compile(r"\b(" + "|".join(IMAGE_WORDS) + r")\b")

# Task-type heuristic: lookup/computational questions have an objective
# answer derivable mechanically (counting, summing, comparing numbers,
# graph paths). Everything else is treated as 'descriptive'
# (interpretation, atmosphere, relations between elements, aesthetics).
COMPUTATIONAL_WORDS = [
    "shortest", "weight", "total", "how many", "percentage", "node",
    "topological", "spanning tree", "distance", "calculate", "compute",
    "sum", "average", "count", "probability", "maximum", "minimum",
]
COMPUTATIONAL_RE = re.compile(r"\b(" + "|".join(COMPUTATIONAL_WORDS) + r")\b")


def has_visual_anchor(text: str) -> bool:
    t = text.lower()
    if any(ph in t for ph in IMAGE_PHRASES):
        return True
    return bool(ANCHOR_RE.search(t))


def is_computational(text: str) -> bool:
    return bool(COMPUTATIONAL_RE.search(text.lower()))


def build_user_message(image, question: str):
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }


def find_last_subsequence(haystack: torch.Tensor, needle: list) -> int:
    n, m = haystack.shape[0], len(needle)
    for start in range(n - m, -1, -1):
        if haystack[start:start + m].tolist() == needle:
            return start + m
    return -1


model = AutoModelForCausalLM.from_pretrained(
    CKPT, trust_remote_code=True, dtype=torch.bfloat16,
).to("cuda")
model.eval()
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
assistant_header_ids = processor.tokenizer.encode(
    "<|im_start|>assistant\n", add_special_tokens=False
)

ds = load_dataset(DATASET_ID)["train"].shuffle(seed=SEED)
eval_ds_raw = ds.select(range(EVAL_SIZE))

random.seed(SEED)
cells = {
    ("anchor", "computational"): [],
    ("anchor", "descriptive"): [],
    ("no_anchor", "computational"): [],
    ("no_anchor", "descriptive"): [],
}
for i in range(len(eval_ds_raw)):
    prompt = (eval_ds_raw[i]["prompt"] or "").strip()
    reasoning = (eval_ds_raw[i]["reasoning"] or "").strip()
    answer = (eval_ds_raw[i]["answer"] or "").strip()
    if not prompt or not reasoning or not answer:
        continue
    anchor_key = "anchor" if has_visual_anchor(prompt) else "no_anchor"
    task_key = "computational" if is_computational(prompt) else "descriptive"
    cells[(anchor_key, task_key)].append(i)

for k in cells:
    random.shuffle(cells[k])
    cells[k] = cells[k][:N_PER_CELL]
    print(f"cell {k}: {len(cells[k])} samples")


@torch.no_grad()
def compute_loss(image, prompt, reasoning, answer, use_image, use_real_text):
    text = prompt if use_real_text else GENERIC_PROMPT
    messages = [
        build_user_message(image, text),
        {"role": "assistant", "reasoning": reasoning, "content": answer},
    ]
    full_enc = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
    )
    input_ids = full_enc["input_ids"][0]
    prompt_len = find_last_subsequence(input_ids, assistant_header_ids)
    if prompt_len == -1 or prompt_len >= input_ids.shape[0]:
        return None

    labels = input_ids.clone()
    labels[:prompt_len] = -100

    batch = {
        "input_ids": input_ids.unsqueeze(0).to(model.device),
        "labels": labels.unsqueeze(0).to(model.device),
        "attention_mask": torch.ones_like(input_ids).unsqueeze(0).to(model.device),
    }
    if use_image:
        batch["pixel_values"] = full_enc["pixel_values"].to(model.device)
        batch["image_grid_thw"] = full_enc["image_grid_thw"].to(model.device)
    else:
        batch["pixel_values"] = None
        batch["image_grid_thw"] = None

    out = model(**batch)
    return out.loss.item()


def sem(values):
    if len(values) < 2:
        return float("nan")
    return statistics.stdev(values) / (len(values) ** 0.5)


def run_cell(indices, tag):
    delta_image, delta_text = [], []
    skipped = 0
    for i in indices:
        item = eval_ds_raw[i]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        prompt = (item["prompt"] or "").strip()
        reasoning = (item["reasoning"] or "").strip()
        answer = (item["answer"] or "").strip()

        l_full = compute_loss(image, prompt, reasoning, answer, True, True)
        l_no_image = compute_loss(image, prompt, reasoning, answer, False, True)
        l_no_text = compute_loss(image, prompt, reasoning, answer, True, False)

        if None in (l_full, l_no_image, l_no_text):
            skipped += 1
            continue
        delta_image.append(l_no_image - l_full)   # PAIRED: same item, with vs without image
        delta_text.append(l_no_text - l_full)      # PAIRED: same item, real vs generic prompt
        # We forced a memory cleanup in each iteration, given 
        # that we only had access to limited hardware and it 
        # was the only way to run these ablation studies
        torch.cuda.empty_cache()
        gc.collect()

    n = len(delta_image)
    print(f"\n{tag} (n={n}, skipped={skipped})")
    print(f"  delta image: {statistics.mean(delta_image):+.4f} +/- {sem(delta_image):.4f} (SEM)")
    print(f"  delta text : {statistics.mean(delta_text):+.4f} +/- {sem(delta_text):.4f} (SEM)")
    return {"n": n, "delta_image": delta_image, "delta_text": delta_text}


results = {}
for key, indices in cells.items():
    tag = f"{key[0]} / {key[1]}"
    results[key] = run_cell(indices, tag)

print("\n===== summary table: delta image (mean +/- SEM) =====")
print(f"{'':15s} {'computational':>20s} {'descriptive':>20s}")
for anchor_key in ("anchor", "no_anchor"):
    row = []
    for task_key in ("computational", "descriptive"):
        r = results[(anchor_key, task_key)]
        if r["n"] > 0:
            row.append(f"{statistics.mean(r['delta_image']):+.3f}+/-{sem(r['delta_image']):.3f}")
        else:
            row.append("n/a")
    print(f"{anchor_key:15s} {row[0]:>20s} {row[1]:>20s}")