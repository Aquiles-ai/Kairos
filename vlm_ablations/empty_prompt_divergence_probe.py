"""Divergence probe with content-empty prompts, on real images.

Measures how dependent the model's decoding is on the image when the
prompt names nothing specific to look for ("What do you see?",
"Describe this.").

Methodology, per sample: greedy-decode with the image attached, then
recompute the logits for the same generated sequence WITHOUT pixel
values. The first step where the no-image logits disagree with the
generated token (or where the KL between the two distributions exceeds
the threshold) is the divergence onset; if no step diverges within K
steps, the decoding never needed the image.

Every prompt is probed on the SAME real dataset images, so the only
variable that changes between runs is the prompt text.
"""

import random
import gc
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

CKPT = "Aquiles-ai/Kairos-Proj-Alig-30k"
DATASET_ID = "Aquiles-ai/Kairos-Multimodal-Reasoning"

# Replicate the same eval split used during training, so the model never
# sees examples it was trained on.
EVAL_SIZE = 512
SEED = 42
N_SAMPLES = 12       # real images on which the empty prompts are probed
K_STEPS = 48
KL_THRESHOLD = 0.05

IM_END_ID = 124900
PAD_ID = 124893

EMPTY_PROMPTS = [
    "What do you see?",
    "Describe this.",
    "Tell me what's here.",
    "What's in this image?",  # control: no specific content, BUT with the keyword
]


def build_user_message(image, question: str):
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }


model = AutoModelForCausalLM.from_pretrained(
    CKPT, trust_remote_code=True, dtype=torch.bfloat16,
).to("cuda")
model.eval()
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)

ds = load_dataset(DATASET_ID)["train"].shuffle(seed=SEED)
eval_ds_raw = ds.select(range(EVAL_SIZE))

random.seed(SEED)
sample_idx = random.sample(range(len(eval_ds_raw)), N_SAMPLES)


@torch.no_grad()
def probe(image, prompt_text):
    message = [build_user_message(image, prompt_text)]
    enc = processor.apply_chat_template(
        message, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    prompt_len = enc["input_ids"].shape[1]

    gen_out = model.generate(
        **enc, max_new_tokens=K_STEPS, do_sample=False,
        eos_token_id=IM_END_ID, pad_token_id=PAD_ID,
        output_scores=True, return_dict_in_generate=True,
    )
    generated_ids = gen_out.sequences[0, prompt_len:]
    k_actual = generated_ids.shape[0]
    if k_actual < 2:
        return None
    logits_with_image = torch.stack(gen_out.scores[:k_actual], dim=0).squeeze(1)

    full_ids = gen_out.sequences[:, :prompt_len + k_actual]
    batch = {
        "input_ids": full_ids,
        "attention_mask": torch.ones_like(full_ids),
        "pixel_values": None,
        "image_grid_thw": None,
    }
    out_no_image = model(**batch)
    logits_no_image = out_no_image.logits[0, prompt_len - 1: prompt_len - 1 + k_actual, :]

    divergence_step = None
    for i in range(k_actual):
        p_with = F.log_softmax(logits_with_image[i].float(), dim=-1)
        p_no = F.log_softmax(logits_no_image[i].float(), dim=-1)
        kl = F.kl_div(p_no, p_with, log_target=True, reduction="sum").item()
        chosen_token = generated_ids[i].item()
        agree = logits_no_image[i].argmax().item() == chosen_token
        if not agree or kl > KL_THRESHOLD:
            divergence_step = i
            break

    decoded_full = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {"divergence_step": divergence_step, "k_actual": k_actual, "decoded_full": decoded_full}


def run_prompt(prompt_text, tag):
    results = []
    for i in sample_idx:
        item = eval_ds_raw[i]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        r = probe(image, prompt_text)
        if r:
            results.append(r)
        # We forced a memory cleanup in each iteration, given 
        # that we only had access to limited hardware and it 
        # was the only way to run these ablation studies
        torch.cuda.empty_cache()
        gc.collect()

    n = len(results)
    never = sum(1 for r in results if r["divergence_step"] is None)
    onsets = sorted(r["divergence_step"] for r in results if r["divergence_step"] is not None)

    print(f"\n===== prompt: {tag!r} (n={n}) =====")
    print(f"  never diverges within K={K_STEPS}: {never}/{n} ({100*never/n:.0f}%)")
    if onsets:
        print(f"  divergence onset: min={onsets[0]} median={onsets[len(onsets)//2]} max={onsets[-1]}")
    print(f"  sample output: {results[0]['decoded_full'][:160]!r}")
    return {"never_pct": 100 * never / n, "n": n, "onsets": onsets}


print(f"measuring on the same {N_SAMPLES} real dataset images for every prompt\n")
summary = {}
for p in EMPTY_PROMPTS:
    summary[p] = run_prompt(p, p)

print("\n===== final comparison =====")
print(f"{'prompt':45s} {'% never diverges':>16}")
for p, r in summary.items():
    print(f"{p:45s} {r['never_pct']:15.0f}%")

print("\ninterpretation:")
print("  if the content-empty prompts ('What do you see?', 'Describe this.') show a HIGH")
print("  % never-diverging (ideally ~100%), the decoding barely needs the image when the")
print("  prompt names nothing specific -- content emptiness is what drives the divergence,")
print("  not the absence of the word 'image': compare 'What do you see?' (no keyword, no")
print("  content) against 'What's in this image?' (WITH the keyword, but still no specific")
print("  content) -- if both give a similar high %, the keyword alone does nothing when")
print("  there is no content to name.")