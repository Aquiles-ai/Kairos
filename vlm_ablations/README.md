# VLM Ablations: Conclusions

This README summarizes the conclusions from the ablation scripts in this folder: `ablation_study_2x2.py` and `empty_prompt_divergence_probe.py`.

In internal tests we noticed that in some generations the model said it did not see an image, but a large blank white space. The first suspect was that the model had learned a trigger with something like `"What's in this image?"` (with keyword, no content), but it gives the same 100% as `"Describe this."` (no keyword, no content). That was a decisive comparison we were looking for, and it unambiguously confirms that the keyword alone does nothing when there is no content to name.

In the same internal tests we noticed a data point that does not fit cleanly: `"Tell me what's here."` only gives 25% of samples that never diverge, well below even the "no anchor" bucket with real prompts (67%). The first suspect was that this prompt, by naming nothing to look for, would not activate the image either, but when reviewing the samples we saw something even stranger: the three that did diverge did so at exactly the same step (38, min=median=max). That is not what you would expect from a genuine divergence depending on each image's content; if the image really mattered there, the divergence point should vary depending on what is in each image, as happened with `"What do you see?"` (min=9, max=24, varied). An identical value in all three suggests something more structural: probably a logit tie or a template limit that resolves the same regardless of the image, not real grounding activating.

### Root cause

1. The projector encodes fine detail correctly (masking test, localized negative cosine).
2. The image pipeline is clean (two independent verifications).
3. The LoRA did learn to use the image, but conditioned on the prompt naming something specific to look for, not on the presence of words like "image". Confirmed in three independent experiments that corrected each other (real dataset, 2x2 ablation, divergence probe, and this final direct control).
4. Without named content in the prompt, the model falls deterministically to the LFM2.5-2.6B prior, indistinguishable from whether the image is there or not.
5. Outside the dataset's visual distribution (free photography vs curated benchmark), the model fails much harder even with prompts that do name content: the LoRA r=16 capacity with 30k samples is insufficient to generalize visual composition, only for seen patterns.

### How to fix this

Two different interventions:

- For point 4: add to the dataset examples with generic open prompts (`"What do you see?"`, `"Describe this."`) with correct supervision (a real answer describing the image, not the "blank canvas" pattern the model never saw counteracted). Without this kind of example in training, no volume of the current questions will fix it, because the whole dataset implicitly assumes there is something to look for.
- For point 5: raise `MAX_SAMPLES`/epochs as already planned, monitoring that `eval_loss` flattens this time.