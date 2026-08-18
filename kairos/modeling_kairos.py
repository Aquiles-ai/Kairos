"""Kairos multimodal model: MoonViT-3D vision tower + Kimi-style 2-layer
projector + LiquidAI LFM2.5-2.6B as the LLM.

Layout follows the Kimi_K25 modeling code from transformers main (VisionModel
-> projector over its pooled/merged output -> masked_scatter into `<image>`
placeholder positions of the LLM input embeddings). The vision tower is
frozen in every stage; what trains depends on the stage (see the scripts).

Empirical notes that shape this code (runs on LLaVA-CC3M-80k):
- vs a frozen LLM, a random-init projector emits embeddings ~65x the norm of
  real text embeddings; training stalls at ~ln(vocab). Zero-init of out_proj
  avoids that phase.
- `projector_output_scale` (hard L2 cap, see cofigs.py) blocks the
  magnitude-cheat channel; CE stays the same but inputs to the LLM stay at
  text scale.
- Trainable params must be fp32: bf16 weights + lr ~1e-4 round most Adam
  updates away.
- The loss is the model's own `ForCausalLMLoss` (shifted); combined with
  `accepts_loss_kwargs = False` the HF Trainer applies its standard
  grad-accum normalization.
"""

#from transformers.activations import ACT2FN
import torch
import torch.nn as nn
import torch.nn.functional as F
from .configuration_kairos import KairosVLConfig
from transformers import AutoModel, PreTrainedModel, GenerationMixin, Kimi_K25VisionModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass

class KairosMultimodalProjection(nn.Module):
    def __init__(self, config: KairosVLConfig):
        super().__init__()
        merge_factor = config.vision_config.merge_kernel_size[0] * config.vision_config.merge_kernel_size[1]
        self.hidden_size = config.vision_config.hidden_size * merge_factor
        self.pre_norm = nn.LayerNorm(config.projection_hidden_size, eps=config.projection_layer_norm_eps)
        self.in_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.act = nn.GELU()
        #self.act = ACT2FN[config.vision_config.projector_hidden_act]
        self.out_proj = nn.Linear(self.hidden_size, config.text_config.hidden_size)
        # Hard L2 output cap (None = uncapped): with a frozen LLM the
        # projector's cheapest cheat is cranking magnitudes up (uncapped runs
        # reached token norms of 139 avg / 1614 max vs ~0.9 of real text
        # embeddings) instead of emitting direction-aligned features. The cap
        # costs ~nothing in CE and keeps LoRA-stage inputs stable.
        # NOTE: read at __init__ — train scripts set BOTH this attr and
        # config.projector_output_scale right after from_pretrained, because
        # assigning the config attr alone afterwards is a silent no-op.
        self.output_scale = getattr(config, "projector_output_scale", None)

    def forward(self, hidden_states: torch.Tensor):
        batch_size = hidden_states.shape[0]
        hidden_states = self.pre_norm(hidden_states).view(batch_size, -1, self.hidden_size)
        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        if self.output_scale is not None:
            # eps guards the zero-init row (0/0) — normalize(0) stays 0
            hidden_states = F.normalize(hidden_states, dim=-1, eps=1e-6) * self.output_scale
        return hidden_states

@dataclass
class KairosVLModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Cache | None = None
    hidden_states: tuple | None = None
    attentions: tuple | None = None
    image_hidden_states: torch.FloatTensor | None = None


@dataclass
class KairosCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple | None = None
    attentions: tuple | None = None
    image_hidden_states: torch.FloatTensor | None = None


class KairosPreTrainedModel(PreTrainedModel):
    config: KairosVLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True


class KairosModel(KairosPreTrainedModel):
    def __init__(self, config: KairosVLConfig, vision_model_path=None, text_model_path=None):
        super().__init__(config)

        if vision_model_path is not None:
            self.vision_tower = Kimi_K25VisionModel.from_pretrained(vision_model_path)
        else:
            self.vision_tower = Kimi_K25VisionModel(config.vision_config)

        if text_model_path is not None:
            self.language_model = AutoModel.from_pretrained(text_model_path)
        else:
            self.language_model = AutoModel.from_config(config.text_config)

        self.mm_projector = KairosMultimodalProjection(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def freeze_for_pretraining(self):
        """Freeze vision tower + LLM, keep the projector trainable.

        Entry point for stage-1 (projector-only alignment) and the base
        freeze applied BEFORE wrapping the LLM with LoRA in train_sft.py
        (adapters are trainable by default, so the projector + the LoRA
        deltas end up as the only trainable tensors).
        """
        for p in self.vision_tower.parameters():
            p.requires_grad_(False)
        for p in self.language_model.parameters():
            p.requires_grad_(False)
        for p in self.mm_projector.parameters():
            p.requires_grad_(True)

    def get_image_features(self, pixel_values, image_grid_thw):
        # Cast here and not only in forward(): direct callers of this helper
        # (diagnostics, get_image_features on the LM class) would otherwise
        # feed fp32 pixels into a bf16 patch_embed conv.
        pixel_values = pixel_values.to(self.vision_tower.patch_embed.proj.weight.dtype)
        vision_outputs = self.vision_tower(pixel_values, grid_thw=image_grid_thw)
        # The projector may live in fp32 while the vision tower runs in bf16
        # (fp32 trainable params are required for meaningful Adam updates).
        image_embeds = self.mm_projector(
            vision_outputs.pooler_output.to(self.mm_projector.pre_norm.weight.dtype)
        ).squeeze(1)
        mk = self.config.vision_config.merge_kernel_size
        merge_factor = mk[0] * mk[1]
        split_sizes = (image_grid_thw.prod(-1) // merge_factor).tolist()
        return torch.split(image_embeds, split_sizes)

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
        special_image_mask = input_ids == self.config.image_token_id
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        if n_image_tokens * inputs_embeds.shape[-1] != image_features.numel():
            raise ValueError(
                f"Image tokens ({n_image_tokens}) and features ({image_features.shape[0]}) do not match"
            )
        return special_image_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        pixel_values=None,
        image_grid_thw=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            multimodal_mask = input_ids == self.config.image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[multimodal_mask] = 0
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        image_hidden_states = None
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=inputs_embeds.device, dtype=self.dtype)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(pixel_values.device)
            image_embeds_tuple = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds_tuple, dim=0).to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            image_hidden_states = image_embeds

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return  KairosVLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )


class KairosForConditionalGeneration(KairosPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    # Same as the Kimi_K25 reference: the (shifted) causal-LM loss is computed
    # from `labels` by the model itself; keep the Trainer's standard grad-accum
    # loss normalization on. Without this flag the forward's **kwargs make the
    # Trainer assume we consume `num_items_in_batch`, silently scaling the
    # effective LR by gradient_accumulation_steps.
    accepts_loss_kwargs = False

    def __init__(self, config: KairosVLConfig, vision_model_path=None, text_model_path=None):
        super().__init__(config)
        self.model = KairosModel(config, vision_model_path=vision_model_path, text_model_path=text_model_path)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def prepare_inputs_for_generation(
        self, input_ids, *args, past_key_values=None, pixel_values=None, image_grid_thw=None, **kwargs
    ):
        # Image features are only needed during the prefill step. Once the KV cache exists,
        # `input_ids` gets sliced to the newly generated tokens and no longer contains the
        # image placeholder, so the vision kwargs must be dropped from all decode steps.
        if past_key_values is not None:
            pixel_values = None
            image_grid_thw = None
        return super().prepare_inputs_for_generation(
            input_ids,
            *args,
            past_key_values=past_key_values,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **kwargs,
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple:
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        pixel_values=None,
        image_grid_thw=None,
        logits_to_keep=0,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return KairosCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )