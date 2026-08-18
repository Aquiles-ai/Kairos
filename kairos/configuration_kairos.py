from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig


class KairosVisionConfig(PretrainedConfig):
    """Config of the vision tower: MoonViT-3D (extracted from Kimi-K2.6), used
    FROZEN in every training stage. Values match the real checkpoint
    (Aquiles-ai/MoonViT-3D); image preprocessing lives in the processor's
    preprocessor_config.json (Kimi_K25ImageProcessor, patch 14, merge 2).
    """

    model_type = "kimi_k25_vision"

    def __init__(
            self,
            patch_size: int = 14,
            hidden_size: int = 1152,
            intermediate_size: int = 4304,
            num_attention_heads: int = 16,
            num_hidden_layers: int = 27,
            hidden_act: str = "gelu_pytorch_tanh",
            pos_emb_height: int = 64,
            pos_emb_width: int = 64,
            pos_emb_time: int = 4,
            merge_kernel_size: tuple = (2, 2),
            max_position_embeddings: int | None = None,
            rope_parameters: dict | None = None,
            **kwargs):

        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width
        self.pos_emb_time = pos_emb_time
        self.merge_kernel_size = merge_kernel_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_parameters = rope_parameters if rope_parameters is not None else {
            "rope_theta": 10000.0,
            "rope_type": "default",
        }

        #super().__init__(**kwargs)


class KairosVLConfig(PretrainedConfig):
    """Top-level Kairos config: MoonViT-3D (frozen) + LFM2.5-2.6B + the
    Kimi-style 2-layer projector between them.

    - `image_token_id` (124907): the `<image>` placeholder id the projector's
      outputs get scattered into (must match the tokenizer -- the train
      scripts assert it).
    - `projector_output_scale` (None = uncapped): hard L2 cap applied to
      every image embedding at the projector output. Set to ~0.89 (≈ mean
      norm of LFM2.5 text embeddings) in the trained checkpoints: with a
      (mostly) frozen LLM the projector otherwise cheats through magnitudes
      (stage-1 probes reached token norms of 139/1614 vs 0.89 of real text)
      instead of readable directions. The cap costs ~nothing in CE and keeps
      activations bounded for LoRA-stage training.
    """

    model_type = "kairos_vl"
    has_no_defaults_at_init = True
    sub_configs = {"vision_config": KairosVisionConfig, "text_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=124907,
        video_token_id=None,
        projection_hidden_size=1152,
        projection_layer_norm_eps=1e-5,
        projector_output_scale=None,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            vision_config = KairosVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = KairosVisionConfig()
        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config = AutoConfig.for_model(**text_config)
        elif text_config is None:
            # No safe default exists for the base LLM, fail loud instead of on
            # a later AttributeError.
            raise ValueError(
                "text_config is required. Pass an AutoConfig.from_pretrained(...) "
                "or a valid dict, there is no safe default for the base LLM."
            )
        self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.projection_hidden_size = projection_hidden_size
        self.projection_layer_norm_eps = projection_layer_norm_eps
        self.projector_output_scale = projector_output_scale

        tie_word_embeddings = kwargs.pop("tie_word_embeddings", text_config.tie_word_embeddings)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)