from transformers.processing_utils import ProcessorMixin

IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = 124907


class KairosVLProcessor(ProcessorMixin):
    """Processor for Kairos: Kimi_K25ImageProcessor (NaViT-packed patches +
    image_grid_thw) + the LFM2.5 tokenizer + the LFM2.5 chat template.

    This class is the ONLY owner of '<image>' -> N-placeholder expansion
    (`replace_image_token`), so training and inference follow the exact same
    path. N = prod(t,h,w) / merge_size**2 must match the per-image features
    produced by the model's get_image_features (both use merge 2x2 over a
    grid of 14px patches -- max_patches 16384 and 512x512 resize caps live
    in preprocessor_config.json).
    """

    attributes = ["image_processor", "tokenizer"]
    # Explicit class, not "AutoImageProcessor" (deprecated in transformers 5.x)
    image_processor_class = "Kimi_K25ImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)
        self.image_token = IMAGE_TOKEN
        self.image_token_id = IMAGE_TOKEN_ID

    def replace_image_token(self, image_inputs: dict, image_idx: int, **kwargs) -> str:
        merge_length = self.image_processor.merge_size**2
        num_image_tokens = int(image_inputs["image_grid_thw"][image_idx].prod() // merge_length)
        return self.image_token * num_image_tokens
