from typing import Any

from unicom.vision_transformer import VisionTransformer


class VisionTransformerWithAdditionalVectorInput(VisionTransformer):
    def __init__(self, additional_input_vector_length: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self._additional_input_vector_length = additional_input_vector_length

