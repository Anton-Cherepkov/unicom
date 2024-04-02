from typing import Any, Mapping
import torch
import torch.nn as nn
from unicom.vision_transformer import VisionTransformer


class VisionTransformerWithAdditionalVectorInput(VisionTransformer):
    def __init__(self, additional_input_vector_length: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        assert isinstance(additional_input_vector_length, int)
        assert additional_input_vector_length >= 0
        self._additional_input_vector_length = additional_input_vector_length

        # self.feature = nn.Sequential(
        #     nn.Linear(dim * self.patch_embed.num_patches, dim, False),
        #     nn.BatchNorm1d(dim, eps=2e-5),
        #     nn.Linear(dim, embedding_size, False),
        #     nn.BatchNorm1d(embedding_size, eps=2e-5))

        dim = kwargs["dim"]
        embedding_size = kwargs["embedding_size"]

        self.feature = nn.Sequential(
            nn.Linear(dim * self.patch_embed.num_patches + additional_input_vector_length, dim, False),
            nn.BatchNorm1d(dim, eps=2e-5),
            nn.Linear(dim, embedding_size, False),
            nn.BatchNorm1d(embedding_size, eps=2e-5)
        )

    def forward(self, x, additional_vector: torch.FloatTensor):
        num_images = x.shape[0]

        if additional_vector.ndim != 2 or additional_vector.shape[0] != num_images or additional_vector.shape[1] != self._additional_input_vector_length:
            raise ValueError(f"Get additional_vector of shape {additional_vector.shape}, expected ({num_images}, {self._additional_input_vector_length})")

        x = self.forward_features(x)
        x = torch.cat((x, additional_vector), dim=1)
        x = self.feature(x)
        return x

    @staticmethod
    def _adapt_linear_weights_for_new_in_features(
        weight: torch.Tensor,
        new_in_features: int,
    ):
        assert weight.ndim == 2

        old_in_features = weight.shape[1]
        
        if new_in_features < old_in_features:
            raise NotImplementedError
        
        if new_in_features == old_in_features:
            return weight
        
        # new_in_features > old_in_features
        out_features = weight.shape[0]
        in_features_difference = new_in_features - old_in_features
        assert in_features_difference > 0
        new_random_chunk = torch.randn(size=(out_features, in_features_difference))

        assert weight.shape[0] == new_random_chunk.shape[0]
        old_shape = weight.shape
        weight = torch.cat((weight, new_random_chunk), dim=1)

        print(
            f"[During load_state_dict] extending Linear layer's weights by appending a random chunk of size {new_random_chunk.shape}\n"
            f"[During load_state_dict] Thus, weight matrix change is the following: {old_shape} -> {weight.shape}"
        )

        return weight

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        state_dict = state_dict.copy()

        state_dict["feature.0.weight"] = self._adapt_linear_weights_for_new_in_features(
            weight=state_dict["feature.0.weight"],
            new_in_features=self.feature[0].in_features,
        )

        return super().load_state_dict(state_dict, strict, assign)
