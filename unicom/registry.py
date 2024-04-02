from typing import Optional

from unicom.vision_transformer import VisionTransformer, _transform
from unicom.vision_transformer_extended import VisionTransformerWithAdditionalVectorInput


def _get_architecture_kwargs(name="ViT-L/14@336px"):
    if name == "ViT-B/32":
        parameters = dict(
            input_size=224, patch_size=32, in_channels=3, dim=768, embedding_size=512,
            depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=True)
    elif name == "ViT-B/16":
        parameters = dict(
            input_size=224, patch_size=16, in_channels=3, dim=768, embedding_size=768,
            depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=True)
    elif name == "ViT-L/14":
        parameters = dict(
            input_size=224, patch_size=14, in_channels=3, dim=1024, embedding_size=768,
            depth=24, num_heads=16, drop_path_rate=0.1, using_checkpoint=True)
    elif name == "ViT-L/14@336px":
        parameters = dict(
            input_size=336, patch_size=14, in_channels=3, dim=1024, embedding_size=768,
            depth=24, num_heads=16, drop_path_rate=0.1, using_checkpoint=True)
    else:
        raise ValueError
    return parameters


def build_model(
    name="ViT-L/14@336px",
    additional_input_vector_length: Optional[int] = None,
):
    if additional_input_vector_length is not None:
        assert isinstance(additional_input_vector_length, int)
        assert additional_input_vector_length > 0

    kwargs = _get_architecture_kwargs(name=name)
    
    if additional_input_vector_length is None:
        model = VisionTransformer(**kwargs)
    else:
        model = VisionTransformerWithAdditionalVectorInput(
            additional_input_vector_length=additional_input_vector_length,
            **kwargs
        )
    return model


def load_model_and_transform(
    name="ViT-L/14@336px",
    additional_input_vector_length: Optional[int] = None,
):
    model = build_model(name, additional_input_vector_length=additional_input_vector_length)

    if name == "ViT-B/32":
        return model, _transform(224)
    elif name == "ViT-B/16":
        return model, _transform(224)
    elif name == "ViT-L/14":
        return model, _transform(224)
    elif name == "ViT-L/14@336px":
        return model, _transform(336)
    else:
        raise ValueError
