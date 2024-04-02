from unicom.registry import load_model_and_transform

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch


def get_old_state_dict():
    model, _ = load_model_and_transform(name="ViT-B/16")
    return model.state_dict()


def try_load_old_state_dict_and_infer():
    additional_input_vector_length = 4

    model, transform = load_model_and_transform(name="ViT-B/16", additional_input_vector_length=additional_input_vector_length)

    model.load_state_dict(get_old_state_dict())

    dataset = ImageFolder(root="/home/anton-cherepkov/work/unicom/example_Dataset", transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=2)
    batch = next(iter(dataloader))[0]

    print(type(model))
    print(batch.shape)
    additional_vector = torch.rand(size=(len(batch), additional_input_vector_length))
    features = model(batch, additional_vector)
    print(features.shape)


if __name__ == "__main__":
    try_load_old_state_dict_and_infer()
