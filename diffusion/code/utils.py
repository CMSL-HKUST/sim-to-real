import torch
import torch.nn as nn
import os
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


class CustomImageLabelDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_folder) if f.endswith('.npy')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load images from npy files
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = np.load(img_path).astype(np.uint8)  # Load the .npy file (H, W, 3)

        # Convert to PIL Image (if needed for transforms)
        # image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
        image = Image.fromarray(image).convert("RGB")

        # Load labels
        label_path = os.path.join(self.label_folder, f"{os.path.splitext(img_name)[0]}.txt")
        with open(label_path, 'r') as file:
            label = list(map(float, file.read().strip().split()))

        label = torch.tensor(label, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image) 

        return image, label
    

class LabelOnlyDataset(Dataset):
    def __init__(self, label_list):
        """
        label_list: list of torch.Tensor, each is a label vector.
        """
        self.labels = label_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return None, self.labels[idx]

    
class RFFEmbedder(nn.Module):
    '''Gaussian random features for encoding.'''
    def __init__(self, embed_dim, scale=30.):
        super().__init__()

        g = torch.Generator()
        g.manual_seed(0)

        W = torch.randn(embed_dim // 2, generator=g) * scale
        self.W = nn.Parameter(W, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        embed = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return embed


class SinusoidalEmbedder(nn.Module):
    """Sinusoidal embedding for encoding."""
    def __init__(self, embed_dim):
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even for sinusoidal embedding."
        self.embed_dim = embed_dim

    def forward(self, x):
        if x.dim() == 0:
            x = x.unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(1)
        
        device = x.device
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        ).to(device)

        angles = x * freqs
        embed = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embed
    
    
class LinearEmbedder(nn.Module):
    """Linear embedding for encoding."""
    def __init__(self, embed_dim):
        super().__init__()

        g = torch.Generator()
        g.manual_seed(0)

        self.linear1 = nn.Linear(1, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)

        nn.init.normal_(self.linear1.weight, generator=g)
        nn.init.zeros_(self.linear1.bias)

        nn.init.normal_(self.linear2.weight, generator=g)
        nn.init.zeros_(self.linear2.bias)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.dim() == 0:
            x = x.unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(1)

        x = F.relu(self.linear1(x))
        embed = self.linear2(x)
        return embed # shape: (batch_size, embed_dim)
    

def split_dataset(dataset_size, val_ratio=0.2, shuffle=True):
    '''split validation set from dataset.'''
    indices = list(range(dataset_size))
    if shuffle:
        random.shuffle(indices)
    split = int(dataset_size * (1 - val_ratio))

    train_indices = indices[:split]
    val_indices = indices[split:]
    
    return train_indices, val_indices


def mean_std(dataset):
    '''calculate mean and std of dataset.'''
    mean = 0.
    std = 0.
    total_images = len(dataset)

    for img, _ in dataset:
        if not isinstance(img, torch.Tensor):
            raise ValueError("Expected image in dataset to be a PyTorch tensor.")
        if img.dim() != 3:
            raise ValueError("Expected image tensor to have shape [C, H, W].")

        mean += img.mean(dim=(1, 2))
        std += img.std(dim=(1, 2))

    mean /= total_images
    std /= total_images
    return mean.tolist(), std.tolist()


def label_normalizer(value, max_val, min_val):
    '''normalize input parameters.'''
    return (value - min_val) / (max_val - min_val)


def img_denormalizer(batch, mean, std):
    '''denormalize the predicted images.'''
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)  
    denormalized_batch = batch * std + mean

    return denormalized_batch


def freeze_layers(model):
    """
    Freeze the down_blocks, mid_block, and the first two layers of up_blocks in the UNet2DModel.

    Args:
        model (UNet2DModel): The UNet2DModel instance.
    """
    # Freeze all down_blocks
    for down_block in model.down_blocks:
        for param in down_block.parameters():
            param.requires_grad = False

    # Freeze mid_block
    for param in model.mid_block.parameters():
        param.requires_grad = False

    # Freeze the first two up_blocks
    for up_block in model.up_blocks[:2]:  # Only freeze the first two up_blocks
        for param in up_block.parameters():
            param.requires_grad = False


def check_freeze_status(model):
    """
    Print the freeze status (requires_grad) of all layers in the model.

    Args:
        model (torch.nn.Module): The model (e.g., UNet2DModel) to check.
    """
    print("Layer Name | requires_grad")
    print("-" * 30)
    
    for name, param in model.named_parameters():
        print(f"{name:<40} {'Trainable' if param.requires_grad else 'Frozen'}")


# if __name__ == '__main__': 
#    v = 40
#    max = 40
#    min = 40
#    print(label_normalizer(v, max, min))