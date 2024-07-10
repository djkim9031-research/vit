import math
import torch
from torch import nn


class GELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space
    """

    def __init__(self, config):
        super().__init__()
        self.width = config["image_width"]
        self.height = config["image_height"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.num_patches = (self.width // self.patch_size) * (self.height // self.patch_size)
        # output shape = (hidden_size, height//patch_size, width//patch_size)
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, 
                                    kernel_size = self.patch_size, stride=self.patch_size)
        
    def forward(self, x):
        # B, C, H, W => B, hidden, H/num_patches, W/num_patches
        x = self.projection(x)
        # => B, num_patches, hidden
        x = x.flatten(2).transpose(1, 2)
        return x

