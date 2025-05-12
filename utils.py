"""
Utility module
"""

import torch
import cv2 as cv
import matplotlib.pyplot as plt
from torch import nn
from torchvision.transforms.v2 import Grayscale
from typing import override


import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from globals import CITYSCAPES_RGB


class ToMask(nn.Module):
    def __init__(self):
        super().__init__()
        unique_colors = list(CITYSCAPES_RGB.keys())
        self.color2id = {color: idx for idx, color in enumerate(unique_colors)}
        self.valid_classes = set(CITYSCAPES_RGB.values())

    def forward(self, x: np.ndarray) -> torch.Tensor:
        """
        Converte immagine RGB [H, W, 3] in maschera [1, H, W] con classi numeriche.
        Pixel con colori non riconosciuti vengono assegnati a classe 0 (fallback).
        """
        h, w, _ = x.shape
        mask = np.full((h, w), fill_value=0, dtype=np.uint8)
        print("Mappatura colore → class_id usata:")

        for color, class_id in self.color2id.items():
            print(f"  {color} → {class_id}")
            match = np.all(x == color, axis=-1)
            mask[match] = class_id

        return torch.from_numpy(mask).long().unsqueeze(0)



def show_tensor(tensor: torch.Tensor, cmap: str = "gray") -> None:
    """ """
    t = tensor.numpy(force=True).transpose((1, 2, 0))
    plt.imshow(t, cmap=cmap)
    plt.show()
