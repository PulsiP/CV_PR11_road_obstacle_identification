"""
Utility module
"""

import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from typing import override
from torch import nn

class ToMask(nn.Module):
    """
    Class use to preprocess RGB-Mask for segmentation tasck
    """
    def __init__(self, map_, size):
        super().__init__()

        self.map_ = map_
        self.size = size

    def forward(self, x):
        """
        Converte immagine RGB [H, W, 3] in maschera [1, H, W] con classi numeriche.
        Pixel con colori non riconosciuti vengono assegnati a classe 0 (fallback).
        """

        x = cv.resize(x, dsize=self.size)
        x = image2Map(self.map_, x)
        x = torch.as_tensor(x, dtype=torch.long).unsqueeze(0)

        return x
    
def show_tensor(tensor: torch.Tensor, cmap: str = "gray") -> None:
    """ get Image-Tensor ans show it"""
    t = tensor.numpy(force=True).transpose((1, 2, 0))
    plt.imshow(t, cmap=cmap)
    plt.show()


def image2Map(map_, x, fill=20):
    """
    Convert RGB Image in Quantizited Image using Quantization Table `map_`
    Unquantizited values will map with 20 (unknow by default)
    """
    h, w, _ = x.shape
    mask = np.full((h, w), fill_value=fill, dtype=np.uint8)

    for color, class_id in map_.items():
        #print(f"  {color} → {class_id}")
        match = np.all(x == color, axis=-1)
        mask[match] = class_id

    return mask


def map2Image(map_, x) -> np.ndarray:
    """ Map quantizited Image in RGB Image """
    mask = np.full((x.shape[0], x.shape[1], 3), fill_value=0, dtype=np.uint8)
    # print("Mappatura colore → class_id usata:")

    for class_id, color in map_.items():
        #print(f" {class_id} -> {color}")
        # print(f"  {color} → {class_id}")

        match = x == class_id

        mask[match] = color

    return mask
