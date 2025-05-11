"""
Utility module
"""

import torch
import cv2 as cv
import matplotlib.pyplot as plt
from torch import nn
from torchvision.transforms.v2 import Grayscale
from typing import override


class ToMasck(nn.Module):
    def __init__(self, low: int, high: int) -> None:
        super(ToMasck, self).__init__()
        self.low = low
        self.high = high

    @override
    def forward(self, x):
        """
        Args:
            TODO
        """
        # TODO: Il metodo prende in input un tensore e/o un'immagine RGB e viene convertita in una maschera numerica (un canale)
        #       di valori per le varie classi di segmento
        x = cv.cvtColor(x, code=cv.COLOR_RGB2GRAY)
        x = torch.as_tensor(x, dtype=torch.long)
        x = torch.clamp(x, min=self.low, max=self.high)

        return x


def show_tensor(tensor: torch.Tensor, cmap: str = "gray") -> None:
    """ """
    t = tensor.numpy(force=True).transpose((1, 2, 0))
    plt.imshow(t, cmap=cmap)
    plt.show()
