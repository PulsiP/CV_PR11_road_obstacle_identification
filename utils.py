"""
Utility module
"""
from torch import nn
from typing import override
class ImgToClass(nn.Module):
    def __init__(self) -> None:
        super(ImgToClass, self).__init__()

    @override
    def forward(self, x):
        """
        Args:
            TODO
        """
        # TODO: Il metodo prende in input un tensore e/o un'immagine RGB e viene convertita in una maschera numerica (un canale)
        #       di valori per le varie classi di segmento
        pass 
