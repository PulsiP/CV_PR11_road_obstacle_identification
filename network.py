import abc
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from typing import override
from torchvision.models.vgg import VGG19_Weights

class BaseParams(abc.ABC):
    def __init__(self):
        super().__init__()

class FCNParams(BaseParams):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

class FCN(nn.Module):
    """
    Simple network for semantic segmentation based on encoder-decoder architecture
    with VGG16 network-bac
    """
    def __init__(self, params:FCNParams):
        super(FCN, self).__init__()
        backbone = models.vgg19(weights=VGG19_Weights.DEFAULT)
        self.encoder = backbone.features  # Extracts features from VGG16

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Output: [batch_size, 256, H/16, W/16]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Output: [batch_size, 128, H/8, W/8]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Output: [batch_size, 64, H/4, W/4]
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64, params.num_classes, kernel_size=1
            ),  # Final layer for segmentation map, reduces channels to num_classes
        )

    @override
    def forward(self, x):
        features = self.encoder(x)
        segmentation_map = self.decoder(features)
        segmentation_map = nn.functional.interpolate(
            segmentation_map, size=(96, 256), mode="bilinear", align_corners=False
        )
        return segmentation_map

class BoundaryAwareBCE(nn.Module):
    def __init__(self, lambda_w=3.0):
        super(BoundaryAwareBCE, self).__init__()
        self.lambda_w = lambda_w
    
    def forward(self, pred, target, b_mask):

        pred = F.sigmoid(pred)
        BCE_loss = F.binary_cross_entropy(pred, target, reduction='none')
        BCE_loss = BCE_loss.mean()

        mask_ones = torch.sum(b_mask == 1)
        b = (BCE_loss*b_mask).sum()
        boundary_aware = (self.lambda_w/mask_ones)*b
        
        return BCE_loss + boundary_aware


