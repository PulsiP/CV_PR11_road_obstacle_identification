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
    """
        Initializes parameters for the FCN model.

        Args:
            num_classes (int): Number of output classes for segmentation.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

class FCN(nn.Module):
    """
    Fully Convolutional Network (FCN) for semantic segmentation using an encoder-decoder architecture.
    The encoder is based on the pretrained VGG19 network, and the decoder upsamples the features
    back to the original resolution to produce a segmentation map.
    """
    def __init__(self, params:FCNParams):
        """
        Initializes the FCN model with VGG19 as encoder and a decoder composed of
        transposed convolution layers.

        Args:
            params (FCNParams): Configuration parameters including number of classes.
        """
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
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input image tensor of shape [B, C, H, W].

        Returns:
            Tensor: Segmentation map of shape [B, num_classes, H_out, W_out].
        """
        features = self.encoder(x)
        segmentation_map = self.decoder(features)
        segmentation_map = nn.functional.interpolate(
            segmentation_map, size=(96, 256), mode="bilinear", align_corners=False
        )
        return segmentation_map

class BoundaryAwareBCE(nn.Module):
    def __init__(self, lambda_w=3.0, kernel_size=3):
        """
        Initializes the Boundary-Aware Binary Cross-Entropy (BCE) loss module.

        Args:
            lambda_w (float): Weight applied to the boundary-aware loss term.
            kernel_size (int): Kernel size used for boundary extraction (erosion).
        """
        super(BoundaryAwareBCE, self).__init__()
        self.kernel_size = kernel_size
        self.lambda_w = lambda_w

    
    def _extract_boundary(self, target):
        """
        Extracts boundary regions from the target mask using morphological erosion.

        Args:
            target (Tensor): Ground truth mask tensor of shape [B, C, H, W] or [B, 1, H, W].

        Returns:
            Tensor: Boundary mask highlighting edges in the target.
        """
        pad = self.kernel_size // 2
        inverted = 1 - target.float()
        eroded = F.max_pool2d(inverted, kernel_size=self.kernel_size, stride=1, padding=pad)
        boundary = target - (1 - eroded)
        return boundary.clamp(0, 1)  
    

    def forward(self, pred, target, b_mask):
        """
        Computes the boundary-aware BCE loss.

        Args:
            pred (Tensor): Predicted logits of shape [B, C, H, W].
            target (Tensor): Ground truth tensor of shape [B, C, H, W].
            b_mask (Tensor): Initial boundary mask (optional or dummy input).

        Returns:
            Tensor: Scalar loss value combining standard BCE and boundary-aware BCE.
        """
        pred = torch.sigmoid(pred)  # [B, C, H, W]
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')

        # Make sure b_mask has the same shape
        if b_mask.dim() == 3:
            b_mask = b_mask.unsqueeze(1)
        #b_mask = b_mask.expand_as(bce_loss).float()

        # Standard BCE loss
        bce_mean = bce_loss.mean()

        b_mask = b_mask.expand(-1, pred.shape[1], -1, -1)

        # Boundary-aware weighted BCE
        boundary_mask_sum = b_mask.sum().clamp(min=1.0)
        boundary_loss = ((bce_loss * b_mask).sum() / boundary_mask_sum).mean()

        # Final loss
        return bce_mean  + self.lambda_w * boundary_loss
