import argparse

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.decoders.deeplabv3 import DeepLabV3Plus
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import v2
from data import CSDataset
from globals import *
from network import *
from utils import ToBMask, ToMask, TrainNetwork,miscoverage_loss, plot_report, BoundaryAwareBCE

import torch.nn.functional as F

parser = argparse.ArgumentParser(description="CLI")

# 2. Aggiungi i parametri
parser.add_argument("--epochs", type=int, default=10, help="Numero di epoche")
parser.add_argument(
    "--model", type=str, required=True, help="Nome del modello da usare"
)
parser.add_argument(
    "--log_dir", type=str, default=".", required=True, help="Default log dir"
)
parser.add_argument("--dataset", type=str, required=True, help="Dataset")
# 3. Leggi i parametri dalla linea di comando
args = parser.parse_args()

# 4. Usa i parametri
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
match args.dataset:


    case "CSF512x192_OH":
        ENCODE = "one-hot"
        DATASET_NAME = "CSF192x512"
        MAP_COLOR2LABEL = CS_COLOR2LABEL
        MAP_LABEL2COLOR = CS_LABEL2COLOR
        FILL = 0
        MASK_FN = ToBMask(MAP_COLOR2LABEL, fill=FILL, add_map=CS_PLUS)
        NUM_CLS = len(MAP_LABEL2COLOR) + 1


    case "LAF512x192_OH":
        ENCODE = "one-hot"
        DATASET_NAME = "LAF192x512"
        MAP_COLOR2LABEL = LAF_COLOR2LABEL
        MAP_LABEL2COLOR = LAF_LABEL2COLOR

        FILL = 0
        MASK_FN = ToBMask(MAP_COLOR2LABEL, fill=FILL, add_map=LAF_PLUS)
        NUM_CLS = len(MAP_LABEL2COLOR) + 1

    case _:
        raise ValueError("Dataset Not Found")


data_train = CSDataset(
    f"{DATASET_NAME}/train",
    transform_x=v2.Compose([v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True)]),
    transform_y=v2.Compose([MASK_FN]),
)

data_test = CSDataset(
    f"{DATASET_NAME}/val",
    transform_x=v2.Compose([v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True)]),
    transform_y=v2.Compose([MASK_FN]),
)

scheduler = None

match args.model:
    case "FCN":

        model = FCN(params=FCNParams(NUM_CLS))
        hyper_parameters = {
            "loss": smp.losses.DiceLoss(mode="multiclass", ignore_index=255),
            "optimizer": torch.optim.Adam(
                model.parameters(), lr=0.001, weight_decay=1e-4
            ),
        }

        trainer = TrainNetwork(hyper_parameters, model)
    case "DeepLab":
        # Advanced Encoder-decoder newtwork for segmentation tasck
        model = DeepLabV3Plus(
            encoder_name="resnet50",  # backbone
            encoder_weights="imagenet",  # pre-trained
            in_channels=3,  # in-channels
            classes=NUM_CLS,  # num of classes
            decoder_aspp_dropout=0.3,  # decoder dropout
            # activation="sigmoid"            # activation header
        )

        # Disable encoder tunning
        for encoder_layer in model.encoder.parameters():
            encoder_layer.requires_grad = False

        # Set optimizer
        optim = torch.optim.AdamW(model.parameters())

        # Define base Hyper-parameters
        hyper_parameters = {
            "loss": smp.losses.LovaszLoss(mode="multiclass"),
            "optimizer": optim,
        }

        # Add lr scheduler if needs
        scheduler = StepLR(optimizer=optim, step_size=6, gamma=0.1)

    case "Unet++":
        model = smp.UnetPlusPlus(
            encoder_name="resnet152",
            classes=NUM_CLS,
            encoder_depth=3,
            decoder_channels=(256, 128, 64),
            activation="sigmoid",
        )
        optim = torch.optim.Adam(model.parameters())
        hyper_parameters = {
            "loss": smp.losses.DiceLoss(mode="multiclass"),
            "optimizer": optim,
        }
        scheduler = StepLR(optimizer=optim, step_size=10, gamma=0.1)

    case "MyNetwork":
        model = smp.DeepLabV3Plus(
            encoder_name="se_resnet50",
            encoder_weights="imagenet",
            classes=NUM_CLS,
            activation=None,
        )

        optim = torch.optim.SGD(
            model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9
        )
        

        hyper_parameters = {
            "loss": BoundaryAwareBCE(lambda_w=3.0),
            "optimizer": optim,
        }
        # scheduler = PolynomialLR(optimizer=optim, power=0.9, total_iters=args.epochs)

    case _:
        raise ValueError("Network not found")


trainer = TrainNetwork(
    hyper_parameters,
    model,
    lr_scheduler=scheduler,
    dice_num_classes=NUM_CLS,
    encode=ENCODE,
)
m, _, _ = trainer.train(
    data_train,
    data_test,
    args.log_dir,
    0.1,
    miscoverage_loss,
    0,
    epochs=args.epochs,
    batch_size=16,
    map_cls_to_color=MAP_LABEL2COLOR,
)
plot_report(
    log_file_train=args.log_dir + "/train.csv",
    log_file_valid=args.log_dir + "/eval.csv",
    out_dir=args.log_dir,
)
