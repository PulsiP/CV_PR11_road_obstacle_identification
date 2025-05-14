import argparse
from data import CSDataset
from segmentation_models_pytorch.decoders.unetplusplus import UnetPlusPlus
import segmentation_models_pytorch as smp
import torch
from torchvision.transforms import v2
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torch.optim.lr_scheduler import LinearLR, StepLR

from globals import *
from network import *
from utils import ToMask, TrainNetwork, plot_report

# 1. Crea il parser
parser = argparse.ArgumentParser(description="Esempio di script con parametri CLI")

# 2. Aggiungi i parametri
parser.add_argument('--epochs', type=int, default=10, help='Numero di epoche')
parser.add_argument('--model', type=str, required=True, help='Nome del modello da usare')
parser.add_argument("--log_dir", type=str, default=".", required=True, help='Default log dir')
parser.add_argument("--dataset", type=str, required=True, help="Dataset")
# 3. Leggi i parametri dalla linea di comando
args = parser.parse_args()

# 4. Usa i parametri


data_train = CSDataset(
f"{args.dataset}/train",
transform_x=v2.Compose(
    [v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True), v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
),
transform_y=v2.Compose([ToMask(CS_COLOR2LABEL)]),
)

data_test = CSDataset(
    f"{args.dataset}/val",
    transform_x=v2.Compose(
        [v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True), v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ),
    transform_y=v2.Compose([ToMask(CS_COLOR2LABEL)]),
)

match args.model:
    case "FCN":
        

        model = FCN(params=FCNParams(21))
        hyper_parameters = {
            "loss": smp.losses.DiceLoss(mode='multiclass'),
            "optimizer": torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        }

        trainer = TrainNetwork(hyper_parameters, model)
        m, _, _ = trainer.train(data_train, data_test, args.log_dir, epochs=args.epochs, batch_size=64)
        plot_report(log_file_train=args.log_dir + "/train.csv", log_file_valid=args.log_dir + "/eval.csv", out_dir=args.log_dir)

    case "Unet":
        from data import CSDataset



        model = smp.Unet(
            encoder_name="resnet50",       # backbone
            encoder_weights="imagenet",    # pre-trained
            in_channels=3,
            classes=21,                     # numero di classi
            activation="sigmoid",          # sigmoid → multilabel
        )
        hyper_parameters = {
            "loss": smp.losses.DiceLoss(mode='multiclass'),
            "optimizer": torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        }

        trainer = TrainNetwork(hyper_parameters, model)
        m, _, _ = trainer.train(data_train, data_test, args.log_dir, epochs=args.epochs, batch_size=64)
        plot_report(log_file_train=args.log_dir + "/train.csv", log_file_valid=args.log_dir + "/eval.csv", out_dir=args.log_dir)

    case "Unet++":
        
        model = smp.UnetPlusPlus(encoder_name="resnet50", classes=21, encoder_depth=3, decoder_channels=(256, 128, 64))
        optim = torch.optim.Adam(model.parameters())
        hyper_parameters = {
            "loss": smp.losses.DiceLoss(mode="multiclass"),
            "optimizer": optim
        }

        scheduler = StepLR(optimizer=optim, step_size=10, gamma=0.1)


        trainer = TrainNetwork(hyper_parameters, model, lr_scheduler=scheduler)
        m, _, _ = trainer.train(data_train, data_test, args.log_dir, epochs=args.epochs, batch_size=8)
        plot_report(log_file_train=args.log_dir + "/train.csv", log_file_valid=args.log_dir + "/eval.csv", out_dir=args.log_dir)