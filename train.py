import argparse
import torch
from network import *
from torchvision.transforms import v2
from globals import *
from utils import ToMask, plot_report, TrainNetwork
from train import TrainNetwork
import segmentation_models_pytorch as smp

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


match args.model:
    case "FCN":
        from data import CSDataset

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

        model = FCN(params=FCNParams(21))
        hyper_parameters = {
            "loss": smp.losses.DiceLoss(mode='multiclass'),
            "optimizer": torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        }

        trainer = TrainNetwork(hyper_parameters, model)
        m, _, _ = trainer.train(data_train, data_test, args.log_dir, epochs=args.epochs, batch_size=64)
        plot_report(log_file_train=args.log_dir + "/train.csv", log_file_valid=args.log_dir + "/eval.csv", out_dir=args.log_dir)
        print("loop")
    case "Unet":
        from data import CSDataset

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
       
