import argparse

import segmentation_models_pytorch as smp
import torch
from data import CSDataset
from segmentation_models_pytorch.decoders.deeplabv3 import DeepLabV3Plus
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import v2
from globals import *
from network import *
from utils import ToMask, TrainNetwork, plot_report

# 1. Crea il parser
parser = argparse.ArgumentParser(description="CLI")

# 2. Aggiungi i parametri
parser.add_argument('--epochs', type=int, default=10, help='Numero di epoche')
parser.add_argument('--model', type=str, required=True, help='Nome del modello da usare')
parser.add_argument("--log_dir", type=str, default=".", required=True, help='Default log dir')
parser.add_argument("--dataset", type=str, required=True, help="Dataset")
# 3. Leggi i parametri dalla linea di comando
args = parser.parse_args()

# 4. Usa i parametri

match args.dataset:

    case "Fine256x96":
        DATASET_NAME = "Fine256x96"
        MAP_COLOR2LABEL = CS_COLOR2LABEL
        MAP_LABEL2COLOR = CS_LABEL2COLOR
        NUM_CLS = 21
        FILL = 20
    case "Fine512x192":
        DATASET_NAME = "Fine512x192"
        MAP_COLOR2LABEL = CS_COLOR2LABEL
        MAP_LABEL2COLOR = CS_LABEL2COLOR
        NUM_CLS = 21
        FILL = 20

    case "Obstacle512x192":
        DATASET_NAME = "Obstacle512x192"
        MAP_COLOR2LABEL = OBS_COLOR2LABEL
        MAP_LABEL2COLOR = OBS_LABEL2COLOR
        NUM_CLS = 3
        FILL = 0
    case "Obstacle":
        DATASET_NAME = "ObstacleF"
        MAP_COLOR2LABEL = OBS_COLOR2LABEL
        MAP_LABEL2COLOR = OBS_LABEL2COLOR
        NUM_CLS = 3
        FILL = 0
    case _:
        raise ValueError("Dataset Not Found")
    
    




data_train = CSDataset(
f"{DATASET_NAME}/train",
transform_x=v2.Compose(
    [v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True), v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
),
transform_y=v2.Compose([ToMask(MAP_COLOR2LABEL, fill=FILL)]),
)

data_test = CSDataset(
    f"{DATASET_NAME}/val",
    transform_x=v2.Compose(
        [v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True), v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    ),
    transform_y=v2.Compose([ToMask(MAP_COLOR2LABEL, fill=FILL)]),
)

scheduler = None 

match args.model:
    case "FCN":
        
        model = FCN(params=FCNParams(NUM_CLS))
        hyper_parameters = {
            "loss": smp.losses.DiceLoss(mode='multiclass'),
            "optimizer": torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        }

        trainer = TrainNetwork(hyper_parameters, model)
    case "DeepLab":
        
        model = DeepLabV3Plus(
            encoder_name="resnet101",       # backbone
            encoder_weights="imagenet",    # pre-trained
            in_channels=3,
            classes=NUM_CLS,                     # numero di classi
            decoder_aspp_dropout=0.3,
            activation="sigmoid"
        )

        # Disable encoder tunning
        for encoder_layer in model.encoder.parameters():
            encoder_layer.requires_grad = False
        optim = torch.optim.Adamax(model.parameters())
        hyper_parameters = {
            "loss": smp.losses.LovaszLoss(mode='multiclass'),
            "optimizer": optim
        }
        

        scheduler = StepLR(optimizer=optim, step_size=6, gamma=0.1)
    case "Unet++":
        
        model = smp.UnetPlusPlus(encoder_name="resnet50", classes=NUM_CLS, encoder_depth=3, decoder_channels=(256, 128, 64), activation='sigmoid')
        optim = torch.optim.Adam(model.parameters())
        hyper_parameters = {
            "loss": smp.losses.DiceLoss(mode="multiclass"),
            "optimizer": optim
        }
        scheduler = StepLR(optimizer=optim, step_size=10, gamma=0.1)
    
    case "MobileNet":
        
        model = deeplabv3_mobilenet_v3_large(num_classes=NUM_CLS)
        optim = torch.optim.AdamW(model.parameters())
        hyper_parameters = {
            "loss": smp.losses.DiceLoss(mode="multiclass", classes=[0,1,2]),
            "optimizer": optim
        }
        scheduler = StepLR(optimizer=optim, step_size=5, gamma=0.1)
    
    case _:
        raise ValueError("Network not found")



trainer = TrainNetwork(hyper_parameters, model, lr_scheduler=scheduler, dice_num_classes=NUM_CLS)
m, _, _ = trainer.train(data_train, data_test, args.log_dir, epochs=args.epochs, batch_size=16, map_cls_to_color=MAP_LABEL2COLOR)
plot_report(log_file_train=args.log_dir + "/train.csv", log_file_valid=args.log_dir + "/eval.csv", out_dir=args.log_dir)