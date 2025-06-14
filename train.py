import argparse

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.decoders.deeplabv3 import DeepLabV3Plus
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import v2
from data import CSDataset
from globals import *
from network import BoundaryAwareBCE, FCN, FCNParams
from utils import ToBMask, ToMask, TrainNetwork, miscoverage_loss, plot_report
from evaluation import ObstacleBenchmark
from pathlib import Path
parser = argparse.ArgumentParser(description="CLI")

# 2. Aggiungi i parametri
parser.add_argument(
    "--epochs", type=int, default=10, help="Numero di epoche"
)
parser.add_argument(
    "--model", type=str, required=True, help="Nome del modello da usare"
)
parser.add_argument(
    "--log_dir", type=str, default=".", required=True, help="Default log dir"
)
parser.add_argument("--dataset", type=str, required=True, help="Dataset")
# 3. Leggi i parametri dalla linea di comando


parser.add_argument(
    "--benchmark", type=str, required=False, help="Test the model on a real scenario", default=""
)

parser.add_argument(
    "--train", type=str, required=False, help="Train and Test network", default="True"
)

parser.add_argument(
    "--weights", type=str, required=False, help="path weights", default=""
)

parser.add_argument(
    "--classes", type=str, required=False, help="path weights", default=0
)
# 4. Usa i parametri
args = parser.parse_args()
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

w = Path(args.weights)

train = (args.train == "True") 

if args.weights:
    w = Path(w)
    if not w.exists() or not w.is_file():
        raise FileNotFoundError(f"Weights for network {args.model}")

    
KEEP_IDS = []
match args.dataset:


    case "CSF512x192_OH":
        ENCODE = "one-hot"
        DATASET_NAME = "CSF192x512"
        MAP_COLOR2LABEL = CS_COLOR2LABEL
        MAP_LABEL2COLOR = CS_LABEL2COLOR
        FILL = 0
        MASK_FN = ToBMask(MAP_COLOR2LABEL, fill=FILL, add_map=CS_PLUS)
        NUM_CLS = len(MAP_LABEL2COLOR) + 1

    case "CSC512x192_OH":
        ENCODE = "one-hot"
        DATASET_NAME = "CSC192x512"
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
        KEEP_IDS = [8]
        FILL = 0
        MASK_FN = ToBMask(MAP_COLOR2LABEL, fill=FILL, add_map=None)
        NUM_CLS = len(MAP_LABEL2COLOR)
       

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

if args.classes:
    NUM_CLS = int(args.classes)



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
            encoder_name="resnet101",
            encoder_weights="imagenet",
            classes=NUM_CLS,
            encoder_output_stride=8,
            decoder_atrous_rates = (6,12,18),
            activation=None,
            encoder_depth=5,
            decoder_channels=512,
            decoder_aspp_dropout=0.60
        )

     

        optim = torch.optim.Adamax(
            model.parameters(), lr=0.002, weight_decay=0.01 # 0.02 with Adamax.
        )
        

        hyper_parameters = {
            "loss": BoundaryAwareBCE(lambda_w=3.0),
            "optimizer": optim,
        }
        # scheduler = PolynomialLR(optimizer=optim, power=0.9, total_iters=args.epochs)

    case _:
        raise ValueError("Network not found")


if args.weights:
    try:
        weights = torch.load(w, weights_only=True)
        model.load_state_dict(weights, strict=True)

    except RuntimeError as err:
        print("weights not compatible with selected network")
        print(err)
        exit(-1) 

trainer = TrainNetwork(
    hyper_parameters,
    model=model,
    lr_scheduler=scheduler,
    dice_num_classes=NUM_CLS,
    encode=ENCODE,
    keep_index=KEEP_IDS
)
if train:
    m, _, _ = trainer.train(
        train_set=data_train,
        validation_set=data_test,
        log_dir=args.log_dir,
        epochs=args.epochs,
        batch_size=8,
        map_cls_to_color=MAP_LABEL2COLOR,
    )
else:
    m = model


plot_report(
    log_file_train=args.log_dir + "/train.csv",
    log_file_valid=args.log_dir + "/eval.csv",
    out_dir=args.log_dir,
)


match args.benchmark:
    case "Obstacles":
        benchmark = ObstacleBenchmark(network=m, log_dir=args.log_dir)
        ENCODE = "one-hot"
        DATASET_NAME = "LAF192x512"
        MAP_COLOR2LABEL = LAF_COLOR2LABEL
        MAP_LABEL2COLOR = LAF_LABEL2COLOR
        FILL = 0
        MASK_FN = ToBMask(MAP_COLOR2LABEL, fill=FILL, add_map=LAF_PLUS)
        NUM_CLS = len(MAP_LABEL2COLOR) + 1

        data_test = CSDataset(
            f"{DATASET_NAME}/val",
            transform_x=v2.Compose([v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True)]),
            transform_y=v2.Compose([MASK_FN]),

        )
        
        benchmark.run_benchmark(data_test, hyper_parameters["loss"], keep_index=list(LAF_LABEL2COLOR.keys()), format="one-hot", map_cls_to_color=MAP_LABEL2COLOR, device=DEVICE, repeat_for=5)
    case _:
        print("--benchmark not found--")

print("--End Script--")

