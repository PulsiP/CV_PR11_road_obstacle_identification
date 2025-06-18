import argparse

import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.decoders.deeplabv3 import DeepLabV3Plus
from torch.optim.lr_scheduler import StepLR, PolynomialLR
from torchvision.transforms import v2
from data import CSDataset
from globals import *
from network import BoundaryAwareBCE, FCN, FCNParams
from utils import ToBMask, ToMask, TrainNetwork, miscoverage_loss, plot_report
from evaluation import ObstacleBenchmark
from pathlib import Path
parser = argparse.ArgumentParser(description="CLI")

########################
### Parameter parser ###
########################

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

#####################
## Select Dataset ##
####################
match args.dataset:


    case "CSF512x192_OH":
        ENCODE = "one-hot"
        DATASET_NAME = "CSF192x512"
        MAP_COLOR2LABEL = CS_COLOR2LABEL
        MAP_LABEL2COLOR = CS_LABEL2COLOR
        FILL = 0
        MASK_FN = ToBMask(MAP_COLOR2LABEL, fill=FILL, add_map=CS_PLUS)
        NUM_CLS = len(MAP_LABEL2COLOR) + 1
    
    case "CSF720x288_OH":
        ENCODE = "one-hot"
        DATASET_NAME = "CSF720x288"
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
        #KEEP_IDS = [8]
        FILL = 0
        MASK_FN = ToBMask(MAP_COLOR2LABEL, fill=FILL, add_map=None)
        NUM_CLS = len(MAP_LABEL2COLOR)
    
    case "LAF720x288_OH":
        ENCODE = "one-hot"
        DATASET_NAME = "LAF720x288"
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

##################
## Select model ##
##################

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

    case "Unet":
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=NUM_CLS,
            activation=None,
            encoder_depth=5,
           
        )

        optim = torch.optim.SGD(
            model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9 # 0.02 with Adamax.
        )

        hyper_parameters = {
            "loss": BoundaryAwareBCE(lambda_w=3.0),
            "optimizer": optim,
        }
        scheduler = PolynomialLR(optimizer=optim, power=0.9, total_iters=args.epochs)

    case "DeepLab": 
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=NUM_CLS,
            encoder_output_stride=16,
            decoder_atrous_rates = (3,6,12),
            activation=None,
            encoder_depth=5,
            decoder_channels=64,
            decoder_aspp_dropout=0.3 # 20.0
        )

     

        optim = torch.optim.SGD(
            model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9 # 0.02 with Adamax.
        )
        

        hyper_parameters = {
            "loss": BoundaryAwareBCE(lambda_w=3.0),
            "optimizer": optim,
        }
        scheduler = PolynomialLR(optimizer=optim, power=0.9, total_iters=args.epochs)

    case _:
        raise ValueError("Network not found")


if args.weights:
    try:
        weights = torch.load(w, weights_only=True)
        model.load_state_dict(weights, strict=False)

    except RuntimeError as err:
        print("weights not compatible with selected network")
        print(err)
        exit(-1) 

####################
## Train and Test ##
####################

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
        batch_size=16,
        massimize=False,
        objective="loss",
        map_cls_to_color=MAP_LABEL2COLOR,
    )

    plot_report(
    log_file_train=args.log_dir + "/train.csv",
    log_file_valid=args.log_dir + "/eval.csv",
    out_dir=args.log_dir,
)
else:
    m = model


###############
## Benchmark ##
###############

match args.benchmark:
    case "Obstacles":
        benchmark = ObstacleBenchmark(network=m, log_dir=args.log_dir)
        ENCODE = "one-hot"
        DATASET_NAME = "LAF720x288"
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
        
        benchmark.run_benchmark(data_test, hyper_parameters["loss"], keep_index=[8], format="one-hot", map_cls_to_color=MAP_LABEL2COLOR, device=DEVICE, repeat_for=3)
    case _:
        print("--benchmark not found--")

print("--End Script--")

