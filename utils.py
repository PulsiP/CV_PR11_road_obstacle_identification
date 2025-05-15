"""
Utility module
"""

import os
from collections import OrderedDict
from pathlib import Path
from random import randint
from typing import Any, override

import cv2 as cv
import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchmetrics.segmentation import DiceScore, MeanIoU
from tqdm.auto import tqdm


class ToMask(nn.Module):
    """
    Class use to preprocess RGB-Mask for segmentation tasck
    """
    def __init__(self, map_, size = None, fill = 0):
        super().__init__()

        self.map_ = map_
        self.size = size
        self.fill = fill

    @override
    def forward(self, x):
        """
        Converte immagine RGB [H, W, 3] in maschera [1, H, W] con classi numeriche.
        Pixel con colori non riconosciuti vengono assegnati a classe 0 (fallback).
        """
        if self.size:
            x = cv.resize(x, dsize=self.size)

        x = image2Map(self.map_, x, fill=self.fill)
        x = torch.as_tensor(x, dtype=torch.long).unsqueeze(0)
        
        
        return x

class TrainNetwork:
    """
    Classe specializzata nell'effettuare l'attività di traning per la rete proposta. Tutti i modelli conformi alle specifiche di rete
    in questo progetto possono utilizzare la classe per effettuare il traning e valutazione contemporaneo del modello secondo i criteri
    configurati
    """

    def __init__(self, hp: dict[str, Any], model: nn.Module, dice_num_classes:int=21, lr_scheduler = None) -> None:
        """
        Inizializza il trainer
        """
        self._model: nn.Module = model
        self._loss: nn.Module = hp["loss"]
        self._optimizer: Optimizer = hp["optimizer"]
        self.dice = DiceScore(num_classes=dice_num_classes, average='micro', input_format="index")
        self.iou = MeanIoU(num_classes=dice_num_classes, input_format="index")
        self._lr_scheduler = lr_scheduler

    def __train(
        self, dataloader_train, model, epoch, loss_fn, device, log_train, train_size
    ):
        ##################
        # Training Phase #
        ##################

        epoch_loss = 0
        loss = 0
        batch = 0

        bar = tqdm(
            desc="training", total=train_size  # type: ignore
        )  # uses len(dataset) instead of dataset.size

        bar.set_postfix({"batch": 0, "loss": 0, "diceScore": 0.0,"IoU": 0.0, 'lr': self._optimizer.param_groups[0]['lr']})
        self.dice = self.dice.to(device)
        self.iou  = self.iou.to(device)
        self.dice.reset()
        self.iou.reset()
        dice_value = 0
        iou_value  = 0
        for x, y, _ in dataloader_train:
            bar.set_description(f"training epoch: {epoch}", refresh=True)
        
            x = x.to(device)
            y = y.squeeze(1).to(device)

            batch += 1
            batch_len = len(y)

            self._optimizer.zero_grad()
            
            y_pred = model(x)
            
            if isinstance(y_pred, OrderedDict):
                y_pred = y_pred["out"] 
                
            y_classes = torch.argmax(y_pred, dim=1)
            dice_value += self.dice(y_classes, y)
            iou_value += self.iou(y_classes, y)
            loss = loss_fn(y_pred, y)
            loss_v = loss.item()
            epoch_loss += loss_v
            loss.backward()

            self._optimizer.step()

            bar.set_postfix(
                {
                    "batch": batch,
                    "loss": epoch_loss / batch,
                    "diceScore": dice_value.numpy(force=True) / batch,
                    "IoU": iou_value.numpy(force=True) / batch,
                    'lr': self._optimizer.param_groups[0]['lr']
                }
            )
            bar.update(batch_len)

        bar.close()
        log_train["epoch"].append(epoch)
        log_train["loss"].append(epoch_loss / batch)
        log_train["diceScore"].append(self.dice.compute().numpy(force=True))
        log_train["IoU"].append(self.iou.compute().numpy(force=True))
        log_train["lr"].append(self._optimizer.param_groups[0]['lr'])
        return log_train

    def __eval(
        self, dataloader_val, model, epoch, loss_fn, device, log_eval, valid_size
    ):
        ####################
        # Validation Phase #
        ####################
        model.eval()
        epoch_loss = 0
        loss = 0
        batch = 0

        bar = tqdm(
            desc="validation", total=valid_size  # type: ignore
        )  # uses len(dataset) instead of dataset.size

        bar.set_postfix(
            {
                "batch": 0,
                "loss": 0,
                "diceScore": 0.0,
                "IoU": 0.0
            }
        )
        self.dice = self.dice.to(device)
        self.iou  = self.iou.to(device)
        self.dice.reset()
        self.iou.reset()

        
        dice_value = 0
        iou_value  = 0
        for x, y, _ in dataloader_val:

            with torch.no_grad():
                bar.set_description(f"validation epoch: {epoch}", refresh=True)
                x = x.to(device)
                y = y.squeeze(1).to(device)

                batch += 1
                batch_len = len(y)

                # print(f"label shape {y.shape}")
                y_pred = model(x)
                if isinstance(y_pred, OrderedDict):
                    y_pred = y_pred["out"] 
                y_classes = torch.argmax(y_pred, dim=1)
                dice_value += self.dice(y_classes, y)
                iou_value  += self.iou(y_classes, y)
                loss = loss_fn(y_pred, y)

                loss_v = loss.item()
                epoch_loss += loss_v

                bar.set_postfix(
                    {
                        "batch": batch,
                        "loss": epoch_loss / batch,
                        "diceScore": dice_value.numpy(force=True) / batch,
                        "IoU" : iou_value.numpy(force=True) / batch 
                    }
                )
                bar.update(batch_len)

        bar.close()
        log_eval["epoch"].append(epoch)
        log_eval["loss"].append(epoch_loss / batch)
        log_eval["diceScore"].append(self.dice.compute().numpy(force=True))
        log_eval["IoU"].append(self.iou.compute().numpy(force=True))
        log_eval["lr"].append(0)

        return log_eval

    def train(
        self,
        train_set: Dataset,
        validation_set: Dataset,
        log_dir: Path | str,
        epochs: int = 1,
        batch_size: int = 1,
        objective: str = "loss",
        massimize: bool = False,
        map_cls_to_color = None,
    ) -> tuple[nn.Module, dict[str, list[float]], dict[str, list[float]]]:
        """
        Train method, effettua il traning della rete specificata utilizzando come datasets di train e validazione quelli specificati.
        La funzione supporta un semplice metodo automatico di loging
        Args:

        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        old_weights = ""
        log_dir = Path(log_dir)
        if not log_dir.exists():
            os.mkdir(log_dir)
        else:
            for f in log_dir.glob("*"):
                os.remove(f)

        dataloader_train = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            num_workers=8,
            
        )
        dataloader_val = DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            num_workers=8,
        )
        log_train = {
            "epoch"    : [],
            "loss"     : [],
            "diceScore": [],
            "lr"       : [],
            "IoU"      : []
        }

        log_eval = {
            "epoch"     : [],
            "loss"      : [],
            "diceScore" : [],
            "lr"        : []  ,
            "IoU"       : [] 
        }

        model = self._model.to(device)
        loss_fn = self._loss.to(device)

        for epoch in range(1, epochs + 1):
            self.__train(
                dataloader_train,
                model,
                epoch,
                loss_fn,
                device,
                log_train,
                len(train_set),
            )
            self.__eval(
                dataloader_val,
                model,
                epoch,
                loss_fn,
                device,
                log_eval,
                len(validation_set),
            )

            if self._lr_scheduler:
                self._lr_scheduler.step()

            print(f"\n==END EPOCH {epoch}==\n")

            # save best weigths in agreement with objective value
            if len(log_eval["epoch"]) < 2 or (
                (
                    massimize and log_eval[objective][-1] > log_eval[objective][-2]
                )  # if objective will be massimize(>) get the best weights that massimize it
                or (
                    not massimize and log_eval[objective][-1] < log_eval[objective][-2]
                )  # if objective will be minimize(<) get  save the best weigths  that minimize it
            ):
                if old_weights:  # old only the best weigths
                    os.remove(old_weights)

                torch.save(
                    self._model.state_dict(),
                    log_dir.joinpath(f"best_weights_epoch_{epoch}.pth"),
                )
                old_weights = log_dir.joinpath(f"best_weights_epoch_{epoch}.pth")

            # Qalitative learning rapresentation
            if map_cls_to_color:
                model.eval()
                with torch.no_grad():

                    x, y, _ = validation_set[randint(0, len(validation_set))]
                    x = x.unsqueeze(0).to(device)

                    y_pred = model(x).argmax(dim=1).cpu().numpy().squeeze(0)
                    y = y.cpu().squeeze(0).numpy()

                    y_pred = map2Image(map_cls_to_color, y_pred)
                    y = map2Image(map_cls_to_color, y)

                    result = np.hstack((y, y_pred))
                    plt.imsave(f"{log_dir}/result_epoch_{epoch}.png", result, cmap="gray")

        ###########
        # Logging #
        ###########

        pd.DataFrame(log_eval).to_csv(
            log_dir.joinpath("eval.csv"), sep=",", index=False
        )
        pd.DataFrame(log_train).to_csv(
            log_dir.joinpath("train.csv"), sep=",", index=False
        )

        # finaly load the best weights
        best_weigths = torch.load(old_weights, weights_only=True)
        self._model.load_state_dict(best_weigths)
        return self._model, log_train, log_eval


def show_tensor(tensor: torch.Tensor, cmap: str = "gray") -> None:
    """ get Image-Tensor ans show it"""
    t = tensor.numpy(force=True).transpose((1, 2, 0))
    plt.imshow(t, cmap=cmap)
    plt.show()


def image2Map(map_, x, fill=20):
    """
    Convert RGB Image in Quantizited Image using Quantization Table `map_`
    Unquantizited values will map with 20 (unknow by default)
    """
    h, w, _ = x.shape
    mask = np.full((h, w), fill_value=fill, dtype=np.uint8)

    for color, class_id in map_.items():
        #print(f"  {color} → {class_id}")
        match = np.all(x == color, axis=-1)
        mask[match] = class_id

    return mask


def map2Image(map_, x) -> np.ndarray:
    """ Map quantizited Image in RGB Image """
    mask = np.full((x.shape[0], x.shape[1], 3), fill_value=0, dtype=np.uint8)
    # print("Mappatura colore → class_id usata:")

    for class_id, color in map_.items():
        #print(f" {class_id} -> {color}")
        # print(f"  {color} → {class_id}")

        match = x == class_id

        mask[match] = color

    return mask


def resize(src_dir: Path | str, out_dir: Path | str, split: Path | str, size:tuple[int, int]|None = None) -> None:
        """
            Ridimensiona il dataset fornito in "src_dir" fornito della struttura e lo salva in out_dir
                src_dir/
                ├── img/
                └── label/
            
        """
        src_dir = Path(src_dir)
        out_dir = Path(out_dir)
        

        src_dir = src_dir.joinpath(split)
        out_dir = out_dir.joinpath(split)


        src_img_path =  src_dir.joinpath("img")
        src_label_path = src_dir.joinpath("label")
        
        out_img_path = out_dir.joinpath("img")
        out_label_path = out_dir.joinpath("label")

        os.makedirs(out_img_path, exist_ok=True)
        os.makedirs(out_label_path, exist_ok=True)

        for file_ in src_img_path.glob("*.png"):

            image = cv.imread(file_)
            image = cv.resize(image, size, interpolation=4)
            #print(file_.name)
            cv.imwrite(out_img_path.joinpath(file_.name), image)


        for file_ in src_label_path.glob("*.png"):

            image = cv.imread(file_)
            image = cv.resize(image, size, interpolation=4)
            cv.imwrite(out_label_path.joinpath(file_.name), image)

def plot_report(log_file_train:Path|str, log_file_valid:Path|str, out_dir:Path|str = "."):
    matplotlib.use("PDF")
    log_file_train = Path(log_file_train)
    log_file_valid = Path(log_file_valid)
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    log_train = pd.read_csv(log_file_train, sep=",").to_dict('list')
    log_valid = pd.read_csv(log_file_valid, sep=",").to_dict('list')

    x = log_train.pop("epoch")
    log_valid.pop("epoch")
    

    for (kt, vt), (_, vv) in zip(log_train.items(), log_valid.items()):
        plt.figure()
        plt.plot(x, vt, marker='o', label="Train")
        plt.plot(x, vv, marker='x', label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel(f"{kt}")
        plt.title(out_dir.joinpath(f"Graphics of {kt}"))
        plt.legend()
        plt.savefig(out_dir.joinpath(f"graphic_{kt}"))


if __name__ == "__main__":
    #resize("./Dataset", "./Dataset256x96", "val", (256,96))
    plot_report("./Sigmoid/train.csv", "./Sigmoid/eval.csv")


        

