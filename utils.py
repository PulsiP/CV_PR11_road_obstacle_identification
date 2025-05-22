"""
Utility module
"""

import os
from collections import OrderedDict
from pathlib import Path
from random import randint
from typing import Any, override, Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
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

class ToBMask(nn.Module):
    """
    Class use to preprocess RGB-Mask for segmentation tasck and produce
    One-hot-encoded masck given map definition that map color in specific
    one-hot layer.
    """
    def __init__(self, map_, size = None, fill = 0, add_map = None):
        """"""
        super().__init__()

        self.map_ = map_
        self.size = size
        self.fill = fill
        self.add_map = add_map

    @override
    def forward(self, x):
        """
        Converte immagine RGB [H, W, 3] in maschera [C, H, W] codificata in one-hot encoding (con C numero di valori one-hot).
        """
        if self.size:
            x = cv.resize(x, dsize=self.size)

        x = image2BMap(self.map_, x, fill=self.fill, add_map=self.add_map)
        x = torch.as_tensor(x, dtype=torch.float32)

        return x
        
        


class TrainNetwork:
    """
    Classe specializzata nell'effettuare l'attività di traning per la rete proposta. Tutti i modelli conformi alle specifiche di rete
    in questo progetto possono utilizzare la classe per effettuare il traning e valutazione contemporaneo del modello secondo i criteri
    configurati
    """

    def __init__(self, hp: dict[str, Any], model: nn.Module, dice_num_classes:int=21, lr_scheduler = None, encode = "index") -> None:
        """
        Inizializza il trainer
        """
        self._model: nn.Module = model
        self._loss: nn.Module = hp["loss"]
        self._optimizer: Optimizer = hp["optimizer"]
        self.dice = DiceScore(num_classes=dice_num_classes, average='micro', input_format=encode)
        self.iou = MeanIoU(num_classes=dice_num_classes, input_format=encode)
        self._lr_scheduler = lr_scheduler
        self.encode = encode

    def __train(
        self, dataloader_train, model, epoch, loss_fn, device, log_train, train_size
    ):
        ##################
        # Training Phase #
        ##################

        epoch_loss = 0
        loss = 0
        batch = 0
        self._loss = self._loss.to(device)
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
        for x, y, m in dataloader_train:
            bar.set_description(f"training epoch: {epoch}", refresh=True)


            
            x = x.to(device)
            y = y.squeeze(1).to(device)
            m = m.to(device)


            batch += 1
            batch_len = len(y)

            self._optimizer.zero_grad()
            
            y_pred = model(x)
            
            
            if isinstance(y_pred, OrderedDict):
                y_pred = y_pred["out"] 
                
            
            
            y_classes = (torch.argmax(y_pred, dim=1) if self.encode == "index" else F.one_hot(torch.argmax(y_pred, dim=1), num_classes=y_pred.shape[1]).permute(0, 3, 1, 2))
            #print(f"x shape: {x.shape}")
            #print(f"y shape: {y.shape}")
            #print(f"y_pred shape: {y_pred.shape}")
            dice_value += self.dice(y_classes, y.long())
            iou_value += self.iou(y_classes, y.long())
            
            loss = loss_fn(y_pred, y, m)
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
        for x, y, m in dataloader_val:

            with torch.no_grad():
                bar.set_description(f"validation epoch: {epoch}", refresh=True)
                x = x.to(device)
                y = y.squeeze(1).to(device)
                m = m.to(device)

                batch += 1
                batch_len = len(y)

                # print(f"label shape {y.shape}")
                y_pred = model(x)
                if isinstance(y_pred, OrderedDict):
                    y_pred = y_pred["out"] 
                y_classes = (torch.argmax(y_pred, dim=1) if self.encode == "index" else F.one_hot(torch.argmax(y_pred, dim=1), num_classes=y_pred.shape[1]).permute(0, 3, 1, 2))
                dice_value += self.dice(y_classes, y.long())
                iou_value  += self.iou(y_classes, y.long())
                loss = loss_fn(y_pred, y, m)

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
            num_workers=16,
            persistent_workers=True
            
        )
        dataloader_val = DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16,
            persistent_workers=True
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
            self.__train(dataloader_train,model,epoch,loss_fn,device,log_train,len(train_set))
            self.__eval(dataloader_val,model,epoch,loss_fn,device,log_eval,len(validation_set))

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

                    img, y, _ = validation_set[randint(0, len(validation_set))]
                    x = img.clone()
                    x = x.unsqueeze(0).to(device)
                    img = img.numpy(force=True)
                    
                    y_pred = model(x)
                    y_pred_logits = unknownObjectnessScore(y_pred.squeeze(0)).unsqueeze(0)
                    if y.shape[-1] > len(map_cls_to_color):
                        #print(y_pred.shape)
                        y_pred = y_pred[:, :len(map_cls_to_color), :, :]
                        #print(y_pred.shape)
                        y_pred = y_pred.argmax(dim=1).cpu().numpy().squeeze(0)
                        y = y.cpu().squeeze(0).numpy()
                        y = y[:len(map_cls_to_color), :, :]
                    
                    #print(y.shape)
                    
                      

                    if self.encode == "index":
                        y_pred = map2Image(map_cls_to_color, y_pred)
                        y = map2Image(map_cls_to_color, y)
                    else:
                         
                        y = np.argmax(y, axis=0)
                        y_pred = bmap2Image(map_cls_to_color, y_pred)
                        y = bmap2Image(map_cls_to_color, y)


                    
                    result = np.hstack((np.permute_dims(img*255, (1,2,0)), y, y_pred)).astype(np.uint8)
                   
                    plt.imsave(f"{log_dir}/result_epoch_{epoch}.png", result)

                    varisco_heatmap = compute_varisco_heatmap_rgb(y_pred_logits.squeeze(0))
                    overlay = overlay_heatmap(np.transpose(img * 255, (1, 2, 0)), varisco_heatmap, alpha=0.5)

                    plt.imsave(f"{log_dir}/heatmap_epoch_{epoch}.png", varisco_heatmap)
                    plt.imsave(f"{log_dir}/heatmap_overlay_epoch_{epoch}.png", overlay)
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


def image2Map(map_, x, fill=0):
    """
    Convert RGB Image in Quantizited Image using Quantization Table `map_`
    Unquantizited values will map with 0 (unknow by default)
    """
    h, w, _ = x.shape
    mask = np.full((h, w), fill_value=fill, dtype=np.uint8)

    for color, class_id in map_.items():
        #print(f"  {color} → {class_id}")
        match = np.all(x == color, axis=-1)
        mask[match] = class_id

    return mask

# from RGB color to layers (indexed using TrainId)
def image2BMap(map_, x, fill=0, add_map=None):
    h, w, _ = x.shape

    

    # # Converti ogni riga in una tupla RGB standard
    # colors = [tuple(map(int, color)) for color in np.unique(x.reshape(-1, 3), axis=0)]
    # mask = np.full((len(set(map_.values())), h, w), fill_value=fill, dtype=np.uint8)
    # for color, class_id in map_.items():
        
    #     #print(f"  {color} → {class_id}")
        
    #     match = np.all(x == color, axis=-1)
        
    #     #if match.sum() == 0: print(color)
    #     #print(f"{color} -> {match.sum()}")
        
    #     mask[class_id][match] = 1

    label_map = np.full((h, w), fill_value=fill,  dtype=np.uint8)

    for color, class_idx in map_.items():
        mask = np.all(x == color, axis=-1)
        label_map[mask] = class_idx

    # One-hot encoding
    mask = np.eye(len(set(map_.values())), dtype=np.uint8)[label_map]
    mask = mask.transpose((2,0,1))
    if add_map:
        layer = np.full((1, h, w), fill_value=fill, dtype=np.uint8)
        for color, _ in add_map.items():
           
            #print(f"  {color} → {class_id}")
            match = np.all(x == color, axis=-1)
            layer[0][match] = 1
        mask = np.concatenate((mask, layer), axis=0)
    
    #i = randint(0, 100)
    #j = randint(0, 100)

    #print(mask[:, i, j])
    #print(x[i, j, :])

    return mask

    

# from layer to RGB color
def bmap2Image(map_, x, fill=0):
    
    h, w = x.shape
    mask = np.full((h, w, 3), fill_value=fill, dtype=np.uint8)

    for class_id, color in map_.items():
        #print(f" {class_id} -> {color}")
        #print(f"  {color} → {class_id}")
        match = x == class_id
        mask[match] = color
    
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
    
def compute_varisco_heatmap_rgb(uos_map: torch.Tensor) -> np.ndarray:
    """Normalizza la mappa UOS e restituisce una heatmap RGB."""

    varisco_map = uos_map.squeeze(0).cpu().numpy()

    
    min_val, max_val = varisco_map.min(), varisco_map.max()
    norm_map = (varisco_map - min_val) / (max_val - min_val + 1e-8)
    return (plt.cm.jet(norm_map)[:, :, :3] * 255).astype(np.uint8)



def overlay_heatmap(img_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Restituisce l’overlay tra immagine RGB e heatmap."""
    import cv2
    img_rgb = img_rgb.astype(np.uint8)
    heatmap_rgb = heatmap_rgb.astype(np.uint8)
    #print(heatmap_rgb.shape)
    #heatmap_resized = cv2.resize(heatmap_rgb, (img_rgb.shape[1], img_rgb.shape[0]))
    return cv2.addWeighted(img_rgb, 1 - alpha, heatmap_rgb, alpha, 0)

def unknownObjectnessScore(pred: torch.Tensor) -> torch.Tensor:
    """Map [1, H, W] with Uknown Objectness Score."""
    if pred.dim() == 4:
        pred = pred.squeeze(0)  # [C, H, W]

    pred = F.sigmoid(pred)
    classes = 1 - pred[:-1]           # [C-1, H, W]
    object_score = pred[-1:]         # [1, H, W]
    return torch.prod(classes, dim=0, keepdim=True)  * object_score  # [1, H, W]


def ObjectnessScore(pred: torch.Tensor) -> torch.Tensor:
    """Map [1, H, W] with Uknown Objectness Score."""
    if pred.dim() == 4:
        pred = pred.squeeze(0) 

    object_score = pred[-1:]         
    return object_score  

def improve_image(x):
    lab = cv.cvtColor(x, cv.COLOR_RGB2LAB)
    # Separare i canali
    l, a, b = cv.split(lab)

    # CLAHE sul canale L
    clahe = cv.createCLAHE(clipLimit=1.4, tileGridSize=(12,12))
    cl = clahe.apply(l)

    # Ricompone e converte di nuovo in BGR
    limg = cv.merge((cl, a, b))
    x = cv.cvtColor(limg, cv.COLOR_LAB2RGB)

    return x

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

def getBCEmask(shape: Tuple[int, int, int, int], dim):
    B, C, H, W = shape


    mask = torch.zeros((B, C, W, H), dtype=torch.uint8)

    mask[:, :, :dim, :] = 1    
    mask[:, :, -dim:, :] = 1    
    mask[:, :, :, :dim] = 1      
    mask[:, :, :, -dim:] = 1  

    return mask

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



if __name__ == "__main__":
    #resize("./Dataset", "./Dataset256x96", "val", (256,96))
    plot_report("./Sigmoid/train.csv", "./Sigmoid/eval.csv")


        

