"""
Utility module
"""

import os
from collections import OrderedDict
from pathlib import Path
from random import randint
import shutil
from typing import Any, List, override, Tuple
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
from torchmetrics.functional.classification import auroc
from tqdm.auto import tqdm

class ToMask(nn.Module):
    """
    Class use to preprocess RGB-Mask for segmentation task
    """
    def __init__(self, map_, size = None, fill = 0):
        super().__init__()

        self.map_ = map_
        self.size = size
        self.fill = fill

    @override
    def forward(self, x):
        """
        Convert RGB image [H, W, 3] to mask [1, H, W] in numeric classes.
        Pixel with unknown colors are assigned to class 0 (fallback)
        """
        if self.size:
            x = cv.resize(x, dsize=self.size)

        x = image2Map(self.map_, x, fill=self.fill)
        x = torch.as_tensor(x, dtype=torch.long).unsqueeze(0)
        
        
        return x

class ToBMask(nn.Module):
    """
    Class use to preprocess RGB-Mask for segmentation task and produce
    one-hot encoded mask given a dictionary that map color in specific
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
        Convert RGB image [H, W, 3] to mask [C, H, W] encoded in one-hot (C is the number of one-hot value)
        """
        if self.size:
            x = cv.resize(x, dsize=self.size)

        x = image2BMap(self.map_, x, fill=self.fill, add_map=self.add_map)
        x = torch.as_tensor(x, dtype=torch.float32)

        return x
        

class TrainNetwork:
    """
    Class for traning generic network in segmentation task and Unknown Object Detection through Unknown Objectness Score (UOS)
    """

    def __init__(self, hp: dict[str, Any], model: nn.Module, dice_num_classes:int=21, keep_index:List[int] = [], lr_scheduler = None, encode = "index") -> None:
        """
        Init trainer 
        """
        self._model: nn.Module = model
        if keep_index:
            self._index = keep_index 
        else:
            self._index = list(range(dice_num_classes))

        
        self._loss: nn.Module = hp["loss"]
        self._optimizer: Optimizer = hp["optimizer"]
        self.dice = DiceScore(num_classes=len(self._index), average='macro', input_format=encode)
        self.iou = MeanIoU(num_classes=len(self._index), input_format=encode)
        #self.AP = MultilabelAveragePrecision(num_labels=len(self._index), average=None)
        #self.auroc = MultilabelAUROC(num_labels=len(self._index))
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

        bar.set_postfix({"batch": 0, "loss": 0, "diceScore": 0.0,"IoU": 0.0, "AVG_P": 0.0, "AUROC": 0.0, 'lr': self._optimizer.param_groups[0]['lr']})

        self.dice = self.dice.to(device)
        self.iou  = self.iou.to(device)

        self.dice.reset()
        self.iou.reset()

        for x, y, m in dataloader_train:
            bar.set_description(f"training epoch: {epoch}", refresh=True)

            # predict
            x = x.to(device)
            y = y.squeeze(1).to(device)
            m = m.to(device)

            batch += 1
            batch_len = len(y)

            self._optimizer.zero_grad()
            
            y_pred = model(x)
            
            
            if isinstance(y_pred, OrderedDict):
                y_pred = y_pred["out"] 
            
            y_pred = y_pred[:, self._index, :, :]
            y = y[:, self._index, :, :]

            # compute loss
            with torch.no_grad():
                y_classes = (torch.argmax(y_pred, dim=1) if self.encode == "index" else F.one_hot(torch.argmax(y_pred, dim=1), num_classes=y_pred.shape[1]).permute(0, 3, 1, 2))

                self.dice.update(y_classes, y.long())
                self.iou.update(y_classes, y.long())
            
            loss = loss_fn(y_pred, y, m)
            loss_v = loss.item()
            epoch_loss += loss_v
            loss.backward()

            self._optimizer.step()

            # record metrics

            bar.set_postfix(
                {
                    "batch": batch,
                    "loss": epoch_loss / batch,
                    "diceScore": self.dice.compute().numpy(force=True),
                    "IoU": self.iou.compute().numpy(force=True),
                    'lr': self._optimizer.param_groups[0]['lr']
                }
            )
            bar.update(batch_len)

        # set log information (for epoch)
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
                "AUROC": 0.0
            }
        )

        self.dice = self.dice.to(device)
        self.iou  = self.iou.to(device)
        self.dice.reset()
        self.iou.reset()

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
                
                y_pred = y_pred[:, self._index, :, :]
                y = y[:, self._index, :, :]

                y_classes = (torch.argmax(y_pred, dim=1) if self.encode == "index" else F.one_hot(torch.argmax(y_pred, dim=1), num_classes=y_pred.shape[1]).permute(0, 3, 1, 2))
                self.dice.update(y_classes, y.long())
                self.iou.update(y_classes, y.long())

                loss = loss_fn(y_pred, y, m)

                loss_v = loss.item()
                epoch_loss += loss_v

                bar.set_postfix(
                    {
                        "batch": batch,
                        "loss": epoch_loss / batch,
                        "diceScore": self.dice.compute().numpy(force=True),
                        "IoU" : self.iou.compute().numpy(force=True),
                        #"AVG_P": AP_value / batch,
                        #"AUROC": AUROC_value / batch, 
                        
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
        Traning `model` for n `epochs` given traning and testing `datasets` then return log file and weights
        Args:
            traning_set(Dataset):
                Dataset using for traning procedure
            
            validation_set(Datase):
                Dataset using for validation procedure during the traning
            
            log_dir(Path):
                Default directory for loggin information (qualitative analysis and metrics report)
            
            epochs(int):
                Number traning epochs, default 1 epoch
            
            batch_size(int):
                Number of elements for batch, default 1 element
            
            objective(str):
                Values used for monitoring the model during traning
            
            massimize(bool):
                Massimize/Minimize objective value
            
            map_cls_to_color(dict[int, tuple[int, int, int]):
                Map predicted value in RGB space for visualization purposes

        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        old_weights = ""
        log_dir = Path(log_dir)
        if not log_dir.exists():
            os.mkdir(log_dir)
        else:
            for f in log_dir.glob("*"):
                if f.is_dir():
                    shutil.rmtree(f)
                else:
                    os.remove(f)

        dataloader_train = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=24,
            persistent_workers=True
            
        )
        dataloader_val = DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=24,
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
            "lr"        : [],
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
                    massimize and log_eval[objective][-1] >= max(log_eval[objective])
                )  # if objective will be massimize(>) get the best weights that massimize it
                or (
                    not massimize and log_eval[objective][-1] <= min(log_eval[objective])
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
                    
                    # Add batch for computing
                    x = x.unsqueeze(0).to(device)
                    y = y.unsqueeze(0)
        
                    y_pred = model(x)
                    uos = unknownObjectnessScore(y_pred)
                    
                    # Remove Object map if present
                    
                        #print(y.shape)
                    y_pred = y_pred[:, :len(map_cls_to_color), :, :]    
                    y = y[:, :len(map_cls_to_color), :, :]
                    
                    #y_pred = y_pred[:, self._index, :, :]
                    #print(y.shape)

                    # remove batchs
                    y = y.cpu().squeeze(0)
                    y_pred = y_pred.cpu().squeeze(0)
                    x = x.cpu().squeeze(0)

                    # Convert in numpy objects
                    y = y.numpy()
                    y_pred = y_pred.numpy()
                    img = x.numpy()

                    if self.encode == "index":
                        pred_img = map2Image(map_cls_to_color, y_pred)
                        y_img = map2Image(map_cls_to_color, y)
                    else:
                        # convert logits in classes
                        y_pred_prb = np.argmax(y_pred, axis=0)
                        y_prb= np.argmax(y, axis=0)

                        pred_img = bmap2Image(map_cls_to_color, y_pred_prb)
                        y_img = bmap2Image(map_cls_to_color, y_prb)


                    result = np.hstack((np.permute_dims(img*255, (1,2,0)), y_img, pred_img)).astype(np.uint8)
                   
                    plt.imsave(f"{log_dir}/result_epoch_{epoch}.png", result)

                    varisco_heatmap = compute_varisco_heatmap_rgb(uos)
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
    """ Get and show Image-Tensor"""
    t = tensor.numpy(force=True).transpose((1, 2, 0))
    plt.imshow(t, cmap=cmap)
    plt.show()


def image2Map(map_, x, fill=0) -> np.ndarray:
    """
    Convert RGB Image in quantized Image using Quantization Table `map_`
    Unquantized values are mapped to 0 (unknown by default)
    """
    h, w, _ = x.shape
    mask = np.full((h, w), fill_value=fill, dtype=np.uint8)

    for color, class_id in map_.items():
        match = np.all(x == color, axis=-1)
        mask[match] = class_id

    return mask

# from RGB color to layers (indexed using TrainId)
def image2BMap(map_, x, fill=0, add_map=None):
    """Converts an RGB label image into a one-hot encoded mask using a color-to-class mapping."""
    h, w, _ = x.shape
    label_map = np.full((h, w), fill_value=fill,  dtype=np.uint8)
    
    for color, class_idx in map_.items():
        mask = np.all(x == color, axis=-1)
        label_map[mask] = class_idx

    max_index = max(set(map_.values()))
    # One-hot encoding
    mask = np.eye(max_index + 1 , dtype=np.uint8)[label_map]
    mask = mask.transpose((2,0,1))
    if add_map:
        layer = np.full((1, h, w), fill_value=fill, dtype=np.uint8)
        for color, _ in add_map.items():
           
            match = np.all(x == color, axis=-1)
            layer[0][match] = 1
        mask = np.concatenate((mask, layer), axis=0)

    return mask

    

# from layer to RGB color
def bmap2Image(map_:np.ndarray, x:dict, fill:int=0):
    """
    Convert a one-hot mask into an RGB image via a color map.

    Args:
        map_ (dict[int, tuple[int,int,int]]): class -> RGB mapping.
        x (np.ndarray): shape (H, W) of class IDs.
        fill (int): default RGB fill value.

    Returns:
        np.ndarray: RGB image of shape (H, W, 3).
    """
    h, w = x.shape
    mask = np.full((h, w, 3), fill_value=fill, dtype=np.uint8)

    for class_id, color in map_.items():
        match = x == class_id
        mask[match] = color
    
    return mask
    

def map2Image(map_, x) -> np.ndarray:
    """ Map quantized Image in RGB Image """
    mask = np.full((x.shape[0], x.shape[1], 3), fill_value=0, dtype=np.uint8)

    for class_id, color in map_.items():
        match = x == class_id
        mask[match] = color

    return mask
    
def compute_varisco_heatmap_rgb(uos_map: np.ndarray|torch.Tensor) -> np.ndarray:
    """ Normalize a score map to [0,1] and apply a jet colormap. """
    if isinstance(uos_map, torch.Tensor):
        uos_map = uos_map.squeeze(0).numpy(force=True)
    
    varisco_map = uos_map    
    min_val, max_val = varisco_map.min(), varisco_map.max()
    norm_map = (varisco_map - min_val) / (max_val - min_val + 1e-8)
    return (plt.cm.jet(norm_map)[:, :, :3] * 255).astype(np.uint8)


def overlay_heatmap(img_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """ Blend a heatmap over an RGB image. """
    import cv2
    img_rgb = img_rgb.astype(np.uint8)
    heatmap_rgb = heatmap_rgb.astype(np.uint8)
    return cv2.addWeighted(img_rgb, 1 - alpha, heatmap_rgb, alpha, 0)

def unknownObjectnessScore(pred: torch.Tensor) -> torch.Tensor:
    """Map [1, H, W] with Uknown Objectness Score."""
    if pred.dim() == 4:
        pred = pred.squeeze(0)  # [C, H, W]

    pred = F.sigmoid(pred)
    
    classes = 1 - pred[:-1]           # [C-1, H, W]
    object_score = pred[-1:]         # [1, H, W]
    return torch.prod(classes, dim=0, keepdim=True)  * object_score  # [1, H, W]

def unknownObjectnessScoreMB(pred: torch.Tensor) -> torch.Tensor:
    """
    Computes the Unknown Objectness Score for a batch of predictions.

    Args:
        pred (Tensor): A tensor of shape [B, C, H, W], where C = number of classes including the 'unknown object' class (last one).

    Returns:
        Tensor: A tensor of shape [B, 1, H, W] containing the unknown objectness scores.
    """
    #assert pred.dim() == 4, "Expected input shape [B, C, H, W]"

    pred = F.sigmoid(pred)  # Apply sigmoid to logits

    classes = 1 - pred[:, :-1, :, :]        # All known classes: [B, C-1, H, W]
    object_score = pred[:, -1:, :, :]       # Unknown object class score: [B, 1, H, W]

    # Multiply the inverse of known class probabilities and scale by the unknown score
    unknown_score = torch.prod(classes, dim=1, keepdim=True) * object_score  # [B, 1, H, W]

    return unknown_score


def ObjectnessScore(pred: torch.Tensor) -> torch.Tensor:
    """Map [1, H, W] with Objectness Score."""
    if pred.dim() == 4:
        pred = pred.squeeze(0) 

    object_score = pred[-1:]         
    return object_score  

def improve_image(x:np.ndarray) -> np.ndarray:
    """
    Apply CLAHE on L channel in LAB color space to boost contrast.

    Args:
        x (H,W,3) RGB image.

    Returns:
        np.ndarray: contrast-enhanced RGB image.
    """
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
            Resize the dataset located in `src_dir`, which must have the following structure:

                src_dir/
                ├── img/
                └── label/

            Save the resized images and labels to `out_dir`.
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

def plot_report(log_file_train:Path|str|pd.DataFrame, log_file_valid:Path|str|pd.DataFrame|None = None, out_dir:Path|str = ".") -> None:
    """Generate training (and validation) metric plots and save them as PDF."""
    matplotlib.use("PDF")
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    def _comparative_report(log_train, log_valid, out_dir):
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

    def _plot_graphics(log_train, out_dir):
        x = log_train.pop("epoch")
        for (kt, vt) in log_train.items():
            plt.figure()
            plt.plot(x, vt, marker='o', label="Test")
            
            plt.xlabel("Epochs")
            plt.ylabel(f"{kt}")
            plt.title(out_dir.joinpath(f"Graphics of {kt}"))
            plt.legend()
            plt.savefig(out_dir.joinpath(f"graphic_{kt}"))

    if isinstance(log_file_train, pd.DataFrame):
        log_train = log_file_train
    else:
        log_file_train = Path(log_file_train)
        log_train = pd.read_csv(log_file_train, sep=",").to_dict('list')

    
    
    if log_file_valid:
        if isinstance(log_file_valid, pd.DataFrame):
            log_valid = log_file_valid
        else:
            log_file_valid = Path(log_file_valid)        
            log_valid = pd.read_csv(log_file_valid, sep=",").to_dict('list')
        _comparative_report(log_train, log_valid, out_dir)
    else:
        _plot_graphics(log_train, out_dir)


    
    
    

    

def getBCEmask(shape: Tuple[int, int, int, int], dim:int) -> torch.Tensor:
    """
    Generate a binary mask with 1s along the border of each image in a batch.
    Useful for applying BCE loss only near the edges for boundary-aware training.
    """
    B, C, H, W = shape


    mask = torch.zeros((B, C, W, H), dtype=torch.uint8)

    mask[:, :, :dim, :] = 1    
    mask[:, :, -dim:, :] = 1    
    mask[:, :, :, :dim] = 1      
    mask[:, :, :, -dim:] = 1  

    return mask


class AUROC_metric():
    """
    Class to compute and accumulate AUROC (Area Under ROC Curve) scores
    across multiple batches for multi-class segmentation.
    """
    def __init__(self, num_classes):
        self.__num_classes = num_classes

        self.auroc_value = 0.0
        self.num_updates = 0
      
    def update(self, preds, target):
        B, C, H, W = preds.shape

        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        preds_flat = preds.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = target.permute(0, 2, 3, 1).reshape(-1, C)

        auroc_scores = auroc(preds_flat, target_flat.int(), task="multiclass", num_classes=C, average=None)

        auroc_score = auroc_scores.cpu().mean().item()

        self.auroc_value += auroc_score
        self.num_updates +=1

        return auroc_score
    
    def compute(self):
        return self.auroc_value/self.num_updates


def miscoverage_loss(Z_mask:torch.Tensor, Y_mask:torch.Tensor) -> float:
    """
    Compute the miscoverage loss between binary multi-label mask.
    """
    #print(Z_mask.shape)
    
    try:
        C, H, W = Z_mask.shape
    except Exception:
        print(f"error {Z_mask.shape}")
        return 1
    
    coverage = torch.sum(Z_mask & Y_mask).float() / (H * W)
    return 1.0 - coverage.item()

def calibration2(Z:torch.Tensor, Y:torch.Tensor, alpha:float, B:float, loss_fn:Any, verbose:bool=True, num_points:int=5000) -> float:
    """
    Post-hoc calibration: find the smallest lambda_hat such that:
    (n/(n+1)) * R_n(lambda) + B/(n+1) <= alpha
    
    Args:
        Z (torch.Tensor): logits or score [B,C,H,W]
        Y (torch.Tensor): label ground-truth [B,H,W]
        alpha (float): significance level (es. 0.01)
        B (float): conformal parameter (es. quantile)
        loss_fn (callable): function to compute R_n(lambda)
        num_points (int): number of points in the grid [0, 1]
    
    Returns:
        float: optimal lambda_hat 
    """
    n = len(Z)
    if n == 0:
        if verbose:
            print(" Calibration fail")
        return 1.0

    # Candidati λ ∈ [0, 1]
    lambda_grid = np.linspace(0., 1.0, num_points)
    
    lambda_hat = 1.0  # fallback (più conservativo)
    found = False
    
    for lam in lambda_grid:
        R_n = empirical_risk(Z, Y, loss_fn, lam)
        condition = (n / (n + 1)) * R_n + (B / (n + 1))

        if verbose:
            print(f"  Test lambda={lam:.4f}, R_n={R_n:.4f}, ValCond={condition:.4f}, Target α={alpha:.4f}")
        
        if condition <= alpha:
            lambda_hat = lam
            found = True
            break  # abbiamo trovato l'infimo λ

    if not found and verbose:
        R_n_at_1 = loss_fn(Z, Y)
        cond_at_1 = (n / (n + 1)) * R_n_at_1 + (B / (n + 1))
        alpha_min_possible = B / (n + 1)
        print(f"  lambda not found,  α={alpha:.4f}")
        print(f"  Min risk value: {alpha_min_possible:.4f}")
        print(f"  set lambda= 1.0 (fallback)")

    return lambda_hat


# Conformal Semantic Image Segmentation: Post-hoc Quantification of Predictive Uncertainty
def lasc(Z:torch.Tensor, lamb:float) -> torch.Tensor: # or Least Ambiguous Set-Valued Classifiers (LAC)
    """
    Least Ambiguous Set-valued Classifier (LASC):
    Returns a binary mask with classes whose score >= 1 - lambda,
    """
    
    # X:[B,C,H,W]
    Z = torch.permute(Z, dims=(0,2,3,1)) # X:[B,H,W,C]
    Z = F.softmax(Z, dim=-1) # pixel-wise softmax-scores

    #  fallback

    _ , top1_indices = Z.max(dim=-1) # top1_indices will be (B, H, W)

    # Create a tensor for the top-1 classes, initialized to zeros
    # This will be used to ensure at least one class is active for each pixel
    # Use torch.zeros_like for shape matching
    top1_mask_fill = torch.zeros_like(Z, requires_grad=False).long()

    # Fill the top-1_mask_fill at the top-1 indices with 1s
    # Unsqueeze top1_indices to match dimensions for scatter_
    # top1_indices needs to be (B, H, W, 1) for scatter_
    top1_mask_fill.scatter_(dim=-1, index=top1_indices.unsqueeze(-1), value=1)
    
    # Combine the thresholded mask with the top-1 fallback
    # For each pixel, if the thresholded mask is all zeros (empty),
    # then include the top-1 class.
    # This can be done by taking the element-wise OR (max) of the two masks.
    # If mask[b, h, w, k] is 1 OR top1_mask_fill[b, h, w, k] is 1, then the final mask element is 1.
    mask = (Z >= 1 - lamb).long()

    final_mask = torch.max(mask, top1_mask_fill)
    final_mask = torch.permute(final_mask, (0,3,1,2))

    return final_mask





def empirical_risk(Z:torch.Tensor, Y:torch.Tensor, loss:Any, lamb:float) -> float:
    """
    Compute average empirical risk using a set-valued classifier (LASC)
    with confidence level lambda.
    """
    # 0 1 2  3
    # X:[B,C,H,W]
    # Y:[B,C,H,W]
    # loss: f(cx, y) -> float
    cx = lasc(Z, lamb)
    R = 0
    n = Z.shape[0]

    for  z, y in zip(cx, Y): # for i=1 to i=n
        R += loss(z, y)

    return R/n


def binary_loss(Z:torch.Tensor, Y:torch.Tensor) -> float:
    """
    Return 1 if prediction misses any true class (Z < Y), else 0.
    """
    # Z: [C,H,W]
    # Y: [C,H,W]
    
    return int(torch.all(Z < Y))

def binary_loss_threshold (Z:torch.Tensor, Y:torch.Tensor, t:float=0.001) -> float:
    """
    Returns 1 if the overlap between prediction Z and ground truth Y is below threshold t, else 0.
    """
    return int((torch.sum(Z*Y) / torch.sum(Y)) < t)
    
def uncertainty_loss_simple(Y_pred:torch.Tensor, Y_true:torch.Tensor, delta:float=0.1)-> float:
    """
    Penalize predictions falling outside the interval [Y_true ± delta].
    """
    lower = Y_true - delta
    upper = Y_true + delta
    outside = ((Y_pred < lower) | (Y_pred > upper)).float()
    return outside.mean()

def img2UQH(img:np.ndarray, Y_pred:torch.Tensor, lam:float) -> np.ndarray:
    """
    Compute Uncertain Quantization Heatmap (UQH)
        Y_pred :        [B, C, H, W]
        Y      : Tensor [B, C, H, W]
        
    """
    
    if len(Y_pred.shape) == 3:
        
        Y_pred = Y_pred[np.newaxis, :, :, :]
    
    Y_pred = torch.tensor(Y_pred, requires_grad=False)
    mask = lasc(Y_pred, lam) # [B, C, H, W]
    mask = mask.squeeze(0).numpy() # [C,H,W]
    #print(mask[:, 0:10,:10])
    mask = np.sum(mask, axis=0)
    
    vh = compute_varisco_heatmap_rgb(mask)
    #print(vh.shape)
    
    hm = overlay_heatmap(np.transpose(img * 255, (1, 2, 0)), vh, 0.5)

    return hm


if __name__ == "__main__":
    pass