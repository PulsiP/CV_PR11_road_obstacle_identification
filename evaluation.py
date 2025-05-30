from matplotlib import pyplot as plt
import torch
import pandas as pd
import numpy as np
import os
from pathlib import Path
from torch.nn import Module
from torch.nn import functional as F
from typing import *
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchmetrics.segmentation import DiceScore, MeanIoU
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from torch.utils.data import Dataset
from utils import (\
    bmap2Image,
    compute_varisco_heatmap_rgb,
    map2Image,
    overlay_heatmap,
    plot_report,
    unknownObjectnessScore
)


class ObstacleBenchmark:
    def __init__(self,network:Module,log_dir, weights:str|Path|None = None):
        self.log = Path(log_dir).joinpath("ObstacleBenchmark")
        self._network = network
        os.makedirs(self.log, exist_ok=True)
        if weights:
            try:
                w = torch.load(f=weights, weights_only=True)
                self._network.load_state_dict(w, strict=True)

            except FileNotFoundError as err:
                print(err)
                exit(-1)
            except Exception as exp:
                print(exp)
                exit(-1)
        else:
            self._network = network

        
        
    def run_benchmark(self, test_loader:Dataset,loss_fn:Module, keep_index:List[int] = [], format:str="one-hot", map_cls_to_color:Dict[int,Tuple[int,int,int]]|None =None, device:str="cpu", repeat_for:int= 1, target:bool=True):
        
        with torch.no_grad():
            
            dice = DiceScore(num_classes=len(keep_index), average="macro", input_format=format)
            iou  = MeanIoU(num_classes=len(keep_index), input_format=format, per_class=False)
            AP = MultilabelAveragePrecision(num_labels=len(keep_index), average="macro")
            auroc = MultilabelAUROC(num_labels=len(keep_index), average="macro")
            
            test_loader = DataLoader(
                test_loader,
                batch_size=8,
                shuffle=True,
                pin_memory=True,
                num_workers=16,
                persistent_workers=True
            
            )

            valid_size = len(test_loader.dataset)
            log_eval = {
                
                
                "diceScore":    [],
                "IoU":          [],
                "AVG_P":        [],
                "AUROC":        [],
                "epoch":        []

            }
            model = self._network.to(device)
            
            

            for epoch in range(1, repeat_for + 1):
                ####################
                # Validation Phase #
                ####################
                model.eval()
                epoch_loss = 0
                loss = 0
                batch = 0

                bar = tqdm(
                    desc="run_benchmark time", total=valid_size  # type: ignore
                )  # uses len(dataset) instead of dataset.size

                dice = dice.to(device)
                iou  = iou.to(device)
                AP = AP.to(device)
                auroc = auroc.to(device)

                dice.reset()
                iou.reset()
                AP.reset()
                auroc.reset()

                AP_value = 0
                AUROC_value = 0
                for item in test_loader:
                    
                    if target:
                        x, y, m = item
                        y = y.to(device)

                    else:
                        x, m = item
                    bar.set_description(f"run_benchmark time: {epoch}", refresh=True)
                    x = x.to(device)
                    m = m.to(device)

                    batch += 1
                    # print(f"label shape {y.shape}")
        
                    y_pred = model(x)
                    if isinstance(y_pred, OrderedDict):
                        y_pred = y_pred["out"] 

                    if len(keep_index) > 0:
                        
                        y_pred = y_pred[:, keep_index, :, :]
                        
                        if target:
                            y = y[:, keep_index, :, :]

                    if target:
                        y_classes = (torch.argmax(y_pred, dim=1) if format == "index" else F.one_hot(torch.argmax(y_pred, dim=1), num_classes=y_pred.shape[1]).permute(0, 3, 1, 2))
                        dice.update(y_classes, y.long())
                        iou.update(y_classes, y.long())

                        AP.reset()
                        AP_value += AP(y_pred, y.long()).numpy(force=True).mean()

                        auroc.reset()
                        AUROC_value += auroc(y_pred, y.long()).numpy(force=True)

                        loss = loss_fn(y_pred, y, m)

                        loss_v = loss.item()
                        epoch_loss += loss_v

                    
                    #
                    bar.update(len(x))

                bar.close()
                if target:
                    #log_eval["epoch"].append(epoch)
                    #log_eval["loss"].append(epoch_loss / batch)
                    log_eval["diceScore"].append(dice.compute().numpy(force=True))
                    log_eval["IoU"].append(iou.compute().numpy(force=True))
                    log_eval["AVG_P"].append(AP_value / batch)
                    log_eval["AUROC"].append(AUROC_value / batch)
                    log_eval["epoch"].append(epoch)

                if map_cls_to_color:
                    x = x[-1].unsqueeze(0)
                    y_pred = y_pred[-1].unsqueeze(0)
                    uos = unknownObjectnessScore(y_pred)
                    
                    # Remove object layer if exist
                    y_pred = y_pred[:len(map_cls_to_color), :, :]
                    if target :
                        y = y[-1].unsqueeze(0)
                        y = y[:, :len(map_cls_to_color), :, :]

                    
                    

                    img = x.cpu().squeeze(0).numpy()
                    y =   y.cpu().squeeze(0).numpy()
                    y_pred = y_pred.cpu().squeeze(0).numpy()
                    
                    
                    if format == "index":
                        y_pred_img = map2Image(map_cls_to_color, y_pred)
                    else:
                        y_pred_prb = np.argmax(y_pred, axis=0)
                        
                        y_pred_img = bmap2Image(map_cls_to_color, y_pred_prb)


                    if target:
                        if format == "index":
                            y_img = map2Image(map_cls_to_color, y)
                        else:
                            y_prb = np.argmax(y, axis=0)
                            y_img = bmap2Image(map_cls_to_color, y_prb)

                    if target:
                        result = np.hstack((np.permute_dims(img*255, (1,2,0)), y_img, y_pred_img)).astype(np.uint8)
                    else:
                        result = np.hstack((np.permute_dims(img*255, (1,2,0)), y_pred_img)).astype(np.uint8)


                    varisco_heatmap = compute_varisco_heatmap_rgb(uos)
                    overlay = overlay_heatmap(np.transpose(img * 255, (1, 2, 0)), varisco_heatmap, alpha=0.5)

                    head_map = np.hstack( (np.transpose(img * 255, (1, 2, 0)), varisco_heatmap, overlay) ).astype(np.uint8)
                    
                    plt.imsave(f"{self.log}/result_epoch_{epoch}.png", result)
                    plt.imsave(f"{self.log}/heatmap_epoch_{epoch}.png", head_map)
            if target:
                df = pd.DataFrame(log_eval)
                plot_report(df, out_dir=self.log)
                
            return log_eval