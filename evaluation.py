from random import randint
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
from torchmetrics.classification import AUROC, AveragePrecision
from torch.utils.data import Dataset
from utils import (\
    binary_loss_threshold,
    bmap2Image,
    calibration2,
    compute_varisco_heatmap_rgb,
    map2Image,
    overlay_heatmap,
    plot_report,
    uncertainty_loss_simple,
    unknownObjectnessScore, unknownObjectnessScoreMB,
    img2UQH,
    miscoverage_loss
)


class ObstacleBenchmark:
    """
    Benchmark based on LostAndFound dataset
    """
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

        
        
    def run_benchmark(self, test_dataset:Dataset,loss_fn:Module|None = None, keep_index:List[int] = [], format:str="one-hot", map_cls_to_color:Dict[int,Tuple[int,int,int]]|None =None, device:str="cpu", repeat_for:int= 1, target:bool=True):
        
        with torch.no_grad():
            
            dice = DiceScore(num_classes=len(keep_index), average="macro", input_format=format)
            iou  = MeanIoU(num_classes=len(keep_index), input_format=format, per_class=False)
            #ap = MultilabelAveragePrecision(len(keep_index),"macro", **{"compute_on_cpu": True})
            auroc = AUROC(task="binary", **{"compute_on_cpu": True})
            ap = AveragePrecision(task="binary", **{"compute_on_cpu": True})
            #auroc = MultilabelAUROC(len(keep_index),"macro", **{"compute_on_cpu": True})
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=True,
                pin_memory=True,
                num_workers=6,
                persistent_workers=True
            
            )

            valid_size = len(test_dataset)
            log_eval = {
                
                
                "diceScore":    [],
                "IoU":          [],
                "mAP":        [],
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

                bar.set_postfix({
                    "lambda" : 0,
                    "dice_score": 0,
                    "IoU": 0,
                    "mAP" : 0.0,
                    "AUROC": 0.0
                    
                }, refresh=True)
                dice = dice.to(device)
                iou  = iou.to(device)
                ap = ap.to(device)
                auroc = auroc.to(device)

                dice.reset()
                iou.reset()
                ap.reset()
                auroc.reset()

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

                    
                    # Select Only subset of classes
                    if len(keep_index) > 0:
                        y_pred = y_pred[:, keep_index, :, :]
                        
                        if target:
                            y = y[:, keep_index, :, :]

                    if target:
                        y_classes = (torch.argmax(y_pred, dim=1) if format == "index" else F.one_hot(torch.argmax(y_pred, dim=1), num_classes=y_pred.shape[1]).permute(0, 3, 1, 2))
                        dice.update(y_classes, y.long())
                        iou.update(y_classes, y.long())
                        
                        ap.update(unknownObjectnessScoreMB(y_pred).squeeze(1), y[:,-1,:,:].long())
                        auroc.update(unknownObjectnessScoreMB(y_pred).squeeze(1), y[:,-1,:,:].long())

                        
                        bar.set_postfix({
                            #"lambda" : lambda_,
                            "dice_score": dice.compute().numpy(force=True),
                            "IoU": iou.compute().numpy(force=True),
                            "mAP": ap.compute().numpy(force=True),
                            "AUROC": auroc.compute().numpy(force=True)
                            
                        }, refresh=True)
                        
                        loss = loss_fn(y_pred, y, m)

                        loss_v = loss.item()
                        epoch_loss += loss_v

                    
                    #
                    bar.update(len(x))
                


                bar.close()
                
                if target:
                    lambda_ = calibration2(Y=y.long(), Z=y_pred, B=1, alpha=0.99, loss_fn=miscoverage_loss, verbose=False, num_points=2000)
                    log_eval["diceScore"].append(dice.compute().numpy(force=True))
                    log_eval["IoU"].append(iou.compute().numpy(force=True))
                    log_eval["mAP"].append(ap.compute().numpy(force=True))
                    log_eval["AUROC"].append(auroc.compute().numpy(force=True)) # TODO: aggiornare
                    log_eval["epoch"].append(epoch)               #  

                x, y, _ = test_dataset[randint(0, len(test_loader.dataset))]
                
                x = torch.unsqueeze(x, 0).to(device)
                y = torch.unsqueeze(y, 0).to(device)
                y_pred = model(x)

                
                if map_cls_to_color:
                    
                    uos = unknownObjectnessScore(y_pred)
                    
                    # Remove object layer if exist
                    y_pred_full = y_pred
                    y_pred = y_pred[:, :len(map_cls_to_color), :, :]
                    if target :
                        y = y[:, :len(map_cls_to_color), :, :]

                    
                    
                    y_pred_full = y_pred_full.cpu().squeeze(0).numpy()
                    img = x.cpu().squeeze(0).numpy()
                    y =   y.cpu().squeeze(0).numpy()
                    y_pred = y_pred.cpu().squeeze(0).numpy()
                    
                    
                    if format == "index":
                        y_pred_img = map2Image(map_cls_to_color, y_pred)
                    else:
                        y_pred_prb = np.argmax(y_pred, axis=0)
                        y_pred_img = bmap2Image(map_cls_to_color, y_pred_prb)


                    if target:
                        uhm = img2UQH(img, y_pred_full, lambda_)
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

                    head_map = np.hstack( (np.transpose(img * 255, (1, 2, 0)), varisco_heatmap, overlay, uhm) ).astype(np.uint8)
                    
                    plt.imsave(f"{self.log}/result_epoch_{epoch}.png", result)
                    plt.imsave(f"{self.log}/heatmap_epoch_{epoch}.png", head_map)
            if target:
                df = pd.DataFrame(log_eval)
                plot_report(df, out_dir=self.log)
        
            return log_eval