"""
Questo modulo contiene gli script fondamentali per interagire con il dataset scelto. Gli script presenti
permettono il caricamento del dataset e l'integrazione con il framework Pytorch.
"""

import os
import shutil
from pathlib import Path
from typing import Tuple, override

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose

from utils import improve_image


class DatasetFactory:
    def __init__(self, out_dir: Path | str) -> None:
        """
        Initialize the DatasetFactory with an output directory.

        Args:
            out_dir (Path | str): Root path where the processed dataset will be saved.
        """
        self.out_dir = Path(out_dir)
    
    def produce_Obs(
        self, src_img_dir: Path | str, src_label_dir: Path | str, split: Path | str
    , copy_size:tuple[int,int]|None = None) -> None:
        """
        Produce a reformatted LostAndFound dataset structure. The output will be structured as:

            datasetPath/
            ├── img/
            └── label/

        Args:
            src_img_dir (Path | str): Path to the original images directory.
            src_label_dir (Path | str): Path to the original labels directory.
            split (Path | str): Subdirectory name under `out_dir` (e.g., "train", "val").
            copy_size (tuple[int, int], optional): If provided, resizes images and labels to this size.
        """
        src_img_dir = Path(src_img_dir)
        src_label_dir = Path(src_label_dir) 
        base_dir = self.out_dir.joinpath(split) 

        img_path = base_dir.joinpath("img")
        label_path = base_dir.joinpath("label")
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        for dir_ in src_img_dir.iterdir():
            for file_ in dir_.glob("*leftImg8bit.png"):
                att = file_.stem.split("_")
                ids = "".join(att[1:3]) + "".join(att[5:7])
               
                nn = ids + file_.suffix
                if copy_size:
                    image = cv.imread(file_)
                    image = cv.resize(image, copy_size, interpolation=cv.INTER_LANCZOS4)
                    cv.imwrite(img_path.joinpath(nn), image)
                else:    
                    shutil.copy2(file_, img_path.joinpath(nn))
                
                

        for dir_ in src_label_dir.iterdir():
            for file_ in dir_.glob("*color.png"):
                att = file_.stem.split("_")
                ids = "".join(att[1:3]) + "".join(att[5:7])
                nn = ids + file_.suffix

                if copy_size:
                    image = cv.imread(file_)
                    image = cv.resize(image, copy_size, interpolation=cv.INTER_NEAREST)
                    cv.imwrite(label_path.joinpath(nn), image)
                     
                else:
                    shutil.copy2(file_, label_path.joinpath(nn))

        return None

    def produce_CS(
        self, src_img_dir: Path | str, src_label_dir: Path | str, split: Path | str
    , copy_size:tuple[int,int]|None = None) -> None:
        """
        Costruisce un nuovo dataset a partire dalla struttura base del dataset Cityscape nella forma

            datasetPath/
            ├── img/
            └── label/

        """
        src_img_dir = Path(src_img_dir)
        src_label_dir = Path(src_label_dir) 
        base_dir = self.out_dir.joinpath(split) 

        img_path = base_dir.joinpath("img")
        label_path = base_dir.joinpath("label")
        os.makedirs(img_path, exist_ok=True)
        os.makedirs(label_path, exist_ok=True)

        for dir_ in src_img_dir.iterdir():

            for file_ in dir_.glob("*leftImg8bit.png"):
                att = file_.stem.split("_")
                ids = "".join(att[1:3])
                idds = att[0]

                nn = ids + idds + file_.suffix
                if copy_size:
                    image = cv.imread(file_)
                    image = cv.resize(image, copy_size, interpolation=cv.INTER_LANCZOS4)
                    cv.imwrite(img_path.joinpath(nn), image)
                else:    
                    shutil.copy2(file_, img_path.joinpath(nn))

        for dir_ in src_label_dir.iterdir():
            for file_ in dir_.glob("*color.png"):
                att = file_.stem.split("_")
                ids = "".join(att[1:3])
                idds = att[0]
                nn = ids + idds + file_.suffix

                if copy_size:
                    image = cv.imread(file_)
                    image = cv.resize(image, copy_size, interpolation=cv.INTER_NEAREST)
                    cv.imwrite(label_path.joinpath(nn), image)
                     
                else:
                    shutil.copy2(file_, label_path.joinpath(nn))

        return None

class CSDataset(Dataset):
    """
    Class for load dataset **Cityscapes** `in expected format`.

    `Expected format`
    
    datasetPath/
        ├── img/
        └── label/
    """

    def __init__(
        self,
        dataset_path: Path | str,
        transform_x: Compose | None = None,
        transform_y: Compose | None = None,
        mask_s: Tuple[int, int] = (2,2)
    ):
        """
        """
        self.base_path = Path(dataset_path)

        if not self.base_path.exists():
            raise FileNotFoundError()

        self.transform_x = transform_x
        self.transform_y = transform_y
        self.mask_s = mask_s
        self._x = list(self.base_path.joinpath("img").glob("*.png"))
        self._y = list(self.base_path.joinpath("label").glob("*.png"))
        self._label = [path.stem for path in self._x]
        
        self._x = sorted(self._x, key=lambda path: path.stem)
        self._y = sorted(self._y, key=lambda path: path.stem)
        self._label.sort()
        

        
    def __len__(self):
        return len(self._label)

    @override
    def __getitem__(self, index) -> Tuple[ndarray, ndarray, int]:
        x = cv.cvtColor(cv.imread(self._x[index]), cv.COLOR_BGR2RGB)
        y = cv.cvtColor(cv.imread(self._y[index]), cv.COLOR_BGR2RGB)
        l = cv.Canny(y, 0.2,0.5)
        
        l = np.asarray(l == 255, dtype=np.uint8)
        l = cv.dilate(l, np.ones(self.mask_s, np.uint8), iterations=1) 
        
        
        

        x = improve_image(x)

        if self.transform_x:
            x = self.transform_x(x)

        if self.transform_y:
            y = self.transform_y(y)
        
        _, h, w = y.shape
        l = cv.resize(l, dsize=(w,h), interpolation=cv.INTER_LINEAR)
        l = torch.as_tensor(l, dtype=torch.float32).unsqueeze(0)
        

        return (x, y, l)


if __name__ == "__main__":
    #DatasetFactory("CSF720x288").produce_CS("Datasets/CSF/train/img", "Datasets/CSF/train/label", "train", copy_size=(720,288))
    #DatasetFactory("CSF720x288").produce_CS("Datasets/CSF/val/img", "Datasets/CSF/val/label", "val", copy_size=(720,288))
    DatasetFactory("LAF720x288").produce_Obs("Datasets/LAF/train/img", "Datasets/LAF/train/label", "train", copy_size=(720,288))
    DatasetFactory("LAF720x288").produce_Obs("Datasets/LAF/val/img", "Datasets/LAF/val/label", "val", copy_size=(720,288))
    

