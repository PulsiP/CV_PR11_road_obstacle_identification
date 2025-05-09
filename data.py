"""
    Questo modulo contiene gli script fondamentali per interagire con il dataset scelto. Gli script presenti
    permettono il caricamento del dataset e l'integrazione con il framework Pytorch.
"""
import os
import shutil
from pathlib import Path
from typing import Tuple, override

import cv2 as cv
from numpy import ndarray
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose


class DatasetFactory:
    def __init__(self, out_dir:Path|str) -> None:
        self.out_dir = Path(out_dir)

    def produce(self, src_img_dir:Path|str, src_label_dir:Path|str, split:Path|str) -> None:
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
           
            for file_ in dir_.glob("*eftImg8bit.png"):
                att = file_.stem.split("_")
                ids = "".join(att[1:3])
                idds = att[0]
            
                nn = ids+idds + file_.suffix
               
                shutil.copy2(file_, img_path.joinpath(nn))
             
        for dir_ in src_label_dir.iterdir():
            for file_ in dir_.glob("*color.png"):
                att = file_.stem.split("_")
                ids = "".join(att[1:3])
                idds = att[0]
                nn = ids+idds + file_.suffix
               
                shutil.copy2(file_, label_path.joinpath(nn))
                
        return None

class CSDataset(Dataset):
    """
        Classe per il caricamento del dataset **Cityscapes_kaggle** `valido`. Un dataset
        è definito valido se risulta coerente con la seguente struttura:

        datasetPath/
            ├── img/
            └── label/
    """

    def __init__(self, dataset_path:Path, transform:Compose|None = None):

        """
        
            Args:

            ...
        """
        self.base_path = dataset_path

        if not self.base_path.exists():
            raise FileNotFoundError()
        
        self.transform = transform
        self._x = list(self.base_path.joinpath("img").glob("*.png"))
        self._y = list(self.base_path.joinpath("label").glob("*.png"))
        self._label  = [path.stem for path in self._x]
        
    
    
    def __len__(self):
        return len(self._label)
    
    @override
    def __getitem__(self, index) -> Tuple[ndarray, ndarray, int]:
        x = cv.cvtColor(cv.imread(self._x[index]), cv.COLOR_BGR2RGB)
        y = cv.cvtColor(cv.imread(self._y[index]), cv.COLOR_BGR2RGB)
        l = self._label[index]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return (x, y, l)
    

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    builder = DatasetFactory("LDataset")
    builder.produce("Dataset_full/imgs/val", "Dataset_full/label/val", 'val')
    builder.produce("Dataset_full/imgs/train", "Dataset_full/label/train", 'train')
    exit(0)

    path = Path("./dataset/train")
    d = CSDataset(path)

    i = 0

    for img, _, _ in d:
        if i > 5:
            break 
        
        plt.imshow(img)
        plt.show()

        i +=1
