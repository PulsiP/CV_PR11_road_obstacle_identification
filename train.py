# log tools
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.segmentation import DiceScore
from tqdm.auto import tqdm

from data import CSDataset
from network import FCN


class TrainNetwork:
    """
    Classe specializzata nell'effettuare l'attività di traning per la rete proposta. Tutti i modelli conformi alle specifiche di rete
    in questo progetto possono utilizzare la classe per effettuare il traning e valutazione contemporaneo del modello secondo i criteri
    configurati
    """

    def __init__(self, model: nn.Module, loss: nn.Module, optimizer: Optimizer) -> None:
        """
        Inizializza il trainer
        """
        self._model = model
        self._loss = loss
        self._optimizer = optimizer

    def train(
        self,
        train_set: Dataset,
        validation_set: Dataset,
        log_dir: Path | str,
        epochs: int = 1,
        batch_size: int = 1,
    ):
        """
        Train method, effettua il traning della rete specificata utilizzando come datasets di train e validazione quelli specificati.
        La funzione supporta un semplice metodo automatico di loging
        Args:

        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log_dir = Path(log_dir)
        dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(
            validation_set, batch_size=batch_size, shuffle=False
        )
        log_train = {
            "epoch": [],
            "loss": [],
        }

        log_eval = {
            "epoch": [],
            "loss": [],
        }

        model = self._model.to(device)
        loss_fn = self._loss.to(device)

        for epoch in range(1, epochs + 1):

            ##################
            # Training Phase #
            ##################

            epoch_loss = 0
            loss = 0
            batch = 0

            bar = tqdm(
                desc="training", total=len(train_set)  # type: ignore
            )  # uses len(dataset) instead of dataset.size

            bar.set_postfix(
                {
                    "batch": batch,
                    "loss": loss,
                }
            )

            for x, y, _ in dataloader_train:
                bar.set_description(f"training epoch: {epoch}", refresh=True)

                x = x.to(device)
                y = y.to(device)
                batch += 1
                batch_len = len(y)

                self._optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_fn(y, y_pred)
                epoch_loss += loss.item()
                loss.backward()

                self._optimizer.step()
                bar.update(batch_len)

            log_train["epoch"].append(epoch)
            log_train["loss"].append(epoch_loss / batch)

            ####################
            # Validation Phase #
            ####################

            epoch_loss = 0
            loss = 0
            batch = 0

            bar.close()

            bar = tqdm(
                desc="validation", total=len(validation_set)  # type: ignore
            )  # uses len(dataset) instead of dataset.size

            bar.set_postfix(
                {
                    "batch": batch,
                    "loss": loss,
                }
            )

            for x, y, _ in dataloader_val:

                with torch.no_grad():
                    bar.set_description(f"validation epoch: {epoch}", refresh=True)

                    x = x.to(device)
                    y = y.to(device)
                    batch += 1
                    batch_len = len(y)

                    y_pred = model(x)
                    loss = loss_fn(y, y_pred)
                    epoch_loss += loss.item()

                    bar.update(batch_len)

            log_eval["epoch"].append(epoch)
            log_eval["loss"].append(epoch_loss / batch)
            bar.close()

        ###########
        # Logging #
        ###########

        pd.DataFrame(log_eval).to_csv(
            log_dir.joinpath("eval.csv"), sep=",", index=False
        )
        pd.DataFrame(log_train).to_csv(
            log_dir.joinpath("train.csv"), sep=",", index=False
        )


if __name__ == "__main__":
    from torchvision.transforms.v2 import Compose, ToTensor

    data_train = CSDataset("Dataset/train", Compose([ToTensor()]))
    data_test = CSDataset("Dataset/val")
    model = FCN(4)
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    trainer = TrainNetwork(model, loss, optim)
    trainer.train(data_train, data_test, ".", epochs=10, batch_size=8)
