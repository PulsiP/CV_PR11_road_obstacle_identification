# log tools
from typing import Any
import pandas as pd
import torch
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.segmentation import DiceScore
from tqdm.auto import tqdm
from data import CSDataset
from network import FCN, FCNParams
from utils import ToMasck, show_tensor


from matplotlib import pyplot as plt


class TrainNetwork:
    """
    Classe specializzata nell'effettuare l'attività di traning per la rete proposta. Tutti i modelli conformi alle specifiche di rete
    in questo progetto possono utilizzare la classe per effettuare il traning e valutazione contemporaneo del modello secondo i criteri
    configurati
    """

    def __init__(self, hp: dict[str, Any], model: nn.Module) -> None:
        """
        Inizializza il trainer
        """
        self._model: nn.Module = model
        self._loss: nn.Module = hp["loss"]
        self._optimizer: Optimizer = hp["optimizer"]

    def __train(
        self, dataloader_train, model, epoch, loss_fn, device, log_train, train_size
    ):
        ##################
        # Training Phase #
        ##################

        epoch_loss = 0
        loss = 0
        batch = 0
        model.train()
        bar = tqdm(
            desc="training", total=train_size  # type: ignore
        )  # uses len(dataset) instead of dataset.size

        bar.set_postfix(
            {
                "batch": 0,
                "loss": 0,
            }
        )

        for x, y, _ in dataloader_train:
            bar.set_description(f"training epoch: {epoch}", refresh=True)

            x = x.to(device)
            y = y.squeeze(1).to(device)

            batch += 1
            batch_len = len(y)

            self._optimizer.zero_grad()
            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            loss_v = loss.item()
            epoch_loss += loss_v
            loss.backward()

            self._optimizer.step()

            bar.set_postfix(
                {
                    "batch": batch,
                    "loss": epoch_loss / batch,
                }
            )
            bar.update(batch_len)

        bar.close()
        log_train["epoch"].append(epoch)
        log_train["loss"].append(epoch_loss / batch)
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
            }
        )

        for x, y, _ in dataloader_val:

            with torch.no_grad():
                bar.set_description(f"validation epoch: {epoch}", refresh=True)

                x = x.to(device)
                y = y.squeeze(1).to(device)

                batch += 1
                batch_len = len(y)

                # print(f"label shape {y.shape}")
                y_pred = model(x)

                loss = loss_fn(y_pred, y)
                loss_v = loss.item()
                epoch_loss += loss_v

                bar.set_postfix(
                    {
                        "batch": batch,
                        "loss": epoch_loss / batch,
                    }
                )
                bar.update(batch_len)

        bar.close()
        log_eval["epoch"].append(epoch)
        log_eval["loss"].append(epoch_loss / batch)

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

            # save best weigths in agreement with objective value
            if len(log_eval["epoch"]) < 2 or (
                (massimize and log_eval[objective][-1] > log_eval[objective][-2])         # if objective will be massimize(>) get the best weights that massimize it
                or (not massimize and log_eval[objective][-1] < log_eval[objective][-2])  # if objective will be minimize(<) get  save the best weigths  that minimize it
            ):
                if old_weights: # old only the best weigths
                    os.remove(old_weights)

                torch.save(
                    self._model.state_dict(),
                    log_dir.joinpath(f"best_weights_epoch_{epoch}.pth"),
                )
                old_weights = log_dir.joinpath(f"best_weights_epoch_{epoch}.pth")

            # Qalitative learning rapresentation
            model.eval()
            with torch.no_grad():

                x, y, _ = validation_set[0]
                x = x.unsqueeze(0).to(device)

                y_pred = model(x).argmax(dim=1).cpu().numpy().squeeze(0)
                y = y.cpu().squeeze(0).numpy()

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


if __name__ == "__main__":
    from torchvision.transforms import v2

    # TODO: add normalization for input images
    data_train = CSDataset(
        "Dataset/train",
        transform_x=v2.Compose(
            [v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True)]
        ),
        transform_y=v2.Compose([ToMasck(0, 255)]),
    )

    data_test = CSDataset(
        "Dataset/val",
        transform_x=v2.Compose(
            [v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True)]
        ),
        transform_y=v2.Compose([ToMasck(0, 255)]),
    )
    param = FCNParams(256)
    model = FCN(param)

    hyper_parameters = {
        "loss": nn.CrossEntropyLoss(),
        "optimizer": torch.optim.Adamax(model.parameters()),
    }

    trainer = TrainNetwork(hyper_parameters, model)
    m, _, _ = trainer.train(data_train, data_test, "./FCN", epochs=10, batch_size=64)
