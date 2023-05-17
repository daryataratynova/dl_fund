import lightning as L
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
import pandas as pd
import torch
from shared_utilities import LightningModel, MNISTDataModule, PyTorchMLP


train_val_diff = []

class CallbackTrainValDiff(L.Callback):
    def on_train_epoch_end(self, trainer, LightningModel):
        diff = LightningModel.train_acc.compute() - LightningModel.val_acc.compute()
        train_val_diff.append(diff)

if __name__ == "__main__":

    torch.manual_seed(123)

    dm = MNISTDataModule()

    pytorch_model = PyTorchMLP(num_features=784, num_classes=10)

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    trainer = L.Trainer(
        max_epochs=10, accelerator="cpu", devices=1, deterministic=True,
        logger=CSVLogger(save_dir="logs/", name="my-model"), callbacks=[CallbackTrainValDiff()]
    )
    trainer.fit(model=lightning_model, datamodule=dm)

    train_acc = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_acc"]
    val_acc = trainer.validate(datamodule=dm)[0]["val_acc"]
    test_acc = trainer.test(datamodule=dm)[0]["test_acc"]
    print(
        f"Train Acc {train_acc*100:.2f}%"
        f" | Val Acc {val_acc*100:.2f}%"
        f" | Test Acc {test_acc*100:.2f}%"
    )

for diff in train_val_diff:
    print(f"Train-Validation accuracy difference: {diff*100:.2f}%",)
