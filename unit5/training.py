import lightning as L
import torch
from shared_utilities import LightningModel, AmesHousingDataModule, PyTorchMLP

#Train mse 0.24 | Val mse 0.24 | Test mse 0.29 with 3*25+25+25*1+1 = 26 params
#Train mse 0.18 | Val mse 0.15 | Test mse 0.17 with 3*25+25+25*1+1 = 126 params
#
if __name__ == "__main__":

    torch.manual_seed(123)

    dm = AmesHousingDataModule()

    pytorch_model = PyTorchMLP(num_features=3)

    lightning_model = LightningModel(model=pytorch_model, learning_rate=0.05)

    trainer = L.Trainer(
        max_epochs=10, accelerator="cpu", devices="auto", deterministic=True, default_root_dir="~/Desktop/DL_course/unit5"
    )
    trainer.fit(model=lightning_model, datamodule=dm)

    train_mse = trainer.validate(dataloaders=dm.train_dataloader())[0]["val_mse"]
    val_mse = trainer.validate(datamodule=dm)[0]["val_mse"]
    test_mse = trainer.test(datamodule=dm)[0]["test_mse"]
    print(
        f"Train mse {train_mse:.2f}"
        f" | Val mse {val_mse:.2f}"
        f" | Test mse {test_mse:.2f}"
    )