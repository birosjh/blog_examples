import torch
import hydra
from lightning.pytorch import Trainer

from fasion_mnist_dm import FashionMNISTDataModule
from resnet_lm import ResnetLightningModule


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(config):

    datamodule = FashionMNISTDataModule(config.datamodule)

    model = ResnetLightningModule(config.model)

    trainer = Trainer(**config.trainer)
    trainer.fit(model, datamodule)

    input_sample = torch.ones(1, 3, 28, 28)
    model.to_onnx("export.onnx", input_sample, export_params=True)


if __name__ == "__main__":
    train()