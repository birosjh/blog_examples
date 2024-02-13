import hydra
from lightning.pytorch import Trainer

from fasion_mnist_dm import FashionMNISTDataModule
from resnet_lm import ResnetLightningModule


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(config):

    datamodule = FashionMNISTDataModule(config.datamodule)

    model = ResnetLightningModule(config.model)

    trainer = Trainer()
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()