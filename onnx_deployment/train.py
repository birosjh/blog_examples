import torch

from lightning import Trainer

from model import ResnetModule
from datamodule import CIFAR10DataModule

def main():

    torch.set_float32_matmul_precision('medium')

    model = ResnetModule()
    datamodule = CIFAR10DataModule()

    trainer = Trainer(
        max_epochs=30,
        accelerator='gpu',
        devices=1,
        logger=False,
        enable_progress_bar=True
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":

    main()