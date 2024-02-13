import lightning as L

from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class FashionMNISTDataModule(L.LightningDataModule):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.prepare_data_per_node = True

    def prepare_data(self):
        # download
        FashionMNIST("data", train=True, download=True)
        FashionMNIST("data", train=False, download=True)

    def setup(self, stage):

        self.train_dataset = FashionMNIST("data", train=True, transform=ToTensor())

        self.val_dataset = FashionMNIST("data", train=False, transform=ToTensor())

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )