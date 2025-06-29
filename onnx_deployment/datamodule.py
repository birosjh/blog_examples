import torch
import lightning as L
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import CIFAR10
from torchvision import transforms

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [45000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=32, num_workers=6, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=32, num_workers=6)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=1, num_workers=6)