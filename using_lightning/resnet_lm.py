import torch
import torch.nn as nn
import lightning as L
from torchvision.models import resnet18

from torchmetrics import Accuracy


class ResnetLightningModule(L.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.model = resnet18(weights='IMAGENET1K_V1')

        # Change the first layer to use only 1 channel for FashionMNIST
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Change the fully connected layer to output 10 features, one for each MNIST class
        self.model.fc = nn.Linear(in_features=512, out_features=config.num_classes, bias=True)

        self.loss_function = nn.CrossEntropyLoss()

        self.metric = Accuracy(task="multiclass", num_classes=config.num_classes)

        self.training_outputs = []
        self.validation_outputs = []

        self.training_ground_truths = []
        self.validation_ground_truths = []
        

    def forward(self, inputs):
        return self.model(inputs)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.loss_function(output, target)

        self.log("train_loss", loss)

        self.training_outputs.append(output)
        self.training_ground_truths.append(target)
        
        return loss
    
    def on_train_epoch_end(self):
        all_preds = torch.cat(self.training_outputs, dim=0)
        all_targets = torch.cat(self.training_ground_truths, dim=0)

        accuracy = self.metric(all_preds, all_targets)

        self.log("train_accuracy", accuracy)

        self.training_outputs.clear()

    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs)
        loss = self.loss_function(output, target)

        self.log("val_loss", loss)

        self.validation_outputs.append(output)
        self.validation_ground_truths.append(target)
        
        return loss
    
    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_outputs, dim=0)
        all_targets = torch.cat(self.validation_ground_truths, dim=0)

        accuracy = self.metric(all_preds, all_targets)

        self.log("val_accuracy", accuracy)

        self.validation_outputs.clear()
        self.validation_ground_truths.clear()

