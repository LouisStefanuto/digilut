import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import F1Score


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        pos_weight: float = 1.0,
        lr: float = 1e-3,
    ):
        super(ClassificationModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the original classifier

        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.threshold = threshold

        self.lr = lr

        # Instantiate the loss function
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))
        self.f1score = F1Score("binary", threshold=0.5)

        # Initialize a list to hold validation outputs
        self.train_losses = []  # type:ignore
        self.val_outputs = []  # type:ignore

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images).squeeze(dim=1)  # Remove the second dimension
        labels = labels.float()  # Ensure labels are of type float
        loss = self.loss_fn(logits, labels)

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images).squeeze(dim=1)
        labels = labels.float()  # Ensure labels are of type float
        loss = self.loss_fn(logits, labels)
        preds = torch.sigmoid(logits) > self.threshold

        # Convert tensors to numpy arrays for f1_score calculation
        f1 = self.f1score(labels, preds)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_f1_score",
            f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
