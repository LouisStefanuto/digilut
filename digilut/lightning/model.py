import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchmetrics.classification import BinaryAUROC
from torchvision import models


class PatchClassificationModel(pl.LightningModule):
    def __init__(
        self,
        class_weights: torch.Tensor,
        learning_rate: float,
        dropout: float,
    ):
        self.save_hyperparameters()
        super(PatchClassificationModel, self).__init__()
        self.model = models.resnet18(weights="DEFAULT")

        # Two-layer classifier with ReLU activation
        self.dropout = dropout
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(self.model.fc.in_features, 1),
            # torch.nn.Linear(self.model.fc.in_features, 128),
            # torch.nn.BatchNorm1d(128),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(self.dropout),
            # torch.nn.Linear(128, 32),
            # # torch.nn.BatchNorm1d(32),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(self.dropout),
            # torch.nn.Linear(32, 1),
        )

        # Calculate the pos_weight
        self.criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        # self.criterion = FocalLoss(weight=class_weights)
        self.learning_rate = learning_rate

        # Initialize the AUC metric
        self.val_auc = BinaryAUROC()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img_paths, inputs, labels = batch
        labels = labels.float().view(-1, 1)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.sigmoid(outputs) > 0.5
        preds = preds.int()
        acc = torch.sum(preds == labels.data).float() / len(labels)
        f1 = f1_score(labels.cpu(), preds.cpu())
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img_paths, inputs, labels = batch
        labels = labels.float().view(-1, 1)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.sigmoid(outputs) > 0.5
        preds = preds.int()
        acc = torch.sum(preds == labels.data).float() / len(labels)
        f1 = f1_score(labels.cpu(), preds.cpu())

        # Compute AUC
        auc = self.val_auc(outputs, labels.int())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True)
        self.log("val_auc", auc, on_epoch=True, prog_bar=True)  # Log the AUC metric
        return loss

    def predict_step(self, batch, batch_idx):
        _, inputs, _ = batch
        return self(inputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
