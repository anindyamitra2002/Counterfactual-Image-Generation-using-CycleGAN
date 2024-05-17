import torch
import torch.nn as nn
import lightning as pl
import torchvision.models as models
from torchmetrics import Accuracy



class Classifier(pl.LightningModule):
    def __init__(self, transfer=True):
        super(Classifier, self).__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)  # Adjust input channels to 3
        self.model = models.efficientnet_b1(weights='IMAGENET1K_V1')
        if transfer:
            # layers are frozen by using eval()
            self.model.eval()
            # freeze params
            for p in self.model.parameters() : 
                p.requires_grad = False
        num_ftrs = 1280
        self.model.classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(p=0.3), 
            nn.Linear(in_features=num_ftrs , out_features=2),
            nn.Softmax(dim=1)  
        )

        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')

    def forward(self, x):
        x = self.conv(x)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        # Calculate and log accuracy
        _, preds = torch.max(outputs, 1)
        acc = self.train_accuracy(preds, labels)
        self.log('train_acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        # Calculate and log accuracy
        _, preds = torch.max(outputs, 1)
        acc = self.val_accuracy(preds, labels)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        self.val_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            },
            'monitor': 'val_loss'
        }