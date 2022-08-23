
import torch
from torch import nn
from torch.nn import functional as F
import timm

import pytorch_lightning as pl
import torchmetrics

class Base_CNN(torch.nn.Module):
    def __init__(self, model_name="resnet18", pretrained=False, num_classes=2, in_chans=3, checkpoint_path=''):
        super(Base_CNN,self).__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            checkpoint_path=checkpoint_path
        )
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input_image):
        out = self.model(input_image)
        logits = self.LogSoftmax(out)
        return logits

class CNN(pl.LightningModule):
    def __init__(self, model_name="resnet18", pretrained=False, num_classes=2, in_chans=3, checkpoint_path='', loss_name="NLLLoss", learning_rate=0.1, optimizer_config={"type":"SGD","lr": 0.1,"momentum":0.9}, scheduler_config={"type": "ReduceLROnPlateau","mode": "min","factor": 0.1,"patience": 15}, scheduler_monitor="train_loss_epoch"):
        super(CNN, self).__init__()
        # actual model
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            checkpoint_path=checkpoint_path
        )
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        # training
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.train_cm = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.valid_cm = torchmetrics.ConfusionMatrix(num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(num_classes=num_classes)
        self.valid_f1 = torchmetrics.F1Score(num_classes=num_classes)
        self.loss_criterion = getattr(torch.nn, loss_name)()
        self.learning_rate = learning_rate
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.scheduler_monitor = scheduler_monitor

    def forward(self, input_image):
        out = self.model(input_image)
        logits = self.LogSoftmax(out)
        return logits

    def configure_optimizers(self):
        self.optimizer = getattr(torch.optim, self.optimizer_config["type"])(params=self.parameters(), **{**{k:v for k,v in self.optimizer_config.items() if k!="type"}, **{"lr":self.learning_rate}})
        self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_config["type"])(optimizer=self.optimizer, **{k:v for k,v in self.scheduler_config.items() if k!="type"})
        self.lr_scheduler = {
            'scheduler': self.scheduler,
            'name': 'learning_rate',
            'interval':'epoch',
            'monitor': self.scheduler_monitor,
            'frequency': 1}
        return [self.optimizer], [self.lr_scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_criterion(logits, y)
        y_hat_prob = F.softmax(logits, dim=-1)
        # metrics
        self.train_acc(y_hat_prob, y)
        self.train_cm(y_hat_prob, y)
        self.train_f1(y_hat_prob, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_criterion(logits, y)
        y_hat_prob = F.softmax(logits, dim=-1)
        # metrics
        self.valid_acc(y_hat_prob, y)
        self.valid_cm(y_hat_prob, y)
        self.valid_f1(y_hat_prob, y)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss_train = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch', loss_train, on_step=False, on_epoch=True)
        train_acc = self.train_acc.compute()
        self.log('train_acc_epoch', train_acc, on_step=False, on_epoch=True)
        train_f1 = self.train_f1.compute()
        self.log('train_f1_epoch', train_f1, on_step=False, on_epoch=True)
        lr = float(self.optimizer.param_groups[0]['lr'])
        self.log('learning_rate', lr, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('valid_loss_epoch', loss_val, on_step=False, on_epoch=True)
        valid_acc = self.valid_acc.compute()
        self.log('valid_acc_epoch', valid_acc, on_step=False, on_epoch=True)
        valid_f1 = self.valid_f1.compute()
        self.log('valid_f1_epoch', valid_f1, on_step=False, on_epoch=True)
