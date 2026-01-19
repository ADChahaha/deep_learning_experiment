import torch
import torch.nn as nn
import lightning
from model import AutoEncoder_Net


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')
    
    def forward(self, data, label):
        oriData, _, reconData = data
        return self.mse(oriData, reconData)


class MyModel(lightning.LightningModule):
    def __init__(self, in_channel, embed_channel, lr=1e-3):
        super().__init__()
        self.model = AutoEncoder_Net(in_channel, embed_channel)
        self.criterion = VAELoss()
        self.lr = lr

    def training_step(self, batch):
        img, _ = batch
        x, lowData, reconData = self.model(img)
        
        # 打包成 data 传给 loss
        data = (x, lowData, reconData)
        loss = self.criterion(data, None)
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def validation_step(self, batch):
        img, _ = batch
        x, lowData, reconData = self.model(img)
        
        data = (x, lowData, reconData)
        loss = self.criterion(data, None)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)