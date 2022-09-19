import torch
import torchvision
import lightly
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models import alexnet
from pl_bolts.optimizers import LARS
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss

class SimCLRModel(pl.LightningModule):
    def __init__(self,
                 dataset_path,
                 backbone = alexnet(pretrained=False),
                 transform = torchvision.transforms.Resize((64,64)),
                 batch_size = 512,
                 temp = 0.1, 
                 learning_rate = 1e-2,  
                 momentum = 0.9, 
                 embedding_dim = 128,
                 input_size = 64,
                 weight_decay = 1e-6):
        super().__init__()
        hidden_dim = 9216
        self.backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.dataset_path = dataset_path
        self.transform = transform
        self.batch_size = batch_size
        self.temp = temp
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, embedding_dim)
        self.criterion = NTXentLoss(temperature=temp)
        self.collate_fn = lightly.data.SimCLRCollateFunction(input_size=input_size,gaussian_blur=0.0,cj_prob=0.0)
        self.save_hyperparameters()
    def forward(self, x):
        h=self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = LARS(params=self.parameters(),lr=self.lr,momentum=self.momentum,weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10, threshold=0.1, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
        monitor =  {"scheduler": scheduler, "monitor": "train_loss_ssl",'interval':'epoch',"frequency":10}
        return [optim],[monitor]

    def train_dataloader(self):
        #normalize={'mean':[0, 0, 0],'std':[1, 1, 1]})
        dataset = lightly.data.LightlyDataset(input_dir=self.dataset_path,transform=self.transform)
        return DataLoader(
                        dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        collate_fn=self.collate_fn,
                        drop_last=True,
                        num_workers=8,
                        pin_memory=False
                    )