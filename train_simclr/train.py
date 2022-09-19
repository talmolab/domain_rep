import torch
import torchvision
from torchvision.models import alexnet
import argparse
import wandb
import simclr_alexnet
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", action = "store", help="Path to input dataset", type=str)
    parser.add_argument('-bs', '--batch_size', default='4096', action='store', help="Batch size during training", type=int)
    parser.add_argument("--temp", default=0.1, action='store', help="Temperature for NXent Loss", type=float)
    parser.add_argument("-lr", "--learning_rate", action='store', default=4.8, help="Learning Rate to be used", type=float)  
    parser.add_argument("-m", "--momentum", action='store', default=0.9, help="Momentum to be used in optimization", type=float)
    parser.add_argument("-ed", "--embedding_dim", action='store', default = 128, help="Embedding Dimension size for SimCLR projection head MLP", type=int)
    parser.add_argument("-i", "--input_size", action='store', default = 64, help="input image size", type=int)
    parser.add_argument("-wd", "--weight_decay", action='store', default = 1e-6, help="Weight Decay to be used", type=float)
    parser.add_argument("-n", "--n_gpus", action='store', default=0, help="Number of GPUS to be used during training", type=int)
    parser.add_argument("-rn", "--run_name", action='store', default="train_simclr", help="Run Name for wandb logging", type=str)
    parser.add_argument("-id", "--log_id", action='store', default="simclr_model", help="Run ID for wandb", type=str)
    parser.add_argument("-me", "--max_epochs", action='store', default=600, help="Maximum number of epochs to be run during training", type=int)
    parser.add_argument("-r", "--resume", action='store', default=None, help="Path to checkpoint to resume training from", type=str)
    args = parser.parse_args()
    simclr_model = simclr_alexnet.SimCLRModel(dataset_path = args.dataset_path,
                                              transform = None,
                                              batch_size = args.batch_size,
                                              temp = args.temp,
                                              learning_rate = args.learning_rate,
                                              momentum = args.momentum,
                                              embedding_dim = args.embedding_dim,
                                              input_size = args.input_size,
                                              weight_decay = args.weight_decay,
                                              )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print('GPU is not available. Using CPU')
    logger = WandbLogger(name=args.run_name, 
                         id=args.log_id, 
                         project = 'rep_data', 
                         log_model='all'
                        )
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='train_loss_ssl', 
                                              mode='min', 
                                              save_last=True, 
                                              every_n_epochs = 50)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    logger.watch(simclr_model, log_graph=False)
    trainer = pl.Trainer(gpus=args.n_gpus, 
                         strategy="dp", 
                         max_epochs=args.max_epochs, 
                         progress_bar_refresh_rate=100, 
                         logger=logger, 
                         callbacks=[checkpoint, lr_monitor], 
                         log_every_n_steps=1)
    trainer.fit(simclr_model, ckpt_path=args.resume)
    wandb.finish()