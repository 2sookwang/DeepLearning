# Weights & Biases
import wandb
from pytorch_lightning.loggers import WandbLogger

# Pytorch modules
import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl

# Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms

class LitCIFAR10(LightningModule):

    def __init__(self, n_classes=10, lr=1e-3):
        '''method used to define our model parameters'''
        super().__init__()

        # cifar10 images are (3, 32, 32) (channels, width, height)
        self.conv1 = torch.nn.Conv2d(3, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.layer_1 = torch.nn.Linear(5*5*50, 500)
        self.dropout = torch.nn.Dropout(0.5)
        self.layer_2 = torch.nn.Linear(500, 10)

        self.lr = lr
        # metrics
        self.accuracy = pl.metrics.Accuracy()

        # optional - save hyper-parameters to self.hparams
        # they will also be automatically logged as config parameters in W&B
        self.save_hyperparameters()

    def forward(self, x):
        '''method used for inference input -> output'''

        batch_size, channels, width, height = x.size()

        # (b, 1, 32, 32) -> (b, 1*32*32)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 5*5*50)
        x = F.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.layer_2(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        '''needs to return a loss from a single batch'''
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # Log training loss
        self.log('train_loss', loss)

        # Log metrics
        #self.log('train_acc', self.accuracy(logits, y))

        return loss

    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # Log validation loss (will be automatically averaged over an epoch)
        self.log('valid_loss', loss)

        # Log metrics
        #self.log('valid_acc', self.accuracy(logits, y))

    def test_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # Log test loss
        self.log('test_loss', loss)

        # Log metrics
        #self.log('test_acc', self.accuracy(logits, y))

    def configure_optimizers(self):
        '''defines model optimizer'''
        return Adam(self.parameters(), lr=self.lr)

class CIFAR10DataModule(LightningDataModule):

    def __init__(self, data_dir='./', batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            cifar10_train = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_train, [45000, 5000])
        if stage == 'test' or stage is None:
            self.cifar10_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        '''returns training dataloader'''
        cifar10_train = DataLoader(self.cifar10_train, batch_size=self.batch_size)
        return cifar10_train

    def val_dataloader(self):
        '''returns validation dataloader'''
        cifar10_val = DataLoader(self.cifar10_val, batch_size=self.batch_size)
        return cifar10_val

    def test_dataloader(self):
        '''returns test dataloader'''
        cifar10_test = DataLoader(self.cifar10_test, batch_size=self.batch_size)
        return cifar10_test

wandb.login()
wandb_logger = WandbLogger(project='2022317007_Leesookwang_cifar10')

# setup data
cifar10 = CIFAR10DataModule()

# setup model - choose different hyperparameters per experiment
model = LitCIFAR10(lr=1e-3)

trainer = Trainer(
    logger=wandb_logger,    # W&B integration
    gpus=1,                # use all GPU's
    max_epochs=20            # number of epochs
    )

trainer.fit(model, cifar10)
trainer.test(model, datamodule=cifar10)

wandb.finish()
