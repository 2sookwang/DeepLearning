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
from torchvision.datasets import MNIST
from torchvision import transforms

class LitMNIST(LightningModule):

    def __init__(self, n_classes=10, n_layer_1=128, n_layer_2=256, lr=1e-3):
        '''method used to define our model parameters'''
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, n_layer_1)
        self.layer_2 = torch.nn.Linear(n_layer_1, n_layer_2)
        self.layer_3 = torch.nn.Linear(n_layer_2, n_classes)

        # optimizer parameters
        self.lr = lr

        # metrics
        self.accuracy = pl.metrics.Accuracy()

        # optional - save hyper-parameters to self.hparams
        # they will also be automatically logged as config parameters in W&B
        self.save_hyperparameters()

    def forward(self, x):
        '''method used for inference input -> output'''

        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

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

class MNISTDataModule(LightningDataModule):

    def __init__(self, data_dir='./', batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        '''returns training dataloader'''
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
        return mnist_train

    def val_dataloader(self):
        '''returns validation dataloader'''
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
        return mnist_val

    def test_dataloader(self):
        '''returns test dataloader'''
        mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size)
        return mnist_test

wandb.login()
wandb_logger = WandbLogger(project='2022317007_Leesookwang_mnist')

# setup data
mnist = MNISTDataModule()

# setup model - choose different hyperparameters per experiment
model = LitMNIST(n_layer_1=128, n_layer_2=256, lr=1e-3)

trainer = Trainer(
    logger=wandb_logger,    # W&B integration
    gpus=1,                # use all GPU's
    max_epochs=20            # number of epochs
    )

trainer.fit(model, mnist)
trainer.test(model, datamodule=mnist)

wandb.finish()
