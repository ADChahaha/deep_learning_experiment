from lightning.pytorch.cli import LightningCLI
from modelmodule import MyModel
from datamodule import MyDataModule
from utils import load_glove


if __name__ == "__main__":
    LightningCLI(MyModel, MyDataModule)
