from lightning.pytorch.cli import LightningCLI
from modelmodule import MyModel
from datamodule import MyDataModule


if __name__ == "__main__":
    LightningCLI(MyModel, MyDataModule)


    
    
