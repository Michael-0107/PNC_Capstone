import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from Hypers import Config


class Trainer:
    def __init__(self) -> None:
        pass

    def train_one_epoch(self, model, optimizer, dataloader: DataLoader):
        pass

    def validate_one_epoch(self, model, dataloader: DataLoader):
        pass

    def train_loop(self, model, optimizer, dataloader: DataLoader):
        for epoch in range(Config.epochs):
            self.train_one_epoch(model, optimizer, dataloader)
            self.validate_one_epoch(model, dataloader)



if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_loop()