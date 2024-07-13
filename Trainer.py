import os
from pyexpat import features
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

import Hypers
from Hypers import Config

import utils


class Trainer:
    def __init__(self, model, criterion, optimizer, device, train_loader, test_loader) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Statistics
        self.train_loss_history = []
        self.train_acccuracy_history = []
        self.test_loss_history = []
        self.test_acccuracy_history = []
        

    def train_one_epoch(self):
        loss_record = []
        accuracy_record = []
        for idx, (features_b, labels_b, mask_b) in enumerate(tqdm(self.train_loader)):
            optimizer.zero_grad()
            # features_b: (B, L, len(features))
            # labels_b: (B, L)
            # mask_b: (B, L)
            features_b = features_b.to(self.device)
            labels_b = labels_b.to(self.device)
            mask_b = mask_b.to(self.device)

            # Forward Pass
            c_0 = torch.zeros(1, Config.batch_size, Config.hidden_size).to(self.device)
            h_0 = torch.zeros(1, Config.batch_size, Config.hidden_size).to(self.device)
            output_b = model(features_b,h_0, c_0) 
            assert output_b.shape == (features_b.shape[0], features_b.shape[1], len(Hypers.rating_to_category)) # (B, L, len(categories))

            output_flat = output_b.view(-1, len(Hypers.rating_to_category))
            labels_flat = labels_b.view(-1)
            mask_flat = mask_b.view(-1)

            output_masked = output_flat[mask_flat == 1]
            labels_masked = labels_flat[mask_flat == 1]

            loss = criterion(output_masked, labels_masked)
            loss.backward()
            optimizer.step()

            # Statistics
            loss_record.append(loss.item())

            pred = torch.argmax(output_masked, dim=1)
            accuracy = (pred == labels_masked).sum() / len(labels_masked)
            accuracy_record.append(accuracy.item())
        
        return sum(loss_record)/len(loss_record), sum(accuracy_record)/len(accuracy_record)


    def validate_one_epoch(self):
        loss_record = []
        accuracy_record = []
        with torch.no_grad():
            for idx, (features_b, labels_b, mask_b) in enumerate(tqdm(self.test_loader)):
                features_b = features_b.to(self.device)
                labels_b = labels_b.to(self.device)
                mask_b = mask_b.to(self.device)

                c_0 = torch.zeros(1, Config.batch_size, Config.hidden_size).to(self.device)
                h_0 = torch.zeros(1, Config.batch_size, Config.hidden_size).to(self.device)
                output_b = model(features_b,h_0, c_0) 
                assert output_b.shape == (features_b.shape[0], features_b.shape[1], len(Hypers.rating_to_category)) # (B, L, len(categories))

                output_flat = output_b.view(-1, len(Hypers.rating_to_category))
                labels_flat = labels_b.view(-1)
                mask_flat = mask_b.view(-1)

                output_masked = output_flat[mask_flat == 1]
                labels_masked = labels_flat[mask_flat == 1]

                loss = criterion(output_masked, labels_masked)

                # Statistics
                loss_record.append(loss.item())
                pred = torch.argmax(output_masked, dim=1)
                accuracy = (pred == labels_masked).sum() / len(labels_masked)
                accuracy_record.append(accuracy.item())

        return sum(loss_record)/len(loss_record), sum(accuracy_record)/len(accuracy_record)
    
    
    def summarize_one_epoch(self, train_loss, train_accuracy, test_loss, test_accuracy):
        self.train_loss_history.append(train_loss)
        self.train_acccuracy_history.append(train_accuracy)
        self.test_loss_history.append(test_loss)
        self.test_acccuracy_history.append(test_accuracy)

        logging.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        

    def train_loop(self):
        for epoch in range(Config.epochs):
            train_loss, train_accuracy = self.train_one_epoch()
            test_loss, test_accuracy = self.validate_one_epoch()

            self.summarize_one_epoch(train_loss, train_accuracy, test_loss, test_accuracy)
            

if __name__ == "__main__":
    from PredictorModel import PredictorModel
    model = PredictorModel(input_size=len(Hypers.feature_list), hidden_size=Config.hidden_size, output_size=len(Hypers.rating_to_category))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from RatingSet import RatingSet
    merged_dict = utils.load_pickle(os.path.join(Hypers.Config.data_path, "merged_dict.pkl"))
    train_set = RatingSet(merged_dict)
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True, collate_fn=utils.custom_collate_fn)
    test_set = None
    test_loader = None

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, test_loader=test_loader)
    trainer.train_loop()
