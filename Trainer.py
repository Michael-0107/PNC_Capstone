from multiprocessing import reduction
import os
from numpy import dtype
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from datetime import datetime

import Hypers
from Hypers import Config
import utils


class Trainer:
    def __init__(self, model, criterion, optimizer, device, train_loader, test_loader) -> None:
        # Training Related
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.current_epoch = 0
        # Statistics
        self.start_time = datetime.now().strftime("%m%d%H%M")
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=os.path.join(Config.log_path, f"train_{self.start_time}.log"), level=logging.INFO)

        self.best_test_loss = torch.inf
        self.train_loss_history = []
        self.train_acccuracy_history = []
        self.test_loss_history = []
        self.test_acccuracy_history = []
        self.progress_bar = tqdm(range(Config.epochs))
        

    def train_one_epoch(self):
        loss_accumulated = 0
        hit_accumulated = 0
        total_items = 0

        self.model.train()
        
        self.progress_bar.set_description(f"Train Epoch {self.current_epoch}")
        for idx, (features_b, labels_b, mask_b) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            # features_b: (B, L, len(features))
            # labels_b: (B, L)
            # mask_b: (B, L)
            features_b = features_b.to(self.device)
            labels_b = labels_b.to(self.device)
            mask_b = mask_b.to(self.device)

            # Forward Pass
            c_0 = torch.zeros(1, features_b.shape[0], Config.hidden_size).to(self.device) # (directions*num_layers, batch_size, hidden_size)
            h_0 = torch.zeros(1, features_b.shape[0], Config.proj_size).to(self.device) # (directions*num_layers, batch_size, hidden_size if proj_size=0 else proj_size)
            
            output_b, h_end, c_end = self.model(features_b, h_0, c_0)
            assert output_b.shape == (features_b.shape[0], features_b.shape[1], Config.proj_size)

            output_flat = output_b.reshape(-1)
            labels_flat = labels_b.reshape(-1)
            mask_flat = mask_b.reshape(-1)

            output_masked = output_flat[mask_flat == 1]
            labels_masked = labels_flat[mask_flat == 1]

            loss = self.criterion(output_masked, labels_masked)
            loss.backward()
            self.optimizer.step()

            # Statistics
            loss_accumulated += loss.item()

            pred = torch.round(output_masked)
            hit_count = (pred == labels_masked).sum().item()
            hit_accumulated += hit_count
            total_items += len(labels_masked)

            self.progress_bar.set_postfix_str(f"Loss: {loss_accumulated/total_items:.3f}")
        
        return loss_accumulated/total_items, hit_accumulated/total_items


    def validate_one_epoch(self):
        loss_accumulated = 0
        hit_accumulated = 0
        total_items = 0

        self.model.eval()
        self.progress_bar.set_description(f"Vaild Epoch {self.current_epoch}")
        with torch.no_grad():
            for idx, (features_b, labels_b, mask_b) in enumerate(self.test_loader):
                features_b = features_b.to(self.device)
                labels_b = labels_b.to(self.device)
                mask_b = mask_b.to(self.device)

                c_0 = torch.zeros(1, features_b.shape[0], Config.hidden_size).to(self.device) 
                h_0 = torch.zeros(1, features_b.shape[0], Config.proj_size).to(self.device) 
                output_b, h_end, c_end = self.model(features_b,h_0, c_0) 
                assert output_b.shape == (features_b.shape[0], features_b.shape[1], Config.proj_size) # (B, L, 1)

                output_flat = output_b.reshape(-1)
                labels_flat = labels_b.reshape(-1)
                mask_flat = mask_b.reshape(-1)

                output_masked = output_flat[mask_flat == 1]
                labels_masked = labels_flat[mask_flat == 1]

                loss = self.criterion(output_masked, labels_masked)

                # Statistics
                loss_accumulated += loss.item()
                pred = torch.round(output_masked)
                hit_count = (pred == labels_masked).sum().item()
                hit_accumulated += hit_count
                total_items += len(labels_masked)

                self.progress_bar.set_postfix_str(f"Loss: {loss_accumulated/total_items:.3f}")

        return loss_accumulated/total_items, hit_accumulated/total_items
    
    
    def summarize_one_epoch(self, train_loss, train_accuracy, test_loss, test_accuracy):
        self.train_loss_history.append(train_loss)
        self.train_acccuracy_history.append(train_accuracy)
        self.test_loss_history.append(test_loss)
        self.test_acccuracy_history.append(test_accuracy)

        logging_string = f"Epoch {self.current_epoch}: Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}"
        if test_loss is not None and test_accuracy is not None:
            logging_string += f", Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}"
        logging.info(logging_string)
        

        if test_loss is not None and test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.save_model()
        

    def train_loop(self, train_only=False):
        for self.current_epoch in self.progress_bar:
            train_loss, train_accuracy = self.train_one_epoch()

            test_loss, test_accuracy = None, None
            if not train_only:
                test_loss, test_accuracy = self.validate_one_epoch()
                
            self.summarize_one_epoch(train_loss, train_accuracy, test_loss, test_accuracy)

        return self.train_loss_history, self.train_acccuracy_history, self.test_loss_history, self.test_acccuracy_history
    
    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(Config.model_path, f"model_{self.start_time}.ckpt"))

if __name__ == "__main__":
    from PredictorModel import PredictorModel
    model = PredictorModel(input_size=len(Hypers.feature_list), hidden_size=Config.hidden_size, proj_size=len(Hypers.rating_to_category))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from RatingSet import RatingSet
    merged_dict = utils.load_pickle(os.path.join(Hypers.Config.data_path, "merged_dict.pkl"))
    train_set = RatingSet(merged_dict)
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True, collate_fn=utils.custom_collate_fn)
    test_set = None
    test_loader = None

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device, train_loader=train_loader, test_loader=test_loader)
    trainer.train_loop()
