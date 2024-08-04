import os
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
    def __init__(self, 
                 model=None, 
                 criterion=None, 
                 optimizer=None, 
                 device=None, 
                 train_loader=None, 
                 valid_loader=None, 
                 max_seq_len=None, 
                 model_type="") -> None:
        # Training Related
        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.current_epoch = 0
        self.max_seq_len = max_seq_len
        
        # Statistics
        self.start_time = datetime.now().strftime("%m%d%H%M")
        self.logger = logging.getLogger(__name__)
        self.identifier = f"{model_type}_{self.start_time}"
        logging.basicConfig(filename=os.path.join(Config.log_path, f"train_{self.identifier}.log"), level=logging.INFO)

        self.best_valid_loss = torch.inf
        self.train_loss_history = []
        self.train_acccuracy_history = []
        self.valid_loss_history = []
        self.valid_acccuracy_history = []
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
            labels_b = labels_b.to(self.device).long()
            mask_b = mask_b.to(self.device)

            # Forward Pass
            output_b = self.model(features_b)  # (B, L, C)
            # print("Output shape:", output_b.shape, "Labels shape:", labels_b.shape, "Mask shape:", mask_b.shape)

            # Flatten the output, labels, and mask
            B, L, C = output_b.shape
            output_flat = output_b.view(B * L, C).float()  # Ensure float type for output
            labels_flat = labels_b.view(B * L).long()  # Ensure long type for labels
            mask_flat = mask_b.view(B * L)  # (B*L)

            # Ensure the shapes match
            # print("Flattened shapes - Output:", output_flat.shape, "Labels:", labels_flat.shape, "Mask:", mask_flat.shape)
            assert output_flat.size(0) == labels_flat.size(0) == mask_flat.size(0), "Flattened shapes mismatch"

            # Apply the mask
            output_masked = output_flat[mask_flat == 1]  # Only keep valid time steps
            labels_masked = labels_flat[mask_flat == 1]  # Only keep valid time steps

            # Ensure the shapes match after masking
            # print("Masked shapes - Output:", output_masked.shape, "Labels:", labels_masked.shape)
            assert output_masked.size(0) == labels_masked.size(0), "Output and labels shape mismatch after masking"

            # Ensure data types are correct
            assert output_masked.dtype == torch.float32, f"Output dtype should be float32, but got {output_masked.dtype}"
            assert labels_masked.dtype == torch.int64, f"Labels dtype should be int64, but got {labels_masked.dtype}"

            # Calculate loss
            loss = self.criterion(output_masked, labels_masked)
            loss.backward()
            self.optimizer.step()

            # Statistics
            loss_accumulated += loss.item() * labels_masked.size(0)  # Multiply by the number of valid items to accumulate loss correctly

            # Calculate accuracy
            pred = torch.argmax(output_masked, 1)
            hit_count = (pred == labels_masked).sum().item()
            hit_accumulated += hit_count
            total_items += labels_masked.size(0)

            self.progress_bar.set_postfix_str(f"Loss: {loss_accumulated/total_items:.3f}, Accuracy: {hit_accumulated/total_items:.3f}")
        
        return loss_accumulated / total_items, hit_accumulated / total_items


    def validate_one_epoch(self):
        loss_accumulated = 0
        hit_accumulated = 0
        total_items = 0

        self.model.eval()
        
        with torch.no_grad():
            self.progress_bar.set_description(f"Validating Epoch {self.current_epoch}")
            for idx, (features_b, labels_b, mask_b) in enumerate(self.valid_loader):
                # features_b: (B, L, len(features))
                # labels_b: (B, L)
                # mask_b: (B, L)
                features_b = features_b.to(self.device)
                labels_b = labels_b.to(self.device).long()
                mask_b = mask_b.to(self.device)

                # Forward Pass
                output_b = self.model(features_b)  # (B, L, C)
                # print("Output shape:", output_b.shape, "Labels shape:", labels_b.shape, "Mask shape:", mask_b.shape)

                # Flatten the output, labels, and mask
                B, L, C = output_b.shape
                output_flat = output_b.view(B * L, C).float()  # Ensure float type for output
                labels_flat = labels_b.view(B * L).long()  # Ensure long type for labels
                mask_flat = mask_b.view(B * L)  # (B*L)

                # Ensure the shapes match
                # print("Flattened shapes - Output:", output_flat.shape, "Labels:", labels_flat.shape, "Mask:", mask_flat.shape)
                assert output_flat.size(0) == labels_flat.size(0) == mask_flat.size(0), "Flattened shapes mismatch"

                # Apply the mask
                output_masked = output_flat[mask_flat == 1]  # Only keep valid time steps
                labels_masked = labels_flat[mask_flat == 1]  # Only keep valid time steps

                # Ensure the shapes match after masking
                # print("Masked shapes - Output:", output_masked.shape, "Labels:", labels_masked.shape)
                assert output_masked.size(0) == labels_masked.size(0), "Output and labels shape mismatch after masking"

                # Calculate loss
                loss = self.criterion(output_masked, labels_masked)

                # Statistics
                loss_accumulated += loss.item() * labels_masked.size(0)  # Multiply by the number of valid items to accumulate loss correctly

                # Calculate accuracy
                pred = torch.argmax(output_masked, 1)
                hit_count = (pred == labels_masked).sum().item()
                hit_accumulated += hit_count
                total_items += labels_masked.size(0)

                self.progress_bar.set_postfix_str(f"Validation Loss: {loss_accumulated/total_items:.3f}, Accuracy: {hit_accumulated/total_items:.3f}")
        
        return loss_accumulated / total_items, hit_accumulated / total_items

    
    
    def summarize_one_epoch(self, train_loss, train_accuracy, valid_loss, valid_accuracy):
        self.train_loss_history.append(train_loss)
        self.train_acccuracy_history.append(train_accuracy)
        self.valid_loss_history.append(valid_loss)
        self.valid_acccuracy_history.append(valid_accuracy)

        logging_string = f"Epoch {self.current_epoch}: Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}"
        if valid_loss is not None and valid_accuracy is not None:
            logging_string += f", valid Loss: {valid_loss:.3f}, valid Accuracy: {valid_accuracy:.3f}"
        logging.info(logging_string)
        

        if valid_loss is not None and valid_loss < self.best_valid_loss:
            self.best_valid_loss = valid_loss
            self.save_model()
        

    def train_loop(self, train_only=False):
        for self.current_epoch in self.progress_bar:
            train_loss, train_accuracy = self.train_one_epoch()

            valid_loss, valid_accuracy = None, None
            if not train_only:
                valid_loss, valid_accuracy = self.validate_one_epoch()
                
            self.summarize_one_epoch(train_loss, train_accuracy, valid_loss, valid_accuracy)

        return self.train_loss_history, self.train_acccuracy_history, self.valid_loss_history, self.valid_acccuracy_history
    
    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(Config.model_path, f"{self.identifier}.ckpt"))

if __name__ == "__main__":
    from LSTMDataset import RatingSet
    train_dict = utils.load_pickle(os.path.join(Config.data_path, "train_dict.pkl"))
    valid_dict = utils.load_pickle(os.path.join(Config.data_path, "test_dict.pkl"))
    train_set = RatingSet(train_dict)
    valid_set = RatingSet(valid_dict)
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True, collate_fn=RatingSet.custom_collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=Config.batch_size, shuffle=True, collate_fn=RatingSet.custom_collate_fn)
    max_seq_len = max_seq_len = max(max([len(entries) for entries in train_dict.values()]), max([len(entries) for entries in test_dict.values()]))


    from LSTMModel import PredictorModel
    model = PredictorModel(input_size=len(Hypers.feature_list), 
                            hidden_size=Hypers.Config.hidden_size,
                            num_layers=max_seq_len,
                            proj_size=Hypers.Config.proj_size)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(model=model, 
                      criterion=criterion, 
                      optimizer=optimizer, 
                      device=device, 
                      train_loader=train_loader, 
                      valid_loader=valid_loader, 
                      max_seq_len=max_seq_len)
    trainer.train_loop()
