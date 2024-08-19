import os
import torch
from torch.utils.data import Dataset

from Hypers import Config

class LSTMDataset(Dataset):
    def __init__(self, merged_dict, max_seq_len=4):
        self.merged_dict = merged_dict
        self.max_seq_len = max_seq_len
        
        self.data_indices = []
        for company, entries in self.merged_dict.items():
            num_entries = len(entries)
            num_chunks = (num_entries + self.max_seq_len - 1) // self.max_seq_len
            for i in range(num_chunks):
                self.data_indices.append((company, i))

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        company, chunk_idx = self.data_indices[idx]
        entries = list(self.merged_dict[company].values())
        
        start = chunk_idx * self.max_seq_len
        end = start + self.max_seq_len
        
        features = [x[0] for x in entries[start:end]]
        labels = [x[1] for x in entries[start:end]]
        labels_normalized = [x[2] for x in entries[start:end]]

        if len(features) < self.max_seq_len:
            pad_len = self.max_seq_len - len(features)
            features.extend([torch.zeros(features[0].shape)] * pad_len)
            labels.extend([torch.zeros(labels[0].shape)] * pad_len)
            labels_normalized.extend([torch.zeros(labels_normalized[0].shape)] * pad_len)

        features = torch.stack(features)
        labels = torch.stack(labels)
        labels_normalized = torch.stack(labels_normalized)

        return features, labels, labels_normalized
    
    @staticmethod
    def custom_collate_fn(batch):
        features, labels, labels_normalized = zip(*batch)
        
        features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        label_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        label_normalized_padded = torch.nn.utils.rnn.pad_sequence(labels_normalized, batch_first=True)
        
        label_padded = label_padded.squeeze(-1)
        label_normalized_padded = label_normalized_padded.squeeze(-1)

        mask = torch.zeros((features_padded.shape[0], features_padded.shape[1]))
        for idx in range(len(features)):
            mask[idx, :len(features[idx])] = 1

        return features_padded, label_padded, label_normalized_padded, mask
    

            
    
if __name__ == "__main__":
    pass