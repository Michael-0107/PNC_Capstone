import os
import torch
from torch.utils.data import Dataset

from Hypers import Config

class LSTMDataset(Dataset):
    def __init__(self, merged_dict):
        self.merged_dict = merged_dict
        
        companies = list(self.merged_dict.keys())
        indicies = [i for i in range(len(companies))]

        self.company_idx_to_name = dict(zip(indicies, companies))

        self.max_seq_len = max([len(entries) for entries in self.merged_dict.values()])


    def __len__(self):
        return len(self.merged_dict)

    def __getitem__(self, idx):
        entries = self.merged_dict[self.company_idx_to_name[idx]]

        features = [x[0] for _, x in entries.items()]
        labels = [x[1] for _, x in entries.items()]

        features = torch.stack(features)
        labels = torch.stack(labels)

        return features, labels
    
    @staticmethod
    def custom_collate_fn(batch):
        features, labels = zip(*batch)
        
        features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        label_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
        label_padded = label_padded.squeeze(-1)

        mask = torch.zeros((features_padded.shape[0], features_padded.shape[1]))
        for idx, _ in enumerate(features):
            mask[idx, 0:len(labels[idx])] = 1

        return features_padded, label_padded, mask
            
    
if __name__ == "__main__":
    pass