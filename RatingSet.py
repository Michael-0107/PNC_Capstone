import os
import torch
from torch.utils.data import Dataset

from Hypers import Config

class RatingSet(Dataset):
    def __init__(self, merged_dict):
        self.merged_dict = merged_dict
        
        companies = list(self.merged_dict.keys())
        indicies = [i for i in range(len(companies))]

        self.company_idx_to_name = dict(zip(indicies, companies))


    def __len__(self):
        return len(self.merged_dict)

    def __getitem__(self, idx):
        entries = self.merged_dict[self.company_idx_to_name[idx]]

        features = [x[0] for _, x in entries.items()]
        labels = [x[1] for _, x in entries.items()]

        features = torch.stack(features)
        labels = torch.stack(labels)

        return features, labels

            
    
if __name__ == "__main__":
    RatingSet()