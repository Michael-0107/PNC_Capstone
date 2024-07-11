import torch
from torch.utils.data import Dataset

class RatingSet(Dataset):
    def __init__(self, record_dict, truth_dict):
        self.record_dict = record_dict
        
        companies = list(self.record_dict.keys())
        indicies = [i for i in range(len(companies))]

        self.idx_to_company = dict(zip(indicies, companies))


    def __len__(self):
        return len(self.record_dict)

    def __getitem__(self, idx):
        return self.record_dict[self.idx_to_company[idx]]
    
if __name__ == "__main__":
    pass