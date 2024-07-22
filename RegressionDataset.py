import os
import torch 
from torch.utils.data import Dataset

from Hypers import Config

class RegressionDataset(Dataset):
    def __init__(self, merged_dict):
        self.merged_dict = merged_dict

        self.feature_list = []
        self.label_list = []

        for comapny, entries in self.merged_dict.items():
            for period, (feature, label) in entries.items():
                self.feature_list.append(feature)
                self.label_list.append(label)



    def __len__(self):
        assert(len(self.feature_list) == len(self.label_list))
        return len(self.feature_list)


    def __getitem__(self, idx):
        return self.feature_list[idx], self.label_list[idx]
    
    @staticmethod
    def custom_collate_fn(batch):
        features, labels = zip(*batch)

        features_b = torch.stack(features)
        labels_b = torch.stack(labels)
        mask_b = torch.ones_like(labels_b)

        return features_b, labels_b, mask_b
        


if __name__ == "__main__":
    import pickle
    import utils
    
    train_dict = utils.load_pickle(os.path.join(Config.data_path, "train_dict.pkl"))
    train_set = RegressionDataset(train_dict)

    print(len(train_set))
    print(train_set[0]) 