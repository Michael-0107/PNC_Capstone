import os
import torch 
from torch.utils.data import Dataset

from Hypers import Config

class RegressionDataset(Dataset):
    def __init__(self, merged_dict):
        self.merged_dict = merged_dict

        self.feature_list = []
        self.label_list = []
        self.label_normalized_list = []

        for comapny, entries in self.merged_dict.items():
            for period, (feature, label, label_normalized) in entries.items():
                self.feature_list.append(feature)
                self.label_list.append(label)
                self.label_normalized_list.append(label_normalized)

        assert len(self.feature_list) == len(self.label_list) == len(self.label_normalized_list)


    def __len__(self):
        return len(self.feature_list)


    def __getitem__(self, idx):
        return self.feature_list[idx], self.label_list[idx], self.label_normalized_list[idx]
    
    @staticmethod
    def custom_collate_fn(batch):
        features, labels, labels_normalized = zip(*batch)

        features_b = torch.stack(features)
        labels_b = torch.stack(labels)
        labels_normalized_b = torch.stack(labels_normalized)
        mask_b = torch.ones_like(labels_b)

        return features_b, labels_b, labels_normalized_b, mask_b
        


if __name__ == "__main__":
    import pickle
    import utils
    
    train_dict = utils.load_pickle(os.path.join(Config.data_path, "dataset_US_8.pkl"))
    train_set = RegressionDataset(train_dict)

    print(len(train_set))
    print(train_set[0]) 