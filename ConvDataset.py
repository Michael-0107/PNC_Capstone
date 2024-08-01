import torch
from torch.utils.data import Dataset

import Hypers
from Hypers import Config


class ConvDataset(Dataset):
    def __init__(self, merged_dict, window_size=4) -> None:
        super(ConvDataset, self).__init__()

        self.merged_dict = merged_dict

        self.feature_list = []
        self.label_list = []

        for comapny, entries in self.merged_dict.items():
            for period, (feature, label) in entries.items():
                self.feature_list.append(feature)
                self.label_list.append(label)

        assert len(self.feature_list) == len(self.label_list)

        self.window_size = window_size
    
    def __len__(self):
        return len(self.feature_list)


    def __getitem__(self, idx):
        return self.feature_list[idx].reshape(self.window_size,-1), self.label_list[idx]

    @staticmethod
    def custom_collate_fn(batch):
        features, labels = list(zip(*batch))
        features = torch.stack(features)
        labels = torch.stack(labels)
        mask = torch.ones_like(labels)
        return features, labels, mask


if __name__ == "__main__":
    import os
    import utils
    train_dict = utils.load_pickle(os.path.join(Config.data_path, "train_dict_RetInd_4.pkl"))

    train_set = ConvDataset(train_dict)
    print(len(train_set))
    print(train_set[0][0].shape, train_set[0][1].shape)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=ConvDataset.custom_collate_fn)

    for features, labels, mask in train_loader:
        print(features.shape, labels.shape, mask.shape)
        break