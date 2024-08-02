import os 
import pickle
import json
import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

from Hypers import Config, rating_to_category

def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def read_dict_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_pickle(save_dict, save_path):
    with open(save_path, 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(load_path):
    with open(load_path, 'rb') as handle:
        return pickle.load(handle)



def spilt_train_valid(merged_dict, random_select=False, save=True, suffix=None):
    train_dict = {}
    test_dict = {}
    if not random_select:
        train_length = int(len(merged_dict) * Config.train_ratio)
        idx = 0
        for company_name in merged_dict:
            if idx < train_length:
                train_dict[company_name] = merged_dict[company_name]
            else:
                test_dict[company_name] = merged_dict[company_name]
            idx += 1
    else:     
        for company_name in merged_dict:
            if random.random() < Config.train_ratio:
                train_dict[company_name] = merged_dict[company_name]
            else:
                test_dict[company_name] = merged_dict[company_name]
    
    if save:
        if suffix is not None:
            save_pickle(train_dict, os.path.join(Config.data_path, f"train_dict_{suffix}.pkl"))
            save_pickle(test_dict, os.path.join(Config.data_path, f"test_dict_{suffix}.pkl"))
        else:
            save_pickle(train_dict, os.path.join(Config.data_path, "train_dict.pkl"))
            save_pickle(test_dict, os.path.join(Config.data_path, "test_dict.pkl"))

    return train_dict, test_dict

def plot_graph(train_loss, train_accuracy, test_loss, test_accuracy, identifier:str=""):
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(["Train Loss", "Test Loss"])
    plt.title(f"Loss, {identifier}")
    plt.grid()
    plt.show()

    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.legend(["Train Accuracy", "Test Accuracy"])
    plt.title(f"Accuracy, {identifier}")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    pass
    