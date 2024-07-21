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
    

def merge_input_output_dicts(input_dict, output_dict, verbose=True):
    merged_dict = {}
    for company_name in output_dict:
        if not company_name in input_dict:
            continue
        
        for period in output_dict[company_name]:
            if not period in input_dict[company_name]:
                continue
            
            if company_name not in merged_dict:
                merged_dict[company_name] = {}
            
            # transform to one hot
            rating = output_dict[company_name][period]
            category = rating_to_category[rating]
            # output_dict[company_name][period] = F.one_hot(torch.tensor(category), num_classes=len(rating_to_category))

            merged_dict[company_name][period] = (input_dict[company_name][period], torch.FloatTensor([category]))
    
    if verbose:
        print(f"input_dict: {len(input_dict)}")
        print(f"output_dict: {len(output_dict)}")
        print(f"merged_dict: {len(merged_dict)}")

    return merged_dict


def save_pickle(save_dict, save_path):
    with open(save_path, 'wb') as handle:
        pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(load_path):
    with open(load_path, 'rb') as handle:
        return pickle.load(handle)


def custom_collate_fn(batch):
    features, labels = zip(*batch)
    
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    label_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    label_padded = label_padded.squeeze(-1)

    mask = torch.zeros((features_padded.shape[0], features_padded.shape[1]))
    for idx, _ in enumerate(features):
        mask[idx, 0:len(labels[idx])] = 1


    return features_padded, label_padded, mask


def spilt_train_valid(merged_dict, random_select=False, save=True):
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
        save_pickle(train_dict, os.path.join(Config.data_path, "train_dict.pkl"))
        save_pickle(test_dict, os.path.join(Config.data_path, "test_dict.pkl"))

    return train_dict, test_dict

def plot_graph(train_loss, train_accuracy, test_loss, test_accuracy):
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(["Train Loss", "Test Loss"])
    plt.title("Loss")
    plt.grid()
    plt.show()

    plt.plot(train_accuracy)
    plt.plot(test_accuracy)
    plt.legend(["Train Accuracy", "Test Accuracy"])
    plt.title("Accuracy")
    plt.grid()
    plt.show()


if __name__ == "__main__":

    input_dict = load_pickle(os.path.join(Config.data_path, "features_retail_indus_normalized_dict.pkl"))
    output_dict = load_pickle(os.path.join(Config.data_path, "ratings_retail_indus.pkl"))

    merged_dict = merge_input_output_dicts(input_dict, output_dict)
    save_pickle(merged_dict, os.path.join(Config.data_path, "dataset_retail_indus.pkl"))
    