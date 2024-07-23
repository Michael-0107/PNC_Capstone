import os
import pandas as pd
from collections import OrderedDict
import pickle
import torch

from utils import *
from Hypers import Config

def expand_quarter(filepath):

    data = load_pickle(filepath)
    print(list(data.values())[0]['2009Q4'])

    return

import pickle
from collections import OrderedDict
import torch

def concatenate_features(data, k=4):
    new_data = {}

    for company, time_feature_dict in data.items():
        new_time_feature_dict = OrderedDict()
        sorted_times = sorted(time_feature_dict.keys())
        feature_length = next(iter(time_feature_dict.values())).shape[0]  # Assuming all features have the same length

        for i, current_time in enumerate(sorted_times):
            start_index = max(0, i - 3)
            features_to_concatenate = [time_feature_dict[sorted_times[j]] for j in range(start_index, i + 1)]
            
            # If there are less than 4 quarters, pad with zeros
            if len(features_to_concatenate) < 4:
                padding_count = 4 - len(features_to_concatenate)
                padding_tensors = [torch.zeros(feature_length) for _ in range(padding_count)]
                features_to_concatenate = padding_tensors + features_to_concatenate

            concatenated_feature = torch.cat(features_to_concatenate, dim=0)
            new_time_feature_dict[current_time] = concatenated_feature

        new_data[company] = new_time_feature_dict

    return new_data

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":

    k = 4
    # Example usage
    input_file_path = os.path.join(Config.data_path, "features_retail_indus_normalized_dict.pkl")
    output_file_path = os.path.join(Config.data_path, "features_retail_indus_normalized_dict_{}.pkl".format(k))

    data = load_pickle(input_file_path)
    new_data = concatenate_features(data, k)
    save_pickle(new_data, output_file_path)
