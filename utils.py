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

def merge_input_output_dicts_k(input_dict, output_dict,all_rating, k, verbose=True):
    merged_dict = {}
    k = k - 1 
    
    for company_name in output_dict:
        if company_name not in input_dict:
            continue
        
        for period in output_dict[company_name]:
            if period not in input_dict[company_name]:
                continue
            
            if company_name not in merged_dict:
                merged_dict[company_name] = {}
            
            # get current rating and category
            current_rating = output_dict[company_name][period]
            current_category = rating_to_category[current_rating]
            
            # get current features
            current_features = input_dict[company_name][period]
            
            # get k quarters features
            features_list = []
            current_year, current_quarter = map(int, period.split('Q'))
            
            for i in range(k, -1, -1):  # From previous k to current quarter
                if current_quarter - i <= 0:
                    prev_quarter = 4 + (current_quarter - i)
                    prev_year = current_year - 1
                else:
                    prev_quarter = current_quarter - i
                    prev_year = current_year
                
                prev_period = f"{prev_year}Q{prev_quarter}"
                
                if prev_period in input_dict[company_name]:
                    prev_features = input_dict[company_name][prev_period]
                else:
                    prev_features = [0] * len(current_features)  
                
                if prev_period in all_rating[company_name]:
                    prev_rating = all_rating[company_name][prev_period]
                    prev_category = rating_to_category[prev_rating]
                else:
                    prev_category = -1  
                
                prev_features_tensor = torch.FloatTensor(prev_features)
                prev_category_tensor = torch.FloatTensor([prev_category])
                combined_tensor = torch.cat((prev_features_tensor, prev_category_tensor))
                
                features_list.append(combined_tensor)
            
            # Combine features
            combined_features = torch.stack(features_list)
            
            # Only current Rating
            combined_ratings = torch.FloatTensor([current_category])
            
            merged_dict[company_name][period] = (combined_features, combined_ratings)
    
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
    postfix = "RetInd"
    input_dict = load_pickle(os.path.join(Config.data_path, f"features_{postfix}_4.pkl"))
    output_dict = load_pickle(os.path.join(Config.data_path, f"ratings_{postfix}.pkl"))

    merged_dict = merge_input_output_dicts(input_dict, output_dict)
    save_pickle(merged_dict, os.path.join(Config.data_path, f"dataset_{postfix}_4.pkl"))
    