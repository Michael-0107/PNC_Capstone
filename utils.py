import os 
import pickle
import json
import torch
import torch.nn.functional as F

from Hypers import Config, rating_to_category



def read_dict_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    

def merge_input_output_dicts(input_dict, output_dict):
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




if __name__ == "__main__":
    output_dict = read_dict_json(os.path.join(Config.data_path, "retail_ratings.json"))
    input_dict = load_pickle(os.path.join(Config.data_path, "Retail_07111221.pkl"))

    merged_dict = merge_input_output_dicts(input_dict, output_dict)
    save_pickle(merged_dict, os.path.join(Config.data_path, "merged_dict.pkl"))
    