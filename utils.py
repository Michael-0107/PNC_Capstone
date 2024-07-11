import os 
import pickle
import json

from Hypers import Config


def read_dict_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    


if __name__ == "__main__":
    sample_dict = read_dict_json(os.path.join(Config.data_path, "retail_ratings.json"))
    print(sample_dict)