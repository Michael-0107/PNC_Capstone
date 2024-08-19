import os 
import pickle
import json
import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

import Hypers
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
    
def rating_to_one_hot_encoding(rating_to_category):
    categories = list(rating_to_category.values())

    categories_tensor = torch.tensor(categories)

    one_hot_encodings = F.one_hot(categories_tensor, num_classes=len(rating_to_category))

    rating_to_one_hot = {rating: one_hot for rating, one_hot in zip(rating_to_category.keys(), one_hot_encodings)}

    return rating_to_one_hot
    
def merge_input_output_dicts_k(input_dict, output_dict,all_rating, k, verbose=True):
    merged_dict = {}
    rating_to_one_hot = rating_to_one_hot_encoding(rating_to_category)
    k = k-1
    
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
            current_category = rating_to_category[current_rating.strip("+-")]
            
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
                
                if prev_period in input_dict[company_name] and prev_period in all_rating[company_name]:
                    prev_features = input_dict[company_name][prev_period]
                    prev_rating = all_rating[company_name][prev_period]
                    prev_category = rating_to_category[prev_rating.strip("+-")]
                else:
                    prev_features = [0] * len(current_features)
                    # prev_category = rating_to_one_hot['NG']  
                    prev_category = -1
                
                prev_features_tensor = torch.FloatTensor(prev_features)
                prev_category_tensor = torch.FloatTensor([prev_category])
                if i == 0:
                    combined_tensor = prev_features_tensor
                else:
                    combined_tensor = torch.cat((prev_features_tensor, prev_category_tensor))
                
                features_list.append(combined_tensor)
            
            # Combine features
            combined_features = torch.cat(features_list)
            
            # Only current Rating
            combined_ratings = torch.FloatTensor([current_category])
            combined_ratings_normalized = combined_ratings / (len(Hypers.rating_to_category)-1)
            
            merged_dict[company_name][period] = (combined_features, combined_ratings,combined_ratings_normalized)
    
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

from collections import defaultdict

def remove_companies_with_few_quarters(train_dict, min_quarters=3):
    filtered_dict = {}

    for company, periods in train_dict.items():
        if len(periods) >= min_quarters:
            filtered_dict[company] = periods
    
    return filtered_dict

def count_labels_in_train_dict(train_dict):
    label_count = defaultdict(int)
    
    for company, periods in train_dict.items():
        for period, (features, label, normalized_label) in periods.items():
            label_value = label.item()  # 获取标签值
            label_count[label_value] += 1
    
    return label_count

def balance_train_dict(train_dict, target_count=1300, min_quarters=3, labels_to_balance=[3, 4, 5]):
    # 第一步：删除季度数少于 min_quarters 的公司
    train_dict = remove_companies_with_few_quarters(train_dict, min_quarters=min_quarters)
    
    # 统计初始标签数量
    label_counts = count_labels_in_train_dict(train_dict)
    
    # 第一次删除：删除那些所有季度都属于目标标签的公司
    for label in labels_to_balance:
        companies_to_delete = []
        if label_counts[label] > target_count:
            for company_id in list(train_dict.keys()):
                if label_counts[label] <= target_count:
                    break  # 如果标签数量已经达到目标，停止删除

                # 检查该公司的所有季度是否全都是该标签
                company_labels = [content[1].item() for content in train_dict[company_id].values()]
                if all(l == label for l in company_labels):
                    # 如果该公司的所有季度数据都具有同一个标签，删除该公司
                    label_counts[label] -= len(company_labels)
                    companies_to_delete.append(company_id)

            # 执行删除操作
            for company_id in companies_to_delete:
                del train_dict[company_id]

    # 第二轮删除：删除那些包含目标标签的公司，直到数量符合要求
    for label in labels_to_balance:
        companies_to_delete = []
        if label_counts[label] > target_count:
            for company_id in list(train_dict.keys()):
                if label_counts[label] <= target_count:
                    break  # 如果标签数量已经达到目标，停止删除

                # 检查公司是否有任何一个标签为当前目标标签
                company_labels = [content[1].item() for content in train_dict[company_id].values()]
                if any(l == label for l in company_labels):
                    # 记录该公司中有多少个该标签
                    label_occurrences = company_labels.count(label)
                    # 标记删除公司
                    label_counts[label] -= label_occurrences
                    companies_to_delete.append(company_id)

            # 执行删除操作
            for company_id in companies_to_delete:
                del train_dict[company_id]

    # 重新统计每个标签的数量
    final_label_counts = defaultdict(int)
    for company_id, quarters in train_dict.items():
        for quarter, content in quarters.items():
            label = content[1].item()
            if label in labels_to_balance:
                final_label_counts[label] += 1

    # 输出最终结果
    print(f"The number of each labels: {dict(final_label_counts)}")
    
    return train_dict

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


def prepare_cpi_dict(cpi_path, start_year=2010, end_year=2021, save=True):
    cpi_df = pd.read_csv(cpi_path, parse_dates=["Yearmon"], dayfirst=True)

    cpi_dict = {}
    for year in range(start_year, end_year):
        for quarter in range(1, 5):
            sample_date = pd.Timestamp(year=year, month=3 * quarter - 2, day=1)
            cpi_dict[f"{year}Q{quarter}"] = float(cpi_df[cpi_df["Yearmon"] == sample_date]["CPI"].values[0])
    
    if save:
        save_pickle(cpi_dict, os.path.join(Config.data_path, "cpi.pkl"))

    return cpi_dict



if __name__ == "__main__":
    cpi_dict = prepare_cpi_dict(os.path.join(Config.data_path, "US_CPI.csv"), save=True, start_year=1979)
    print(cpi_dict)
    