import os
import pandas as pd
import torch
from collections import OrderedDict
import pickle

from Hypers import Config, feature_list

class CompustatExtractor:
    def __init__(self):
        pass

    @staticmethod
    def append_financial_ratio(record_df):
        record_filled = record_df.dropna().copy()

        record_filled["GrossProfitRatio"] = (record_filled["revtq"]-record_filled["cogsq"])/record_filled["revtq"]
        record_filled["NetProfitRatio"] = record_filled["niq"]/record_filled["revtq"]

        record_filled["CurrentRatio"] = record_filled["actq"]/record_filled["lctq"]
        record_filled["QuickAcidRatio"] = (record_filled["actq"]-record_filled["invtq"])/record_filled["lctq"]
        record_filled["CashRatio"] = record_filled["cheq"]/record_filled["lctq"]
        
        record_filled["EquityMultiplier"] = record_filled["atq"]/record_filled["teqq"]
        record_filled["ReturnOnAsset"] = record_filled["niq"]/record_filled["actq"]
        record_filled["ReturnOnEquity"] = record_filled["niq"]/record_filled["teqq"]

        record_filled["InventoryTurnover"] = record_filled["cogsq"]/record_filled["invtq"]
        # record_filled["ReceivablesTurnover"] = record_filled["revtq"]/record_filled["rectq"]

        return record_filled

    @staticmethod
    def get_feature_tensor_dict(record_df: pd.DataFrame) -> OrderedDict:
        record_sorted_df = record_df.sort_values(["tic", "fyearq", "fqtr"], ascending=[False, True, True]).copy()

        ret_dict = OrderedDict()
        for idx, row in record_sorted_df.iterrows():
            ticker = row["tic"]
            year = row["fyearq"]
            quarter = row["fqtr"]

            # check no Nan in row
            if (row.isnull().sum() != 0):
                continue
            
            features = [row[feature_name] for feature_name in feature_list]

            if ticker not in ret_dict:
                ret_dict[ticker] = {}
            
            peroid_str = f"{year}Q{quarter}"

            feature_tensor = torch.tensor(features)
            assert(feature_tensor.shape[0] == len(feature_list))
            ret_dict[ticker][peroid_str] = feature_tensor
        
        return ret_dict
        
    @staticmethod
    def process_compustat_data(csv_path, save=True, filestem="compustat"):
        record_df = pd.read_csv(csv_path)

        record_appended = CompustatExtractor.append_financial_ratio(record_df)
        feature_dict = CompustatExtractor.get_feature_tensor_dict(record_appended)

        if save:
            CompustatExtractor.save_pickle(feature_dict, os.path.join(Config.data_path, f"{filestem}.pkl"))

        return feature_dict
        
            

if __name__ == "__main__":
    feature_dict = CompustatExtractor().process_compustat_data(os.path.join(Config.data_path, "WRDS", "Retailer_07041810.csv"), 
                                                               filestem="Retail_07111221")
    print(feature_dict)






