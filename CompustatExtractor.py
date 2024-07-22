import os
import pandas as pd
import torch
from collections import OrderedDict
import pickle

from sklearn.preprocessing import RobustScaler

from Hypers import Config, feature_list, numeric_features, derived_features
import utils

class CompustatExtractor:
    def __init__(self):
        pass

    @staticmethod
    def append_financial_ratio(record_df):
        record_filled = record_df.dropna().copy()
        
        # Gross Profit Ratio
        record_filled["GrossProfitRatio"] = (record_filled["revtq"]-record_filled["cogsq"])/record_filled["revtq"]
        record_filled.loc[record_filled["revtq"] == 0, "GrossProfitRatio"] = 0
        # Net Profit Ratio
        record_filled["NetProfitRatio"] = record_filled["niq"]/record_filled["revtq"]
        record_filled.loc[record_filled["revtq"] == 0, "NetProfitRatio"] = 0
        
        # Current Ratio
        record_filled["CurrentRatio"] = record_filled["actq"]/record_filled["lctq"]
        record_filled.loc[record_filled["lctq"] == 0, "CurrentRatio"] = 0
        # Quick Acid Ratio
        record_filled["QuickAcidRatio"] = (record_filled["actq"]-record_filled["invtq"])/record_filled["lctq"]
        record_filled.loc[record_filled["lctq"] == 0, "QuickAcidRatio"] = 0
        # Cash Ratio
        record_filled["CashRatio"] = record_filled["cheq"]/record_filled["lctq"]
        record_filled.loc[record_filled["lctq"] == 0, "CashRatio"] = 0

        # EM
        record_filled["EquityMultiplier"] = record_filled["atq"]/record_filled["teqq"]
        record_filled.loc[record_filled["teqq"] == 0, "EquityMultiplier"] = 0

        # ROA ROE
        record_filled["ReturnOnAsset"] = record_filled["niq"]/record_filled["actq"]
        record_filled.loc[record_filled["actq"] == 0, "ReturnOnAsset"] = 0
        record_filled["ReturnOnEquity"] = record_filled["niq"]/record_filled["teqq"]
        record_filled.loc[record_filled["teqq"] == 0, "ReturnOnEquity"] = 0

        # Inventory Turnover
        record_filled["InventoryTurnover"] = record_filled["cogsq"]/record_filled["invtq"]
        record_filled.loc[record_filled["invtq"] == 0, "InventoryTurnover"] = 0
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
            # if (row.isnull().sum() != 0):
            #     continue
            
            features = [row[feature_name] for feature_name in feature_list]

            if ticker not in ret_dict:
                ret_dict[ticker] = {}
            
            peroid_str = f"{year}Q{quarter}"

            feature_tensor = torch.tensor(features)
            assert(feature_tensor.shape[0] == len(feature_list))
            ret_dict[ticker][peroid_str] = feature_tensor
        
        return ret_dict

    @staticmethod
    def normalize_features(record_df):
        for feature_name in feature_list:
            record_df[feature_name] = RobustScaler().fit_transform(record_df[[feature_name]])
        return record_df

    @staticmethod
    def process_compustat_data(csv_path, save=True, filestem="compustat"):
        record_df = pd.read_csv(csv_path, dtype={"fyearq": str, "fqtr": str})
        record_df = record_df.dropna(axis=0, how='any', subset=['tic', 'fyearq', 'fqtr']+numeric_features)

        record_appended = CompustatExtractor.append_financial_ratio(record_df)
        if save:
            record_appended.drop_index().to_csv(os.path.join(Config.data_path, f"{filestem}_scaler_df.csv"))

        record_appended = CompustatExtractor.normalize_features(record_appended)
        feature_dict = CompustatExtractor.get_feature_tensor_dict(record_appended)

        if save:
            utils.save_pickle(feature_dict, os.path.join(Config.data_path, f"{filestem}_normalized_dict.pkl"))

        return feature_dict

    @staticmethod 
    def get_ratings_by_quarter(df, start_date='2009-10-01', end_date='2024-03-31'):
        df = df[(df['datadate'] >= start_date) & (df['datadate'] <= end_date)]

        # df['Quarter'] = df['datadate'].dt.to_period('Q')
        df.loc[:, 'Quarter'] = df['datadate'].dt.to_period('Q')

        quarter_rating_dict = {}

        quarters = pd.period_range(start=start_date, end=end_date, freq='Q')

        for quarter in quarters:
            quarter_start_date = quarter.start_time
            closest_dates = df[df['datadate'] < quarter_start_date].sort_values(by='datadate', ascending=False)
            if not closest_dates.empty:
                closest_date = closest_dates.iloc[0]
                quarter_rating_dict[str(quarter)] = closest_date['splticrm']

        return quarter_rating_dict

    @staticmethod
    def process_compustat_ratings(csv_path, save=True, filestem="ratings"):
        record_df = pd.read_csv(csv_path, parse_dates=["datadate"])
        # drop Nan in "splticrm" column
        record_df = record_df.dropna(subset=["splticrm"])
        grouped = record_df.groupby("tic")

        ret_rating_dict = {}
        for ticker, group in grouped:
            ratings_by_quarter = CompustatExtractor.get_ratings_by_quarter(group)
            ret_rating_dict[ticker] = ratings_by_quarter

        if save:
            utils.save_pickle(ret_rating_dict, os.path.join(Config.data_path, f"{filestem}.pkl"))

        return ret_rating_dict

if __name__ == "__main__":
    postfix = "retail_indus"
    feature_dict = CompustatExtractor().process_compustat_data(os.path.join(Config.data_path, "WRDS", f"features_{postfix}.csv"),
                                                                save=True, 
                                                                filestem=f"features_{postfix}")

    rating_dict = CompustatExtractor().process_compustat_ratings(os.path.join(Config.data_path, "WRDS", f"ratings_{postfix}.csv"), 
                                                                 save=True, filestem=f"ratings_{postfix}")






