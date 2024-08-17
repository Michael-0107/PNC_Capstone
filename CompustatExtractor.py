import os
import pandas as pd
import torch
from collections import OrderedDict
import pickle
import numpy as np

from sklearn.preprocessing import RobustScaler

import Hypers
from Hypers import Config
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
    def append_period_change(record_df, target_cols=None):
        if target_cols is None:
            target_cols = Hypers.numeric_features

        record_df_grouped = record_df.groupby("tic")
        for col in target_cols:
            record_df[col + "_change"] = record_df_grouped[col].pct_change()

        record_df = record_df.replace([np.inf, -np.inf, np.nan], 0)
        return record_df



    @staticmethod
    def append_cpi(record_df, cpi_dict):
        record_df["year_quarter"] = record_df["fyearq"].astype(str) + "Q" + record_df["fqtr"].astype(str)
        record_df["CPI"] = record_df["year_quarter"].apply(lambda x: cpi_dict.get(x, 0))

        record_df.drop(columns=["year_quarter"], inplace=True)
        return record_df

    @staticmethod
    def get_feature_tensor_dict(record_df: pd.DataFrame, target_features=None, add_cpi=False) -> OrderedDict:
        record_sorted_df = record_df.sort_values(["tic", "fyearq", "fqtr"], ascending=[False, True, True]).copy()

        if target_features is None:
                target_features = Hypers.feature_list.copy()
        if add_cpi:
            target_features.append("CPI")

        ret_dict = OrderedDict()
        for idx, row in record_sorted_df.iterrows():
            ticker = row["tic"]
            year = int(row["fyearq"])
            quarter = int(row["fqtr"])
            peroid_str = f"{year}Q{quarter}"
            
            if ticker not in ret_dict:
                ret_dict[ticker] = {}

            
            
            features = [row[feature_name] for feature_name in target_features]       

            feature_tensor = torch.tensor(features)
            ret_dict[ticker][peroid_str] = feature_tensor
        
        return ret_dict

    @staticmethod
    def normalize_features(record_df, target_cols=None):
        if target_cols is None:
            target_cols = Hypers.feature_list
        for feature_name in target_cols:
            record_df[feature_name] = RobustScaler().fit_transform(record_df[[feature_name]])
        return record_df

    @staticmethod
    def process_compustat_features(csv_path, save=True, filestem="compustat", add_cpi=True):
        """Process features from Compustat csv files, into nested dictionaries (ticker->(period->features))

        Args:
            csv_path (_type_): path to the raw compustat feature csv file
            save (bool, optional): whether to save the intermediate and final processed results. Defaults to True.
            filestem (str, optional): The filestem of the saved result, will be save in {data_path}/{filestem}.pkl. Defaults to "compustat".
            add_cpi (bool, optional): Whether to add CPI as additional feature. Defaults to True.

        Returns:
            _type_: Nested dictionary. First layer with key: ticker of the comapny, value: entries. Second layer (entries): key: period, value: feature vector
        """
        record_df = pd.read_csv(csv_path, parse_dates=["datadate"]) # may have nan in fyearq, fqtr
        record_df = record_df.dropna(axis=0, how='any', subset=['tic', 'fyearq', 'fqtr']+Hypers.numeric_features)
        record_df['fyearq'] = record_df['fyearq'].astype(int)
        record_df['fqtr'] = record_df['fqtr'].astype(int)
        record_df = record_df.sort_values(["tic", "fyearq", "fqtr"], ascending=[True, True, True])

        # filter time
        record_df = record_df[(record_df['datadate'] >= Config.record_begin_threshold) & \
                              (record_df['datadate'] <= Config.record_end_threshold)]

        # Add period change
        record_appended = CompustatExtractor.append_period_change(record_df)

        # Add derived financial ratios
        record_appended = CompustatExtractor.append_financial_ratio(record_appended)

        # Add Macro-economic features
        if add_cpi:
            cpi_dict = utils.load_pickle(Config.cpi_path)
            record_appended = CompustatExtractor.append_cpi(record_appended, cpi_dict)
            
        # Save human-readable intermediate results before normalizing
        if save:
            record_appended.to_csv(os.path.join(Config.data_path, f"{filestem}_scaler.csv"), index=False)

        # Normalize the features
        record_appended = CompustatExtractor.normalize_features(record_appended)
        
        # Transform to dictionary
        feature_dict = CompustatExtractor.get_feature_tensor_dict(record_appended, add_cpi=add_cpi)
        

        if save:
            utils.save_pickle(feature_dict, os.path.join(Config.data_path, f"{filestem}.pkl"))

        return feature_dict

    @staticmethod 
    def get_ratings_by_quarter(df):
        df = df[(df['datadate'] >= Config.record_begin_threshold) & (df['datadate'] <= Config.record_end_threshold)]

        df = df.copy()
        df['Quarter'] = df['datadate'].dt.to_period('Q')
        
        quarter_rating_dict = {}

        quarters = pd.period_range(start=Config.record_begin_threshold, end=Config.record_end_threshold, freq='Q')

        for quarter in quarters:
            quarter_start_date = quarter.start_time
            closest_dates = df[df['datadate'] < quarter_start_date].sort_values(by='datadate', ascending=False)
            if not closest_dates.empty:
                closest_date = closest_dates.iloc[0]
                quarter_rating_dict[str(quarter)] = closest_date['splticrm']

        return quarter_rating_dict


    @staticmethod
    def process_compustat_ratings(csv_path, 
                                  save=True, 
                                  filestem="ratings", 
                                  start_date='2010-01-01', 
                                  end_date='2017-01-01'):
        """Process ratings from Compustat csv files, into dictionary (ticker->(period->features))

        Args:
            csv_path (_type_): path to the raw compustat rating csv file
            save (bool, optional): whether to save the results into pkl. Defaults to True.
            filestem (str, optional): The filestem of the saved result, will be save in {data_path}/{filestem}.pkl. Defaults to "ratings".

        Returns:
            _type_: Nested dictionary. First layer with key: ticker of the comapny, value: entries. Second layer (entries): key: period, value: rating vector
        """
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
    
    
    @staticmethod
    def process_compustat_omni(csv_path, feature_list_path, save=True, postfix="omni", verbose=True, add_cpi=True):
        """OMNI stands for 205 extremely raw financial data from Compustat. Very alike process_compustat_features()

        Args:
            csv_path (_type_): path to the raw compustat rating csv file
            feature_list_path (_type_): path to a document that contain what features to treat as features. See data/WRDS/float_features for example
            save (bool, optional): Whether to save the results into pkl. Defaults to True.
            postfix (str, optional): identifier of the dataset. Defaults to "omni".
            verbose (bool, optional): Whether to print information. Defaults to True.
            add_cpi (bool, optional): Whether to add CPI. Defaults to True.

        Returns:
            _type_: Nested dictionary. First layer with key: ticker of the comapny, value: entries. Second layer (entries): key: period, value: feature vector
        """
        float_features = []
        with open(feature_list_path, 'r') as f:
            for line in f:
                float_features.append(line.strip())

        record_df = pd.read_csv(csv_path, parse_dates=["datadate"])
        record_df = CompustatExtractor.clean_omni(record_df)
        
        record_df["fyearq"] = record_df["fyearq"].astype(int)
        record_df["fqtr"] = record_df["fqtr"].astype(int)

        record_df = record_df[(record_df['datadate'] >= Config.record_begin_threshold) & \
                              (record_df['datadate'] <= Config.record_end_threshold)]
        float_features.remove("fyearq")
        float_features.remove("fqtr")

        # find columns in feature_list
        feature_cols = [col for col in record_df.columns if col in float_features]
        if verbose:
            print(f"len(feature_cols): {len(feature_cols)}")
            print(feature_cols)

        record_df = record_df[["tic", "fyearq", "fqtr"] + feature_cols]

        if add_cpi:
            cpi_dict = utils.load_pickle(Config.cpi_path)
            record_df = CompustatExtractor.append_cpi(record_df, cpi_dict)

        if save:
            record_df.to_csv(os.path.join(Config.data_path, f"features_{postfix}_scaler.csv"), index=False)

        record_df = CompustatExtractor.normalize_features(record_df, target_cols=feature_cols)
        features_dict = CompustatExtractor.get_feature_tensor_dict(record_df, target_features=feature_cols, add_cpi=add_cpi)

        if save:
            utils.save_pickle(features_dict, os.path.join(Config.data_path, f"features_{postfix}.pkl"))

        return features_dict
    
    @staticmethod
    def clean_omni(data: pd.DataFrame):
        # Define thresholds
        small_portion_threshold = 0.2  # Columns with less than 20% NaNs will be filled
        large_portion_threshold = 0.5  # Rows with more than 50% NaNs will be dropped

        # Load the data
        # data = pd.read_csv(file, low_memory=False)

        # 1. Delete columns that are entirely NaN
        data = data.dropna(axis=1, thresh=int((1 - small_portion_threshold) * data.shape[0]))
        print("Empty columns deleted.")

        # 2. Delete columns with only one unique value
        data = data.loc[:, data.nunique() > 1]
        print("Columns with one unique value deleted.")

        # 3. Delete companies if there's one feature missing for each quarter
        companies_to_drop = []
        ## Identify companies to drop
        for company in data['tic'].unique():
            company_data = data[data['tic'] == company]
            if company_data.isna().all(axis=0).any():
                companies_to_drop.append(company)

        ## Drop identified companies
        data = data[~data['tic'].isin(companies_to_drop)]
        print("Companies with a missing feature for every quarter deleted.")

        # 4. Forward Filling
        data = data.groupby('tic').apply(lambda group: group.ffill().bfill()).reset_index(drop=True)
        print("NaNs in each columns are filled")

        # Save the cleaned data
        # data.to_csv('RawData/cleaned_financial_data.csv', index=False)

        print("Data cleaning complete.")
        print("Totally {} companies are included.".format(len(data['tic'].unique())))
        print("Cleaned data saved to 'cleaned_financial_data.csv'.")

        return data


    @ staticmethod
    def merge_input_output_dicts(input_dict, output_dict, save=True, filestem="compustat", verbose=True):
        """Pair the input features to output ratings (Note: also chages the ratings to numbers). 
            The function will find the intersection of two dictionaries.

        Args:
            input_dict (_type_): Nested dictionary with features
            output_dict (_type_): Nested dictionary with ratings
            verbose (bool, optional): Whether to print information. Defaults to True.

        Returns:
            _type_: Nested dictionary. First layer with key: ticker of the comapny, value: entries.
                    Second layer (entries): key: period, value: tuple(feature vector, rating vector)
        """
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
                category = Hypers.rating_to_category[rating.strip("+-")]
                category_normalized = category / (len(Hypers.rating_to_category)-1)

                merged_dict[company_name][period] = (input_dict[company_name][period], torch.FloatTensor([category]), torch.FloatTensor([category_normalized]))
        
        if save:
            utils.save_pickle(merged_dict, os.path.join(Config.data_path, f"{filestem}.pkl"))
            
        if verbose:
            print(f"input_dict: {len(input_dict)}")
            print(f"output_dict: {len(output_dict)}")
            print(f"merged_dict: {len(merged_dict)}")

        return merged_dict
    
    @staticmethod
    def concatenate_features(data, k=4):
        """Generate a feature tensor by concatenating the features from the last k quarters.

        Args:
            data (_type_): Nested dictionary. First layer with key: ticker of the comapny, value: entries. Second layer (entries): key: period, value: feature vector
            k (int, optional): window size, add k-1 previous quarters with the current quarter. Defaults to 4.

        Returns:
            _type_: Nested dictionary. First layer with key: ticker of the comapny, value: entries. Second layer (entries): key: period, value: feature vector
        """
        new_data = {}

        for company, time_feature_dict in data.items():
            new_time_feature_dict = OrderedDict()
            sorted_times = sorted(time_feature_dict.keys())
            feature_length = next(iter(time_feature_dict.values())).shape[0]  # Assuming all features have the same length

            for i, current_time in enumerate(sorted_times):
                start_index = max(0, i - k + 1)
                features_to_concatenate = [time_feature_dict[sorted_times[j]] for j in range(start_index, i + 1)]
                
                # If there are less than k quarters, pad with zeros
                if len(features_to_concatenate) < k:
                    padding_count = k - len(features_to_concatenate)
                    padding_tensors = [torch.zeros(feature_length) for _ in range(padding_count)]
                    features_to_concatenate = padding_tensors + features_to_concatenate

                concatenated_feature = torch.cat(features_to_concatenate, dim=0)
                new_time_feature_dict[current_time] = concatenated_feature

            new_data[company] = new_time_feature_dict

        return new_data


if __name__ == "__main__":
    postfix = "All"
    # Extract the features and ratings
    # feature_dict = CompustatExtractor().process_compustat_features(os.path.join(Config.data_path, "WRDS", f"features_{postfix}.csv"),
    #                                                             save=True, 
    #                                                             filestem=f"features_{postfix}")
    
    # rating_dict = CompustatExtractor().process_compustat_ratings(os.path.join(Config.data_path, "WRDS", f"ratings_{postfix}.csv"), 
    #                                                              save=True, filestem=f"ratings_{postfix}")
    rating_dict = utils.load_pickle(os.path.join(Config.data_path, f"ratings_{postfix}.pkl"))
    
    # If you want to use all 205 raw features
    feature_omni_dict = CompustatExtractor.process_compustat_omni(os.path.join(Config.data_path, "WRDS", f"features_{postfix}_omni.csv"),
                                               os.path.join(Config.data_path, "WRDS", "float_features.txt"))

    # If you want to concatenate the features from the last k quarters. 
    # If you don't do so, it will only contain the features from the current quarter
    k=8
    # feature_windowed_dict = CompustatExtractor.concatenate_features(feature_dict, k=k)
    feature_windowed_dict = CompustatExtractor.concatenate_features(feature_omni_dict, k=k)
    
    # Merge features(probably windowed) and ratings for dataset
    merged_dict = CompustatExtractor.merge_input_output_dicts(feature_windowed_dict, rating_dict)
    utils.save_pickle(merged_dict, os.path.join(Config.data_path, f"dataset_{postfix}_omni_{k}.pkl"))








