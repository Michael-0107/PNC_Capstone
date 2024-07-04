import os
import pandas as pd

from Config import Config, tags_interested
from SEC_Processor import SEC_Processor



class FeatureProcessor:
    def __init__(self):
        pass

    def _collect_numbers(self, company_name_list, start_year, end_year, verbose=False):
        ret_feature_dict = {}
        for company_name in company_name_list:
            ret_feature_dict[company_name] = {}

        for year in range(start_year, end_year+1):
            processor = SEC_Processor(year=year, quarter=1)
            for company_name in company_name_list:
                company_key = processor.get_company_key(company_name, form='10-K')
                company_nums = processor.get_company_nums(company_key)

                for timestamp, num_dict in company_nums.items():
                    if (timestamp>=(end_year+1)*10000 or timestamp<(start_year)*10000):
                        if verbose: print(f"Timestamp {timestamp} is not in range {start_year}-{end_year}")
                        continue
                    if (len(num_dict)!=len(tags_interested)):
                        if verbose: print(f"Timestamp {timestamp} has {len(num_dict)} tags, expected {len(tags_interested)}")
                        continue
                    ret_feature_dict[company_name][timestamp] = num_dict
        return ret_feature_dict
    
    def _append_financial_ratio(self, company_numbers_dict):
        for company_name in company_numbers_dict.keys():
            for timestamp, num_dict in company_numbers_dict[company_name].items():
                assert(len(num_dict)==len(tags_interested))
                
                num_dict["GrossProfitRatio"] = (num_dict["Revenues"]-num_dict["CostOfRevenue"])/num_dict["Revenues"]
                num_dict["OperatingProfitRatio"] = num_dict["OperatingIncomeLoss"]/num_dict["Revenues"]
                num_dict["NetProfitRatio"] = num_dict["NetIncomeLoss"]/num_dict["Revenues"]

                num_dict["CurrentRatio"] = num_dict["AssetsCurrent"]/num_dict["LiabilitiesCurrent"]
                num_dict["QuickAcidRatio"] = (num_dict["AssetsCurrent"]-num_dict["InventoryNet"])/num_dict["LiabilitiesCurrent"]
                num_dict["CashRatio"] = num_dict["CashAndCashEquivalentsAtCarryingValue"]/num_dict["LiabilitiesCurrent"]
                
                num_dict["EquityMultiplier"] = num_dict["Assets"]/num_dict["StockholdersEquity"]
                num_dict["ReturnOnAsset"] = num_dict["NetIncomeLoss"]/num_dict["Assets"]
                num_dict["ReturnOnEquity"] = num_dict["NetIncomeLoss"]/num_dict["StockholdersEquity"]

                num_dict["InventoryTurnover"] = num_dict["CostOfRevenue"]/num_dict["InventoryNet"]
                num_dict["ReceivablesTurnover"] = num_dict["Revenues"]/num_dict["AccountsReceivableNet"]
        
        return company_numbers_dict
                
                


    def generate_feature_dict(self, company_name_list, start_year=2019, end_year=2023, verbose=False):
        """Generate features for the companies, getting a dictionary

        Args:
            company_name_list (iterable): An iterable of company names
            start_year (int, optional): start year, INCLUSIVE. Defaults to 2019.
            end_year (int, optional): end year, INCLUSIVE. Defaults to 2023.
            verbose (bool, optional): verbose. Defaults to False.
        """
        company_numbers_dict = self._collect_numbers(company_name_list, start_year, end_year, verbose=verbose)

        company_numbers_with_ratio_dict = self._append_financial_ratio(company_numbers_dict)

        return company_numbers_with_ratio_dict
    

if __name__ == "__main__":
    company_name_list = ["WALMART INC."]
    feature_processor = FeatureProcessor()
    feature_dict = feature_processor.generate_feature_dict(company_name_list, start_year=2019, end_year=2023, verbose=True)
    print(feature_dict)
        
        
        

            
        


                

        

