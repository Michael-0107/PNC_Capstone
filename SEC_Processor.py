import os
import pandas as pd

from Config import Config, tags_interested


class SEC_Processor:
    def __init__(self, year, quarter):
        target_folder = os.path.join(Config.data_path, "SEC_Fillings", f"{year}q{quarter}")

        sub_path = os.path.join(target_folder, "sub.txt")
        num_path = os.path.join(target_folder, "num.txt")

        self.sub_df = pd.read_csv(sub_path, sep="\t")
        self.num_df = pd.read_csv(num_path, sep="\t")

    def get_company_key(self, company_name, form='10-K'):
        target_row = self.sub_df[(self.sub_df["name"]==company_name) & (self.sub_df["form"] == "10-K")]
        return target_row['adsh'].values[0]
    
    def get_company_nums(self, company_key):
        target_df = self.num_df[(self.num_df["adsh"] == company_key) & (self.num_df["tag"].isin(tags_interested))]

        years_distinct = sorted(list(set(target_df["ddate"])), reverse=True)

        ret_dict = {}
        for year in years_distinct:
            ret_dict[year] = {}

        for idx, row in target_df.iterrows():
            year = row["ddate"]
            tag = row["tag"]
            ret_dict[year][tag] = row["value"]

        return ret_dict
    

if __name__ == "__main__":
    processor = SEC_Processor(year=2024, quarter=1)
    walmart_key = processor.get_company_key("WALMART INC.", form='10-K')
    print(walmart_key)