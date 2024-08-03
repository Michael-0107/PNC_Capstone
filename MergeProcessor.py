import pickle
import re
import pandas as pd
import numpy as np
import random
from utils import * 

class RatingProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_num = None
        self.df_diff = None
        self.final_ratings = {}

    def load_data(self):
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
        self.df = pd.DataFrame(data).T

    def rating_to_numeric(self, rating):
        rating_order = [
            'F3', 'D', 'RD', 'C', 'CC', 'CCC-', 'CCC', 'CCC+',  
            'B-', 'B', 'B+', 'BB-', 'BB', 'BB+', 
            'BBB-', 'BBB', 'BBB+', 'A-', 'A', 'A+', 
            'AA-', 'AA', 'AA+', 'AAA'
        ]
        rating_to_num = {rating: i for i, rating in enumerate(rating_order)}
        if pd.isna(rating):
            return None
        return rating_to_num.get(rating, None)

    def convert_ratings(self):
        self.df_num = self.df.applymap(self.rating_to_numeric)

    def calculate_diff(self):
        self.df_diff = self.df_num.diff(axis=1)
        first_quarter = self.df.iloc[:, 0]
        self.df_diff.iloc[:, 0] = first_quarter.apply(lambda x: 0 if pd.notna(x) else None)

    def count_changes(self):
        no_change = (self.df_diff == 0).sum().sum()
        increase = (self.df_diff > 0).sum().sum()
        decrease = (self.df_diff < 0).sum().sum()
        return no_change, increase, decrease

    def extract_ratings_by_change_type(self):
        increase = []
        decrease = []
        no_change = []
        for company in self.df.index:
            for quarter in self.df.columns:
                if self.df_diff.loc[company, quarter] > 0:
                    increase.append((company, quarter))
                elif self.df_diff.loc[company, quarter] < 0:
                    decrease.append((company, quarter))
                elif self.df_diff.loc[company, quarter] == 0:
                    no_change.append((company, quarter))
        
        return increase, decrease, no_change

    def select_samples(self, increase, decrease, no_change):
        num_samples = min(len(increase), len(decrease), len(no_change))
        selected_increase = random.sample(increase, num_samples)
        selected_decrease = random.sample(decrease, num_samples)
        selected_no_change = random.sample(no_change, num_samples)
        return selected_increase + selected_decrease + selected_no_change

    def create_final_ratings(self, selected_data):
        for company, quarter in selected_data:
            if company not in self.final_ratings:
                self.final_ratings[company] = {}
            self.final_ratings[company][quarter] = self.df.loc[company, quarter]

    def save_final_ratings(self, output_path):
        pd.to_pickle(self.final_ratings, output_path)

    def process_ratings(self, output_path):
        self.load_data()
        self.convert_ratings()
        self.calculate_diff()
        no_change, increase, decrease = self.count_changes()
        print(f"No Change: {no_change}")
        print(f"Increase: {increase}")
        print(f"Decrease: {decrease}")

        increase, decrease, no_change = self.extract_ratings_by_change_type()
        selected_data = self.select_samples(increase, decrease, no_change)
        self.create_final_ratings(selected_data)
        self.save_final_ratings(output_path)
        return self.final_ratings