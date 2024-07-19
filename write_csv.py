import csv
import pandas as pd
import numpy as np

from utils import *
import transformers
from transformers import pipeline
import torch

from huggingface_hub import login

login("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")

print("Company Info read from CSV file")
df = pd.read_csv('comp.csv')

with torch.no_grad():

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.float32},
        device="cuda:0",
    )
    for index, row in df.iterrows():
        if row['news_summary'] != 0: continue
        try:
            df.at[index, 'news_summary'] = summarize_news(pipeline, row['company'], row['ticker'], row['startdate'], row['enddate'])
        except:
            df.at[index, 'news_summary'] = "There are no information of this company for this quarter."
        if index % 10 == 0:
            df.to_csv('comp.csv', index=False)

            
### Build base csv from temp.txt
# q = {
#     1: ("-01-01", "-03-31"),
#     2: ("-04-01", "-06-30"),
#     3: ("-07-01", "-09-30"),
#     4: ("-10-01", "-12-31")
#     }

# year = list(range(2009,2018))

# with open('temp.txt', 'r') as in_file:
#     stripped = (line.strip() for line in in_file)
#     lines = [line.split("\t")[:3] for line in stripped if line]
#     new_lines = []
#     for line in lines:
#         for y in year:
#             for quar in q:
#                 new_line = line.copy()
#                 new_line.append(str(y)+q[quar][0])
#                 new_line.append(str(y)+q[quar][1])
#                 new_lines.append(new_line)
        
#     with open('comp.csv', 'w') as out_file:
#         writer = csv.writer(out_file)
#         writer.writerow(('number', 'ticker', 'company', 'startdate', 'enddate', 'news_summary'))
#         writer.writerows(new_lines)