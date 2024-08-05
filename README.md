# PNC_Capstone

## Quick start
- Abbreviations: 
  - US: all companies in US
  - RetInd: comapnies in either retailing or industrial sectors
  - Ret: Retail only
  - Omni: full 205 raw feature
- Make sure you have raw csv from WRDS, named as features_{subset}.csv or ratings_{subset}.csv in data/WRDS folder. (e.g., features_US.csv, ratings_US.csv, features_RetInd.csv)
- Use CompustatExtractor.py
  - Call process_compustat_features() to process and normalize features into a nested dict
  - Call process_compustat_ratings() to process ratings into a nested dict
  - Call concatenate_features() to concatenate features within k quarters (or else you will be predicting the labels only using the current quarter)
  - Call merge_input_output_dicts() to merge features and ratings

- Check the custom dataset and model for your model is alright (e.g., RegressionDataset.py or LSTMModel.py)

- Use the TrainerDriverCode.ipynb to train
  - Run the prerequisites
  - (If first time or changed the features/ratings) split the dataset into training/testing
  - Find the model you are responsible for, and train it
  - Infer and analyze the results

## 7/31~8/4 Homeworks
- Try out different combinations for results (at least five combinations): 
  - Features: different scaling, normalization, more derived features, more macroecnomic features, balance the dataset, PCA, etc.
  - Model architecture: input size, output size, hidden size, layers, dropout, batchnorm etc. 
  - Hyperparameters: learning rate, batch size, epochs, etc.
- Each of the attempts should surpass the baseline as far as possible.
- Save the graphs and resutls, create tables for comparison.
- Do analysis and write your section in the report

