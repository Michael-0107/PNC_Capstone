import pandas as pd

def calculate_nan_percentage(df):
    return df.isna().mean().mean() * 100

def preprocess(file):

    # Define thresholds
    small_portion_threshold = 0.2  # Columns with less than 20% NaNs will be filled
    large_portion_threshold = 0.5  # Rows with more than 50% NaNs will be dropped

    # Load the data
    data = pd.read_csv(file, low_memory=False)

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
    data.to_csv('RawData/cleaned_financial_data.csv', index=False)

    print("Data cleaning complete.")
    print("Totally {} companies are included.".format(len(data['tic'].unique())))
    print("Cleaned data saved to 'cleaned_financial_data.csv'.")

    return

if __name__ == "__main__":

    preprocess('RawData/features_omni.csv')