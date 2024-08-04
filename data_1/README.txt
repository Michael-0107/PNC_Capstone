{train/test}_dict.pkl: 
    Training/Testing data, contains a nested dict: {company_ticker: {period: (feature: rating)}} (e.g. {AMZN: {2024Q1: tuple(Tensor, Tensor)}})
    The features are already normalized
    The ratings are transformed into integer

feature_retail_indus_normalized_dict.pkl:
    Contains all the features for companies in retailing/industrial category. Nested dict with {company_ticker: {period: feature}}
    The features are already normalized (e.g. AAA->0)

ratings_retail_indus.pkl
    Contains all the ratings for companies in retailing/industrial category. Nested dict with {company_ticker: {period: ratings}}
    Ratings are still 

features_retail_indus_scaler_df.csv
    Contains the features, not normalized yet. FYI.
