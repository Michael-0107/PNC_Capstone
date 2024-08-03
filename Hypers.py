import os

numeric_features = ["actq",
    "ancq", 
    "atq",
    "cheq",
    "cogsq",
    "invtq",
    "lctq",
    "lltq",
    "ltq",
    "niq",
    "ppentq",
    "revtq",
    "teqq",
    "xsgaq"
]

derived_features = [
    "GrossProfitRatio",
    "NetProfitRatio",
    "CurrentRatio",
    "QuickAcidRatio",
    "CashRatio",
    "EquityMultiplier",
    "ReturnOnAsset",
    "ReturnOnEquity",
    "InventoryTurnover"
]

feature_list = numeric_features + derived_features

rating_to_category = {
    "AAA": 0, 
    "AA+": 1,
    "AA": 2,
    "AA-": 3, 
    "A+": 4,
    "A": 5,
    "A-": 6, 
    "BBB+": 7,
    "BBB": 8,
    "BBB-": 9, 
    "BB+": 10,
    "BB": 11,
    "BB-": 12, 
    "B+": 13,
    "B": 14,
    "B-": 15, 
    "CCC+": 16,
    "CCC": 17,
    "CCC-": 18, 
    "CC": 19,
    "C": 20,
    "RD": 21,
    "D": 22, 
    "F3": 23,
    'SD': 24,
    'NG': 25
}



category_to_rating = {v: k for k, v in rating_to_category.items()}

class Config:
    base_path = os.getcwd()
    data_path = os.path.join(base_path, "data")
    model_path = os.path.join(base_path, "model")
    log_path = os.path.join(base_path, "log")

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    seed = 15
    
    # Data preprocessing hyperparameters
    record_begin_threshold = "2010-01-01"
    record_end_threshold = "2017-01-01"
    cpi_path = os.path.join(data_path, "cpi.pkl")


    # Training hyperparameters
    epochs = 500
    batch_size = 32
    train_ratio = 0.5

    # LSTM hyperparameters
    learning_rate = 1e-3
    hidden_size = 256
    proj_size = 0




