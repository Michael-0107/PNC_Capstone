import os

feature_list = [
    "actq",
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
    "xsgaq",
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
    "F3": 23
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


    epochs = 1000
    batch_size = 4
    learning_rate = 0.001
    hidden_size = 128
    proj_size = len(rating_to_category)

    train_ratio = 0.8




