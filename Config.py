import os


class Config:
    base_path = os.getcwd()
    data_path = os.path.join(base_path, "data")
    model_path = os.path.join(base_path, "model")

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)


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

