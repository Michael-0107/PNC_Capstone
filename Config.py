import os


class Config:
    base_path = os.getcwd()
    data_path = os.path.join(base_path, "data")
    model_path = os.path.join(base_path, "model")

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)


tags_interested = [
    "Revenues", 
    "CostOfRevenue", 
    "SellingGeneralAndAdministrativeExpense", 
    "OperatingIncomeLoss", 
    "NetIncomeLoss", 
    "Assets",
    "AssetsCurrent",
    "InventoryNet",
    "CashAndCashEquivalentsAtCarryingValue",
    "NoncurrentAssets",
    "PropertyPlantAndEquipmentNet",
    "LiabilitiesCurrent",
    "LongTermDebt",
    "LongTermDebtCurrent",
    "LongTermDebtNoncurrent",
    "StockholdersEquity"
]
