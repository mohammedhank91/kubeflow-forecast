import pandas as pd

# ---------------------- LOAD DATA
def load_data():
    csv_file = 'dataScraped/F107_27day_averaged.csv'
    full_data = pd.read_csv(csv_file, sep=';')
    data_copy = full_data.copy()  # Copy that will be useful to save the forecasts at the end
    full_data.index = pd.to_datetime(full_data["epoch"])
    full_data.drop('epoch', axis=1, inplace=True)
    return full_data, data_copy