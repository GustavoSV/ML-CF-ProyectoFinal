import pandas as pd
from preprocess import create_features, load_data
import os

def load_data(path):
    return pd.read_csv(path, encoding="latin1")

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "uniformes_ignifugos.csv")
df = load_data(data_path)
df = create_features(df)
print(df.head())