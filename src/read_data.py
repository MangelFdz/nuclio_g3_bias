
import pandas as pd

def read_scores_raw(path):
    df =  pd.read_csv(path)
    print(f"scores raw shape: {df.shape}")

    return df

def read_scores_2y(path):
    df =  pd.read_csv(path)
    print(f"scores 2y shape: {df.shape}")

    return df