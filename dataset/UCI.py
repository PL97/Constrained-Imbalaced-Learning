import numpy as np
import pandas as pd
from collections import Counter
import torch


def random_split(df, r=0.1):
    idx = list(range(df.shape[0]))
    np.random.shuffle(idx)
    df1 = df.iloc[idx[int(df.shape[0]*r):], :].reset_index(drop=True)
    df2 = df.iloc[idx[:int(df.shape[0]*r)], :].reset_index(drop=True)
    return df1, df2
        

def get_data(name='adult', device=None):
    df = pd.read_csv(f"binary_data/{name}.csv")
    
    X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()
    print("label distribution:", Counter(df.iloc[:, -1]))
    ## cast data to float type
    X = X.astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1)
    
    return torch.from_numpy(X).to(device), torch.from_numpy(y).to(device), X, y
    