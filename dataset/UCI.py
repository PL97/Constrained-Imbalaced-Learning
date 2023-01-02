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
    df_train, df_test = random_split(df, r=0.1)
    print(df_train.shape, df_test.shape)
    
    X_train, y_train = df_train.iloc[:, :-1].to_numpy(), df_train.iloc[:, -1].to_numpy()
    X_test, y_test = df_test.iloc[:, :-1].to_numpy(), df_test.iloc[:, -1].to_numpy()
    print("label distribution:", Counter(df.iloc[:, -1]))
    # print(df.head())
    # print(X_train.shape, y_train.shape)
    # asdf
    ## cast data to float type
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32).reshape(-1, 1)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32).reshape(-1, 1)
    
    
    return torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).to(device), X_train, y_train
    
