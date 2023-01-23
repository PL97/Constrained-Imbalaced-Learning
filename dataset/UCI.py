import numpy as np
import pandas as pd
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
        
        
class UCI_dataset(Dataset):
    def __init__(self, X, y):
        """UCI dataset instance

        Args:
            X (numpy array or tensor): features
            y (numpy array or tensor): labels
        """
        if isinstance(X, torch.Tensor):
            self.X, self.y = X, y
        else:
            self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
        self.set_ret_idx(ret=True)
    
    def __len__(self):        
        return self.X.shape[0]
    
    def set_ret_idx(self, ret):
        self.ret_idx = ret
    
    def __getitem__(self, index):
        if self.ret_idx:
            return index, self.X[index], self.y[index]
        else:
            return self.X[index], self.y[index]
  

def get_data(name='adult', batch_size=10, random_seed=1997):
    """get UCI dataset and prepare dataloader

    Args:
        name (str, optional): name of dataset in UCI, select from binary_data/ folder. Defaults to 'adult'.
        batch_size (int, optional): batch size of the returned dataloader. Defaults to 10.
        random_seed (int, optional): random seed to generate train val test split. Defaults to 1997.

    Returns:
        dataloaders and data stats (as a dict)
    """
    df = pd.read_csv(f"binary_data/{name}.csv")
    
    X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy()
    ## cast data to float type
    X = X.astype(np.float32)
    y = y.astype(np.float32).reshape(-1, 1)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, \
                                                      y, \
                                                      test_size=0.2, \
                                                      stratify=y, \
                                                      random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, \
                                                    y_tmp, \
                                                    test_size=0.5, \
                                                    stratify=y_tmp, \
                                                    random_state=random_seed)

    stats = {
        'feature_dim': X.shape[1], \
        'label_num': len(np.unique(y)), \
        'total_num': X.shape[0], \
        'train_num': X_train.shape[0],\
        'test_num': X_test.shape[0], \
        'val_num': X_val.shape[0], \
        'label_distribution': Counter(df.iloc[:, -1])
    }
    trainloader = DataLoader(UCI_dataset(X=X_train, y=y_train), \
                            batch_size=batch_size, \
                            shuffle=True, \
                            num_workers=4)
    
    valloader = DataLoader(UCI_dataset(X=X_val, y=y_val), \
                            batch_size=batch_size, \
                            shuffle=False, \
                            num_workers=4) 
    
    testloader = DataLoader(UCI_dataset(X=X_test, y=y_test), \
                            batch_size=batch_size, \
                            shuffle=False, \
                            num_workers=4) 
    
    return trainloader, valloader, testloader, stats
    
    




if __name__ == "__main__":
    pass
    # from torch.utils.data import DataLoader
    # train, val, test = get_data()
    # for i, tmp_X, tmp_y in train:
    #     print(i, tmp_X, tmp_y)
    #     exit("finished")