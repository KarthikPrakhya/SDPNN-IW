import torch
from torch.utils.data import Dataset


class PrepareData(Dataset):
    """
    The PrepareData class preprocesses a dataset to store as a torch.utils.data.Dataset object to iterate over.
    """

    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X

        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y).float()
        else:
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
