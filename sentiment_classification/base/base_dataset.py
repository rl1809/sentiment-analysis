from abc import abstractmethod

from torch.utils.data import Dataset


class Dataset(Dataset):
    """Base dataset class"""
    def __init__(self, contents, labels):
        self.contents = contents
        self.labels = labels

    def __len__(self):
        return len(self.contents)

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError
