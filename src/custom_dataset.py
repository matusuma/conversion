import torch
from torch.utils.data import Dataset
from loguru import logger
import numpy as np


class CustomDataset:
    def __init__(self, pixel_data, target_data, date):
        """
        Initialize the dataset
        :param pixel_data: np.ndarray of shape (identities, n_events, features(=7))
        :param target_data: np.ndarray of shape (identities)
        :param date: date of datablock
        """
        self.x = [torch.from_numpy(x) for x in pixel_data]
        self.y = torch.from_numpy(target_data)
        self.date = date

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CompleteDataset(Dataset):
    def __init__(self, pixel_datablocks, target_datablocks, dates):
        """
        :param pixel_datablocks: list(per day) of np.ndarrays of shape (identities, n_events, features(=7))
        :param target_datablocks: list(per day) of np.ndarray of shape (identities)
        :param dates: list of dates of corresponding datablocks
        """
        super().__init__()
        self.datasets = []
        self.lengths = [0]
        # create subdataset for each day
        for i in np.arange(len(pixel_datablocks)):
            dataset = CustomDataset(pixel_datablocks[i], target_datablocks[i], dates[i])
            self.lengths += [len(dataset)]
            self.datasets.append(dataset)
        self.indeces = np.cumsum(self.lengths)
        logger.info(f'{self.lengths} -> {self.indeces}')

    def __len__(self):
        return np.sum(self.lengths)

    def __getitem__(self, idx):
        blocknum = np.argmax(idx < self.indeces)
        blockidx = idx - self.indeces[blocknum - 1]
        user_events = self.datasets[blocknum - 1][blockidx]
        return user_events[0], user_events[1]
