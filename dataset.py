import os
import pandas as pd
import pickle
import numpy as np

from settings import Config
from torch.utils.data import Dataset


class SWaTDataset(Dataset):
    def __init__(self, config, force_flush=False, training=True,orders=None):
        super(SWaTDataset, self).__init__()
        pickle_path = config.pickle_path
        train_or_test = "train" if training is True else "test"
        df = pickle.load(open(os.path.join(pickle_path, "swat_{}.pkl".format(train_or_test)), "rb"))
        if orders is not None:
            df = df.reindex(orders)
        self.df = df

    def __getitem__(self, item):
        data = self.df.iloc[item,].to_list()
        return {
            "data": np.array(data[:-1], dtype=np.float32),
            "label": np.array(data[-1], dtype=np.long)
        }

    def __len__(self):
        return self.df.shape[0]

    @property
    def input_size(self):
        """InputSize= # of input features"""
        return len(self.df.columns) - 1


class BATADALDataset(Dataset):
    def __init__(self, config, training=True,orders=None):
        super(BATADALDataset, self).__init__()
        pickle_path = config.pickle_path
        train_or_test = "train" if training is True else "test"
        df = pickle.load(open(os.path.join(pickle_path, "batadal_{}.pkl".format(train_or_test)), "rb"))
        if orders is not None:
            df=df.reindex(orders)
        self.df = df

    def __getitem__(self, item):
        data = self.df.iloc[item,].to_list()
        return {
            "data": np.array(data[:-1], dtype=np.float32),
            "label": np.array(data[-1], dtype=np.float32)
        }

    def __len__(self):
        return self.df.shape[0]

    @property
    def input_size(self):
        """InputSize= # of input features"""
        return len(self.df.columns) - 1


class WADIDataset(Dataset):
    def __init__(self, config, training=True,orders=None):
        super(WADIDataset, self).__init__()
        pickle_path = config.pickle_path
        train_or_test = "train" if training is True else "test"
        df = pickle.load(open(os.path.join(pickle_path, "wadi_{}.pkl".format(train_or_test)), "rb"))
        if orders is not None:
            df=df.reindex(orders)
        self.df = df

    def __getitem__(self, item):
        data = self.df.iloc[item,].to_list()
        return {
            "data": np.array(data[:-1], dtype=np.float32),
            "label": np.array(data[-1], dtype=np.float32)
        }

    def __len__(self):
        return self.df.shape[0]

    @property
    def input_size(self):
        """InputSize= # of input features"""
        return len(self.df.columns) - 1


class PHMDataset(Dataset):
    def __init__(self,config,training=True,orders=None):
        pickle_path = config.pickle_path
        train_or_test = "train" if training is True else "test"
        df = pickle.load(open(os.path.join(pickle_path, "phm_{}.pkl".format(train_or_test)), "rb"))
        if orders is not None:
            df=df.reindex(orders)
        self.df = df

    def __getitem__(self, item):
        data = self.df.iloc[item,].to_list()
        return {
            "data": np.array(data[:-1], dtype=np.float32),
            "label": np.array(data[-1], dtype=np.float32)
        }

    def __len__(self):
        return self.df.shape[0]

    @property
    def input_size(self):
        return len(self.df.columns) - 1

class GASDataset(Dataset):
    def __init__(self,config,training=True,orders=None):
        pickle_path = config.pickle_path
        train_or_test = "train" if training is True else "test"
        df = pickle.load(open(os.path.join(pickle_path, "gas_{}.pkl".format(train_or_test)), "rb"))
        if orders is not None:
            df=df.reindex(orders)
        self.df = df

    def __getitem__(self, item):
        data = self.df.iloc[item,].to_list()
        return {
            "data": np.array(data[:-1], dtype=np.float32),
            "label": np.array(data[-1], dtype=np.float32)
        }

    def __len__(self):
        return len(self.df.columns) - 1


if __name__ == '__main__':
    config = Config()
    swat_dataset = SWaTDataset(config, force_flush=False)
    # todo train test split
