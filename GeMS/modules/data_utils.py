GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch
import pytorch_lightning as pl

from recordclass import recordclass
from typing import List, Dict
import random
from tqdm import tqdm

Trajectory = recordclass("Trajectory", ("obs", "action", "reward", "next_obs", "done"))


class SlateDataset(torch.utils.data.Dataset):
    '''
        Dataset containing slates and corresponding clicks, outside of any trajectory structure.
    '''
    def __init__(self, device : torch.device, filename : str = None,
                            data : Dict = {}, full_traj : bool = False) -> None:

        if full_traj:
            self.data = data
        else:
            self.data = {}
            compt = 0
            for key, val in data.items():
                for i, (slate, clicks) in enumerate(zip(val["slate"], val["clicks"])):
                    self.data[compt + i] = {"slate" : slate, "clicks" : clicks}
                compt = len(self.data)

        self.filename = filename

        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class SlateDataModule(pl.LightningDataModule):
    '''
        DataModule that serves interactions from a SlateDataset
    '''
    def __init__(self, batch_size : int, full_traj : bool, device : str, data : Dict = {}, 
                        n_train_ep : int = 0, n_val_ep : int = 0, **kwargs) -> None:
        super().__init__()

        n = len(data)
        if n > 0:
            val_data = {k : val for k, val in enumerate(list(data.values())[:n//10])}
            train_data = {k : val for k, val in enumerate(list(data.values())[n//10:])}
            self.train_dataset = SlateDataset(device, data = train_data, full_traj = full_traj)
            self.val_dataset = SlateDataset(device, data = val_data, full_traj = full_traj)
        else:
            raise ValueError("data is empty")

        self.num_workers = 0
        self.batch_size = batch_size

    def collate_fn(self, batch : List[Dict]) -> Trajectory:
        obs = {"slate" : [b["slate"] for b in batch],
                "clicks" : [b["clicks"] for b in batch]}
        return Trajectory(obs, None, None, None, None)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.train_dataset, collate_fn = self.collate_fn,
                                            batch_size = self.batch_size, num_workers = self.num_workers)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.val_dataset, collate_fn = self.collate_fn,
                                            batch_size = self.batch_size, num_workers = self.num_workers)


class MFDataset(torch.utils.data.Dataset):
    '''
        Dataset used for the pre-training of item embeddings using Matrix Factorization.
    '''
    def __init__(self, data : Dict):
        self.data = [(u_id,i_id.item()) for u_id, user_traj in data.items()
                                        for k, i_id in enumerate(user_traj["slate"].flatten())
                                        if user_traj["clicks"].flatten()[k] == 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
