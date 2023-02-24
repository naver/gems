GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch
import pytorch_lightning as pl

from collections import deque
from recordclass import recordclass
from typing import List, Tuple, Dict
import copy
import random
from tqdm import tqdm

from RecSim.simulators import TopicRec

Trajectory = recordclass("Trajectory", ("obs", "action", "reward", "next_obs", "done", "info"))

class ReplayBuffer():
    '''
        This ReplayBuffer class supports both tuples of experience and full trajectories,
        and it allows to never discard environment transitions for Offline Dyna.
    '''
    def __init__(self, offline_data : List[Trajectory], capacity : int) -> None:

        self.buffer_env = deque(offline_data, maxlen = capacity)
        self.buffer_model = deque([], maxlen = capacity)

    def push(self, buffer_type : str, *args) -> None:
        """Save a trajectory or tuple of experience"""
        if buffer_type == "env" :
            self.buffer_env.append(Trajectory(*args))
        elif buffer_type == "model":
            self.buffer_model.append(Trajectory(*args))
        else:
            raise ValueError("Buffer type must be either 'env' or 'model'.")

    def sample(self, batch_size : int, from_data : bool = False) -> List[Trajectory]:
        if from_data:
            return random.sample(self.buffer_env, batch_size)
        else:
            if len(self.buffer_env + self.buffer_model) < batch_size:
                return -1
            return random.sample(self.buffer_env + self.buffer_model, batch_size)

    def __len__(self) -> int:
        return len(self.buffer_env) + len(self.buffer_model)

class BufferDataset(torch.utils.data.IterableDataset):
    def __init__(self, buffer: ReplayBuffer, batch_size: int) -> None:
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.buffer.sample(self.batch_size)

class BufferDataModule(pl.LightningDataModule):
    '''
        DataModule that serves batches to the agent.
    '''
    def __init__(self, batch_size : int, capacity : int, offline_data : List[Trajectory] = [], **kwargs) -> None:
        super().__init__()

        self.buffer = ReplayBuffer(offline_data, capacity)
        self.buffer_dataset = BufferDataset(self.buffer, batch_size)
        self.num_workers = 0

    def collate_fn(self, batch):
        if batch == [-1]:
            # Special case of num_steps < batch_size
            return 0
        batch = Trajectory(*zip(*batch[0]))
        if batch.next_obs[0] is None:   ## POMDP
            batch.obs = {key : [obs[key] for obs in batch.obs] for key in batch.obs[0].keys()}
            batch.next_obs = None
            batch.action = torch.cat(batch.action, dim = 0)
            batch.reward = torch.cat(batch.reward, dim = 0)
            batch.done = torch.cat(batch.done, dim = 0)
            if batch.info[0] is not None:
                batch.info = torch.cat(batch.info, dim = 0)
        else:                           ## MDP
            batch.obs = torch.stack(batch.obs)
            batch.next_obs = torch.stack(batch.next_obs)
            batch.action = torch.stack(batch.action)
            batch.reward = torch.stack(batch.reward, dim = 0).squeeze()
            batch.done = torch.stack(batch.done, dim = 0).squeeze()
        return batch

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.buffer_dataset, collate_fn = self.collate_fn,
                                                num_workers = self.num_workers)

    def val_dataloader(self)-> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.buffer_dataset, collate_fn = self.collate_fn,
                                                num_workers = self.num_workers, shuffle = False)

    def test_dataloader(self)-> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.buffer_dataset, collate_fn = self.collate_fn,
                                                num_workers = self.num_workers, shuffle = False)


class EnvWrapper():
    '''
        This classprovides a unified interface for gym environments, custom PyTorch environments, and model in model-based RL.
    '''
    def __init__(self, buffer : ReplayBuffer, device : torch.device, env_name : str, dyn_model : pl.LightningModule = None, **kwargs) -> None:

        self.device = device
        self.buffer = buffer
        self.obs = None
        self.done = True

        if env_name is not None:
            if env_name in ["topics", "TopicRec"]:
                self.gym = False
                if env_name in ["topics", "TopicRec"]:
                    env_class = TopicRec
                else:
                    raise NotImplementedError("This environmenet has not been implemented.")
                self.env = env_class(device = device, **kwargs)
            self.dynmod = False
        elif dyn_model is not None:
            self.dynmod = True
            self.gym = False
            self.env = dyn_model
        else:
            raise ValueError("You must specify either a gym ID or a dynamics model.")

    def reset(self) -> torch.FloatTensor:
        self.done = False
        if self.dynmod:
            traj = self.buffer.sample(batch_size = 1, from_data = True)
            self.obs =  traj.obs[0, :]
        else:
            self.obs, info = self.env.reset()
        return self.obs

    def step(self, action : torch.Tensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, Dict]:
        next_obs, reward, done, info = self.env.step(action)
        self.obs =  copy.deepcopy(next_obs)

        self.done = done
        return self.obs, reward.float(), torch.tensor(done, device = self.device).long(), info

    def get_obs(self) -> Tuple[torch.FloatTensor, bool]:
        return self.obs, self.done

    def get_dimensions(self) -> Tuple[int, int]:
        return self.env.get_dimensions()

    def get_item_embeddings(self) -> torch.nn.Embedding:
        return self.env.get_item_embeddings()

    def get_random_action(self):
        return self.env.get_random_action()



def get_file_name(arg_dict):
    filename = arg_dict["agent"] + "_"
    if arg_dict["env_name"] != "Walker2DBulletEnv-v0":
        filename += arg_dict["ranker"] + "_"
        if arg_dict["env_probs"] == [0.0, 1.0, 0.0]:
            cm = "DBN_"
        else:
            cm = "MixDBN_"
        filename += cm
        if arg_dict["ranker"] in ["GeMS"]:
            ranker_checkpoint = arg_dict["ranker_checkpoint"]
            logging_policy, dataset_size, beta = ranker_checkpoint.split("_")[2:5]
            item_embedds = "_".join(ranker_checkpoint.split("_")[5:])
            filename += logging_policy + "_" + dataset_size + "_" + beta + "_" + item_embedds + "_"
        elif arg_dict["MF_checkpoint"] is not None:
            mf_checkpoint = arg_dict["MF_checkpoint"]
            mf_checkpoint = mf_checkpoint.split(".")[0] # Remove suffix .pt
            logging_policy, dataset_size = mf_checkpoint.split("_")[1:3]
            item_embedds = "mf"
            filename += logging_policy + "_" + dataset_size + "_" + item_embedds + "_"
        else: # True or from-scratch embeddings
            item_embedds = arg_dict["item_embedds"]
            filename += item_embedds + "_"
    else:
        filename += "walker_"
    return filename + "seed" + str(arg_dict["seed"]) + ".pt"
