GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch
import pytorch_lightning as pl

import os
from torch.nn import Embedding
from typing import List, Tuple, Dict
from collections import namedtuple
from pathlib import Path

from .MatrixFactorization.models import BPRMatrixFactorization
from .data_utils import MFDataset
from .argument_parser import MyParser


class ItemEmbeddings(pl.LightningModule):
    '''
        Base Embedding class.
    '''
    def __init__(self, num_items : int, item_embedd_dim : int, device : torch.device, weights = None, **kwargs) -> None:
        super().__init__()

        self.num_items = num_items
        self.embedd_dim = item_embedd_dim
        self.embedd = Embedding(num_items, item_embedd_dim, _weight = weights).to(device)

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--MF_checkpoint', type=str, default = None)
        arguments = [action.option_strings[0] for action in parser._actions]
        if '--num_items' not in arguments:
            parser.add_argument('--num_items', type=int, default = 1000)
        if '--item_embedd_dim' not in arguments:
            parser.add_argument('--item_embedd_dim', type=int, default = 20)
        return parser

    def forward(self, items : torch.LongTensor) -> torch.FloatTensor:
        return self.embedd(items)

    @classmethod
    def from_pretrained(cls, checkpoint_path : str, device : torch.device):
        weights = torch.load(checkpoint_path)
        num_items, embedd_dim = weights.size()
        return cls(num_items, embedd_dim, weights = weights, device = device)

    @classmethod
    def get_from_env(cls, env, device, data_dir : str = None, embedd_path : str = None):
        embedd_weights = env.get_item_embeddings()
        num_items, embedd_dim = embedd_weights.size()
        return cls(num_items, embedd_dim, weights = embedd_weights, device = device)

    @classmethod
    def from_scratch(cls, num_items : int, embedd_dim : int, device : torch.device):
        return cls(num_items, embedd_dim, device = device)

    def clone_weights(self):
        return self.embedd.weight.data.clone()

    def get_weights(self):
        return self.embedd.weight.data

    def freeze(self):
        self.embedd.requires_grad_(False)

class MFEmbeddings(ItemEmbeddings):
    '''
        Matrix factorization with a BPR loss and trained with SGD. Courtesy of Thibaut Thonet.
    '''
    def __init__(self, train_val_split_MF : float, batch_size_MF : int, lr_MF : float, num_neg_sample_MF : int,
                    weight_decay_MF : float, patience_MF : int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.train_val_split = train_val_split_MF
        self.batch_size = batch_size_MF
        self.lr = lr_MF
        self.num_neg_sample = num_neg_sample_MF
        self.weight_decay = weight_decay_MF
        self.patience = patience_MF

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[ItemEmbeddings.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--MF_dataset', type=str, default = None)
        parser.add_argument('--train_val_split_MF', type=float, default = 0.1)
        parser.add_argument('--batch_size_MF', type=int, default = 256)
        parser.add_argument('--lr_MF', type=float, default = 1e-4)
        parser.add_argument('--num_neg_sample_MF', type=int, default = 1)
        parser.add_argument('--weight_decay_MF', type=float, default = 0)
        parser.add_argument('--patience_MF', type=int, default = 3)
        return parser

    def collate_fn(self, batch : List[Tuple]) -> Dict:
        return {"user_ids" : torch.tensor([b[0] for b in batch], dtype = torch.long, device = self.device),
                "item_ids" : torch.tensor([b[1] for b in batch], dtype = torch.long, device = self.device)}

    def train(self, dataset_path : str, data_dir : str) -> None:
        '''
            Train MF item embeddings on pre-collected dataset.
        '''
        ### Loading the data and pre-processing
        data = torch.load(dataset_path)
        num_user = len(data)

        train_data = {k : val for k, val in enumerate(list(data.values())[int(num_user * self.train_val_split):])}
        val_data = {k : val for k, val in enumerate(list(data.values())[:int(num_user * self.train_val_split)])}

        train_dataset = MFDataset(data = train_data)
        val_dataset = MFDataset(data = val_data)

        print("Number of interactions :")
        print("In training set : ", len(train_dataset))
        print("In validation set : ", len(val_dataset))

        train_gen = torch.utils.data.DataLoader(train_dataset, batch_size = self.batch_size,
                                                    shuffle = True, collate_fn = self.collate_fn)
        val_gen = torch.utils.data.DataLoader(val_dataset, batch_size = self.batch_size,
                                                    shuffle = True, collate_fn = self.collate_fn)

        Options = namedtuple("Options", ["lr_embedd", "embedd_dim", "num_neg_sample", "weight_decay_embedd"])
        options = Options(self.lr, self.embedd_dim, self.num_neg_sample, self.weight_decay)
        model = BPRMatrixFactorization(num_user, self.num_items, options, self.device, self.device)

        ### Training
        epoch = 0
        min_val_loss = 1e10
        count = 0

        while True :
            model.train()
            epoch_loss = 0.0
            train_loss = 0.0
            for (n, batch) in enumerate(train_gen):
                if n % 1000 == 0:
                    model.eval()
                    val_loss = 0.0
                    for (k, val_batch) in enumerate(val_gen):
                        loss = model(val_batch)
                        val_loss += loss.item()
                    val_loss /= (k + 1) # Divide by the number of batches
                    model.train()
                    print("     After %d batches : train_loss = %.4f | val loss = %.4f" % (n, train_loss / 1000, val_loss))

                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        Path(data_dir).mkdir(parents=True, exist_ok=True)
                        torch.save(model.item_embeddings.weight.data, data_dir + dataset_path.split("/")[-1])
                        count = 0
                    else:
                        count += 1
                    if count == self.patience:
                        break

                    train_loss = 0.0

                loss = model(batch)

                epoch_loss += loss.item()
                train_loss += loss.item()

            if count == self.patience:
                break

            epoch_loss /= (n + 1) # Divide by the number of batches


            print('Epoch {}: train loss {}'.format(epoch, epoch_loss))
            epoch += 1
