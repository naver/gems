GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch.nn as nn
import torch
import numpy as np
from .layers import DotProdScorer
from .utils.loss import bpr_loss
from .utils.sample import sample_items

class BPRMatrixFactorization(nn.Module):
    """
        Implementation of the matrix factorization with a BPR loss and trained with SGD
    """
    def __init__(self, num_user, num_item, options, device_embed, device_ops):
        super(BPRMatrixFactorization, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.embed_dim = options.embedd_dim
        self.lr = options.lr_embedd
        self.num_neg_sample = options.num_neg_sample
        self.device_embed = device_embed
        self.device_ops = device_ops

        # Embeddings
        self.user_embeddings = nn.Embedding(num_user, self.embed_dim) # User embeddings to be learned
        self.item_embeddings = nn.Embedding(num_item, self.embed_dim) # Item embeddings to be learned
        nn.init.xavier_uniform_(self.user_embeddings.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embeddings.weight, gain=1)
        self.user_embeddings = self.user_embeddings.to(device_embed)
        self.item_embeddings = self.item_embeddings.to(device_embed)

        # Components of the model
        self.scorer = DotProdScorer(device_ops).to(device_ops)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=options.weight_decay_embedd)

    def predict(self, user_ids, item_ids=None):
        """
        Compute the score predictions at test time
        Args:
            user_ids: (array<int>) users for whom to recommend items
            item_ids: (array<int>) items for which prediction scores are desired; if not provided, predictions for all
            items will be computed
        Returns:
            scores: (tensor<float>) predicted scores for all items in item_ids
        """
        batch_user_embeddings = self.user_embeddings(user_ids).to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)

        if item_ids is None:
            item_ids = np.arange(self.num_item)
        item_ids = torch.tensor(item_ids, dtype=torch.long, device=self.device_embed)
        batch_item_embeddings = self.item_embeddings(item_ids).to(self.device_ops)
        ## Shape of batch_item_embeddings: (num_item, embed_dim)

        scores = self.scorer(batch_user_embeddings, batch_item_embeddings)
        ## Shape of scores: (batch_size, num_item)
        return scores

    def forward(self, batch):
        # Unpack the content of the minibatch
        user_ids = batch['user_ids']
        item_ids = batch['item_ids']

        # Fetch the user embeddings for the minibatch
        batch_user_embeddings = self.user_embeddings(user_ids).to(self.device_ops)
        ## Shape of batch_user_embeddings: (batch_size, embed_dim)

        # Fetch the (positive) item embeddings for the minibatch
        batch_item_embeddings = self.item_embeddings(item_ids).to(self.device_ops)
        ## Shape of batch_item_embeddings: (batch_size, embed_dim)

        # Calculate the recommendation loss on the minibatch using BPR
        positive_score = self.scorer(batch_user_embeddings, batch_item_embeddings)

        ## Shape of positive_score: (batch_size)
        loss = torch.tensor(0.0, dtype=torch.float, device=self.device_ops)
        for i in range(self.num_neg_sample):
            # Negative sampling
            negative_item_ids = sample_items(self.num_item, item_ids.size())
            negative_item_ids = torch.tensor(negative_item_ids, dtype=torch.long, device=self.device_embed)
            batch_negative_item_embeddings = self.item_embeddings(negative_item_ids).to(self.device_ops)
            ## Shape of batch_negative_item_embeddings: (batch_size, embed_dim)
            negative_score = self.scorer(batch_user_embeddings, batch_negative_item_embeddings)
            ## Shape of negative_score: (batch_size)
            # Compute the BPR loss on the positive and negative scores while masking padded elements in the sequences
            loss += bpr_loss(positive_score, negative_score)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
