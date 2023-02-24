GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch.nn as nn
import torch

class DotProdScorer(nn.Module):
    def __init__(self, device):
        super(DotProdScorer, self).__init__()

        # Define the device_ops
        self.device = device

    def forward(self, user_embeddings, item_embeddings):
        # Scores based on the learned user/item embeddings
        if self.training:
            assert user_embeddings.size()[0] == item_embeddings.size()[0] # Equals to batch_size
            # Score user-item pairs aligned in user_embeddings and item_embeddings
            scores = (user_embeddings * item_embeddings).sum(-1).squeeze()
            ## Shape of scores: (batch_size)
        else:
            # Score every pair made of a row from user_embeddings and a row from item_embeddings
            scores = torch.mm(user_embeddings, item_embeddings.t())
            ## Shape of scores: (batch_size, num_item)

        return scores
