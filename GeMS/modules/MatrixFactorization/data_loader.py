GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

from torch.utils.data import Dataset
import torch

def user_collate_fn(batch):
    batch_size = len(batch)
    batch_max_size = max([len(sample['interactions']) for sample in batch]) # Max #interactions per user in the batch

    # Collate interactions, sequence sizes, and queries from the current minibatch
    collated_user_ids = torch.zeros(batch_size, dtype=torch.long) # IDs of the users in the batch
    collated_user_interactions = torch.zeros((batch_size, batch_max_size), dtype=torch.long) # Collated interactions
    user_seq_sizes = torch.zeros(batch_size, dtype=torch.long) # Users' #interactions for future unpadding
    for (i, sample) in enumerate(batch):
        # Save the ID of the user in the sample to retrieve the corresponding embedding later
        collated_user_ids[i] = sample['user_id']

        # Pad interactions
        user_interactions = sample['interactions']
        current_seq_size = min(batch_max_size, len(user_interactions)) # Cut off sequences that are too long
        latest_interactions = user_interactions[-current_seq_size:] # Keep only the latest interactions if cut off
        latest_interactions = torch.tensor(latest_interactions, dtype=torch.long) # Convert to a tensor
        collated_user_interactions[i, :current_seq_size] = latest_interactions

        # Save the size of user sequences
        user_seq_sizes[i] = current_seq_size

    return {'user_ids': collated_user_ids, 'interactions': collated_user_interactions, 'seq_sizes': user_seq_sizes}

def interaction_collate_fn(batch):
    batch_size = len(batch)

    # Collate interactions, sequence sizes, and queries from the current minibatch
    collated_user_ids = torch.zeros(batch_size, dtype=torch.long) # IDs of the users in the batch
    collated_item_ids = torch.zeros(batch_size, dtype=torch.long)  # IDs of the interacted items in the batch

    for (i, sample) in enumerate(batch):
        # Save the ID of the user in the sample
        collated_user_ids[i] = sample['user_id']

        # Save the ID of the interacted item in the sample
        collated_item_ids[i] = sample['item_id']


    return {'user_ids': collated_user_ids, 'item_ids': collated_item_ids}

class UserDataset(Dataset):
    """
        Dataset where a user (with all her interactions on items) is a single sample
    """
    def __init__(self, user_interactions, options):
        self.user_interactions = user_interactions
        self.user_ids = list(user_interactions.keys())

    def __len__(self):
        return len(self.user_interactions)

    def __getitem__(self, idx):
        sample = {'user_id': self.user_ids[idx], 'interactions': self.user_interactions[self.user_ids[idx]]}
        return sample

class InteractionDataset(Dataset):
    """
        Dataset where an interaction (user + clicked item) is a single sample
    """
    def __init__(self, user_interactions, data_size, options):

        # Build the user and clicked item vectors by considering each interaction as a sample
        (self.num_user, self.num_item) = data_size
        self.user_ids = []
        self.item_ids = []
        for user in user_interactions.keys():
            for item in user_interactions[user]:
                self.item_ids.append(item)
            self.user_ids.extend([user] * len(user_interactions[user]))

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        sample = {'user_id': self.user_ids[idx], 'item_id': self.item_ids[idx]}
        return sample
