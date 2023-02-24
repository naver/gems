GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import pandas as pd
import numpy as np
import torch
import os
import sys
from option_parser import OptParser
from data_preprocess import process_interactions, load_data
from data_loader import UserDataset, user_collate_fn
from torch.utils.data import RandomSampler, DataLoader
from evaluation import predict_evaluate
from models import BPRMatrixFactorization
from datetime import datetime
import random

# Option definition
optparser = OptParser()
options = optparser.parse_args()[0]

# Seeds for reproducibility
seed = int(options.seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

# Settings
## Model
model_type = options.model
if model_type == "BPRMatrixFactorization":
    from data_loader import InteractionDataset as Dataset
    from data_loader import interaction_collate_fn as collate_fn
else:
    print("Model unknown!")
    exit()
## Training
num_epoch = options.num_epoch
batch_size = options.batch_size
num_workers = options.num_workers
## Evaluation
use_valid = options.use_valid # Use a validation set, otherwise the validation set is integrated to the training set
spl_valid_size = int(options.valid_size) # Sample of users for fast evaluation on the validation set during training
valid_level = "user" # Evaluation level used for model selection on validation set (macro-level or micro-level)
metrics = [("precision", 1), ("precision", 10), ("precision", 20), ("recall", 1), ("recall", 10), ("recall", 20),
           ("ndcg", 20), ("map", 20)]
## Pre-training
use_pre = options.use_pre # Indicates whether pre-trained embeddings should be used
save_embed = options.save_embed # Indicates whether the embeddings should be saved at the end

# Data
## Paths
dataset = options.dataset
data_dir = "data" + os.sep + dataset
data_size_path = data_dir + os.sep + "data_size.txt"
train_path = data_dir + os.sep + "train.txt"
valid_path = data_dir + os.sep + "valid.txt"
test_path = data_dir + os.sep + "test.txt"
## Load the dataset
data_size = [int(e) for e in open(data_size_path, "r").readline().split("\t")] # Gives (num_user, num_item)
num_user = data_size[0]
num_item = data_size[1]
print("Number of users:", num_user)
print("Number of items:", num_item)
print("Loading datasets...", datetime.now())
fields = ['user_id', 'item_id', 'frequency', 'time', 'lat', 'long', 'query'] 
delimiter = '\t'
train_data = load_data(train_path, fields, delimiter)
valid_data = load_data(valid_path, fields, delimiter)
if not use_valid:
    train_data = pd.concat([train_data, valid_data]) # Use training and validation data as training set
test_data = load_data(test_path, fields, delimiter)

# Initialize the gpu usage
device_embed = torch.device(options.device_embed + ":" + str(options.cuda)) if options.device_embed == "cuda" \
    else torch.device(options.device_embed)
device_ops = torch.device(options.device_ops + ":" + str(options.cuda)) if options.device_ops == "cuda" \
    else torch.device(options.device_ops)
print("Using device_embed: {0}".format(device_embed))
print("Using device_ops: {0}".format(device_ops))

# Preprocess data
print("Preprocessing data...", datetime.now())
## Interactions
train_user_interactions = process_interactions(train_data)
if use_valid:
    valid_user_interactions = process_interactions(valid_data)
    spl_valid_user_interactions = {user: valid_user_interactions[user] # Sample of the validation set's interactions
                                   for user in random.sample(valid_user_interactions.keys(), spl_valid_size)}
test_user_interactions = process_interactions(test_data)
train_valid_user_interactions = process_interactions(pd.concat([train_data, valid_data])) # Used in test-time evaluation
## Dataset and DataLoader variables
### Training set
train_dataset = Dataset(train_user_interactions, data_size, options)
train_random_sampler = RandomSampler(train_dataset, replacement=False) # No replacement to process all users
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_random_sampler,
                               num_workers=num_workers, collate_fn=lambda batch: collate_fn(batch))
if use_valid:
    ### Validation set -- full set
    valid_dataset = UserDataset(valid_user_interactions, options)
    valid_random_sampler = RandomSampler(valid_dataset, replacement=False) # No replacement to process all users
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_random_sampler,
                                   num_workers=num_workers, collate_fn=lambda batch: user_collate_fn(batch))
    ### Sample validation set -- sample of the full validation set for fast evaluation at each epoch
    spl_valid_dataset = UserDataset(spl_valid_user_interactions, options)
    spl_valid_random_sampler = RandomSampler(spl_valid_dataset, replacement=False) # No replacement to process all users
    spl_valid_data_loader = DataLoader(spl_valid_dataset, batch_size=batch_size, sampler=spl_valid_random_sampler,
                                       num_workers=num_workers, collate_fn=lambda batch: user_collate_fn(batch))
### Test set
test_dataset = UserDataset(test_user_interactions, options)
test_random_sampler = RandomSampler(test_dataset, replacement=False) # No replacement to ensure processing all samples
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_random_sampler,
                              num_workers=num_workers, collate_fn=lambda batch: user_collate_fn(batch))

# Setting up log files
## Log file name
experiment_id = dataset
experiment_id += "-" + model_type
experiment_id += "-valid" if use_valid else "-novalid"
experiment_id += "-pre" if use_pre else "-nopre"
experiment_id += "-e" + str(num_epoch) + "-b" + str(batch_size) + "-r" + str(options.lr) + \
                 "-h" + str(options.embed_dim) + "-n" + str(options.num_neg_sample) + \
                 "-w" + str(options.weight_decay)
## File creation
if not os.path.isdir('logs'):
    os.mkdir('logs')
if not os.path.isdir('res'):
    os.mkdir('res')
logfile = open('./logs/%s.log' % experiment_id, 'w')
resfile = open('./res/%s.tsv' % experiment_id, 'w')
outputs = [sys.stdout, logfile]

# Model initialization
print("Initializing the model...", datetime.now())
if model_type == "BPRMatrixFactorization":
    model = BPRMatrixFactorization(num_user, num_item, options, device_embed, device_ops)
else:
    print("Model unknown!")
    exit()

# Training
print("Starting training...", datetime.now())
best_eval = -1 # For model selection
for epoch in range(num_epoch):
    # Train for an epoch
    model.train()
    epoch_loss = 0.0
    for (n, batch) in enumerate(train_data_loader):
        batch = {k: v.to(device_embed) for (k, v) in batch.items()}
        loss = model(batch)
        epoch_loss += loss
    epoch_loss /= (n + 1) # Divide by the number of batches

    [print('Epoch {}: train loss {} -- {}'.format(epoch, epoch_loss, datetime.now()), file=f) for f in outputs]

    if use_valid:
        # Evaluate on the sample validation set
        model.eval()
        eval_results = predict_evaluate(spl_valid_data_loader, model, device_embed, train_user_interactions, metrics, (valid_level,))
        eval_results = eval_results[valid_level]
        eval_results_str = ["{}@{} {:.5f}".format(metric[0], metric[1], eval_results[metric]) for metric in metrics]
        eval_results_str = ", ".join(eval_results_str)
        [print('Epoch {}: sample valid, {}-level -- {} -- {}'.format(epoch, valid_level, eval_results_str, datetime.now()),
               file=f) for f in outputs]

        # Save the model if the performance on validation set improved, if use_valid is True
        current_eval = eval_results[("precision", 10)]  # model selection based on Precision@10
        if current_eval > best_eval:
            print('Saving...')
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, 'checkpoint/%s.t7' % experiment_id)
            best_eval = current_eval
    else:
        # Save only for the last epoch if no validation set is used for model selection
        if epoch == num_epoch - 1:
            print('Saving...')
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, 'checkpoint/%s.t7' % experiment_id)

    logfile.flush()

# Model selection
if use_valid:
    # Load the selected model based on the best performance on the validation set
    print("Selected model evaluation...", datetime.now())
    checkpoint = torch.load('checkpoint/%s.t7' % experiment_id)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']

# Save embeddings
if save_embed:
    if not os.path.isdir('embeddings'):
        os.mkdir('embeddings')
    embeddings = {"user": model.user_embeddings.weight.data.cpu().numpy(),
                  "item": model.item_embeddings.weight.data.cpu().numpy()}
    np.save('embeddings/%s.npy' % experiment_id, embeddings)

# Evaluation
print("Starting evaluation...", datetime.now())
model.eval()
## Header of the result file
metrics_str = ["{}@{}".format(metric[0], metric[1]) for metric in metrics]
metrics_str = "\t".join(metrics_str)
print("set\tlevel\t{}".format(metrics_str), file=resfile)
## Evaluate on the validation set
if use_valid:
    eval_results = predict_evaluate(valid_data_loader, model, device_embed, train_user_interactions, metrics, ('user', 'interaction'))
    for level in ('user', 'interaction'):
        level_eval_results = eval_results[level]
        eval_results_str = ["{}@{} {:.5f}".format(metric[0], metric[1], level_eval_results[metric]) for metric in metrics]
        eval_results_str = ", ".join(eval_results_str)
        [print('Epoch {}: valid, {}-level -- {} -- {}'.format(epoch, level, eval_results_str, datetime.now()),
               file=f) for f in outputs]
        eval_results_str = ["{:.5f}".format(level_eval_results[metric]) for metric in metrics]
        eval_results_str = "\t".join(eval_results_str)
        print("valid\t{}-level\t{}".format(level, eval_results_str), file=resfile)
## Evaluate on the test set
eval_results = predict_evaluate(test_data_loader, model, device_embed, train_valid_user_interactions, metrics, ('user', 'interaction'))
for level in ('user', 'interaction'):
    level_eval_results = eval_results[level]
    eval_results_str = ["{}@{} {:.5f}".format(metric[0], metric[1], level_eval_results[metric]) for metric in metrics]
    eval_results_str = ", ".join(eval_results_str)
    [print('Epoch {}: test, {}-level -- {} -- {}'.format(epoch, level, eval_results_str, datetime.now()),
           file=f) for f in outputs]
    eval_results_str = ["{:.5f}".format(level_eval_results[metric]) for metric in metrics]
    eval_results_str = "\t".join(eval_results_str)
    print("test\t{}-level\t{}".format(level, eval_results_str), file=resfile)

logfile.close()
