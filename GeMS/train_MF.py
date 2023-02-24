GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch
import pytorch_lightning as pl
import random

from modules.argument_parser import MainParser
from modules.item_embeddings import MFEmbeddings

argparser = MainParser() # Program-wide parameters
argparser = MFEmbeddings.add_model_specific_args(argparser)  # Agent-specific parameters
args = argparser.parse_args()
arg_dict = vars(args)

# Seeds for reproducibility
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

item_embeddings = MFEmbeddings(**arg_dict)
dataset_path = args.data_dir + "/../RecSim/datasets/" + args.MF_dataset
item_embeddings.train(dataset_path, args.data_dir + "/../MF_embeddings/")
