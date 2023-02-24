GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch
import pytorch_lightning as pl
import random
from pathlib import Path

import sys
import os
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from aim.pytorch_lightning import AimLogger
from argparse import ArgumentParser

from modules.data_utils import SlateDataModule
from modules.rankers import GeMS
from modules.argument_parser import MainParser
from modules.item_embeddings import ItemEmbeddings, MFEmbeddings

main_parser = ArgumentParser()
main_parser.add_argument("--ranker", type = str, required = True, choices = ["GeMS"], help = "Ranker type")
main_parser.add_argument("--dataset", type = str, default = "data/RecSim/datasets/focused_topdown_moving_env.pt", help = "Path to dataset")
main_parser.add_argument("--item_embedds", type=str, required = True, choices=["scratch", "mf_init", "mf_fixed"], help = "Item embeddings.")

def get_elem(l, ch):
    for i,el in enumerate(l):
        if el.startswith(ch):
            return el
ranker_name = get_elem(sys.argv, "--ranker=")
dataset_path = get_elem(sys.argv, "--dataset=")
item_embedds = get_elem(sys.argv, "--item_embedds=")
main_args = main_parser.parse_args([ranker_name, dataset_path, item_embedds])
sys.argv.remove(ranker_name)
sys.argv.remove(dataset_path)
sys.argv.remove(item_embedds)

if main_args.ranker == "GeMS":
    ranker_class = GeMS
else:
    raise NotImplementedError("This ranker is not trainable or has not been implemented yet.")

if main_args.item_embedds in ["scratch"]:
    item_embedd_class = ItemEmbeddings
elif main_args.item_embedds in ["mf_init", "mf_fixed"]:
    item_embedd_class = MFEmbeddings
else :
    raise NotImplementedError("This type of item embeddings has not been implemented yet.")

argparser = MainParser() # Program-wide parameters
argparser = ranker_class.add_model_specific_args(argparser)  # Agent-specific parameters
argparser = item_embedd_class.add_model_specific_args(argparser)  # Item embeddings-specific parameters
args = argparser.parse_args(sys.argv[1:])
args.MF_dataset = main_args.dataset.split("/")[-1]
embedd_dir = args.data_dir + "embeddings/"
if os.path.isfile(embedd_dir + args.MF_dataset): # Check if the MF checkpoint already exists
    args.MF_checkpoint = args.MF_dataset
else:
    args.MF_checkpoint = None
arg_dict = vars(args)

# Seeds for reproducibility
seed = int(args.seed)
pl.seed_everything(seed)

logger_arg_dict = {**vars(args), **vars(main_args)}
aim_logger = AimLogger(experiment=arg_dict["exp_name"], log_system_params=False)
aim_logger.log_hyperparams(logger_arg_dict)

# Item embeddings
arg_dict["item_embedds"] = main_args.item_embedds
if arg_dict["item_embedds"][-5:] == "fixed":
    arg_dict["fixed_embedds"] = True
else:
    arg_dict["fixed_embedds"] = False
if main_args.item_embedds in ["scratch"]:
    item_embeddings = ItemEmbeddings.from_scratch(args.num_items, args.item_embedd_dim, device = args.device)
elif main_args.item_embedds.startswith("mf"):
    if args.MF_checkpoint is None:
        item_embeddings = MFEmbeddings(**arg_dict)
        print("Pre-training MF embeddings ...")
        dataset_path = "/" + os.path.join(*main_args.dataset.split("/")[:-1]) + "/" + args.MF_dataset
        item_embeddings.train(dataset_path)
        arg_dict["MF_checkpoint"] = args.MF_dataset
        print("Pre-training done.")
    item_embeddings = ItemEmbeddings.from_pretrained(embedd_dir + arg_dict["MF_checkpoint"], args.device)
    if main_args.item_embedds == "mf_fixed":
        item_embeddings.freeze()
else:
    raise NotImplementedError("This type of item embeddings have not been implemented yet.")

ranker = ranker_class(item_embeddings = item_embeddings, **arg_dict)
ckpt_dir =  args.data_dir + "/checkpoints/"
Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
ckpt_name = main_args.ranker + "_" + main_args.dataset.split("/")[-1][:-3] + "_latentdim" + str(arg_dict["latent_dim"]) + \
            "_beta" + str(arg_dict["lambda_KL"]) + "_lambdaclick" + str(arg_dict["lambda_click"]) + \
            "_lambdaprior" + str(arg_dict["lambda_prior"]) + "_" + arg_dict["item_embedds"] + "_seed" + str(args.seed)
trainer = pl.Trainer(enable_progress_bar = arg_dict["progress_bar"], logger=aim_logger, 
                    callbacks = [RichProgressBar(),
                    ModelCheckpoint(monitor = 'val_loss', dirpath = ckpt_dir, filename = ckpt_name)],
                    gpus = 1 if arg_dict["device"] == "cuda" else None, max_epochs = args.max_epochs)


print("### Loading data and initializing DataModule ...")
data = torch.load(main_args.dataset, map_location = arg_dict["device"])
datamod = SlateDataModule(env = None, data = data, full_traj = False, **arg_dict)

print("### Launch training")
trainer.fit(ranker, datamod)
