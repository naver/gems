GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch
import random
import pytorch_lightning as pl

import sys
import os
from pathlib import Path
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from argparse import ArgumentParser

from modules.data_utils import BufferDataModule, EnvWrapper, get_file_name
from RecSim.simulators import TopicRec
from modules.agents import DQN, SAC, SlateQ, REINFORCE, REINFORCESlate, EpsGreedyOracle, RandomSlate, STOracleSlate, WolpertingerSAC
from modules.argument_parser import MainParser
from modules.belief_encoders import BeliefEncoder, GRUBelief
from GeMS.modules.rankers import Ranker, TopKRanker, kHeadArgmaxRanker, GeMS
from GeMS.modules.item_embeddings import ItemEmbeddings, MFEmbeddings
from modules.loops import TrainingEpisodeLoop, ValEpisodeLoop, TestEpisodeLoop, ResettableFitLoop

######################
## Argument parsing ##
######################

main_parser = ArgumentParser()
main_parser.add_argument('--agent', type=str, required = True,
                            choices=['DQN', 'SAC', 'WolpertingerSAC', 'SlateQ', 'REINFORCE', 'REINFORCESlate',
                                        'EpsGreedyOracle', 'RandomSlate', 'STOracleSlate'], help='RL Agent.')
main_parser.add_argument('--belief', type=str, required = True,
                            choices=['none', 'GRU'], help='Belief encoder.')
main_parser.add_argument('--ranker', type=str, required = True,
                            choices=['none', 'topk', 'kargmax', 'GeMS'], help='Ranker.')
main_parser.add_argument('--item_embedds', type=str, required = True,
                            choices=['none', 'scratch', 'mf', 'ideal'], help='Item embeddings.')
main_parser.add_argument('--env_name', type=str, required = True, help='Environment.')

def get_elem(l, ch):
    for i,el in enumerate(l):
        if el.startswith(ch):
            return el
agent_name = get_elem(sys.argv, "--agent=")
belief_name = get_elem(sys.argv, "--belief=")
ranker_name = get_elem(sys.argv, "--ranker=")
embedd_name = get_elem(sys.argv, "--item_embedds=")
env_name = get_elem(sys.argv, "--env_name=")
main_args = main_parser.parse_args([agent_name, belief_name, ranker_name, embedd_name, env_name])
sys.argv.remove(agent_name)
sys.argv.remove(belief_name)
sys.argv.remove(ranker_name)
sys.argv.remove(embedd_name)

if main_args.agent == "DQN":
    agent_class = DQN
elif main_args.agent == "SAC":
    agent_class = SAC
elif main_args.agent == "WolpertingerSAC":
    agent_class = WolpertingerSAC
elif main_args.agent == "SlateQ":
    agent_class = SlateQ
elif main_args.agent == "REINFORCE":
    agent_class = REINFORCE
elif main_args.agent == "REINFORCESlate":
    agent_class = REINFORCESlate
elif main_args.agent == "EpsGreedyOracle":
    agent_class = EpsGreedyOracle
elif main_args.agent == "RandomSlate":
    agent_class = RandomSlate
elif main_args.agent == "STOracleSlate":
    agent_class = STOracleSlate
else :
    raise NotImplementedError("This agent has not been implemented yet.")

if main_args.belief in ["none"]:
    belief_class = None
elif main_args.belief == "GRU":
    belief_class = GRUBelief
else :
    raise NotImplementedError("This belief encoder has not been implemented yet.")

if main_args.ranker in ["none"]:
    ranker_class = None
elif main_args.ranker == "topk":
    ranker_class = TopKRanker
elif main_args.ranker == "kargmax":
    ranker_class = kHeadArgmaxRanker
elif main_args.ranker == "GeMS":
    ranker_class = GeMS
else :
    raise NotImplementedError("This ranker has not been implemented yet.")

if main_args.item_embedds in ["none", "ideal", "scratch"]:
    item_embedd_class = ItemEmbeddings
elif main_args.item_embedds == "mf":
    item_embedd_class = MFEmbeddings
else :
    raise NotImplementedError("This type of item embeddings has not been implemented yet.")

if main_args.env_name in ["TopicRec", "topics"]:
    env_class = TopicRec
else:
    env_class = None


argparser = MainParser() # Program-wide parameters
argparser = agent_class.add_model_specific_args(argparser)  # Agent-specific parameters
argparser = TrainingEpisodeLoop.add_model_specific_args(argparser)  # Loop-specific parameters
argparser = ValEpisodeLoop.add_model_specific_args(argparser)  # Loop-specific parameters
argparser = TestEpisodeLoop.add_model_specific_args(argparser)  # Loop-specific parameters
if belief_class is not None:
    argparser = belief_class.add_model_specific_args(argparser) # Belief-specific parameters
if env_class is not None:
    argparser = env_class.add_model_specific_args(argparser) # Env-specific parameters
if ranker_class is not None:
    argparser = ranker_class.add_model_specific_args(argparser) # Ranker-specific parameters
argparser = item_embedd_class.add_model_specific_args(argparser)  # Item embeddings-specific parameters


args = argparser.parse_args(sys.argv[1:])
arg_dict = vars(args)
arg_dict["item_embedds"] = main_args.item_embedds
logger_arg_dict = {**vars(args), **vars(main_args)}


# Seeds for reproducibility
seed = int(args.seed)
pl.seed_everything(seed)

is_pomdp = (belief_class is not None)

####################
## Initialization ##
####################

# Environement and Replay Buffer
buffer = BufferDataModule(offline_data = [], **arg_dict)
env = EnvWrapper(buffer = buffer, **arg_dict)
arg_dict["env"] = env

# Item embeddings
if main_args.item_embedds in ["none"]:
    item_embeddings = None
elif main_args.item_embedds in ["scratch"]:
    item_embeddings = ItemEmbeddings.from_scratch(args.num_items, args.item_embedd_dim, device = args.device)
elif main_args.item_embedds in ["ideal"]:
    item_embeddings = ItemEmbeddings.get_from_env(env, device = args.device)
    item_embeddings.freeze()    # No fine-tuning when we already have the ideal embeddings
elif main_args.item_embedds in ["mf", "mf_fixed", "mf_init"]:
    if args.MF_checkpoint is None:
        item_embeddings = MFEmbeddings(**arg_dict)
        print("Pre-training MF embeddings ...")
        dataset_path = args.data_dir + "datasets/" + args.MF_dataset
        item_embeddings.train(dataset_path)
        arg_dict["MF_checkpoint"] = args.MF_dataset
        print("Pre-training done.")
    item_embeddings = ItemEmbeddings.from_pretrained(args.data_dir + "MF_embeddings/" + arg_dict["MF_checkpoint"] + ".pt", args.device)
    if main_args.item_embedds == "mf_fixed":
        item_embeddings.freeze()
else:
    raise NotImplementedError("This type of item embeddings have not been implemented yet.")

# Belief encoder
if is_pomdp:
    if ranker_class is None:
        ranker = None
        _, action_dim, num_actions = env.get_dimensions()
    else:
        if ranker_class in [GeMS]:
            arg_dict["fixed_embedds"] = True
            if args.ranker_dataset is None :
                ranker_checkpoint = main_args.ranker + "_" + args.click_model + "_" + args.logging_policy + "_" + args.pretrain_size
            else:
                ranker_checkpoint = main_args.ranker + "_" + args.ranker_dataset
            ranker_checkpoint += "_latentdim" + str(arg_dict["latent_dim"]) + "_beta" + str(arg_dict["lambda_KL"]) + "_lambdaclick" + str(arg_dict["lambda_click"]) + \
                                    "_lambdaprior" + str(arg_dict["lambda_prior"]) + "_" + args.ranker_embedds + "_seed" + str(args.ranker_seed)
            ranker = ranker_class.load_from_checkpoint(args.data_dir + "GeMS/checkpoints/" + ranker_checkpoint + ".ckpt",
                                                    map_location = args.device, item_embeddings = item_embeddings, **arg_dict)
            ranker.freeze()
            print("Getting action bounds ...")
            if args.ranker_dataset is None :
                ranker.get_action_bounds(args.data_dir + "RecSim/datasets/" + args.click_model + "_" + args.logging_policy + "_10K.pt")
            else:
                ranker.get_action_bounds(args.data_dir + "RecSim/datasets/" + args.ranker_dataset + ".pt")
                            ### We find the appropriate action bounds from the aggregated posterior.
        else:
            ranker = ranker_class(item_embeddings = item_embeddings, **arg_dict)
            ranker_checkpoint = main_args.ranker
        action_dim, num_actions = ranker.get_action_dim()
    belief = belief_class(item_embeddings = ItemEmbeddings.from_scratch(args.num_items, args.item_embedd_dim, device = args.device),
                            ranker = ranker, **arg_dict)
    state_dim = belief.get_state_dim()
else:
    belief = None
    ranker = None
    state_dim, action_dim, num_actions = env.get_dimensions()

# Agent
agent = agent_class(belief = belief, ranker = ranker, state_dim = state_dim, action_dim = action_dim, num_actions = num_actions, **arg_dict)


########################
## Training procedure ##
########################

### Logger
aim_logger = AimLogger(experiment = args.exp_name, log_system_params=False)
aim_logger.log_hyperparams(logger_arg_dict)

### Checkpoint
ckpt_dir =  args.data_dir + "checkpoints/" + args.MF_checkpoint + "/"
Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
if ranker is not None:
    ckpt_name = args.name + "_" + ranker_checkpoint + "_agentseed" + str(seed) + "_gamma" + str(args.gamma)
    if ranker.__class__ not in [GeMS]:
        ckpt_name += "_rankerembedds-" + arg_dict["item_embedds"]
else:
    ckpt_name = args.name + "_seed" + str(seed)
    if agent.__class__ not in [RandomSlate, EpsGreedyOracle]:
        ckpt_name += "_gamma" + str(args.gamma)
ckpt = ModelCheckpoint(monitor = 'val_reward', dirpath = ckpt_dir, filename = ckpt_name, mode = 'max')

### Agent
trainer_agent = pl.Trainer(logger=aim_logger, enable_progress_bar = args.progress_bar, callbacks = [RichProgressBar(), ckpt],
                            log_every_n_steps = args.log_every_n_steps, max_steps = args.max_steps + 1,
                            check_val_every_n_epoch = args.check_val_every_n_epoch,
                            gpus = 1 if args.device == "cuda" else None, enable_model_summary = False)

fit_loop = ResettableFitLoop(max_epochs_per_iter = args.iter_length_agent)
episode_loop = TrainingEpisodeLoop(env, buffer.buffer, belief, agent, ranker, random_steps = args.random_steps,
                                            max_steps = args.max_steps + 1, device = args.device)

res_dir = args.data_dir + "results/" + args.MF_checkpoint + "/"
Path(res_dir).mkdir(parents=True, exist_ok=True)
val_loop = ValEpisodeLoop(belief = belief, agent = agent, ranker = ranker, trainer = trainer_agent,
                            filename_results = res_dir + ckpt_name + ".pt", **arg_dict)
test_loop = TestEpisodeLoop(belief = belief, agent = agent, ranker = ranker, trainer = trainer_agent,
                            filename_results = res_dir + ckpt_name + ".pt", **arg_dict)
trainer_agent.fit_loop.epoch_loop.val_loop.connect(val_loop)
trainer_agent.test_loop.connect(test_loop)
episode_loop.connect(batch_loop = trainer_agent.fit_loop.epoch_loop.batch_loop, val_loop = trainer_agent.fit_loop.epoch_loop.val_loop)
fit_loop.connect(episode_loop)
trainer_agent.fit_loop = fit_loop

if agent.__class__ not in [EpsGreedyOracle, RandomSlate, STOracleSlate]:
    trainer_agent.fit(agent, buffer)

    env.env.reset_random_state()
    res = trainer_agent.test(model=agent, ckpt_path=ckpt_dir + ckpt_name + ".ckpt", verbose=True, datamodule=buffer)

    ### Test reward in checkpoint
    ckpt = torch.load(ckpt_dir + ckpt_name + ".ckpt")
    list(ckpt["callbacks"].values())[0]["test_reward"] = res[0]["test_reward"]
    torch.save(ckpt, ckpt_dir + ckpt_name + ".ckpt")
else:
    env.env.reset_random_state()
    res = trainer_agent.test(model=agent, verbose=True, datamodule=buffer)
