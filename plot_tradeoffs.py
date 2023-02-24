GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch
import os
from argparse import ArgumentParser
import pandas as pd
import scipy.stats as st
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


parser = ArgumentParser()
parser.add_argument('path', type=str, help='Path to results.')
parser.add_argument('--betas', type=float, nargs = '+', default = [0.1, 0.5, 1.0, 2.0], help='Betas.')
parser.add_argument('--lambdas', type=float, nargs = '+', default = [0.0, 0.2, 0.5, 1.0], help='Lambdas.')
parser.add_argument('--extension', type=str, default = "png", help='File extension.')


args = parser.parse_args()
arg_dict = vars(args)


try:
    assert len(args.betas) == 1 or len(args.lambdas) == 1
except AssertionError:
    raise ValueError("Either beta or lambda must be fixed.")

if args.path == "all":
    runs = os.listdir("data/checkpoints/")
else:
    runs = [args.path]

params = ["agentseed", "seed", "gamma", "rankerembedds-", "latentdim", "beta", "lambdaclick", "lambdaprior"]
rows = []
overall_rows = []
for run in runs:
    path = "data/results/" + run + "/"
    files = os.listdir(path)

    for f in files:
        if "testtraj" not in f:
            f_split = f[:-5].split("_")
            if f_split[0] != "SAC+GeMS":
                continue
            model_params = {"name" : f_split[0]}
            ind = []
            for i, ch in enumerate(f_split):
                if ch.startswith("seed") or ch.startswith("agentseed"):
                    ind.append(i)
            overall_split = f_split[:]
            for i in sorted(ind, reverse=True):
                overall_split.pop(i)
            overall_model_params = {"model" : "_".join(overall_split), "seed" : f_split[ind[0]].split("seed")[1]}
            for spec in f_split[1:]:
                val = None
                for i, p in enumerate(params):
                    if spec.startswith(p):
                        val = spec[len(p):]
                        if val == "":
                            val = "True"
                        else:
                            try :
                                val = int(val)
                            except ValueError:
                                try:
                                    val = float(val)
                                except ValueError:
                                    break
                        break
                model_params[p] = val

            res = torch.load(path + f)
            rows.append({**model_params, "reward" : res.item()})
            overall_rows.append({**overall_model_params, "reward" : res.item()})

results = pd.DataFrame.from_dict(rows, orient = "columns")
overall_results = pd.DataFrame.from_dict(overall_rows, orient = "columns")


# ----------------------------- #
#           Plotting            #
# ----------------------------- #
if len(args.betas) == 1:
    res = results[results['beta'] == args.betas[0]]
    param1 = "beta"
    param2 = "lambdaclick"
    label2 = "lambda"
    values2 = args.lambdas
else:
    res = results[results['lambdaclick'] == args.lambdas[0]]
    param1 = "lambdaclick"
    param2 = "beta"
    label2 = "beta"
    values2 = args.betas

#sns.set(font_scale = 2)
plt.figure(figsize=(15,4))
myplot = sns.catplot(x = param2, y = "reward", hue = "latentdim", data = res, ci=95, 
                        palette="muted", capsize=.2, kind = "point")
myplot._legend.set_title("Latent dim")
#myplot._legend.remove()
myplot.set(xlabel = label2, ylabel = 'Cumulative number of clicks')
sns.despine(offset=10, trim=True)

myplot.fig.set_size_inches(8,4)
myplot.fig.savefig("catplot_" + label2 + "." + args.extension, bbox_inches = 'tight') 
