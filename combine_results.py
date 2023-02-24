GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch
import os
from argparse import ArgumentParser, BooleanOptionalAction
import pandas as pd
import scipy.stats as st
import numpy as np


parser = ArgumentParser()
parser.add_argument('path', type=str, help='Path to results.')
parser.add_argument('--set', type=str, choices = ["val", "test"], default = "test", help='Best validation score or test score.')
parser.add_argument("--t_test", type = str, nargs = '+', help = "Models and metrics on which to perform a statistical test.", default = None)
parser.add_argument('--median', action=BooleanOptionalAction)

args = parser.parse_args()
arg_dict = vars(args)

if args.path == "all":
    if args.set=="test":
        runs = os.listdir("data/results/")
    else:
        runs = os.listdir("data/checkpoints/")
else:
    runs = [args.path]

params = ["agentseed", "seed", "gamma", "rankerembedds-", "latentdim", "beta", "lambdaclick", "lambdaprior"]
rows = []
overall_rows = []
for run in runs:
    if args.set=="test":
        path = "data/results/" + run + "/"
    else:
        path = "data/checkpoints/" + run + "/"

    files = os.listdir(path)

    for f in files:
        if "testtraj" not in f:
            if args.set=="test":
                f_split = f[:-3].split("_")
            else:
                f_split = f[:-5].split("_")
            model_params = {"name" : f_split[0]}
            ind = []
            for i, ch in enumerate(f_split):
                if ch.startswith("seed") or ch.startswith("agentseed"):
                    ind.append(i)
            overall_split = f_split[:]
            for i in sorted(ind, reverse=True):
                overall_split.pop(i)
            overall_model_params = {"model" : "_".join(overall_split), "seed" : int(f_split[ind[0]].split("seed")[1])}
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
                # if val is None:
                #     model_params["Model"] = f_split[0] + "_" + spec
                # else:
                model_params[p] = val

            if args.set=="test":
                res = torch.load(path + f)
            else:
                ckpt = torch.load(path + f)
                res = list(ckpt["callbacks"].values())[0]["best_model_score"].cpu()
            rows.append({**model_params, "reward" : res.item()})
            overall_rows.append({**overall_model_params, "reward" : res.item()})

results = pd.DataFrame.from_dict(rows, orient = "columns")
overall_results = pd.DataFrame.from_dict(overall_rows, orient = "columns")


# ----------------------------- #
#           Printing            #
# ----------------------------- #

models = overall_results['model'].unique()

for i, model in enumerate(models):
    print('--------------------------------------')
    print(model + " : ")
    data = overall_results[(overall_results['model'] == model)]
    mean = data['reward'].mean()
    if len(data) > 1:
        interval = st.t.interval(alpha=0.95, df=len(data)-1, loc=mean, scale=st.sem(data['reward'])) 
    else:
        interval = (mean, mean)
    if np.isnan(interval[0]):
        thresh = 0.0
    else:
        thresh = interval[1] - mean
    print("\t reward : %.4f (+- %.4f)" % (mean, thresh))
    if args.median:
        print("\t median : %.4f" % (data['reward'].median()))
    print("Computed on %d seeds." % len(data))

if args.t_test is not None:
    print("\n\n\n")
    tests = args.t_test
    ## Format {Ref Model}|{Model1}&{Model2}&{...}

    for t in tests:
        t_split = t.split("|")
        ref_model = t_split[0]
        target_models = t_split[1].split("&")

        for model in target_models:
            data_ref = overall_results[(overall_results['model'] == ref_model)]
            data = overall_results[(overall_results['model'] == model)]

            t_test_res = st.ttest_ind(data_ref["reward"], data["reward"], equal_var=False)
            # data_ref = data_ref.sort_values(by=['seed'])
            # data = data.sort_values(by=['seed'])
            # t_test_res = st.ttest_rel(data_ref["reward"], data["reward"])
            print("'-------------------------")
            print("%s vs %s :" % (ref_model, model))
            print(t_test_res)
            if t_test_res.pvalue < 0.05:
                print("Statistically significant")
            else:
                print("Not statistically significant")
