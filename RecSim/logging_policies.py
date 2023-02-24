GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch
from typing import List, Dict, Union

from modules.argument_parser import MyParser

class LoggingPolicy():
    '''
        Base class for logging policies
    '''
    def __init__(self, num_items : Union[int, Dict], rec_size : int, device : torch.device = torch.device("cpu"),
                        frequencies : str = None, data_dir : str = None, **kwargs):
        self.num_items = num_items  # Number of items or number of items per query
        self.rec_size = rec_size
        self.device = device
        self.data_dir = data_dir

        if frequencies is not None:
            self.frequencies = torch.load(data_dir + frequencies, map_location = self.device)  # Dict of query and doc frequencies
            self.docs_in_q = {qid : torch.arange(len(freqs), device = self.device)[torch.nonzero(freqs).squeeze()]
                                        for qid, freqs in obs_freq_q.items()}
        else:
            self.frequencies = None
            if type(self.num_items)==int:
                self.docs_in_q = torch.arange(self.num_items, device = self.device)
            else:
                self.docs_in_q = {qid : torch.arange(self.num_items[qid], device = self.device)}

    @staticmethod
    def add_model_specific_args(parent_parser : MyParser) -> MyParser:
        parser = MyParser(parents=[parent_parser], add_help=False)
        return parser

    def get_items(self, obs : Dict) -> torch.LongTensor:
        if "query" in obs:  # Search
            return self.docs_in_q[obs["query"].item()]
        else:   # Recommendation
            return self.docs_in_q


    def forward(self, obs : Dict, info : Dict) -> torch.LongTensor:
        '''
            Common function for observing and choosing an action

            Parameters :
             - obs : Dict[str:torch.Tensor]
                Observation (query, previous clicks, etc)
             - info : None or dictionary
                Info returned by the simulator

            Output :
             - rec_list : torch.LongTensor(rec_size)
                New recommendation list

        '''
        return torch.zeros(self.rec_size, device = self.device, dtype = torch.long)

class RandomPolicy(LoggingPolicy):
    '''
        Returns a random list
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, obs : Dict, info : Dict) -> torch.LongTensor:
        items = self.get_items(obs)
        return items[torch.randperm(len(items), device = self.device)[:self.rec_size]]

class EpsGreedyPolicy(LoggingPolicy):
    '''
        Epsilon-Optimal Policy
    '''
    def __init__(self, epsilon_pol : float, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon_pol

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[LoggingPolicy.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--epsilon_pol', type=float, default = 0.2)
        return parser

    def forward(self, obs : Dict, info : Dict) -> torch.LongTensor:
        items = self.get_items(obs)

        eps_greedy = torch.bernoulli(self.epsilon * torch.ones(self.rec_size, device = self.device))
        random = items[torch.randperm(len(items), device = self.device)[:self.rec_size]]
        greedy = - torch.ones(self.rec_size, dtype = torch.long, device = self.device)
        return torch.where(eps_greedy.bool(), random, greedy)

class LoggedPolicy(LoggingPolicy):
    '''
        Policy read from file. Ranks by decreasing order of relevance according to the file.
        Only Search for now
    '''
    def __init__(self, policy_name : str, reverse : bool, sample10 : bool, **kwargs):
        super().__init__(**kwargs)

        self.policy_name = policy_name
        self.reverse = reverse
        self.sample10 = sample10

        is_policy_from_click_model = (len(policy_name.split("/")) > 1)
        if is_policy_from_click_model:  # generated policies
            # Not compatible with Plackett-Luce, sample10, etc
            self.sorted_docs = torch.load(self.data_dir + policy_name + ".pt", map_location = self.device)
        else:
            self.relevances = torch.load(self.data_dir + policy_name + ".pt", map_location = self.device)
                                                # Should be a dict{q_id : torch.LongTensor(num_items_for_q)}
                                                # For each query, we need the argsorted relevances
            self.relevances = {qid : rels[self.docs_in_q[qid]] for qid, rels in self.relevances.items()}


            if sample10:    # Relevance-stratified sampling
                self.docs_in_q = {}
                for qid, rels in self.relevances.items():
                    rescaled_rels = torch.log2(rels * 15 + 1).long()
                    rels1 = rescaled_rels[rescaled_rels >= 1]
                    rels1_idx = torch.arange(len(rescaled_rels))[rescaled_rels >= 1]
                    if len(rels1) < self.rec_size:
                        rels1 = torch.cat([rels1, rescaled_rels[rescaled_rels == 0]])[:self.rec_size]
                        rels1_idx = torch.cat([rels1_idx, torch.arange(len(rescaled_rels))[rescaled_rels == 0]])[:self.rec_size]
                    sample = []
                    count_rels = torch.zeros(5)
                    for idx, rel in zip(rels1_idx, rels1):
                        if len(sample) == self.rec_size:
                            break
                        if count_rels[rel] < 3:
                            sample.append(idx)
                            count_rels[rel] += 1
                    if len(sample) < self.rec_size:
                        poorly_relevant = torch.cat([torch.arange(len(rescaled_rels))[rescaled_rels == 1], torch.arange(len(rescaled_rels))[rescaled_rels == 0]])
                        unique = torch.tensor([pr for pr in poorly_relevant if pr not in sample], dtype = torch.long)
                        sample = torch.cat([torch.tensor(sample, dtype = torch.long, device = self.device), unique ])[:self.rec_size]
                    else:
                        sample = torch.tensor(sample, device = self.device, dtype = torch.long)
                    self.docs_in_q[qid] = sample
                    if len(sample) < self.rec_size:
                        print("shit")

                    self.relevances = {qid : rels[self.docs_in_q[qid]] for qid, rels in self.relevances.items()}

            self.sorted_docs = {q_id : self.docs_in_q[q_id][torch.argsort(rels[self.docs_in_q[q_id]], descending = True)]
                                            for q_id, rels in self.relevances.items()}


    @staticmethod
    def add_model_specific_args(parent_parser : MyParser) -> MyParser:
        parser = MyParser(parents=[LoggingPolicy.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--policy_name', type=str, default = "bm25")
        parser.add_argument('--reverse', type=parser.str2bool, default = False)
        parser.add_argument('--sample10', type=parser.str2bool, default = False)
        return parser

    def forward(self, obs : Dict, info : Dict) -> torch.LongTensor:
        rl = self.sorted_docs[obs["query"].item()][:self.rec_size]
        if self.reverse:
            rl = torch.flip(rl, [0])
        return rl

class PlackettLuceLoggedPolicy(LoggedPolicy):
    '''
        Transform a logged policy into a stochastic policy using a Plackett-Luce model.
    '''
    def __init__(self, temperature : float, **kwargs):
        super().__init__(**kwargs)

        self.T = temperature

    @staticmethod
    def add_model_specific_args(parent_parser : MyParser) -> MyParser:
        parser = MyParser(parents=[LoggedPolicy.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--temperature', type=float, default = 0.1)
        return parser

    def forward(self, obs : Dict, info : Dict) -> torch.LongTensor:
        log_relevances = torch.log(self.relevances[obs["query"].item()] + 1e-2)    # Assumes rescaled relevances
        docs = self.docs_in_q[obs["query"].item()]

        ## We sample using the Gumbel-trick
        noise = torch.rand(len(log_relevances), device = self.device)
        gumbel_noise = - torch.log(- torch.log(noise))
        perturbed_softmax = torch.nn.Softmax(dim = 0)(log_relevances + gumbel_noise * self.T)
        ranked_list = docs[torch.topk(perturbed_softmax, k = self.rec_size).indices]

        if self.reverse:
            ranked_list = torch.flip(ranked_list, [0])
        return ranked_list

class PlackettLuceNoisyOracle(PlackettLuceLoggedPolicy):
    '''
        Gaussian perturbation of oracle policy with Plackett Luce stochasticity
    '''
    def __init__(self, noise_var : float, policy_name : str, **kwargs):
        super().__init__(policy_name = "relevances", **kwargs)
        self.noise_var = noise_var

        self.relevances = {qid : torch.clip(rels + torch.randn(len(rels), device = self.device) * noise_var, 0, 1)
                                    for qid, rels in self.relevances.items()}

    @staticmethod
    def add_model_specific_args(parent_parser : MyParser) -> MyParser:
        parser = MyParser(parents=[PlackettLuceLoggedPolicy.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--noise_var', type=float, default = 0.1)
        return parser
