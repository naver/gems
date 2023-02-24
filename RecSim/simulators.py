GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch
import math
from typing import List, Dict, Tuple
from tqdm import tqdm
from pathlib import Path
from collections import deque

from modules.argument_parser import MyParser


class RecSim():
    '''
        Base class for Recommendation simulator
    '''
    def __init__(self, num_items : int, rec_size : int, dataset_name : str, sim_seed : int, filename : str,
                 device : torch.device = torch.device("cpu"), **kwargs):
        self.num_items = num_items
        self.rec_size = rec_size
        self.dataset_name = dataset_name
        self.filename = filename
        self.device = device

        self.sim_seed = sim_seed
        self.rd_gen = torch.Generator(device = device)
        self.rd_gen.manual_seed(sim_seed)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_items', type=int, default=1000)
        parser.add_argument('--rec_size', type=int, default=10)
        parser.add_argument('--filename', type=str, default=None)
        parser.add_argument('--dataset_name', type = str, default = None)
        parser.add_argument('--sim_seed', type = int, default = 24321357327)
        return parser

    def reset_random_state(self):
        self.rd_gen.manual_seed(self.sim_seed)

    def set_policy(self, MyPolicy, kwargs):
        self.logging_policy = MyPolicy(**{**kwargs, **vars(self)})

    def get_dimensions(self) -> Tuple[int, int]:
        return self.num_items, self.rec_size

    def generate_dataset(self, n_sess : int, path : str, chunksize : int = 0) -> None:
        '''
            Builds a dataset for model training, with 'policy' as the logging policy.

            Parameters :
            - n_sess : int
                Size of the dataset (in number of sessions !)
            - path : string
                Name of the file to save
            - chunksize : int
                Size of each chunk of data. If chunksize = 0, data is saved in one chunk

        '''
        dataset = {}
        chunk_count = 1

        for sess_id in tqdm(range(n_sess)):
            ## Get initial query and/or recommendation
            obs, info = self.reset()
            done = False

            slates, clicks = [], []
            while not done:
                rl = self.logging_policy.forward(obs, info)
                ## Let the simulated user interact with the ranking
                obs, reward, done, info = self.step(rl)
                ## Store the interaction
                slates.append(obs["slate"])
                clicks.append(obs["clicks"])

            ## Form session dictionary
            sess = {"slate"     : torch.stack(slates),
                    "clicks"    : torch.stack(clicks)}
            ## Add it to dataset
            dataset[sess_id] = sess

            if chunksize * sess_id != 0 and sess_id % chunksize == 0:
                Path(path).mkdir(parents=True, exist_ok=True)
                torch.save(dataset, path + "/chunk" + str(chunk_count) + ".pt")
                dataset = {}
                chunk_count +=1

        if chunk_count == 1:
            Path("/".join(path.split("/")[:-1])).mkdir(parents=True, exist_ok=True)
            torch.save(dataset, path + ".pt")
        else:
            torch.save(dataset, path + "/chunk" + str(chunk_count) + ".pt")

class TopicRec(RecSim):
    '''
        Topic-based slate-recommendation simulator with boredom and user preference drifting.
    '''
    def __init__(self, topic_size : int, num_topics : int, episode_length : int, env_alpha : float, env_propensities : List[float],
                 env_offset : float, env_slope: float, env_omega : float, recent_items_maxlen : int, short_term_boost : float,
                 boredom_threshold : int, boredom_moving_window : int, env_embedds : str, click_model : str, click_only_once : bool,
                 rel_threshold : float, prop_threshold : float, diversity_penalty : float, diversity_threshold : int, **kwargs) -> None:
        super().__init__(**kwargs)

        # Topics and horizon length
        self.topic_size = topic_size
        self.num_topics = num_topics
        self.H = episode_length

        if click_model=="tdPBM":
            self.alpha = 1.0
            gammas = torch.tensor(0.85).pow(torch.arange(self.rec_size, device = self.device)).unsqueeze(1)
            self.propensities = gammas * torch.ones(self.rec_size, self.rec_size + 1, device = self.device)
        elif click_model=="mixPBM":
            probs = [0.5, 0.5]
            self.alpha = 1.0
            gammas = torch.tensor(0.85).pow(torch.arange(self.rec_size, device = self.device)).unsqueeze(1)
            props = gammas * torch.ones(self.rec_size, self.rec_size + 1, device = self.device)
            self.propensities = probs[0] * props + probs[1] * torch.flip(props, dims = [0])
        else:
            self.alpha = env_alpha
            self.propensities = torch.tensor(env_propensities, device = self.device).reshape(self.rec_size, self.rec_size + 1)
        if prop_threshold is not None:
            self.propensities = torch.where(self.propensities > prop_threshold, torch.ones_like(self.propensities), torch.zeros_like(self.propensities))

        # User preference model
        self.offset = env_offset
        self.slope = env_slope
        self.omega = env_omega
        self.rel_threshold = rel_threshold
        self.diversity_penalty = diversity_penalty
        self.diversity_threshold = diversity_threshold

        # Boredom model
        self.recent_items_maxlen = recent_items_maxlen
        self.short_term_boost = short_term_boost
        self.boredom_thresh = boredom_threshold
        self.boredom_moving_window = boredom_moving_window
        self.click_only_once = click_only_once

        self.env_embedds = env_embedds
        if self.env_embedds is None:
            #### First distribution over topics
            comp_dist = torch.rand(size = (self.num_items, self.num_topics), device = self.device)
            comp_dist /= torch.sum(comp_dist, dim = 1).unsqueeze(1)     ### An item cannot be good in every topic.
            #### Then topic_specific quality and position
            self.item_embedd = torch.abs(torch.clamp(0.4 * torch.randn(self.num_items, self.num_topics, self.topic_size, device = self.device), -1, 1))
            #### Then we can normalize
            self.item_embedd *= comp_dist.unsqueeze(2)
            # For focused embeddings:
            self.item_embedd = self.item_embedd.flatten(start_dim = 1).pow(1.5)
            embedd_norm = torch.linalg.norm(self.item_embedd, dim = 1)
            self.item_embedd /= embedd_norm.unsqueeze(1)
            torch.save(self.item_embedd, "data/RecSim/embeddings/item_embeddings_focused.pt")
        else:
            self.item_embedd = torch.load("data/RecSim/embeddings/" + self.env_embedds, map_location = self.device)



        # with m > 1:
        self.item_comp = torch.argmax(torch.stack([torch.linalg.norm(topic, dim = 1)
                                                   for topic in torch.split(self.item_embedd, self.topic_size, dim = 1)]), dim = 0)
        self.max_score = torch.max(torch.linalg.norm(self.item_embedd, dim = 1))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[RecSim.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--click_model', type=str, default="tdPBM")

        parser.add_argument('--topic_size', type=int, default=2)
        parser.add_argument('--num_topics', type=int, default=10)
        parser.add_argument('--episode_length', type = int, default = 100)

        parser.add_argument('--env_alpha', type=float, default=1.0)
        parser.add_argument('--env_propensities', type=float, nargs = '+', default=None)
        parser.add_argument('--rel_threshold', type=float, default=None)
        parser.add_argument('--prop_threshold', type=float, default=None)
        parser.add_argument('--diversity_penalty', type=float, default=1.0)
        parser.add_argument('--diversity_threshold', type=float, default=5)

        parser.add_argument('--click_only_once', type=parser.str2bool, default=False)
        parser.add_argument('--env_offset', type=float, default=0.15)
        parser.add_argument('--env_slope', type=float, default=20)
        parser.add_argument('--env_omega', type = float, default = 0.9)

        parser.add_argument('--recent_items_maxlen', type=int, default=10)
        parser.add_argument('--short_term_boost', type=float, default=3.0)
        parser.add_argument('--boredom_threshold', type = int, default = 4)
        parser.add_argument('--boredom_moving_window', type = int, default = 5)

        parser.add_argument('--env_embedds', type=str, default=None)
        return parser

    def get_random_action(self) -> torch.LongTensor:
        return torch.randint(self.num_items, size = (self.rec_size, ), device = self.device)

    def get_dimensions(self) -> Tuple[int, int, int]:
        return 0, 0, self.num_items

    def get_item_embeddings(self) -> torch.FloatTensor:
        return self.item_embedd

    def click_model(self, rels : torch.FloatTensor, comps : torch.LongTensor) -> torch.LongTensor:
        '''
            UBM click model
        '''
        attr = self.alpha * rels
        if torch.max(torch.unique(comps, return_counts = True)[1]) >= self.diversity_threshold:
            attr /= self.diversity_penalty ### When too many similar are in the slate, the overall attractiveness of the slate decreases.
        clicks = torch.empty(self.rec_size, device = self.device, dtype = torch.long)
        rank_latest_click = -1
        for rank in range(self.rec_size):
            click_prob = attr[rank] * self.propensities[rank, rank_latest_click]
            clicks[rank] = torch.bernoulli(click_prob, generator = self.rd_gen)
            if clicks[rank]:
                rank_latest_click = rank
        return clicks

    def reset(self) -> Tuple[Dict, Dict]:
        '''
            The initial ranker returns the most qualitative document in each topic (or the 10 first topics, or multiple top_docs per topic)
        '''
        self.boredom_counter = 0
        self.t = 0  # Index of the trajectory-wide timestep
        self.clicked_items = deque([], self.recent_items_maxlen)
        self.clicked_step = deque([], self.recent_items_maxlen)
        self.all_clicked_items = []
        self.bored = torch.zeros(self.num_topics, dtype = torch.bool, device = self.device)
        self.bored_timeout = 5 * torch.ones(self.num_topics, dtype = torch.long, device = self.device)

        ## User embeddings
        self.user_embedd = torch.abs(torch.clamp(0.4 * torch.randn(self.num_topics, self.topic_size,
                                                                   device = self.device, generator = self.rd_gen), -1, 1))

        user_comp_dist = torch.rand(self.num_topics, device = self.device, generator = self.rd_gen).pow(3)
        user_comp_dist /= torch.sum(user_comp_dist)
        self.user_embedd *= user_comp_dist.unsqueeze(1)

        topic_norm = torch.linalg.norm(self.user_embedd, dim = 1)
        self.user_embedd = self.user_embedd.flatten() / torch.sum(topic_norm)
        # We normalize the user embedd (in the future we could have users with different click propensities)

        ## Initial recommendation
        quality = [torch.linalg.norm(topic, dim = 1) for topic in torch.split(self.item_embedd, self.topic_size, dim = 1)]
        if self.num_topics >= self.rec_size:
            ## The following assumes K >= rec_size
            most_quality = torch.cat([torch.argmax(topic).unsqueeze(0) for topic in quality])   # num_topics
            # user_preference = torch.cat([torch.linalg.norm(topic).unsqueeze(0) for topic in torch.split(self.user_embedd, self.topic_size)])
            # #print("user_preference", user_preference)
            # sorted_user_pref, ind_user_pref = torch.topk(user_preference, self.rec_size, sorted = True)
            rl = most_quality[:self.rec_size]    # rec_size
        else:
            rl = torch.cat([torch.topk(topic, k = self.rec_size // self.num_topics)[1] for topic in quality])

        ## Compute relevances
        rl_embedd = self.item_embedd[rl]    # rec_size, num_topics * topic_size
        score = torch.matmul(rl_embedd, self.user_embedd)   # rec_size
        norm_score = score / self.max_score # Normalize score
        if self.rel_threshold is None:
            relevances = 1 / (1 + torch.exp(-(norm_score - self.offset) * self.slope))    ## Rescale relevance
        else:
            relevances = torch.where(norm_score > self.rel_threshold,
                                        torch.ones_like(norm_score),
                                        torch.zeros_like(norm_score))

        ## First interaction
        clicks = self.click_model(relevances, self.item_comp[rl])
        clicked_items = torch.where(clicks)[0]
        self.clicked_items.extend(self.item_comp[rl[clicked_items]])
        self.all_clicked_items.extend(rl[clicked_items])
        if torch.sum(clicks) > 0:
            self.user_short_term_comp = self.item_comp[rl[clicked_items[-1]]]
        else:
            self.user_short_term_comp = torch.randint(self.num_topics, size = (1,), device = self.device, generator = self.rd_gen)

        info = {'user_state' : self.user_embedd, "done"  : False}
        obs = {'slate' : rl, 'clicks' : clicks}
        return obs, info

    def step(self, slate : torch.LongTensor, return_scores : bool = False) -> Tuple[Dict, torch.LongTensor, bool, Dict]:
        '''
            Simulates user interaction.
        '''
        self.t+=1
        info = {}
        self.bored_timeout -= self.bored.long() # Remove one to timeout
        self.bored = self.bored & (self.bored_timeout != 0) # "Unbore" timed out components
        self.bored_timeout[self.bored == False] = 5 # Reset timer for "unbored" components
        ## Bored anytime recently items from one topic have been clicked more than boredom_threshold
        if len(self.clicked_items) > 0:
            recent_items = torch.cat([it.unsqueeze(0) for it in self.clicked_items])
            recent_comps = torch.histc(recent_items.float(), bins = self.num_topics, min = 0, max = self.num_topics - 1).long()
            #recent_comps = torch.bincount(recent_items, minlength = self.num_topics).long()
            #print("recent_comps", recent_comps)
            bored_comps = torch.arange(self.num_topics)[recent_comps >= self.boredom_thresh]
            ## Then, these 2 components are put to 0 for boredom_timeout steps
            self.bored[bored_comps] = True
        #print("bored", self.bored.nonzero(as_tuple=True)[0])


        u_embedd = self.user_embedd.clone()
        bored_comps = torch.nonzero(self.bored).flatten()
        ### Set bored component to 0
        for bc in bored_comps:
            u_embedd[self.topic_size * bc : self.topic_size * (bc + 1)] = 0
        ### Boost short-term component
        u_embedd[self.topic_size * self.user_short_term_comp : self.topic_size * (self.user_short_term_comp + 1)] *= self.short_term_boost

        oracle = (-1 in slate)
        if oracle:
            #### Compute oracle
            scores = torch.matmul(self.item_embedd, u_embedd)
            if self.click_only_once:
                scores[torch.tensor(self.all_clicked_items, device = self.device).long()] = 0.0 # Items already clicked
            topk, oracle_slate = torch.topk(scores, k = self.rec_size, sorted = True)
            slate = torch.where(slate == -1, oracle_slate, slate)
        #print("item_comp", self.item_comp[slate])

        info["slate"] = slate
        info["slate_components"] = self.item_comp[slate]

        ## Compute relevances
        score = torch.matmul(self.item_embedd[slate], u_embedd)   # rec_size
        norm_score = score / self.max_score # Normalize score
        if self.rel_threshold is None:
            relevances = 1 / (1 + torch.exp(-(norm_score - self.offset) * self.slope))    ## Rescale relevance
        else:
            relevances = torch.where(norm_score > self.rel_threshold,
                                        torch.ones_like(norm_score),
                                        torch.zeros_like(norm_score))
        if self.click_only_once:
            # Set relevance of already clicked items to 0
            already_click_items = torch.tensor(self.all_clicked_items, device = self.device).unsqueeze(0)
            relevances[torch.logical_not(torch.all(slate.unsqueeze(1).expand(-1, len(self.all_clicked_items)) - already_click_items, dim = 1))] = 0.0

        info["scores"] = norm_score
        info["bored"] = self.bored

        ## Interaction
        clicks = self.click_model(relevances, self.item_comp[slate])
        clicked_items = torch.where(clicks)[0]
        self.clicked_items.extend(self.item_comp[slate[clicked_items]])
        self.clicked_step.extend(self.t * torch.ones_like(clicked_items))
        while len(self.clicked_step) > 0 and self.clicked_step[0] < self.t - self.boredom_moving_window:
            # We remove old clicks from boredom "log"
            self.clicked_items.popleft()
            self.clicked_step.popleft()
        self.all_clicked_items.extend(slate[clicked_items])


        ## Let clicked items influence user behavior
        #for it in clicked_items:
        for it in slate[clicked_items]:
            self.user_embedd = self.omega * self.user_embedd + (1 - self.omega) * self.item_embedd[it]
            topic_norm = torch.linalg.norm(self.user_embedd.reshape(self.num_topics, self.topic_size), dim = 1)
            self.user_embedd /= torch.sum(topic_norm)

        ## Select new short term comp in case of boredom on this component
        if self.bored[self.user_short_term_comp] and torch.sum(clicks) > 0:
            self.user_short_term_comp = self.item_comp[clicked_items[-1]]

        info['user_state'] = self.user_embedd
        info["clicks"] = clicks

        ## 6 - Set done and return
        if self.t >= self.H:
            done = True
            info["done"] = True
        else:
            done = False
            info["done"] = False

        obs = {'slate' : slate, 'clicks' : clicks}
        return obs, torch.sum(clicks), done, info

    def get_relevances(self, user_state : torch.FloatTensor = None) -> torch.FloatTensor:
        '''
            Returns the relevance of all items.
        '''
        if user_state is None:
            u_embedd = self.user_embedd.clone()
            bored_comps = torch.nonzero(self.bored).flatten()
            ### Set bored component to 0
            for bc in bored_comps:
                u_embedd[self.topic_size * bc : (self.topic_size + 1) * bc] = 0
            ### Boost short-term component
            u_embedd[self.topic_size * self.user_short_term_comp : (self.topic_size + 1) * self.user_short_term_comp] *= self.short_term_boost
        else:
            u_embedd = user_state.transpose(0,1)    # (item_embedd_dim, batch_size)

        scores = torch.matmul(self.item_embedd, u_embedd) # (num_items) or (num_items, batch_size)
        norm_scores = scores / self.max_score # Normalize score

        if self.rel_threshold is None:
            relevances = 1 / (1 + torch.exp(-(norm_scores - self.offset) * self.slope))    ## Rescale relevance
        else:
            relevances = torch.where(norm_scores > self.rel_threshold,
                                        torch.ones_like(norm_scores),
                                        torch.zeros_like(norm_scores))
        if user_state is not None:
            relevances = relevances.transpose(0,1)
        return relevances

    def compute_reward(self, relevances : torch.FloatTensor) -> float:
        # At every iteration r of the for loop below :
        # temp[j] = prod_{k = j+1}^{r - 1} ( 1 - alpha_k gamma_k_j )
        #
        batch_size = len(relevances)
        temp = torch.ones(batch_size, 1, device = self.device)
        click_prob = relevances[:, 0].unsqueeze(1) * self.propensities[0, 0]   # (batch_size, 1)
        for r in range(1, self.rec_size):   # In this for loop r represents the current rank and j the index of last rank
            # 1 - Update temp
            gamma_r_minus_1_j = self.propensities[r - 1, torch.arange(start = -1, end = r - 1, device = self.device)].unsqueeze(0) # (1, r)
            alpha_r_minus_1 = relevances[:, r-1].unsqueeze(1)  # (batch_size, 1)
            temp *= (1 - alpha_r_minus_1 * gamma_r_minus_1_j) # (batch_size, r)

            # 2 - For each remaining doc, compute click prob
            gamma_r_j = self.propensities[r, torch.arange(start = -1, end = r, device = self.device)] # (r + 1)
            alpha_r = relevances[:, r]
            cp_r = torch.cat([torch.ones(batch_size, 1, device = self.device), click_prob[:, :-1]], dim = 1) # (batch_size, r)
            cp = torch.sum(cp_r * temp * gamma_r_j[:-1], dim = 1) * alpha_r + \
                                    alpha_r * gamma_r_j[-1] * click_prob[:, -1] # (batch_size)
            click_prob = torch.cat([click_prob, cp.unsqueeze(1)], dim = 1) # (batch_size, r + 1)
            temp = torch.cat([temp, torch.ones(batch_size, 1, device = self.device)], dim = 1) # (batch_size, r + 1)
        return click_prob
