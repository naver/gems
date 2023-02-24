GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

from abc import abstractmethod

import torch
import pytorch_lightning as pl

from typing import List, Tuple, Dict, Union
from torch.nn import Sequential, Embedding, Linear, Softmax, CrossEntropyLoss, BCEWithLogitsLoss, ReLU
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .argument_parser import MyParser
from .item_embeddings import ItemEmbeddings
from .data_utils import Trajectory


class Ranker(pl.LightningModule):
    '''
        Abstract Ranker class.
    '''
    def __init__(self, item_embeddings : ItemEmbeddings, item_embedd_dim : int, device : torch.device,
                    rec_size : int, **kwargs) -> None:
        super().__init__()

        self.my_device = device
        self.rec_size = rec_size
        self.item_embedd_dim = item_embedd_dim
        self.item_embeddings = item_embeddings
        self.num_items = self.item_embeddings.num_items

        action_min = torch.min(self.item_embeddings.embedd.weight.data, dim = 0).values      #item_embedd_dim
        self.action_scale = (torch.max(self.item_embeddings.embedd.weight.data, dim = 0).values - action_min) / 2 #item_embedd_dim
        self.action_center = action_min + self.action_scale

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[parent_parser], add_help=False)
        arguments = [action.option_strings[0] for action in parser._actions]
        if '--num_items' not in arguments:
            parser.add_argument('--num_items', type=int, default = 1000)
        if '--item_embedd_dim' not in arguments:
            parser.add_argument('--item_embedd_dim', type=int, default = 20)
        if '--rec_size' not in arguments:
            parser.add_argument('--rec_size', type=int, default = 10)
        return parser

class TopKRanker(Ranker):
    '''
        Retrieves the k items closest to the latent action.
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.modules = []

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[Ranker.add_model_specific_args(parent_parser)], add_help=False)
        return parser

    def get_action_dim(self) -> Tuple[int, int]:
        return self.item_embedd_dim, 1

    def get_random_action(self) -> torch.FloatTensor:
        return self.action_center + self.action_scale * (torch.rand(self.item_embedd_dim, device = self.device) - 0.5)

    def rank(self, action, clicked : torch.LongTensor = None) -> torch.LongTensor:
        '''
            Translates a latent action into a ranked list of items.
            Here the action is expected to be in the space of item embeddings.
        '''
        with torch.inference_mode():
            similarity = torch.matmul(self.item_embeddings.get_weights(), action)
            #similarity /= torch.linalg.vector_norm(similarity, dim = 1)
        if clicked is None:
            return torch.topk(similarity, k = self.rec_size, sorted = True)[1]
        else:
            unique, counts = torch.cat([torch.arange(self.num_items, device = self.device), clicked]).unique(return_counts = True)
            return unique[counts == 1][torch.topk(similarity[unique[counts == 1]], k = self.rec_size, sorted = True)[1]]

class kHeadArgmaxRanker(TopKRanker):
    '''
        Retrieves the closest item for each slot of the slate
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.action_center = self.action_center.repeat(self.rec_size)
        self.action_scale = self.action_scale.repeat(self.rec_size)

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[TopKRanker.add_model_specific_args(parent_parser)], add_help=False)
        return parser

    def get_action_dim(self) -> Tuple[int, int]:
        return self.item_embedd_dim * self.rec_size, 1

    def get_random_action(self) -> torch.FloatTensor:
        return self.action_center + self.action_scale * (torch.rand(self.item_embedd_dim * self.rec_size, device = self.device) - 0.5)

    def rank(self, action, clicked : torch.LongTensor = None) -> torch.LongTensor:
        '''
            Translates a latent action into a ranked list of items.
            Here the action is expected to be of size item_embedd_dim * rec_size.
        '''
        with torch.inference_mode():
            similarity = torch.matmul(self.item_embeddings.get_weights(), action.reshape(self.item_embedd_dim, self.rec_size))
            #similarity /= torch.linalg.vector_norm(similarity, dim = 1)
        if clicked is None:
            return torch.argmax(similarity, dim = 0)
        else:
            unique, counts = torch.cat([torch.arange(self.num_items, device = self.device), clicked]).unique(return_counts = True)
            return unique[counts == 1][torch.argmax(similarity[unique[counts == 1], :], dim = 0)]

class AbstractGeMS(Ranker):
    '''
        Abstract parent for the GeMS family of model classes.
    '''

    def __init__(self, latent_dim : int, lambda_click : float, lambda_KL : float, lambda_prior : float,
                    ranker_lr : float, fixed_embedds : bool, ranker_sample : bool, **kwargs) -> None:
        super().__init__(**kwargs)

        self.modules = []

        self.latent_dim = latent_dim
        self.lambda_click = lambda_click
        self.lambda_KL = lambda_KL
        self.lambda_prior = lambda_prior
        self.lr = ranker_lr
        self.sample = ranker_sample

        # Item embeddings
        item_pre_embeddings = self.item_embeddings.get_weights() # Pre-trained/random item embeddings from PT Lightning
        self.item_embeddings = Embedding(self.num_items, self.item_embedd_dim)
        self.item_embeddings.weight.data.copy_(item_pre_embeddings)
        if fixed_embedds in ["mf_fixed"]: # Use frozen item embeddings
            self.item_embeddings.weight.requires_grad = False

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[Ranker.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--lambda_click', type=float, default=1.0)
        parser.add_argument('--lambda_KL', type=float, default=1.0)
        parser.add_argument('--lambda_prior', type=float, default=1.0)
        parser.add_argument('--latent_dim', type=int, default=8)
        parser.add_argument('--ranker_lr', type=float, default=3e-3)

        #### For ranker selection in RL4REC
        parser.add_argument('--ranker_dataset', type=str, default=None)
        parser.add_argument('--ranker_embedds', type=str, default=None)
        parser.add_argument('--ranker_seed', type=int, default=None)
        parser.add_argument('--ranker_sample', type=parser.str2bool, default=False)
        return parser

    def get_action_dim(self) -> Tuple[int, int]:
        return self.latent_dim, 1

    def get_random_action(self) -> torch.FloatTensor:
        return self.action_center + self.action_scale * (torch.rand(self.latent_dim, device = self.device) - 0.5)

    def get_action_bounds(self, data_path : str, batch_size : int = 10) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        '''
            Returns the action bounds for continuous control inside GeMS's latent space.
        '''
        data = torch.load(data_path)

        action_min = 1e6 * torch.ones(self.latent_dim)
        action_max = -1e6 * torch.ones(self.latent_dim)
        for i in range(1000 // batch_size):
            # batch = {"slate" : torch.cat([traj["slate"] for traj in list(data.values())[i * batch_size : (i+1) * batch_size]], dim = 0),
            #             "clicks" : torch.cat([traj["clicks"] for traj in list(data.values())[i * batch_size : (i+1) * batch_size]], dim = 0)}
            ### 1 - Pass through embeddings
            slates = torch.stack([traj["slate"] for traj in list(data.values())[i * batch_size : (i+1) * batch_size]], dim = 0)           # batch_size, traj_len, rec_size
            clicks = torch.stack([traj["clicks"] for traj in list(data.values())[i * batch_size : (i+1) * batch_size]], dim = 0).float()  # batch_size, traj_len, rec_size

            ### 2 - Pass through inference model
            with torch.inference_mode():
                latent_mu, log_latent_var = self.run_inference(slates, clicks)

            latent_sigma = torch.exp(log_latent_var / 2)
            latent_min = torch.min(latent_mu - latent_sigma, dim = 0).values
            latent_max = torch.max(latent_mu + latent_sigma, dim = 0).values

            action_min = torch.minimum(action_min, latent_min)
            action_max = torch.maximum(action_max, latent_max)

        self.action_scale = (action_max - action_min).to(self.my_device) / 2
        self.action_center = action_min.to(self.my_device) + self.action_scale
        return self.action_center, self.action_scale

    @abstractmethod
    def rank(self, action) -> torch.LongTensor:
        '''
            Translates a latent action into a ranked list of items.
            Here the action is expected to be in the latent space of the VAE.
        '''
        pass

    @abstractmethod
    def run_inference(self, slates, clicks) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        pass

    @abstractmethod
    def run_decoder(self, latent_sample) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        pass

    @abstractmethod
    def run_prior(self) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        pass

    def training_step(self, batch, batch_idx) -> torch.FloatTensor:
        '''
            Pre-training of the ranker.
        '''
        ### 1 - Extract slates and clicks from batch
        slates = torch.stack(batch.obs["slate"])            # batch_size, rec_size
        clicks = torch.stack(batch.obs["clicks"]).float()            # batch_size, rec_size

        ### 2 - Pass through inference model
        latent_mu, log_latent_var = self.run_inference(slates, clicks)

        ### 3 - Reparameterization trick
        latent_var = log_latent_var.exp()
        latent_sample = latent_mu + torch.randn_like(latent_var) * latent_var    # batch_size, latent_dim

        ### 4 - Pass through decoder model
        item_logits, click_logits = self.run_decoder(latent_sample)

        ### 5 - Pass through prior model
        prior_mu, log_prior_var = self.run_prior()           # batch_size * seq_len, latent_dim * 2
        prior_var = log_prior_var.exp()

        ### 6 - Compute the losses
        slate_loss = CrossEntropyLoss(reduction = 'mean')(item_logits, slates.flatten())   # Softmax is in the CrossEntropyLoss
        click_loss = BCEWithLogitsLoss(reduction = 'mean')(click_logits, clicks.flatten(end_dim = -2))
        mean_term = ((latent_mu - prior_mu) ** 2) / prior_var
        KLLoss = 0.5 * (log_prior_var - log_latent_var + latent_var / prior_var + mean_term - 1).mean()
        prior_reg = torch.sum(prior_mu.pow(2) + log_prior_var.pow(2))

        loss = slate_loss + self.lambda_click * click_loss + self.lambda_KL * KLLoss + self.lambda_prior * prior_reg
        self.log("train_loss", loss)
        self.log("train_slateloss", slate_loss)
        self.log("train_clickloss", click_loss)
        self.log("train_KLloss", KLLoss)
        self.log("train_prior_reg", prior_reg)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.FloatTensor:
        '''
            Validation step during pre-training of the ranker.
        '''
        ### 1 - Pass through embeddings
        slates = torch.stack(batch.obs["slate"])            # batch_size, rec_size
        clicks = torch.stack(batch.obs["clicks"]).float()            # batch_size, rec_size

        ### 2 - Pass through inference model
        latent_mu, log_latent_var = self.run_inference(slates, clicks)

        ### 3 - Reparameterization trick
        latent_var = log_latent_var.exp()
        # latent_sample = latent_mu + torch.randn_like(latent_var) * latent_var    # batch_size, latent_dim

        ### 4 - Pass through decoder model
        item_logits, click_logits = self.run_decoder(latent_mu)

        ### 5 - Pass through prior model
        prior_mu, log_prior_var = self.run_prior()           # batch_size * seq_len, latent_dim * 2
        prior_var = log_prior_var.exp()

        ### 6 - Compute the losses
        slate_loss = CrossEntropyLoss(reduction = 'mean')(item_logits, slates.flatten())   # Softmax is in the CrossEntropyLoss
        click_loss = BCEWithLogitsLoss(reduction = 'mean')(click_logits, clicks.flatten(end_dim = -2))
        mean_term = ((latent_mu - prior_mu) ** 2) / prior_var
        KLLoss = 0.5 * (log_prior_var - log_latent_var + latent_var / prior_var + mean_term - 1).mean()
        prior_reg = torch.sum(prior_mu.pow(2) + log_prior_var.pow(2))

        loss = slate_loss + self.lambda_click * click_loss + self.lambda_KL * KLLoss # + self.lambda_prior * prior_reg

        self.log("val_loss", loss)
        self.log("val_slateloss", slate_loss)
        self.log("val_clickloss", click_loss)
        self.log("val_KLloss", KLLoss)
        self.log("val_prior_reg", prior_reg)
        return loss

    # def encode(self, obs : Dict) -> torch.FloatTensor:
    #     with torch.inference_mode():
    #         ### 1 - Pass through embeddings
    #         slates = obs["slate"]                     # traj_len, rec_size
    #         clicks = obs["clicks"].float()            # traj_len, rec_size

    #         ### 2 - Pass through inference model
    #         latent_mu, log_latent_var = self.run_inference(slates, clicks)

    #     return latent_mu, log_latent_var

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]
        # return {
        #         'optimizer': optimizer,
        #         'lr_scheduler': ReduceLROnPlateau(optimizer, factor = 0.5, patience = 2),
        #         'monitor': 'val_loss',
        #         }

class GeMS(AbstractGeMS):
    '''
        Slate-VAE.
    '''
    def __init__(self, hidden_layers_infer : List[int], hidden_layers_decoder : List[int], **kwargs) -> None:
        super().__init__(**kwargs)

        # Inference
        layers = []
        input_size = self.rec_size * (self.item_embedd_dim + 1)
        out_size = hidden_layers_infer[:]
        out_size.append(self.latent_dim * 2)    # mu and log_sigma
        for i, layer_size in enumerate(out_size):
            layers.append(Linear(input_size, layer_size))
            input_size = layer_size
            if i != len(out_size) - 1:
                layers.append(ReLU())
        self.inference = Sequential(*layers)

        # Decoder
        layers = []
        input_size = self.latent_dim
        out_size = hidden_layers_decoder[:]
        for i, layer_size in enumerate(out_size):
            layers.append(Linear(input_size, layer_size))
            input_size = layer_size
            layers.append(ReLU())
        self.decoder = Sequential(*layers)
        self.slate_decoder = Linear(out_size[-1], self.rec_size * self.item_embedd_dim)
        self.click_decoder = Linear(out_size[-1], self.rec_size)

    @staticmethod
    def add_model_specific_args(parent_parser) -> MyParser:
        parser = MyParser(parents=[AbstractGeMS.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--hidden_layers_infer', type=int, nargs='+', default=[512, 256])
        parser.add_argument('--hidden_layers_decoder', type=int, nargs='+', default=[256, 512])
        return parser

    def rank(self, action, clicked = None) -> torch.LongTensor:
        '''
            Translates a latent action into a ranked list of items.
            Here the action is expected to be in the latent space of the VAE.
        '''
        with torch.inference_mode():
            item_logits = self.slate_decoder(self.decoder(action)).reshape(self.rec_size, self.item_embedd_dim) \
                          @ self.item_embeddings.weight.t()
        if clicked is None:
            if self.sample:
                dist = torch.distributions.categorical.Categorical(logits = item_logits)
                return dist.sample()
            else:
                return torch.argmax(item_logits, dim = 1)   # rec_size
        else:   # Only with sample = False
            unique, counts = torch.cat([torch.arange(self.num_items, device = self.device), clicked]).unique(return_counts = True)
            return unique[counts == 1][torch.argmax(item_logits[:, unique[counts == 1]], dim = 1)]

    def run_inference(self, slates, clicks) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        if len(slates.shape) == 3 : # Batch of trajectories
            slates = slates.flatten(end_dim = 1)
            clicks = clicks.flatten(end_dim = 1)
        embedds = self.item_embeddings(slates).flatten(start_dim = 1)  # batch_size, rec_size * item_embedd_dim
        latent_params = self.inference(torch.cat([embedds, clicks], dim = 1))   # batch_size, latent_dim * 2
        return latent_params[:, :self.latent_dim], latent_params[:, self.latent_dim:]

    def run_decoder(self, latent_sample) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        batch_size = latent_sample.size()[0]
        reconstruction = self.decoder(latent_sample) # batch_size, hidden_layer_size
        item_logits = self.slate_decoder(reconstruction).reshape(batch_size * self.rec_size, self.item_embedd_dim) \
                      @ self.item_embeddings.weight.t().detach() # No backprop to item embeddings for that branch
        # batch_size * rec_size, num_items
        click_logits = self.click_decoder(reconstruction)   # batch_size, rec_size

        return item_logits, click_logits

    def run_prior(self) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return torch.zeros(self.latent_dim, device = self.device), torch.zeros(self.latent_dim, device = self.device)
