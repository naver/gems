GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch
import pytorch_lightning as pl

from torch.nn import Sequential, Linear, ReLU, MSELoss, Softmax
from collections import OrderedDict
from typing import List, Any
import copy

from .argument_parser import MyParser
from .belief_encoders import BeliefEncoder
from .data_utils import EnvWrapper
from GeMS.modules.rankers import Ranker

####
# Agents : DQN, SAC, WolpertingerSAC, SlateQ, REINFORCE, REINFORCESlate

class Agent(pl.LightningModule):
    '''
        Agent abstract class, used only to avoid clogging of subclasses.
    '''
    def __init__(self, belief : BeliefEncoder, ranker : Ranker, state_dim : int, action_dim : int, num_actions : int,
                        device : torch.device, random_steps : int, verbose : bool, **kwargs) -> None:
        super().__init__()

        self.verbose = verbose
        self.my_device = device


        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_actions = num_actions

        if self.num_actions == 1:
            self.action_dtype = torch.float
        else:
            self.action_dtype = torch.long

        self.ranker = ranker
        self.belief = belief
        self.pomdp = (belief is not None)

        self.random_steps = random_steps

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[parent_parser], add_help=False)
        return parser

    def validation_step(self, batch, batch_idx : int):
        return {}

    def test_step(self, batch, batch_idx : int):
        return {}

class EpsGreedyOracle(Agent):
    def __init__(self, epsilon_oracle : float, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon_oracle
        self.rec_size = self.belief.rec_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[Agent.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--epsilon_oracle', type=float, default= 0.0)
        return parser

    def get_action(self, state : torch.FloatTensor, sample : bool = True, clicked : torch.LongTensor = None) -> torch.LongTensor:
        '''
            Action selection using the epsilon-greedy exploration strategy
        '''
        eps_greedy = torch.bernoulli(self.epsilon * torch.ones(self.rec_size, device = self.device))

        if clicked is None:
            random = torch.randperm(self.num_actions, device = self.device)[:self.rec_size]
        else:
            items = torch.arange(self.num_actions, device = self.device)
            unique, counts = torch.cat([items, clicked]).unique(return_counts = True)
            non_clicked_items = unique[counts == 1]
            random = non_clicked_items[torch.randperm(len(non_clicked_items), device = self.device)[:self.rec_size]]

        greedy = - torch.ones(self.rec_size, dtype = torch.long, device = self.device)

        return torch.where(eps_greedy.bool(), random, greedy)

class DQN(Agent):
    '''
        Double Deep Q-Networks (Van Asselt et al., 2015). The target network can be updated either via full copy of the weights or
        via Polyak averaging. Epsilon-greedy with exponentially decreasing epsilon is used as the exploration strategy.

        It also serves as an interface with the environment through on_train_epoch_start and on_train_batch_start.
    '''
    def __init__(self, q_lr : float, gamma : float, tau : float, epsilon_start : float, epsilon_end : float,
                    epsilon_decay : float, hidden_layers_qnet : List[int], target_update_frequency : int, gradient_steps : int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.modules = ["critic"]

        # Q Networks
        layers = []
        input_size = self.state_dim + self.action_dim
        out_size = hidden_layers_qnet[:]
        out_size.append(self.num_actions)
        for i, layer_size in enumerate(out_size):
            layers.append(Linear(input_size, layer_size))
            input_size = layer_size
            if i != len(out_size) - 1:
                layers.append(ReLU())
        self.QNet = Sequential(*layers)

        layers = []
        input_size = self.state_dim + self.action_dim
        out_size = hidden_layers_qnet[:]
        out_size.append(self.num_actions)
        for i, layer_size in enumerate(out_size):
            layers.append(Linear(input_size, layer_size))
            input_size = layer_size
            if i != len(out_size) - 1:
                layers.append(ReLU())
        self.target_QNet = Sequential(*layers)
        self.target_QNet.eval()
        self.target_QNet.load_state_dict(self.QNet.state_dict())
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        self.q_lr = q_lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = torch.tensor(epsilon_decay)

        self.gradient_steps = gradient_steps        # Number of gradient steps per update (NOT IMPLEMENTED YET)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[Agent.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--q_lr', type=float, default=1e-3)
        parser.add_argument('--hidden_layers_qnet', type=int, nargs='+', default=[32, 32])
        parser.add_argument('--target_update_frequency', type=int, default= 10)

        parser.add_argument('--gamma', type=float, default=0.9)
        parser.add_argument('--tau', type=float, default=None)
        parser.add_argument('--epsilon_start', type=float, default=1.0)
        parser.add_argument('--epsilon_end', type=float, default=0.01)
        parser.add_argument('--epsilon_decay', type=float, default=1000)
        parser.add_argument('--gradient_steps', type=int, default= 1)

        return parser

    def get_epsilon(self) -> float:
        '''
            Returns epsilon for the current training step (not necessarily environment step !) according to an exponentially
            decreasing schedule.
        '''
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * torch.exp( - self.trainer.global_step / self.epsilon_decay)
        self.log("train_epsilon", epsilon)
        return epsilon

    def polyak_update(self, params, target_params) -> None:
        '''
            Polyak averaging for target network update (Copy-paste from SB3)
        '''
        with torch.no_grad():
            # zip does not raise an exception if length of parameters does not match.
            for param, target_param in zip(params, target_params):
                target_param.data.mul_(1 - self.tau)
                torch.add(target_param.data, param.data, alpha=self.tau, out=target_param.data)

    def get_action(self, state : torch.FloatTensor, sample : bool = True) -> torch.LongTensor:
        '''
            Action selection using the epsilon-greedy exploration strategy
        '''
        if sample and torch.rand(1) < self.get_epsilon():
            action = torch.randint(self.num_actions, (1,))
        else:
            action = self.QNet(state).argmax().unsqueeze(0)
        return action

    def training_step(self, batch, batch_idx : int) -> OrderedDict:
        '''
            Main training step. Assumes categorical actions.
        '''

        if batch == 0 or self.trainer.global_step < self.random_steps:  # Before the learning starts, do nothing.
            return {}

        if self.pomdp:
            state, next_state = self.belief.forward_batch(batch)
            state, next_state = state["critic"], next_state["critic"]
        else:
            state, next_state = batch.obs, batch.next_obs

        # Q-values of sampled actions
        q_values = self.QNet(state).gather(1, batch.action).squeeze()

        with torch.no_grad():
            next_q_values = torch.zeros_like(batch.reward, device = self.device)
            # Q-values of next actions
            next_q_values[batch.done == 0] = self.target_QNet(next_state[batch.done == 0]).max(1)[0].detach()
            # Q-target
            target_q = batch.reward + (next_q_values * self.gamma)

        # QNet loss
        loss = MSELoss()(q_values, target_q)

        # Update target network
        if self.trainer.global_step % self.target_update_frequency == 0:
            if self.tau is None:
                self.target_QNet.load_state_dict(self.QNet.state_dict())
            else:
                self.polyak_update(self.QNet.parameters(), self.target_QNet.parameters())

        self.log('train_loss', loss)
        return OrderedDict({"loss" : loss})

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Initialize Adam optimizer."""
        optimizers = [torch.optim.Adam(self.parameters(), lr=self.q_lr)]
        return optimizers

class SAC(DQN):
    '''
        Soft-Actor Critic (Original implementation by Haarnoja et al., 2017), with Double Clipped Q-Learning and Squashed Diagonal
        Gaussian Actor. We provide a version for continuous actions and a version for slate recommendation belox (SACSlate).
    '''
    def __init__(self, alpha : float, l2_reg : float, pi_lr : float, hidden_layers_qnet : List[int],
                    hidden_layers_pinet : List[int], auto_entropy : bool, alpha_lr : float, **kwargs):
        super().__init__(hidden_layers_qnet = hidden_layers_qnet, **kwargs)

        self.modules = ["actor", "critic"]

        self.automatic_optimization = False # Here the manual optimization allows to lower the computational burden

        self.alpha = alpha  # Controls the importance of entropy regularization
        self.auto_entropy = auto_entropy
        if self.auto_entropy:
            self.log_alpha = torch.zeros(1, device = self.my_device).requires_grad_(True)
            self.alpha_lr = alpha_lr
            self.target_entropy = - self.action_dim
        self.l2_reg = l2_reg
        self.pi_lr = pi_lr

        if self.ranker is not None:
            self.action_center = self.ranker.action_center
            self.action_scale = self.ranker.action_scale
        else:
            self.action_center = 0
            self.action_scale = 1

        # Policy network
        if self.state_dim > 0:
            layers = []
            input_size = self.state_dim
            out_size = hidden_layers_pinet[:]
            if self.pomdp and self.action_dim == 0:
                out_size.append(self.num_actions)
            else:
                out_size.append(self.action_dim * 2)    # We assume independent gaussian here ...
            for i, layer_size in enumerate(out_size):
                layers.append(Linear(input_size, layer_size))
                input_size = layer_size
                if i != len(out_size) - 1:
                    layers.append(ReLU())
            if self.pomdp and self.action_dim == 0:
                layers.append(Softmax(dim = -1))
            self.PolicyNet = Sequential(*layers)
        else:   # Multi-Armed Bandit agent
            self.policy = torch.zeros(2 * self.action_dim, device = self.my_device).requires_grad_(True)


        # Second Q Network
        layers = []
        input_size = self.state_dim + self.action_dim
        out_size = hidden_layers_qnet[:]
        out_size.append(self.num_actions)
        for i, layer_size in enumerate(out_size):
            layers.append(Linear(input_size, layer_size))
            input_size = layer_size
            if i != len(out_size) - 1:
                layers.append(ReLU())
        self.QNet2 = Sequential(*layers)

        # Second Q Target Network
        layers = []
        input_size = self.state_dim + self.action_dim
        out_size = hidden_layers_qnet[:]
        out_size.append(self.num_actions)
        for i, layer_size in enumerate(out_size):
            layers.append(Linear(input_size, layer_size))
            input_size = layer_size
            if i != len(out_size) - 1:
                layers.append(ReLU())
        self.target_QNet2 = Sequential(*layers)
        self.target_QNet2.load_state_dict(self.QNet2.state_dict())
        self.target_QNet2.eval()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[DQN.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--hidden_layers_pinet', type=int, nargs='+', default = [32, 32])
        parser.add_argument('--pi_lr', type=float, default = 1e-3)

        parser.add_argument('--alpha', type=float, default = 0.2)
        parser.add_argument('--auto_entropy', type=parser.str2bool, default = False)
        parser.add_argument('--alpha_lr', type=float, default = 1e-3)
        parser.add_argument('--l2_reg', type=float, default = 0.001)

        return parser

    def get_action(self, state : torch.FloatTensor, sample : bool = True, return_params : bool = False):
        '''
            Action selection using the Squashed Diagonal Gaussian Actor
        '''
        if len(state.shape) == 1: # When not in a batch
            state = state.unsqueeze(0)
        # Get policy params
        if self.state_dim > 0:
            pol_output = self.PolicyNet(state)
        else:   # Multi-Armed Bandit agent
            pol_output = self.policy.expand(len(state), -1)
        mean = pol_output[:, :self.action_dim].squeeze()
        std = torch.clamp(pol_output[:, self.action_dim:].squeeze(), -20, 2).exp()

        if sample:  # For exploration
            norm = torch.distributions.Normal(mean, std) # Diagonal gaussian
            action = norm.rsample()   # Reparameterization trick
            if return_params: # If we want parameters and log-probability to be returned
                logp = norm.log_prob(action)
                action_squashed = torch.nn.Tanh()(action)
                logp -= torch.log(1 - action_squashed.pow(2) + 1e-6)    # This is because of the Tanh
                logp = torch.sum(logp, dim = 1)
                return self.action_center + self.action_scale * action_squashed, logp, mean, std
            else:
                return self.action_center + self.action_scale * torch.nn.Tanh()(action)
        else: # No exploration
            return self.action_center + self.action_scale * torch.nn.Tanh()(mean)

    def training_step(self, batch, batch_idx : int) -> OrderedDict:
        '''
            Main training step
        '''
        if batch == 0 or self.trainer.global_step < self.random_steps:
            return None
        if self.auto_entropy:
            q_opt, pi_opt, alpha_opt = self.optimizers()
        else:
            q_opt, pi_opt = self.optimizers()

        if self.pomdp:
            state, next_state = self.belief.forward_batch(batch)
        else:
            state, next_state = {"actor" : batch.obs, "critic" : batch.obs}, {"actor" : batch.next_obs, "critic" : batch.next_obs}

        # Q-values of actions in the batch
        q_values = self.QNet(torch.cat([state["critic"], batch.action], dim = 1)).squeeze()
        q_values2 = self.QNet2(torch.cat([state["critic"], batch.action], dim = 1)).squeeze()

        if self.pomdp:      ### Let's compare the estimated Q-function and the true return
            initial_q_values = torch.cat([q_values.detach()[0].unsqueeze(0), q_values.detach()[1:][batch.done[:-1] == 1]], dim = 0)
            initial_q_values2 = torch.cat([q_values2.detach()[0].unsqueeze(0), q_values2.detach()[1:][batch.done[:-1] == 1]], dim = 0)
            k=0
            j=0
            returns = torch.zeros(torch.sum(batch.done), device = self.device)
            for i, r in enumerate(batch.reward):
                returns[k] += self.gamma ** j * r
                j+=1
                if batch.done[i]:
                    k += 1
                    j=0
            self.log('initial_q_values', torch.mean(initial_q_values))
            self.log('initial_q_values2', torch.mean(initial_q_values2))
            self.log('returns', torch.mean(returns))

        # Q targets
        with torch.no_grad():
            # Q-values of next actions according to the policy
            next_q_values = torch.zeros_like(batch.reward, device = self.device)
            next_actions, next_logp, _, _ = self.get_action(next_state["actor"][batch.done == 0], return_params = True)
            nqv = self.target_QNet(torch.cat([next_state["critic"][batch.done == 0], next_actions], dim = 1)).squeeze()
            nqv2 = self.target_QNet2(torch.cat([next_state["critic"][batch.done == 0], next_actions], dim = 1)).squeeze()

            # Q targets
            next_q_values[batch.done == 0] = torch.minimum(nqv, nqv2) - self.alpha * next_logp
            target_q = batch.reward + (next_q_values * self.gamma)

        # Q losses and update
        q_loss1 = MSELoss()(q_values, target_q)
        q_loss2 = MSELoss()(q_values2, target_q)
        q_opt.zero_grad()
        self.manual_backward(0.5 * (q_loss1 + q_loss2))
        q_opt.step()

        # Actions sampled from policy
        # Let's not regularize for now
        policy_actions, policy_logp, _, _ = self.get_action(state["actor"], return_params = True)
        #policy_actions, policy_logp, mean, std = self.get_action(states, return_params = True)

        # Q values of actions from policy
        policy_q_values = self.QNet(torch.cat([state["critic"].detach(), policy_actions], dim = 1)).squeeze()
        policy_q_values2 = self.QNet2(torch.cat([state["critic"].detach(), policy_actions], dim = 1)).squeeze()

        # Policy Loss and update
        # Let's not regularize for now
        pi_loss = (self.alpha * policy_logp - torch.minimum(policy_q_values, policy_q_values2)).mean()
        # pi_loss = (self.alpha * policy_logp - torch.minimum(policy_q_values, policy_q_values2)).mean() + \
        #                 self.l2_reg * ( mean.pow(2).mean() + std.pow(2).mean())
        pi_opt.zero_grad()
        self.manual_backward(pi_loss)
        pi_opt.step()

        if self.auto_entropy:
            alpha_loss = - torch.mean(self.log_alpha.exp() * (policy_logp + self.target_entropy).detach())
            alpha_opt.zero_grad()
            self.manual_backward(alpha_loss)
            alpha_opt.step()
            self.alpha = self.log_alpha.exp().detach()

        # Target QNets update
        if self.trainer.global_step % self.target_update_frequency == 0:
            if self.tau is None:
                self.target_QNet.load_state_dict(self.QNet.state_dict())
                self.target_QNet2.load_state_dict(self.QNet2.state_dict())
            else:
                self.polyak_update(self.QNet.parameters(), self.target_QNet.parameters())
                self.polyak_update(self.QNet2.parameters(), self.target_QNet2.parameters())

        self.log('train_q_loss', 0.5 * (q_loss1 + q_loss2))
        self.log('train_pi_loss', pi_loss)
        loss_dict = OrderedDict({"q_loss" : 0.5 * (q_loss1 + q_loss2).detach(), "pi_loss" : pi_loss.detach()})
        if self.auto_entropy:
            self.log('train_alpha_loss', alpha_loss)
            self.log('train_alpha', self.alpha)
            loss_dict["alpha_loss"] = alpha_loss.detach()
        return loss_dict

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Initialize Adam optimizer."""
        if self.pomdp:
            belief_critic_params = [param[1] for param in self.belief.named_parameters() if param[0].split('.')[1] == "critic"]
            optimizer_QNet = torch.optim.Adam(list(self.QNet.parameters()) + list(self.QNet2.parameters()) + belief_critic_params, lr=self.q_lr)
            if self.state_dim > 0:
                belief_actor_params = [param[1] for param in self.belief.named_parameters() if param[0].split('.')[1] == "actor"]
                optimizer_PolicyNet = torch.optim.Adam(list(self.PolicyNet.parameters()) + belief_actor_params, lr=self.pi_lr)
            else:   # Multi-armed Bandit
                optimizer_PolicyNet = torch.optim.Adam([self.policy], lr=self.pi_lr)
        else:
            optimizer_QNet = torch.optim.Adam(list(self.QNet.parameters()) + list(self.QNet2.parameters()), lr=self.q_lr)
            if self.state_dim > 0:
                optimizer_PolicyNet = torch.optim.Adam(self.PolicyNet.parameters(), lr=self.pi_lr)
            else:   # Multi-armed Bandit
                optimizer_PolicyNet = torch.optim.Adam([self.policy], lr=self.pi_lr)
            
        optimizers = [optimizer_QNet, optimizer_PolicyNet]
        if self.auto_entropy:
            optimizers.append(torch.optim.Adam([self.log_alpha], lr=self.alpha_lr))
        return optimizers

class WolpertingerSAC(SAC):
    def __init__(self, full_slate : bool, wolpertinger_k : int, belief : BeliefEncoder, rec_size : int, 
                        action_dim : int, num_actions : int, **kwargs) -> None:
        super().__init__(belief = belief, rec_size = rec_size, num_actions = 1,
                            action_dim = rec_size * belief.item_embeddings["critic"].embedd_dim, **kwargs)

        if self.ranker is not None:
            raise ValueError("WolpertingerSAC requires ranker = None.")
        if not self.pomdp :
            raise ValueError("Only for POMDPs.")

        self.item_embeddings = copy.deepcopy(belief.item_embeddings["critic"])
        self.item_embedd_dim = self.item_embeddings.embedd_dim
        self.rec_size = rec_size
        self.wolpertinger_k = wolpertinger_k

        action_min = torch.min(self.item_embeddings.embedd.weight.data, dim = 0).values.repeat(rec_size)      #item_embedd_dim
        self.action_scale = (torch.max(self.item_embeddings.embedd.weight.data, dim = 0).values.repeat(rec_size) - action_min) / 2 #item_embedd_dim
        self.action_center = action_min + self.action_scale

        self.full_slate = full_slate
        if not full_slate:  #QNet takes state a single item's embedding as input
            raise NotImplementedError("Only full slates for now.")
            self.QNet[0] = Linear(self.state_dim + self.item_embedd_dim, self.QNet[0].out_features)
            self.target_QNet[0] = Linear(self.state_dim + self.item_embedd_dim, self.traget_QNet[0].out_features)
            self.target_QNet.load_state_dict(self.QNet.state_dict())
            self.QNet2[0] = Linear(self.state_dim + self.item_embedd_dim, self.QNet2[0].out_features)
            self.target_QNet2[0] = Linear(self.state_dim + self.item_embedd_dim, self.target_QNet2[0].out_features)
            self.target_QNet2.load_state_dict(self.QNet2.state_dict())
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[SAC.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--full_slate', type=parser.str2bool, default = True)
        parser.add_argument('--wolpertinger_k', type=int, default = 10)
        return parser
    
    def select_best_items(self, state : torch.FloatTensor, candidates : torch.LongTensor) -> torch.LongTensor:
        '''
            Refines the action selection according to Wolpertinger's method.
            
            Parameters :
             - state : torch.FloatTensor(batch_size, state_dim)
                Current state
             - candidates : torch.LongTensor(batch_size, rec_size, wolpertinger_k)
                All candidate items
            
            Output :
             - best_items : torch.LongTensor(batch_size, rec_size)
                Selected items
        '''
        batch_size, rec_size, wolp_k = candidates.shape
        with torch.no_grad():
            embedds = self.item_embeddings(candidates)      # (batch_size, rec_size, wolpertinger_k, item_embedd_dim)
            expanded_state = state.unsqueeze(1).expand(-1, wolp_k, -1).flatten(end_dim = 1) # (batch_size * wolpertinger_k, state_dim)
            if self.full_slate:
                best_embedds = torch.empty(batch_size, self.rec_size, self.item_embedd_dim, device = self.device)
                best_items_ind = torch.empty(batch_size, self.rec_size, dtype = torch.long, device = self.device)
                for r in range(self.rec_size):
                    best_embedds_r = best_embedds[:, :r].unsqueeze(1).expand(-1, wolp_k, -1, -1)    # (batch_size, wolpertinger_k, r, item_embedd_dim)
                    embedds_repeat = embedds[:, r].unsqueeze(2).expand(-1, -1, self.rec_size - r, -1) # (batch_size, wolpertinger_k, rec_size - r, item_embedd_dim)
                    embedds_r = torch.cat([best_embedds_r, embedds_repeat], dim = 2) # (batch_size, wolpertinger_k, rec_size, item_embedd_dim)
                    embedds_r = embedds_r.flatten(start_dim = -2)   # (batch_size, wolpertinger_k, rec_size * item_embedd_dim)
                    q_inp = torch.cat([expanded_state, embedds_r.flatten(end_dim = 1)], dim = 1) # (batch_size * wolpertinger_k, state_dim + rec_size * item_embedd_dim)
                    q_values = self.target_QNet(q_inp).reshape(batch_size, wolp_k)  # (batch_size, wolpertinger_k)
                    best_items_ind[:, r] = torch.argmax(q_values, dim = 1) # (batch_size)
                    best_embedds[:, r] = embedds[torch.arange(batch_size, device = self.device), r, best_items_ind[:, r]] # (batch_size, item_embedd_dim)
            else:
                flat_embedds = embedds.flatten(end_dim = -1)    # (batch_size * rec_size * wolpertinger_k, item_embedd_dim)
                expanded_state = state.unsqueeze(2).expand(-1, -1, wolp_k).flatten(end_dim = -1) # (batch_size, rec_size, wolpertinger_k, state_dim)
                q_values = self.target_QNet(torch.cat([expanded_state, flat_embedds], dim = 1)).reshape(batch_size, rec_size, wolp_k) # (batch_size, rec_size, wolpertinger_k)
                best_items_ind = torch.argmax(q_values, dim = -1)   # (batch_size, rec_size)
            best_items = torch.gather(candidates, 2, best_items_ind.unsqueeze(2)).squeeze(2)   # (batch_size, rec_size)
        return best_items

    def rank(self, state : torch.FloatTensor, action : torch.FloatTensor) -> torch.LongTensor :
        '''
            Action to slate. We use euclidean distance as a similarity metric, because dot-product would be inconsistent for the training trick
            of plugging policy actions directly in QNet for training.

            Parameters : 
             - state : torch.FloatTensor(batch_size, state_dim)
                Current state
             - action : torch.FloatTensor(batch_size, item_embedd_dim * rec_size)
                Target action given by the policy networ
            
            Output :
             - slates : torch.LongTensor(batch_size, rec_size)
        '''
        if len(action.shape) == 1:
            batch_size = 1
            state = state.unsqueeze(0)
        else:
            batch_size, _ = action.shape
        with torch.no_grad():
            
            embedds = self.item_embeddings.get_weights()
            query = action.reshape(batch_size, self.item_embedd_dim, self.rec_size)
            similarity = torch.norm(embedds.unsqueeze(2).unsqueeze(0).expand(batch_size, -1, -1, 10) - query.unsqueeze(1).expand(-1, 1000, -1, -1), dim = 2)
            values, indices = torch.topk(similarity, k = self.wolpertinger_k, dim = 1, largest = False)
            best_items = self.select_best_items(state, indices.transpose(1,2))

            return best_items

    def get_action(self, state : torch.FloatTensor, sample : bool = True, return_params : bool = False, rank = True):
        '''
            Action selection using the Squashed Diagonal Gaussian Actor
        '''
        if return_params:
            action, logp, mean, std = super().get_action(state, sample, return_params)
        else:
            action = super().get_action(state, sample, return_params)
        if rank:
            slate = self.rank(state, action).squeeze()
            if return_params:
                return slate, logp, mean, std
            else : 
                return slate
        else:
            if return_params:
                return action, logp, mean, std
            else : 
                return action

    def training_step(self, batch, batch_idx : int) -> OrderedDict:
        '''
            Main training step
        '''
        if batch == 0 or self.trainer.global_step < self.random_steps:
            return None
        if self.auto_entropy:
            q_opt, pi_opt, alpha_opt = self.optimizers()
        else:
            q_opt, pi_opt = self.optimizers()

        state, next_state = self.belief.forward_batch(batch)

        # Q-values of actions in the batch
        if self.full_slate:
            embedds = torch.cat([self.item_embeddings(slate) for slate in batch.obs["slate"]])
            q_values = self.QNet(torch.cat([state["critic"], embedds.flatten(start_dim = 1)], dim = 1)).squeeze()
            q_values2 = self.QNet2(torch.cat([state["critic"], embedds.flatten(start_dim = 1)], dim = 1)).squeeze()

        # Q targets
        with torch.no_grad():
            # Q-values of next actions according to the policy
            next_q_values = torch.zeros_like(batch.reward, device = self.device)
            next_slates, next_logp, _, _ = self.get_action(next_state["actor"][batch.done == 0], return_params = True)
            next_embedds = self.item_embeddings(next_slates)
            nqv = self.target_QNet(torch.cat([next_state["critic"][batch.done == 0], next_embedds.flatten(start_dim = 1)], dim = 1)).squeeze()
            nqv2 = self.target_QNet2(torch.cat([next_state["critic"][batch.done == 0], next_embedds.flatten(start_dim = 1)], dim = 1)).squeeze()

            # Q targets
            next_q_values[batch.done == 0] = torch.minimum(nqv, nqv2) - self.alpha * next_logp
            target_q = batch.reward + (next_q_values * self.gamma)

        # Q losses and update
        q_loss1 = MSELoss()(q_values, target_q)
        q_loss2 = MSELoss()(q_values2, target_q)
        q_opt.zero_grad()
        self.manual_backward(0.5 * (q_loss1 + q_loss2))
        q_opt.step()

        # Actions sampled from policy
        # Let's not regularize for now
        policy_actions, policy_logp, _, _ = self.get_action(state["actor"], return_params = True, rank = False)
        #policy_actions, policy_logp, mean, std = self.get_action(states, return_params = True)

        # Q values of actions from policy
        policy_q_values = self.QNet(torch.cat([state["critic"].detach(), policy_actions], dim = 1)).squeeze()
        policy_q_values2 = self.QNet2(torch.cat([state["critic"].detach(), policy_actions], dim = 1)).squeeze()

        # Policy Loss and update
        # Let's not regularize for now
        pi_loss = (self.alpha * policy_logp - torch.minimum(policy_q_values, policy_q_values2)).mean()
        # pi_loss = (self.alpha * policy_logp - torch.minimum(policy_q_values, policy_q_values2)).mean() + \
        #                 self.l2_reg * ( mean.pow(2).mean() + std.pow(2).mean())
        pi_opt.zero_grad()
        self.manual_backward(pi_loss)
        pi_opt.step()

        if self.auto_entropy:
            alpha_loss = - torch.mean(self.log_alpha.exp() * (policy_logp + self.target_entropy).detach())
            alpha_opt.zero_grad()
            self.manual_backward(alpha_loss)
            alpha_opt.step()
            self.alpha = self.log_alpha.exp().detach()

        # Target QNets update
        if self.trainer.global_step % self.target_update_frequency == 0:
            if self.tau is None:
                self.target_QNet.load_state_dict(self.QNet.state_dict())
                self.target_QNet2.load_state_dict(self.QNet2.state_dict())
            else:
                self.polyak_update(self.QNet.parameters(), self.target_QNet.parameters())
                self.polyak_update(self.QNet2.parameters(), self.target_QNet2.parameters())

        self.log('train_q_loss', 0.5 * (q_loss1 + q_loss2))
        self.log('train_pi_loss', pi_loss)
        loss_dict = OrderedDict({"q_loss" : 0.5 * (q_loss1 + q_loss2).detach(), "pi_loss" : pi_loss.detach()})
        if self.auto_entropy:
            self.log('train_alpha_loss', alpha_loss)
            self.log('train_alpha', self.alpha)
            loss_dict["alpha_loss"] = alpha_loss.detach()
        return loss_dict

class SlateQ(DQN):
    '''
        SlateQ (Ie et al., 2019) decomposes the value function of a slate into a combination of item value functions.
        Right now we only perform topk maximization.
    '''
    def __init__(self, env : EnvWrapper, opt_method : str, rec_size : int, **kwargs) -> None:
        super().__init__(**kwargs)

        self.rec_size = rec_size
        self.env = env.env
        self.opt_method = opt_method

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[DQN.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--opt_method', type=str, choices = ["topk", "greedy", "lp"], default= "topk")
        return parser

    def get_action(self, state : torch.FloatTensor, clicked : torch.LongTensor = None, sample : bool = True) -> torch.LongTensor:
        '''
            Action selection using the epsilon-greedy exploration strategy
        '''
        if sample and torch.rand(1) < self.get_epsilon():
            action = torch.randint(self.num_actions, (self.rec_size,), device = self.device)
        elif self.opt_method == "topk":
            relevances = self.env.get_relevances() ## (num_items)
            q_values = self.QNet(state) ## (num_items)
            action = torch.topk(relevances * q_values, self.rec_size).indices ## (rec_size)
        return action

    def training_step(self, batch, batch_idx : int) -> OrderedDict:
        '''
            Main training step.
        '''

        if batch == 0 or self.trainer.global_step < self.random_steps:  # Before the learning starts, do nothing.
            return None

        state, next_state = self.belief.forward_batch(batch)
        state, next_state = state["critic"], next_state["critic"]

        # The following assumes at most one click per slate !
        concat_slates = torch.cat(batch.obs["slate"], dim = 0)
        concat_clicks = torch.cat(batch.obs["clicks"], dim = 0)

        clicked_slates = torch.any(concat_clicks.bool(), dim = 1)
        first_click = torch.argmax(concat_clicks[clicked_slates], dim = 1)
        clicked_items = concat_slates[clicked_slates, first_click]

        # Individual Q-values of clicked_items
        q_values = self.QNet(state[clicked_slates]).gather(1, clicked_items.unsqueeze(1)).squeeze()


        with torch.no_grad():
            next_q_values = torch.zeros_like(q_values, device = self.device)

            # Next slate according to Top-k
            relevances = self.env.get_relevances(batch.info[clicked_slates][batch.done[clicked_slates] == 0]) 
            item_comps = self.env.item_comp
            next_q = self.target_QNet(next_state[clicked_slates][batch.done[clicked_slates] == 0])   # Q(s', j), for all j
            _, next_action = torch.topk(relevances * next_q, k = self.rec_size, dim = 1, sorted = True)  # A'
            next_q = next_q.gather(1, next_action)    # Q(s', j), j in A'

            # Now compute P(j|s', A')
            attractiveness = relevances.gather(1, next_action)
            item_comps = self.env.item_comp[next_action]
            if torch.max(item_comps.unique(return_counts = True)[1]) >= self.env.diversity_threshold:
                attractiveness /= self.env.diversity_penalty
            click_probs = self.env.compute_reward(attractiveness)  # P(j|s', A')

            next_q_values[batch.done[clicked_slates] == 0] = torch.sum(next_q * click_probs, dim = 1)

            target_q = 1 + (next_q_values * self.gamma)

        # QNet loss
        loss = MSELoss()(q_values, target_q)

        # Update target network
        if self.trainer.global_step % self.target_update_frequency == 0:
            if self.tau is None:
                self.target_QNet.load_state_dict(self.QNet.state_dict())
            else:
                self.polyak_update(self.QNet.parameters(), self.target_QNet.parameters())

        self.log('train_loss', loss)
        return OrderedDict({"loss" : loss})

class REINFORCE(Agent):
    '''
        Continuous REINFORCE Agent.
    '''
    def __init__(self, sigma_explo : float, pi_lr : float, hidden_layers_pinet : List[int], gamma : float, **kwargs):
        super().__init__(**kwargs)

        self.modules = ["actor"]

        self.full_traj = True
        self.pi_lr = pi_lr
        self.gamma = torch.tensor(gamma)
        self.sigma_explo = sigma_explo

        # Policy network
        layers = []
        input_size = self.state_dim
        out_size = hidden_layers_pinet[:]
        if self.pomdp and self.action_dim == 0:
            out_size.append(self.num_actions)
        else:
            out_size.append(self.action_dim)    # We assume independent gaussian here.
        for i, layer_size in enumerate(out_size):
            layers.append(Linear(input_size, layer_size))
            input_size = layer_size
            if i != len(out_size) - 1:
                layers.append(ReLU())
        if self.pomdp and self.action_dim == 0:
            layers.append(Softmax(dim = -1))
        self.PolicyNet = Sequential(*layers)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[Agent.add_model_specific_args(parent_parser)], add_help=False)
        parser.add_argument('--hidden_layers_pinet', type=int, nargs='+', default = [32, 32])
        parser.add_argument('--pi_lr', type=float, default = 1e-3)
        parser.add_argument('--gamma', type=float, default=1.0)
        parser.add_argument('--sigma_explo', type=float, default=0.29)
        return parser

    def get_action(self, state : torch.FloatTensor, sample : bool = True, return_params : bool = False) -> torch.LongTensor:
        '''
            Action selection by sampling from softmax
        '''
        if len(state.shape) == 1: # When not in a batch
            state = state.unsqueeze(0)
        # Get policy params
        pol_output = self.PolicyNet(state)
        mean = pol_output[:, :self.action_dim].squeeze()
        std = self.sigma_explo
        #std = torch.clamp(pol_output[:, self.action_dim:].squeeze(), -20, 2).exp()


        if sample:  # For exploration
            try:
                norm = torch.distributions.Normal(mean, std) # Diagonal gaussian
            except ValueError:
                print(mean)
                print(std)
            action = norm.rsample()   # Reparameterization trick
            if return_params: # If we want parameters and log-probability to be returned
                logp = norm.log_prob(action)
                action_squashed = torch.nn.Tanh()(action)
                logp -= torch.log(1 - action_squashed.pow(2) + 1e-6)    # This is because of the Tanh
                logp = torch.sum(logp, dim = 1)
                return action_squashed, logp, mean, std
            else:
                return torch.nn.Tanh()(action)
        else: # No exploration
            return torch.nn.Tanh()(mean)

    def training_step(self, batch, batch_idx : int) -> OrderedDict:
        '''
            Main training step.
        '''
        if batch == 0:
            return

        if self.pomdp:
            states = self.belief.forward_batch(batch)[0]["actor"].squeeze() # traj_len
        else:
            states = batch.obs

        pol_output = self.PolicyNet(states)
        mean = pol_output[:, :self.action_dim].squeeze()
        std = self.sigma_explo
        #std = torch.clamp(pol_output[:, self.action_dim:].squeeze(), -20, 2).exp()

        try:
            norm = torch.distributions.Normal(mean, std) # Diagonal gaussian
        except ValueError:
            print("Mean", mean)
            print("Std", std)
        log_probs = torch.sum(norm.log_prob(torch.atanh(batch.action)) - torch.log(1 - batch.action.pow(2) + 1e-6), dim = 1)

        if self.gamma > 0.05:   ## This is nicer but fails with low gammas
            gamma_pow = self.gamma.pow(torch.arange(len(batch.reward), device = self.device))
            returns_to_go = torch.flip(torch.cumsum(torch.flip(gamma_pow * batch.reward, [0]), dim = 0), [0]) / gamma_pow
        else:
            returns_to_go = torch.zeros(len(batch.reward), device = self.device)
            returns_to_go[-1] = batch.reward[-1]
            for i in range(len(batch.reward)-2, -1, -1):
                returns_to_go[i] = batch.reward[i] + self.gamma * returns_to_go[i+1]

        pseudo_loss = - torch.sum(returns_to_go * log_probs)

        self.log('train_loss', pseudo_loss)
        return OrderedDict({"loss" : pseudo_loss})

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Initialize Adam optimizer."""
        optimizers = [torch.optim.Adam(self.parameters(), lr=self.pi_lr)]
        return optimizers

class REINFORCESlate(REINFORCE):
    '''
        REINFORCE Agent (Slate actions if ranker is None and continuous actions otherwise)
        We still use a replay buffer, it's just that the buffer size is 1
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.pomdp:
            raise ValueError("SOPSlate only works in POMDPs.")
        if self.action_dim != 0:
            raise ValueError("SOPSlate is not compatible with rankers.")
        self.rec_size = self.belief.rec_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[REINFORCE.add_model_specific_args(parent_parser)], add_help=False)
        return parser

    def get_action(self, state : torch.FloatTensor, sample : bool = True, clicked : torch.LongTensor = None) -> torch.LongTensor:
        '''
            Action selection by sampling from softmax
        '''
        item_probs = self.PolicyNet(state)

        if sample:
            item_dist = torch.distributions.categorical.Categorical(item_probs)
            return item_dist.sample((self.rec_size,))   ## We don't avoid duplicates for now
        else:
            if clicked is None:
                return torch.topk(item_probs, k = self.rec_size, sorted = True)[1]
            else:
                unique, counts = torch.cat([torch.arange(self.belief.item_embeddings["actor"].num_items, device = self.device), clicked]).unique(return_counts = True)
                return unique[counts == 1][torch.topk(item_probs[unique[counts == 1]], k = self.rec_size, sorted = True)[1]]

    def training_step(self, batch, batch_idx : int) -> OrderedDict:
        '''
            Main training step.
        '''

        if batch == 0:
            return

        states = self.belief.forward_batch(batch)[0]["actor"].squeeze() #traj_len

        # The following assumes at most one click per slate ! If there are more than 1 click per slate, we only consider the first one.
        clicked_slates = torch.any(batch.obs["clicks"][0], dim = 1)
        first_click = torch.argmax(batch.obs["clicks"][0][clicked_slates], dim = 1)
        clicked_items = batch.obs["slate"][0][clicked_slates, first_click]
        # clicked_items = batch.obs["slate"][0][batch.obs["clicks"][0].bool()]  # traj_len
        # clicked_slates = torch.any(batch.obs["clicks"][0].bool(), dim = 1)
        pi_clicked = self.PolicyNet(states[clicked_slates]).gather(1, clicked_items.unsqueeze(1))
        log_alphas = torch.log(1 - (1 - pi_clicked).pow(self.rec_size) + 1e-6).squeeze()
        if self.gamma > 0.05:   ## This is nicer but fails with low gammas
            gamma_pow = self.gamma.pow(torch.arange(len(batch.reward), device = self.device))
            returns_to_go = torch.flip(torch.cumsum(torch.flip(gamma_pow * batch.reward, [0]), dim = 0), [0]) / gamma_pow
        else:
            returns_to_go = torch.zeros(len(batch.reward), device = self.device)
            returns_to_go[-1] = batch.reward[-1]
            for i in range(len(batch.reward)-2, -1, -1):
                returns_to_go[i] = batch.reward[i] + self.gamma * returns_to_go[i+1]
        returns_to_go = returns_to_go[clicked_slates]

        pseudo_loss = - torch.sum(returns_to_go * log_alphas)

        self.log('train_loss', pseudo_loss)
        return OrderedDict({"loss" : pseudo_loss})

class RandomSlate(Agent):
    '''
        Agent which returns slates with randomly selected items
    '''
    def __init__(self, rec_size : int, **kwargs):
        super().__init__(**kwargs)

        self.rec_size = rec_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[Agent.add_model_specific_args(parent_parser)], add_help=False)
        return parser

    def get_action(self, state : torch.FloatTensor, sample : bool = True, clicked : torch.LongTensor = None) -> torch.LongTensor:
        '''
            Action selection by randomly sampling each item of the slate
        '''
        return torch.randint(low = 0, high = self.num_actions, size = (self.rec_size,), device = self.my_device)

    def training_step(self, batch, batch_idx : int) -> OrderedDict:
        return OrderedDict({"loss" : torch.tensor(0, device = self.my_device)})

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        return []

class STOracleSlate(Agent):
    '''
        Agent which returns slates that are optimal for short term
    '''
    def __init__(self, rec_size : int, **kwargs):
        super().__init__(**kwargs)

        self.rec_size = rec_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[Agent.add_model_specific_args(parent_parser)], add_help=False)
        return parser

    def get_action(self, state : torch.FloatTensor, sample : bool = True, clicked : torch.LongTensor = None) -> torch.LongTensor:
        '''
            Action selection by returning a slate made only of -1's (optimal items are then directly selected in the environment)
        '''
        return -torch.ones(size = (self.rec_size,), device = self.my_device, dtype=torch.long)

    def training_step(self, batch, batch_idx : int) -> OrderedDict:
        return OrderedDict({"loss" : torch.tensor(0, device = self.my_device)})

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        return []
