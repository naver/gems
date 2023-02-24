GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import torch
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from aim import Figure

from typing import Any, List
from pytorch_lightning.loops import FitLoop, TrainingEpochLoop, EvaluationEpochLoop, EvaluationLoop
from pytorch_lightning.loops.utilities import _is_max_limit_reached
import os
from collections import namedtuple
from pathlib import Path

from .data_utils import EnvWrapper, ReplayBuffer
from .belief_encoders import BeliefEncoder
from .agents import Agent, EpsGreedyOracle, SlateQ
from GeMS.modules.rankers import Ranker
from .argument_parser import MyParser


Trajectory = namedtuple("Trajectory", ("obs", "action", "reward", "next_obs", "done"))

### Only for POMDP for now

class TrainingEpisodeLoop(TrainingEpochLoop):
    '''
        This loop replaces the TrainingEpochLoop in RL
    '''
    def __init__(self, env : EnvWrapper, buffer : ReplayBuffer, belief : BeliefEncoder, 
                    agent : Agent, ranker : Ranker, random_steps : int, max_steps : int, device : str, **kwargs) -> None:
        super().__init__(**kwargs)

        self.pomdp = (belief is not None)
        self.env = env
        self.buffer = buffer
        self.belief = belief
        self.agent = agent
        self.ranker = ranker
        self.random_steps = random_steps
        self.device = torch.device(device)
        self.current_iter_step = 0
        self.max_steps_per_iter = max_steps

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--random_steps', type=int, default= 1000)
        return parser

    @property
    def done(self):
        return self.ep_done or super().done

    @property
    def _is_training_done(self) -> bool:
        max_steps_reached = _is_max_limit_reached(self.current_iter_step, self.max_steps_per_iter)
        return max_steps_reached or self._num_ready_batches_reached()

    def reset(self) -> None:
        '''
            Resets the environment.
        '''
        self.obs = self.env.reset()
        self.ep_done = False
        self.cum_reward = 0.0
        self.ep_length = 0
        if self.pomdp:
            self.store(self.obs, None, None, None)
            self.already_clicked = self.obs["slate"][torch.nonzero(self.obs["clicks"]).flatten()]
            self.obs = self.belief.forward(self.obs)
        super().reset()
    
    def store(self, obs, action, reward, done):
        '''
            Store states, actions and reward along the current trajectory.
        '''
        if action is None:   # New trajectory
            self.obs_traj = {key : val.unsqueeze(0) for key, val in obs.items()}
            if self.ranker is None:
                action_size = self.belief.rec_size
            else:
                action_size = self.agent.action_dim
            self.action_traj = torch.empty(0, action_size, device = self.device, dtype = self.agent.action_dtype)
            self.reward_traj = torch.empty(0, device = self.device)
        else:   # We append the new observations to the trajectory
            if not done:
                self.obs_traj = {key : torch.cat([self.obs_traj[key], val.unsqueeze(0)], dim = 0) for key, val in obs.items()}
            self.action_traj = torch.cat([self.action_traj, action.unsqueeze(0)])
            self.reward_traj = torch.cat([self.reward_traj, reward.unsqueeze(0)])

    def advance(self, *args, **kwargs) -> None:
        '''
            Performs one environment step on top on the usual TrainingEpochLoop
        '''
        if self.pomdp:  ### Full trajectory, each epoch only has one training step
            if self.agent.__class__ == SlateQ:
                info_traj = torch.empty(0, self.belief.get_state_dim(), device = self.device)
            else:
                info_traj = None
            while not self.ep_done:
                # Action selection
                if self.trainer.global_step < self.random_steps:
                    if self.ranker is None:
                        action = self.env.get_random_action()
                    else:
                        action = self.ranker.get_random_action()
                else:
                    with torch.inference_mode():
                        if self.ranker is None:
                            action = self.agent.get_action(self.obs)#, clicked = self.already_clicked)
                        else:
                            action = self.agent.get_action(self.obs)
                
                # Slate generation
                if self.ranker is not None:
                    with torch.inference_mode():
                        rec_list = self.ranker.rank(action)#, clicked = self.already_clicked)
                else:
                    rec_list = action

                # Environment step and belief update
                self.obs, reward, self.ep_done, info = self.env.step(rec_list)
                if self.agent.__class__ == SlateQ:
                    info_traj = torch.cat([info_traj, info["user_state"].unsqueeze(0)], dim = 0)
                self.store(self.obs, action, reward, self.ep_done)
                self.already_clicked = torch.cat([self.already_clicked, self.obs["slate"][torch.nonzero(self.obs["clicks"]).flatten()]])
                self.obs = self.belief.forward(self.obs, done = self.ep_done)
                self.cum_reward += reward
                self.ep_length += 1
            
            # Push to buffer
            dones = torch.zeros(self.ep_length, dtype = torch.long, device = self.device)
            dones[-1] += 1
            self.buffer.push("env", self.obs_traj, self.action_traj, self.reward_traj, None, dones, info_traj)
        else:   ### We update at every new state
            # Action selection
            if self.trainer.global_step < self.random_steps:
                action = self.env.get_random_action()
            else:
                with torch.inference_mode():
                    action = self.agent.get_action(self.obs)

            # Environment step and push to buffer
            next_obs, reward, self.ep_done, _ = self.env.step(action)
            self.buffer.push("env", self.obs, action, reward, next_obs, self.ep_done, None)
            self.obs = next_obs.clone()
            self.cum_reward += reward
            self.ep_length += 1

        super().advance(*args, **kwargs)
        if self.ep_done:
            self.batch_progress.is_last_batch = True
  
    def on_advance_end(self):
        super().on_advance_end()
        self.current_iter_step += 1

    def on_run_end(self) -> Any:
        '''
            Pushes to the replay buffer
        '''   
        output = super().on_run_end()
        # Log relevant quantities
        # self.trainer.lightning_module._current_fx_name = "training_step"
        # self.trainer.lightning_module.log("train_reward", self.cum_reward, prog_bar = True)
        # self.trainer.lightning_module.log("train_ep_length", float(self.ep_length))
        return output

class ValEpisodeLoop(EvaluationEpochLoop):
    '''
        Replaces the validation loop in RL.
    '''
    def __init__(self, env : EnvWrapper, belief : BeliefEncoder, agent : Agent, ranker : Ranker, 
                        val_step_length : int, device : str, trainer, filename_results : str, **kwargs) -> None:
        super().__init__()

        self.trainer = trainer

        self.pomdp = (belief is not None)
        self.env = env
        self.belief = belief
        self.agent = agent
        self.ranker = ranker
        if self.ranker is not None:
            self.ranker = self.ranker.to(device)
        self.device = torch.device(device)

        self.val_step_length = val_step_length
        self.count_ep = 0
        self.filename = filename_results
        # if os.path.isfile(self.filename):
        #     os.remove(self.filename)
        # else:
        #     Path("/" + os.path.join(*self.filename.split("/")[:-2]) + "/results/").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--val_step_length', type=int, default= 25)
        return parser

    @property
    def done(self):
        return self.count_ep == self.val_step_length or super().done
    
    def reset(self) -> None:
        '''
            Resets the environment.
        '''
        self.count_ep = 0
        self.cum_rewards = torch.empty(self.val_step_length)
        if self.pomdp:
            self.scores = torch.empty(0, 100, 10, device = self.device)
            self.rewards = torch.empty(0, 100, device = self.device)
            num_topics = self.env.env.num_topics
            self.bored = torch.empty(0, num_topics, dtype = torch.bool, device = self.device)
        super().reset()

    def advance(self, *args, **kwargs) -> None:
        '''
            Performs one validation episode EvaluationLoop
        '''
        super().advance(*args, **kwargs)

        ### Reset the env
        obs = self.env.reset()
        done = False
        cum_reward = 0.0
        ep_length = 0
        already_clicked = None
        if self.pomdp:
            already_clicked = obs["slate"][torch.nonzero(obs["clicks"]).flatten()]
            if self.count_ep == 0:
                slates = torch.empty(0, self.belief.rec_size, device = self.device, dtype = torch.long)
            obs = self.belief.forward(obs)


        with torch.inference_mode():
            traj_scores = torch.empty(0, 10, device = self.device)
            traj_rewards = torch.empty(0, device = self.device)
            while not done:
                # Action selection
                if self.ranker is not None:
                    action = self.agent.get_action(obs, sample = False)
                    rec_list = self.ranker.rank(action)#, clicked = already_clicked)
                else:
                    rec_list = self.agent.get_action(obs, sample = False)#, clicked = already_clicked)

                # Environment step and belief update
                obs, reward, done, info = self.env.step(rec_list)
                if self.pomdp:
                    already_clicked = torch.cat([already_clicked, obs["slate"][torch.nonzero(obs["clicks"]).flatten()]])
                    traj_scores = torch.cat([traj_scores, info["scores"].unsqueeze(0)], dim = 0)
                    traj_rewards = torch.cat([traj_rewards, torch.tensor([reward], device = self.device)], dim = 0)
                    self.bored = torch.cat([self.bored, info["bored"].unsqueeze(0).expand(10, -1)], dim = 0)
                    if self.count_ep == 0:
                        slates = torch.cat([slates, obs["slate"].unsqueeze(0)], dim = 0)
                    obs = self.belief.forward(obs, done = done)

                cum_reward += reward
                ep_length += 1
            self.scores = torch.cat([self.scores, traj_scores.unsqueeze(0)], dim = 0)
            self.rewards = torch.cat([self.rewards, traj_rewards.unsqueeze(0)], dim = 0)
        
        # if self.count_ep == 0:
        #     print(slates)
        self.trainer.lightning_module.log("val_reward", cum_reward)
        self.cum_rewards[self.count_ep] = cum_reward
        self.trainer.lightning_module.log("val_episode_length", float(ep_length)) 
        if self.pomdp: 
            self.trainer.lightning_module.log("val_scores", torch.mean(self.scores))  
        self.count_ep += 1

    def on_run_end(self) -> Any:
        '''
            Pushes to the replay buffer
        '''
        output = super().on_run_end()

        ### Save results to disk
        # cum_reward = torch.mean(self.cum_rewards)
        # if os.path.isfile(self.filename):
        #     val_rewards = torch.load(self.filename)
        # else:
        #     val_rewards = torch.empty(0)
        # val_rewards = torch.cat([val_rewards, cum_reward.unsqueeze(0)], dim = 0)
        # torch.save(val_rewards, self.filename)

        ### Plot boredom status
        if self.pomdp:
            n_bins = 100
            num_topics = self.env.env.num_topics
            norm_factor = len(self.scores.flatten()) * 0.5 / n_bins
            if self.env.env.rel_threshold is None:
                offset = self.env.env.offset
            else:
                offset = self.env.env.rel_threshold
            
            hist, bins = torch.empty(num_topics, n_bins), torch.empty(num_topics, n_bins + 1)
            bored = torch.arange(num_topics).repeat_interleave(n_bins)
            for i in range(num_topics):
                hist[i], bins[i] = torch.histogram(self.scores.flatten()[torch.sum(self.bored, dim = 1) == i].cpu(), bins=100, range=(0., 0.5), density=False)
            
            hist_flatten = hist.flatten()
            bins_flatten = bins[:, :-1].flatten() + 0.5 / (2 * n_bins)

            np_data = torch.stack([hist_flatten / norm_factor, bins_flatten, bored], dim = 1).cpu().numpy()
            df = pd.DataFrame(data = np_data, columns = ["hist", "bin", "boredom"])
            df["boredom"] = df["boredom"].astype(str)
            fig = px.bar(df, x="bin", y="hist", color = "boredom", labels={'bin':'Score', 'hist':'PDF'}, 
                            title = "Density of normalized scores for after %d steps" % self.trainer.global_step,
                            color_discrete_sequence= px.colors.sequential.Plasma_r)
            fig.add_vline(x = offset, line_dash="dash", line_color="black", annotation_text="threshold")

            aim_figure = Figure(fig)
            self.trainer.logger._run.track(aim_figure, step = self.trainer.global_step, name="val_diversity")
        ### Plot scores and reward per timestep
        if self.pomdp:
            if self.env.env.rel_threshold is None:
                offset = self.env.env.offset
            else:
                offset = self.env.env.rel_threshold
            
            mean_score_t = torch.mean(self.scores, dim = (0, 2))
            mean_reward_t = torch.mean(self.rewards, dim = 0)

            np_data = torch.stack([torch.arange(len(mean_score_t)), mean_score_t.cpu(), mean_reward_t.cpu()], dim = 1).numpy()
            df = pd.DataFrame(data = np_data, columns = ["timestep", "score", "reward"])

            subfig = make_subplots(specs=[[{"secondary_y": True}]])

            # create two independent figures with px.line each containing data from multiple columns
            fig = px.line(df, x="timestep", y="score")
            fig.add_hline(y = offset, line_dash="dash", line_color="black", annotation_text="Threshold")
            fig2 = px.line(df, x="timestep", y="reward")

            fig2.update_traces(yaxis="y2")

            subfig.add_traces(fig.data + fig2.data)
            subfig.layout.xaxis.title="Timestep"
            subfig.layout.yaxis.title="Average Score"
            subfig.layout.yaxis2.title="Average Reward"
            subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))

            aim_figure = Figure(subfig)
            self.trainer.logger._run.track(aim_figure, step = self.trainer.global_step, name="val_scores_rewards")
        return output


class TestEpisodeLoop(EvaluationEpochLoop):
    '''
        Replaces the test loop in RL.
    '''
    def __init__(self, env : EnvWrapper, belief : BeliefEncoder, agent : Agent, ranker : Ranker, 
                        test_size : int, device : str, trainer, filename_results : str, **kwargs) -> None:
        super().__init__()

        self.trainer = trainer

        self.pomdp = (belief is not None)
        self.env = env
        self.belief = belief
        self.agent = agent
        self.ranker = ranker
        if self.ranker is not None:
            self.ranker = self.ranker.to(device)
        self.device = torch.device(device)

        self.test_size = test_size
        self.count_ep = 0
        self.filename = filename_results
        # if os.path.isfile(self.filename):
        #     os.remove(self.filename)
        # else:
        #     Path("/" + os.path.join(*self.filename.split("/")[:-2]) + "/results/").mkdir(parents=True, exist_ok=True)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = MyParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--test_size', type=int, default= 100)
        return parser

    @property
    def done(self):
        return self.count_ep == self.test_size or super().done
    
    def reset(self) -> None:
        '''
            Resets the environment.
        '''
        self.count_ep = 0
        self.cum_rewards = torch.empty(self.test_size)
        self.test_trajs = {}

        if self.pomdp:
            self.scores = torch.empty(0, 100, 10, device = self.device)
            self.rewards = torch.empty(0, 100, device = self.device)
            num_topics = self.env.env.num_topics
            self.bored = torch.empty(0, num_topics, dtype = torch.bool, device = self.device)
        super().reset()

    def advance(self, *args, **kwargs) -> None:
        '''
            Performs one validation episode EvaluationLoop
        '''
        super().advance(*args, **kwargs)

        ### Reset the env
        obs = self.env.reset()
        done = False
        cum_reward = 0.0
        ep_length = 0
        already_clicked = None
        if self.pomdp:
            already_clicked = obs["slate"][torch.nonzero(obs["clicks"]).flatten()]
            self.test_trajs[self.count_ep] = {"slates" : torch.empty(0, self.belief.rec_size, dtype = torch.long, device = self.device),
                                            "slate_comps" : torch.empty(0, self.belief.rec_size, dtype = torch.long, device = self.device),
                                            "user_state" : torch.empty(0, self.env.env.num_topics * self.env.env.topic_size, dtype = torch.long, device = self.device),
                                            "bored" : torch.empty(0, self.env.env.num_topics, dtype = torch.bool, device = self.device),
                                            "scores" : torch.empty(0, self.belief.rec_size, dtype = torch.float, device = self.device),
                                            "actions" : torch.empty(0, self.agent.action_dim, dtype = torch.float, device = self.device),
                                            "clicks" : torch.empty(0,self.belief.rec_size, dtype = torch.long, device = self.device)}
            if self.agent.__class__ != EpsGreedyOracle:
                obs = self.belief.forward(obs)

        with torch.inference_mode():
            traj_scores = torch.empty(0, 10, device = self.device)
            traj_rewards = torch.empty(0, device = self.device)
            while not done:
                # Action selection
                if self.ranker is not None:
                    action = self.agent.get_action(obs, sample = False)
                    rec_list = self.ranker.rank(action)#, clicked = already_clicked)
                else:
                    rec_list = self.agent.get_action(obs, sample = False)#, clicked = already_clicked)

                # Environment step and belief update
                obs, reward, done, info = self.env.step(rec_list)
                if self.pomdp:
                    already_clicked = torch.cat([already_clicked, obs["slate"][torch.nonzero(obs["clicks"]).flatten()]])
                    self.test_trajs[self.count_ep]["slates"] = torch.cat([self.test_trajs[self.count_ep]["slates"], info["slate"].unsqueeze(0)], dim = 0)
                    self.test_trajs[self.count_ep]["slate_comps"] = torch.cat([self.test_trajs[self.count_ep]["slate_comps"], info["slate_components"].unsqueeze(0)], dim = 0)
                    self.test_trajs[self.count_ep]["user_state"] = torch.cat([self.test_trajs[self.count_ep]["user_state"], info["user_state"].unsqueeze(0)], dim = 0)
                    self.test_trajs[self.count_ep]["bored"] = torch.cat([self.test_trajs[self.count_ep]["bored"], info["bored"].unsqueeze(0)], dim = 0)
                    self.test_trajs[self.count_ep]["scores"] = torch.cat([self.test_trajs[self.count_ep]["scores"], info["scores"].unsqueeze(0)], dim = 0)
                    if self.ranker is not None:
                        self.test_trajs[self.count_ep]["actions"] = torch.cat([self.test_trajs[self.count_ep]["actions"], action.unsqueeze(0)], dim = 0)
                    self.test_trajs[self.count_ep]["clicks"] = torch.cat([self.test_trajs[self.count_ep]["clicks"], info["clicks"].unsqueeze(0)], dim = 0)
                    traj_scores = torch.cat([traj_scores, info["scores"].unsqueeze(0)], dim = 0)
                    traj_rewards = torch.cat([traj_rewards, torch.tensor([reward], device = self.device)], dim = 0)
                    self.bored = torch.cat([self.bored, info["bored"].unsqueeze(0).expand(10, -1)], dim = 0)

                    if self.agent.__class__ != EpsGreedyOracle:
                        obs = self.belief.forward(obs, done = done)
                    
                cum_reward += reward
                ep_length += 1
            self.scores = torch.cat([self.scores, traj_scores.unsqueeze(0)], dim = 0)
            self.rewards = torch.cat([self.rewards, traj_rewards.unsqueeze(0)], dim = 0)
        
        
        self.trainer.lightning_module.log("test_reward", cum_reward)
        self.cum_rewards[self.count_ep] = cum_reward
        self.trainer.lightning_module.log("test_episode_length", float(ep_length))  
        self.count_ep += 1

    def on_run_end(self) -> Any:
        '''
            Pushes to the replay buffer
        '''
        output = super().on_run_end()

        ### Save results to disk
        if self.filename is not None :
            cum_reward = torch.mean(self.cum_rewards)
            torch.save(cum_reward, self.filename)
            torch.save(self.test_trajs, self.filename[:-3] + "_testtraj.pt")

        # Plot boredom status
        if self.pomdp:
            n_bins = 100
            num_topics = 3#self.env.env.num_topics
            norm_factor = len(self.scores.flatten()) * 0.5 / n_bins
            if self.env.env.rel_threshold is None:
                offset = self.env.env.offset
            else:
                offset = self.env.env.rel_threshold
            
            hist, bins = torch.empty(num_topics, n_bins), torch.empty(num_topics, n_bins + 1)
            bored = torch.arange(num_topics).repeat_interleave(n_bins)
            for i in range(num_topics):
                hist[i], bins[i] = torch.histogram(self.scores.flatten()[torch.sum(self.bored, dim = 1) == i].cpu(), bins=100, range=(0., 0.5), density=False)
            
            hist_flatten = hist.flatten()
            bins_flatten = bins[:, :-1].flatten() + 0.5 / (2 * n_bins)

            np_data = torch.stack([hist_flatten / norm_factor, bins_flatten, bored], dim = 1).cpu().numpy()
            df = pd.DataFrame(data = np_data, columns = ["hist", "bin", "boredom"])
            df["boredom"] = df["boredom"].astype(int).astype(str)
            fig = px.bar(df, x="bin", y="hist", color = "boredom", labels={'bin':'Score', 'hist':'PDF', 'boredom' : "Number of <br>saturated topics"}, 
                            #title = "Density of normalized scores for after %d steps" % self.trainer.global_step,
                            color_discrete_sequence= px.colors.sequential.Plasma_r)
            fig.add_vline(x = offset, line_dash="dash", line_color="black",
                            annotation_text="threshold", #annotation_position="top left",
                            annotation_font_size=15, annotation_font_color="black")
            fig.add_vline(x = self.scores.mean(), line_color="red", 
                            annotation_text="average score", annotation_position="top left",
                            annotation_font_size=15, annotation_font_color="red")
            fig.update_layout(legend=dict(
                                    yanchor="top",
                                    y=0.9,
                                    xanchor="right",
                                    x=0.99,
                                    font_size = 15
                                ),
                                yaxis = dict(tickfont = dict(size=15),
                                            title = dict(font=dict(size=15))),
                                xaxis = dict(tickfont = dict(size=15),
                                            title = dict(font=dict(size=15))),
                                plot_bgcolor='rgb(256,256,256)')
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

            fig.update_traces(marker=dict(line=dict(width=0,
                                        color='DarkSlateGrey')))

            aim_figure = Figure(fig)
            self.trainer.logger._run.track(aim_figure, step = self.trainer.global_step, name="test_diversity")
        ### Plot scores and reward per timestep
        if self.pomdp:
            if self.env.env.rel_threshold is None:
                offset = self.env.env.offset
            else:
                offset = self.env.env.rel_threshold
            
            mean_score_t = torch.mean(self.scores, dim = (0, 2))
            mean_reward_t = torch.mean(self.rewards, dim = 0)

            np_data = torch.stack([torch.arange(len(mean_score_t)), mean_score_t.cpu(), mean_reward_t.cpu()], dim = 1).numpy()
            df = pd.DataFrame(data = np_data, columns = ["timestep", "score", "reward"])

            subfig = make_subplots(specs=[[{"secondary_y": True}]])

            # create two independent figures with px.line each containing data from multiple columns
            fig = px.line(df, x="timestep", y="score")
            fig.add_hline(y = offset, line_dash="dash", line_color="black", annotation_text="Threshold")
            fig2 = px.line(df, x="timestep", y="reward")

            fig2.update_traces(yaxis="y2")

            subfig.add_traces(fig.data + fig2.data)
            subfig.layout.xaxis.title="Timestep"
            subfig.layout.yaxis.title="Average Score"
            subfig.layout.yaxis2.title="Average Reward"
            subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))

            aim_figure = Figure(subfig)
            self.trainer.logger._run.track(aim_figure, step = self.trainer.global_step, name="test_scores_rewards")
        return output


class ResettableFitLoop(FitLoop):
    '''
        This allows us to set a maximumnumber of epochs per fitting iteration while not interfering with logging and other
        inner mechanisms of the trainer.
    '''
    def __init__(self, max_epochs_per_iter : int) -> None:
        super().__init__(max_epochs = -1)

        self.max_epochs_per_iter = max_epochs_per_iter
        self.current_iter_epoch = 0  
    
    @property
    def done(self) -> bool:
        """Evaluates when to leave the loop.

        Returns True if trainer.should_stop was set (e.g. by early stopping) or if the maximum number of steps or epochs
        is reached.
        """
        # TODO(@awaelchli): Move track steps inside training loop and move part of these condition inside training loop
        stop_steps = _is_max_limit_reached(self.epoch_loop.current_iter_step, self.epoch_loop.max_steps_per_iter)
        stop_epochs = _is_max_limit_reached(self.current_iter_epoch, self.max_epochs_per_iter)

        should_stop = False
        if self.trainer.should_stop:
            # early stopping
            met_min_epochs = self.current_epoch >= self.min_epochs if self.min_epochs else True
            met_min_steps = self.global_step >= self.min_steps if self.min_steps else True
            if met_min_epochs and met_min_steps:
                should_stop = True
            else:
                log.info(
                    "Trainer was signaled to stop but required minimum epochs"
                    f" ({self.min_epochs}) or minimum steps ({self.min_steps}) has"
                    " not been met. Training will continue..."
                )
        self.trainer.should_stop = should_stop

        return stop_steps or should_stop or stop_epochs or self.trainer.num_training_batches == 0  
    
    def on_run_end(self) -> None:
        super().on_run_end()
        self.epoch_loop.current_iter_step = 0
        self.current_iter_epoch = 0     

    def on_advance_end(self) -> None:
        super().on_advance_end()
        self.current_iter_epoch += 1
