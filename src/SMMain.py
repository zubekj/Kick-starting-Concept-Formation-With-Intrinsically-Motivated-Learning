import argparse
import glob
import os
import shutil
import sys
import time
import types
from itertools import cycle, repeat, chain
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score, silhouette_score
from sklearn.cluster import KMeans

from params import Parameters
from SMAgent import SMAgent
from SMController import SMController
from SMEnv import SMEnv, SMEnvParasite
from SMGraphs import (comp_map, goal_frequency_map, log, proprio_map,
                      remove_figs, somatosensory_map, trajectories_map,
                      update_weight_data, visual_map)
from tplot import TPlotManager


matplotlib.use("Agg")
np.set_printoptions(formatter={"float": "{:6.4f}".format})

storage_dir = "storage"
site_dir = "www"
simulations_dir = "simulations"
os.makedirs(simulations_dir, exist_ok=True)


class RepeatedGoalPrototypeException(Exception):
    pass


class TimeLimitsException(Exception):
    pass


class SensoryMotorCircle:
    def __init__(self, action_steps=5):
        self.t = 0
        self.action_steps = action_steps

    def step(self, env, agent, state):
        if self.t % self.action_steps == 0:
            self.action = agent.step(state)
        state = env.step(self.action)

        self.t += 1
        return state

    def noisy_step(self, env, agent, state):
        if self.t % self.action_steps == 0:
            self.action = agent.step(state)
        state = env.step(
            self.action
            + np.random.normal(
                scale=params.motor_noise, size=self.action.shape
            )
        )

        self.t += 1
        return state


def softmax(x, t=0.01):
    e = np.exp(x / t)
    return e / (e.sum() + 1e-100)


class Main:
    def __init__(self, params, seed=None, plots=False):

        self.params = params

        print("Main", flush=True)

        if seed is None:
            seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        self.seed = seed

        self.plots = plots
        self.start = time.perf_counter()
        if self.plots is True:
            remove_figs()

        self.random_obj_params = {
            "stretch_conditions": self.params.obj_stretch_conditions,
            "rotation_conditions": self.params.obj_rotation_conditions,
            "pos": [self.params.obj_y, self.params.obj_x],
        }

        self.obj_params_space = [
            {"stretch_conditions": [stretch],
             "rotation_conditions": [rotation],
             "pos": [self.params.obj_y, self.params.obj_x],
             }
            for stretch in self.params.obj_stretch_conditions
            for rotation in self.params.obj_rotation_conditions
        ]

        self.env = SMEnv(
            seed,
            self.params,
            self.params.action_steps,
            rand_obj_params=self.random_obj_params,
        )
        self.agent = SMAgent(self.env)
        self.controller = SMController(
            self.params,
            self.rng,
            load=self.params.load_weights,
            shuffle=self.params.shuffle_weights,
        )
        self.logs = np.zeros([self.params.epochs, 3])
        self.epoch = 0

    def __getstate__(self):
        return {
            "params": self.params.__getstate__(),
            "controller": self.controller.__getstate__(),
            "env": self.env.__getstate__(),
            "rng": self.rng.__getstate__(),
            "seed": self.seed,
            "plots": self.plots,
            "logs": self.logs,
            "epoch": self.epoch,
        }

    def __setstate__(self, state):

        self.params = Parameters()
        self.params.__setstate__(state["params"])
        self.plots = state["plots"]
        self.logs = state["logs"]
        self.epoch = state["epoch"]
        self.seed = state["seed"]
        torch.manual_seed(self.seed)
        self.rng = np.random.RandomState()
        self.rng.__setstate__(state["rng"])

        self.random_obj_params = {
            "stretch_conditions": self.params.obj_stretch_conditions,
            "rotation_conditions": self.params.obj_rotation_conditions,
            "pos": [self.params.obj_y, self.params.obj_x],
        }

        self.obj_params_space = [
            {"stretch_conditions": [stretch],
             "rotation_conditions": [rotation],
             "pos": [self.params.obj_y, self.params.obj_x],
             }
            for stretch in self.params.obj_stretch_conditions
            for rotation in self.params.obj_rotation_conditions
        ]

        nlogs = len(self.logs)
        if self.params.epochs > nlogs:
            tmp = np.zeros([self.params.epochs, 3])
            tmp[:nlogs, :] = self.logs.copy()
            self.logs = tmp
            tmp = np.zeros([self.params.epochs, 2])

        self.env = SMEnv(
            self.seed,
            self.params,
            self.params.action_steps,
            rand_obj_params=self.random_obj_params,
        )
        self.controller = SMController(
            self.params,
            self.rng,
            load=self.params.load_weights,
            shuffle=self.params.shuffle_weights,
        )

        self.controller.__setstate__(state["controller"])
        self.env.__setstate__(state["env"])

        self.agent = SMAgent(self.env)
        self.start = time.perf_counter()
        if self.plots is True:
            remove_figs(self.epoch)

    def reset_model_data(self, controller):

        for key in controller.model_data:
            controller.model_data[key][::] = 0

    def initialize_model_data(self, controller, n_episodes=None):

        if n_episodes is None:
            n_episodes = self.params.batch_size

        controller.model_data = {
            "batch_v": np.zeros(
                [
                    n_episodes,
                    self.params.stime,
                    self.params.visual_size,
                ]
            ),
            "batch_ss": np.zeros(
                [
                    n_episodes,
                    self.params.stime,
                    self.params.somatosensory_size,
                ]
            ),
            "batch_p": np.zeros(
                [
                    n_episodes,
                    self.params.stime,
                    self.params.proprioception_size,
                ]
            ),
            "batch_a": np.zeros(
                [
                    n_episodes,
                    self.params.stime,
                    self.params.policy_size,
                ]
            ),
            "batch_c": np.zeros([n_episodes, self.params.stime, 1]),
            "batch_log": np.zeros([n_episodes, self.params.stime, 1]),
            "batch_g": np.zeros(
                [
                    n_episodes,
                    self.params.stime,
                    self.params.internal_size,
                ]
            ),
            "v_r": np.zeros(
                [
                    n_episodes,
                    self.params.stime,
                    self.params.internal_size,
                ]
            ),
            "ss_r": np.zeros(
                [
                    n_episodes,
                    self.params.stime,
                    self.params.internal_size,
                ]
            ),
            "p_r": np.zeros(
                [
                    n_episodes,
                    self.params.stime,
                    self.params.internal_size,
                ]
            ),
            "a_r": np.zeros(
                [
                    n_episodes,
                    self.params.stime,
                    self.params.internal_size,
                ]
            ),
            "v_p": np.zeros([n_episodes, self.params.stime, 2]),
            "ss_p": np.zeros([n_episodes, self.params.stime, 2]),
            "p_p": np.zeros([n_episodes, self.params.stime, 2]),
            "a_p": np.zeros([n_episodes, self.params.stime, 2]),
            "g_p": np.zeros([n_episodes, self.params.stime, 2]),
            "match_value": np.zeros([n_episodes, self.params.stime]),
            "match_value_per_mod": np.zeros(
                [n_episodes, self.params.stime, 4]
            ),
        }

    def is_object_out_of_taskspace(self, state):
        obj_xy = state["OBJ_POSITION"][0, 0]
        xlim, ylim = (
            self.params.task_space["xlim"],
            self.params.task_space["ylim"],
        )
        return (
            obj_xy[0] < xlim[0]
            or obj_xy[0] > xlim[1]
            or obj_xy[1] < ylim[0]
            or obj_xy[1] > ylim[1]
        )

    def calc_match_inc_within_goal(self, policy_ended, controller):
        def corr(x):
            return np.corrcoef(np.arange(len(x)), x)[0, 1]

        corrs_coeffs_p = []
        corrs_coeffs_ss = []

        for i in range(policy_ended.shape[0]):
            prev_ind = -1
            for curr_ind in np.argwhere(policy_ended[i])[:, 0]:
                corr_ss = corr(
                    controller.model_data["match_value_per_mod"][
                        i, prev_ind + 1 : curr_ind + 1, 1
                    ]
                )
                if not np.isnan(corr_ss):
                    corrs_coeffs_ss.append(corr_ss)
                corr_p = corr(
                    controller.model_data["match_value_per_mod"][
                        i, prev_ind + 1 : curr_ind + 1, 2
                    ]
                )
                if not np.isnan(corr_p):
                    corrs_coeffs_p.append(corr_p)

        episode_match_inc_p = np.mean(corrs_coeffs_p)
        episode_match_inc_ss = np.mean(corrs_coeffs_ss)
        return episode_match_inc_p, episode_match_inc_ss

    def calc_mi_metrics(self, contexts, params_ind, controller, policy_ended):
        """
        Calculate mutual information between object configuration and touch sensors.
        """
        mask = np.ones(policy_ended.shape, dtype=bool)
        mask[:, :self.params.drop_first_n_steps + self.params.policy_selection_steps] = 0

        policies = controller.model_data["batch_a"][mask]
        unique_policies = controller.model_data["batch_a"][policy_ended].reshape(-1, policies.shape[-1])
        policies = policies.reshape(-1, policies.shape[-1])
        km = KMeans(n_clusters=10)
        unique_policies_ind = km.fit_predict(unique_policies)
        policies_ind = km.predict(policies)
        #policies_sscore = silhouette_score(unique_policies, unique_policies_ind)
        policies_dispersion = (((unique_policies - unique_policies.mean(axis=0))**2).sum(axis=1)).mean()
        policies_inertia_norm = km.inertia_ / unique_policies.shape[0] / policies_dispersion

        gripper = controller.model_data["batch_p"][mask]
        gripper = gripper.reshape(-1, gripper.shape[-1])
        km = KMeans(n_clusters=10)
        gripper_ind = km.fit_predict(gripper)
        #gripper_sscore = silhouette_score(gripper, gripper_ind)
        gripper_dispersion = (((gripper - gripper.mean(axis=0))**2).sum(axis=1)).mean()
        gripper_inertia_norm = km.inertia_ / gripper.shape[0] / gripper_dispersion

        touch = controller.model_data["batch_ss"][mask]
        # Discretize touch into 4 regions corresponding to gripper edges
        bins = np.arange(touch.shape[-1], step=10)
        discrete_touch = np.digitize(touch.argmax(axis=-1), bins)
        # No touch is a 5-th category
        discrete_touch[touch.sum(axis=-1) == 0] = 0
        discrete_touch = discrete_touch.reshape(-1)

        contexts = np.repeat(contexts[:, None],
                             self.params.stime, axis=1)[mask].reshape(-1)
        
        # mi_score_context_touch = adjusted_mutual_info_score(contexts, discrete_touch)
        # mi_score_context_policy = adjusted_mutual_info_score(contexts, policies_ind)
        # mi_score_policy_touch = adjusted_mutual_info_score(policies_ind, discrete_touch)
        # mi_score_context_gripper = adjusted_mutual_info_score(contexts, gripper_ind)
        # mi_score_policy_gripper = adjusted_mutual_info_score(policies_ind, gripper_ind)

        res = {
            "mi_score_context_touch": mutual_info_score(contexts, discrete_touch),
            "mi_score_context_policy": mutual_info_score(contexts, policies_ind),
            "mi_score_policy_touch": mutual_info_score(policies_ind, discrete_touch),
            "mi_score_context_gripper": mutual_info_score(contexts, gripper_ind),
            "mi_score_policy_gripper": mutual_info_score(policies_ind, gripper_ind),
            "policies_inertia_norm": policies_inertia_norm,
            "gripper_inertia_norm": gripper_inertia_norm,
        }

        return res 

    def action_outcome_step(
        self,
        episode,
        t,
        smcycles,
        agent,
        envs,
        states,
    ):
        # action-outcome step
        if (
            t >= self.params.drop_first_n_steps
            and t < self.params.drop_first_n_steps + self.params.action_steps
        ):
            state = smcycles[episode].noisy_step(
                envs[episode], agent, states[episode]
            )
        else:
            state = smcycles[episode].step(
                envs[episode], agent, states[episode]
            )

        return state

    def manage_end_of_episode(self, episode, t, controller, state, states):
        if self.is_object_out_of_taskspace(state):
            states[episode] = None
        else:
            states[episode] = state
            controller.model_data["batch_v"][episode, t - 1, :] = state[
                "VISUAL_SENSORS"
            ].ravel()
            controller.model_data["batch_ss"][episode, t - 1, :] = state[
                "TOUCH_SENSORS"
            ]
            controller.model_data["batch_p"][episode, t - 1, :] = state[
                "JOINT_POSITIONS"
            ][:5]

    def run_episodes(
        self,
        agent,
        controller,
        contexts,
        envs,
        states,
    ):
        batch_size = len(contexts)

        # fill all batches with zero policy with is used for first N steps
        controller.model_data["batch_a"][::] = 0

        cum_match = np.zeros((batch_size, self.params.stime), dtype=int)
        episode_len = np.zeros(batch_size, dtype=int)
        max_match = np.zeros((batch_size, self.params.stime))
        matches = np.zeros((batch_size, self.params.stime), dtype=bool)
        policy_changed = np.zeros((batch_size, self.params.stime), dtype=bool)
        bsize = batch_size * self.params.action_steps

        visual_activation = np.zeros((batch_size, self.params.stime))
        goal_activation = np.zeros((batch_size, self.params.stime))

        # Main loop through time steps and episodes
        smcycles = [SensoryMotorCircle(self.params.action_steps)] * batch_size
        for t in range(1, self.params.stime + 1):
            if t < self.params.stime:
                for episode in range(batch_size):
                    # Do not update the episode if it has ended
                    if states[episode] is None:
                        continue
                    episode_len[episode] = t

                    # set correct policy/
                    agent.updatePolicy(
                        controller.model_data["batch_a"][episode, t, :]
                    )

                    # action-outcome step
                    state = self.action_outcome_step(
                        episode,
                        t,
                        smcycles,
                        agent,
                        envs,
                        states,
                    )

                    # End the episode if object moves too far away
                    self.manage_end_of_episode(
                        episode,
                        t,
                        controller,
                        state,
                        states,
                    )

            # Store states and decide action after each params.action_steps
            # interval
            if t % self.params.action_steps == 0 or t == self.params.stime:
                # get Representations for the last N = self.params.action_steps
                # steps
                t0 = t - self.params.action_steps
                sa = np.s_[:, t0:t, :]

                # Use minimal sigma for building within-episode representations
                #controller.updateParams(
                #    self.params.base_internal_sigma, controller.curr_lr
                #)
                # Use representation sigma for building within-episode representations
                controller.updateParams(
                    self.params.representation_sigma, controller.curr_lr
                )

                # Compute state representations
                data_keys = [
                    "batch_v",
                    "batch_ss",
                    "batch_p",
                    "batch_a",
                    "batch_g",
                ]
                reshaped_data = [
                    controller.model_data[key][sa].reshape((bsize, -1))
                    for key in data_keys
                ]
                Rs, Rp = controller.spread(reshaped_data)

                # Store representation in history
                result_keys_r = ["v_r", "ss_r", "p_r", "a_r"]
                result_keys_p = ["v_p", "ss_p", "p_p", "a_p", "g_p"]

                for i, key in enumerate(result_keys_r):
                    controller.model_data[key][sa].flat = Rs[i].flat

                for i, key in enumerate(result_keys_p):
                    controller.model_data[key][sa].flat = Rp[i].flat

                visual_activation[:, t0:t].flat = (
                    controller.stm_v.get_activation(reshaped_data[0])
                    .sum(axis=-1)
                    .flat
                )

                # Do not update match during the initial empty steps
                if t <= self.params.drop_first_n_steps:
                    continue

                # calculate match value
                data = controller.model_data
                (
                    data["match_value"][:, t0:t],
                    data["match_value_per_mod"][sa],
                ) = controller.computeMatchSimple(
                    data["v_p"][sa],
                    data["ss_p"][sa],
                    data["p_p"][sa],
                    data["a_p"][sa],
                    data["g_p"][sa],
                )

                # update cumulative match only after the first policy was selected
                if (t0 >= self.params.drop_first_n_steps
                    + self.params.policy_selection_steps):
                    for i in range(t0, t):
                        # When the policy changes, max_match is set to the current value
                        max_match[policy_changed[:, i], i:] =\
                            controller.model_data["match_value"][policy_changed[:, i], i:]

                        # ####### Dataset Filtering
                        # Select time steps when match in internal representation of touch
                        # increases (touch onset event)
                        touch_mod = 1
                        mmask = (
                            controller.model_data["match_value_per_mod"][
                                :, i, touch_mod
                            ]
                            > controller.model_data["match_value_per_mod"][
                                :, i - 1, touch_mod
                            ]
                        )

                        # Select only timesteps where match increases *globally*
                        # over a certain threshold
                        mmask[
                            controller.model_data["match_value"][:, i]
                            - max_match[:, i]
                            < self.params.match_incr_th
                        ] = 0

                        max_match[mmask, i:] = controller.model_data[
                            "match_value"
                        ][mmask, i, None]

                        # ####### Competence - Option 1
                        # # use controller.model_data['match_value'] as it is

                        # ####### Competence - Option 2
                        # # use change event mask s computed in mmask
                        matches[:, i] = mmask

                        # Match is cumulated within a single policy and reset with
                        # policy change
                        cum_match[:, i] = (
                            cum_match[:, i - 1] * (1 - policy_changed[:, i])
                            + mmask
                        )

                # Counts touch global increases up to threshold
                success_mask = (
                    cum_match[:, t - 1] >= self.params.cum_match_stop_th
                )

                within_action_interval = (
                    t < self.params.stime
                    and t
                    >= self.params.drop_first_n_steps
                    + self.params.policy_selection_steps
                )

                action_interval_onset = (
                    t < self.params.stime
                    and t
                    == self.params.drop_first_n_steps
                    + self.params.policy_selection_steps
                )

                if within_action_interval:

                    # Set initial policy after warmup steps + action selection
                    # steps
                    if action_interval_onset:
                        success_mask[:] = 1

                    # Register every change of policy (including setting the
                    # initial one)
                    policy_changed[success_mask, t] = 1

                    data_slice = slice(
                        t - self.params.policy_selection_steps, t
                    )
                    v_rt = controller.model_data["v_r"][
                        success_mask, data_slice, :
                    ]
                    ss_rt = controller.model_data["ss_r"][
                        success_mask, data_slice, :
                    ]
                    p_rt = controller.model_data["p_r"][
                        success_mask, data_slice, :
                    ]

                    goal_activation[success_mask, t:] = visual_activation[
                        success_mask,
                        t - self.params.policy_selection_steps : t,
                    ].mean(axis=1)[:, None]

                    # choose policy
                    chosen_policy_results = controller.choose_policy(
                        v_rt, ss_rt, p_rt, goal_activation, t
                    )

                    (
                        goals_p,
                        goals,
                        policies,
                        competences,
                        lcompetences,
                        mean_policy_noise,
                    ) = chosen_policy_results

                    # fill successful batches with policies, goals, and
                    # competences (from the current timestep onward)
                    data = controller.model_data
                    data["batch_a"][success_mask, t:, :] = policies[:, None, :]
                    data["batch_g"][success_mask, t:, :] = goals[:, None, :]
                    data["batch_c"][success_mask, t:, :] = competences[
                        :, None, :
                    ]
                    data["batch_log"][success_mask, t:, :] = lcompetences[
                        :, None, :
                    ]

        return (
            matches,
            max_match,
            cum_match,
            episode_len,
            policy_changed,
            goal_activation,
        )

    def train(self, time_limits):

        print(f">>> {self.epoch}")

        if self.epoch == 0:
            print("Training", flush=True)
        else:
            if self.epoch >= self.params.epochs - 1:
                raise TimeLimitsException

        env = self.env
        agent = self.agent
        logs = self.logs
        epoch = self.epoch
        epoch_start = time.perf_counter()
        contexts = (np.arange(self.params.batch_size) % 3) + 1
        gen = cycle(chain.from_iterable(repeat(x, 3) for x in range(len(self.obj_params_space))))
        params_ind = np.array([next(gen) for _ in range(self.params.batch_size)])

        self.initialize_model_data(self.controller)

        cum_match = None
        envs = [None] * self.params.batch_size
        states = [None] * self.params.batch_size

        # Store internal trajectories
        internal_trajectory_data = []

        while epoch < self.params.epochs:

            self.reset_model_data(self.controller)

            total_time_elapsed = time.perf_counter() - self.start
            if total_time_elapsed >= time_limits:
                if self.epoch > 0:
                    raise TimeLimitsException

            print(f"{epoch:6d}", end=" ", flush=True)

            # ----- prepare episodes
            for episode in range(self.params.batch_size):
                # Each environment in each epoch should have a different seed
                env = SMEnv(
                    self.seed + episode + epoch,
                    self.params,
                    self.params.action_steps,
                    rand_obj_params=self.obj_params_space[params_ind[episode]],
                )
                # env.b2d_env.prepare_world(contexts[episode])
                states[episode] = env.reset(contexts[episode])
                envs[episode] = env
                state = states[episode]
                self.controller.model_data["batch_v"][episode, 0, :] = state[
                    "VISUAL_SENSORS"
                ].ravel()
                self.controller.model_data["batch_ss"][episode, 0, :] = state[
                    "TOUCH_SENSORS"
                ]
                self.controller.model_data["batch_p"][episode, 0, :] = state[
                    "JOINT_POSITIONS"
                ][:5]

            n_episodes = self.params.batch_size
            # print(pd.Series(contexts).value_counts() / n_episodes)
            # print(pd.Series(params_ind).value_counts() / n_episodes)
            # d = pd.DataFrame({"contexts": contexts, "params_ind": params_ind})
            # d["v"] = 1
            # print(d.pivot_table(values="v", index="params_ind", columns="contexts", aggfunc="sum") / n_episodes)

            (
                matches,
                max_match,
                cum_match,
                _,
                policy_changed,
                goal_activation,
            ) = self.run_episodes(
                agent,
                self.controller,
                contexts,
                envs,
                states,
            )
        
            # Episode success rate: in how many episodes policy ever changes
            # after the initial one?
            episode_success_rate = (policy_changed.sum(axis=1) >= 2).mean()
            print(f"success rate {episode_success_rate}")

            # How many timesteps involve touching the object?
            touch_freq = (self.controller.model_data["batch_ss"].sum(axis=-1) > 0).mean()
            
            # Mark end of each policy
            policy_ended = np.zeros(policy_changed.shape, dtype=bool)
            policy_ended[:, -1] = (
                1  # End of an episode automatically ends policy
            )
            policy_ended[:, :-1] = policy_changed[:, 1:]
            # Initial policy change does not count
            policy_ended[
                :,
                self.params.drop_first_n_steps
                + self.params.policy_selection_steps
                - 1,
            ] = 0

            # # Calculate within-episode match increase
            # episode_match_inc_p, episode_match_inc_ss = (
            #     self.calc_match_inc_within_goal(
            #         policy_ended,
            #         self.controller,
            #     )
            # )
            #
            # (mi_score_context_touch,
            #  mi_score_context_policy,
            #  mi_score_policy_touch,
            #  mi_score_context_gripper,
            #  mi_score_policy_gripper
            # ) = self.calc_mi_metrics(contexts, params_ind,
            #                          self.controller,
            #                          policy_ended)

            # Local competences based on predictor
            comp_dict = self.controller.get_global_local_competence(
                self.controller.model_data["batch_c"]
            )
            global_competence = comp_dict["global_competence"]
            global_incompetence = comp_dict["global_incompetence"]
            local_incompetences = comp_dict["local_incompetence"]

            bsize = self.params.batch_size * self.params.stime
            local_incompetences = local_incompetences.reshape((bsize, -1))

            def modulate_param(base, limit, prop):
                return base + (limit - base) * prop

            self.controller.match_sigma = modulate_param(
                self.params.base_match_sigma,
                self.params.match_sigma,
                global_incompetence,
            )

            # No global modulation of LR and sigma
            # self.controller.curr_lr = modulate_param(
            #    self.params.base_lr,
            #    self.params.stm_lr,
            #    global_incompetence,
            # )
            self.controller.curr_sigma = modulate_param(
                self.params.base_internal_sigma,
                self.params.internal_sigma,
                global_incompetence,
            )

            # Local sigma is a vector of length batch_size * timesteps
            local_sigma = modulate_param(
                self.params.base_internal_sigma,
                self.params.internal_sigma,
                local_incompetences,
            )

            local_lr = modulate_param(
                self.params.base_lr,
                self.params.max_lr,
                local_incompetences,
            )

            self.controller.updateParams(
                self.controller.curr_sigma, self.controller.curr_lr
            )

            print(f"sigma: {local_sigma.mean()} {self.params.internal_sigma}")

            # ---- end of an epoch: controller update
            (update_items, update_episodes, curr_loss, mean_modulation) = (
                self.controller.update(
                    self.controller.model_data["batch_v"].reshape((bsize, -1)),
                    self.controller.model_data["batch_ss"].reshape(
                        (bsize, -1)
                    ),
                    self.controller.model_data["batch_p"].reshape((bsize, -1)),
                    self.controller.model_data["batch_a"].reshape((bsize, -1)),
                    self.controller.model_data["batch_g"].reshape((bsize, -1)),
                    self.controller.model_data["match_value"].reshape(-1),
                    matches.reshape(-1),
                    max_match.reshape(-1),
                    policy_changed.reshape(-1),
                    policy_ended.reshape(-1),
                    local_lr,
                    local_sigma,
                )
            )
            
            # Store trajectory data
            mvpm = self.controller.model_data["match_value_per_mod"].reshape(
                -1, 4
            )
            internal_trajectory_data.append(
                {
                    "epoch": [epoch]
                    * self.params.batch_size
                    * self.params.stime,
                    "episode": [
                        i
                        for i in range(self.params.batch_size)
                        for _ in range(self.params.stime)
                    ],
                    "context": [
                        c for c in contexts for _ in range(self.params.stime)
                    ],
                    "params_index": [
                        i for i in params_ind for _ in range(self.params.stime)
                    ],
                    "timestep": list(range(self.params.stime))
                    * self.params.batch_size,
                    "v_p": self.controller.model_data["v_p"].reshape((-1, 2))[
                        :, 0
                    ]
                    * 10
                    + self.controller.model_data["v_p"].reshape((-1, 2))[:, 1],
                    "ss_p": self.controller.model_data["ss_p"].reshape(
                        (-1, 2)
                    )[:, 0]
                    * 10
                    + self.controller.model_data["ss_p"].reshape((-1, 2))[
                        :, 1
                    ],
                    "p_p": self.controller.model_data["p_p"].reshape((-1, 2))[
                        :, 0
                    ]
                    * 10
                    + self.controller.model_data["p_p"].reshape((-1, 2))[:, 1],
                    "a_p": self.controller.model_data["a_p"].reshape((-1, 2))[
                        :, 0
                    ]
                    * 10
                    + self.controller.model_data["a_p"].reshape((-1, 2))[:, 1],
                    "g_p": self.controller.model_data["g_p"].reshape((-1, 2))[
                        :, 0
                    ]
                    * 10
                    + self.controller.model_data["g_p"].reshape((-1, 2))[:, 1],
                    "match_value_v": mvpm[:, 0].copy(),
                    "match_value_ss": mvpm[:, 1].copy(),
                    "match_value_p": mvpm[:, 2].copy(),
                    "match_value_a": mvpm[:, 3].copy(),
                    "sensory_change": matches.reshape(-1).copy(),
                }
            )

            # ---- print
            c = np.outer(contexts, np.ones(self.params.stime)).ravel()
            items = [np.sum(update_episodes[c == k]) for k in range(1, 4)]
            items = "".join(
                list(
                    map(
                        lambda x: "{: 6d} {}".format(*x),
                        zip(items, ["f", "m", "c"]),
                    )
                )
            )

            print(f"{update_items:#7d} {items}", end=" ", flush=True)
            print(
                f"{self.controller.model_data['batch_ss'].sum():#10.2f}",
                end=" ",
                flush=True,
            )
            logs[epoch] = [
                self.controller.model_data["batch_log"][policy_ended].min(),
                self.controller.model_data["batch_log"][policy_ended].mean(),
                self.controller.model_data["batch_log"][policy_ended].max(),
            ]
            print(
                ("%8.7f " * 3)
                % (
                    self.controller.model_data["batch_log"][
                        policy_ended
                    ].min(),
                    self.controller.model_data["batch_log"][
                        policy_ended
                    ].mean(),
                    self.controller.model_data["batch_log"][
                        policy_ended
                    ].max(),
                ),
                end="",
            )

            # Do not update statistics if there are no successful timesteps
            if matches.sum() > 0: 
                print(f"  {np.mean(curr_loss):#8.7f}")
                if use_wandb:
                    wandb.log(
                        {
                            "min_comp": logs[epoch][0],
                            "mean_comp": logs[epoch][1],
                            "max_comp": logs[epoch][2],
                            "stm_loss": np.mean(curr_loss),
                            "stm_v_loss": curr_loss[0],
                            "stm_ss_loss": curr_loss[1],
                            "stm_p_loss": curr_loss[2],
                            "stm_a_loss": curr_loss[3],
                            "mean_sigma": local_sigma.mean(),
                            "mean_lr": local_lr.mean(),
                            "mean_cum_match": cum_match[policy_ended].mean()
                            / self.params.cum_match_stop_th,
                            "grid_comp_mean": global_competence,
                            "episode_success_rate": episode_success_rate,
                            "policy_weights_avg": np.abs(
                                self.controller.stm_a.get_weights()
                            ).mean(),
                            "policy_weights_norm": np.linalg.norm(
                                self.controller.stm_a.get_weights(), axis=-1
                            ).mean(),
                            "match_value_v": self.controller.model_data[
                                "match_value_per_mod"
                            ][matches, 0].mean(),
                            "match_value_ss": self.controller.model_data[
                                "match_value_per_mod"
                            ][matches, 1].mean(),
                            "match_value_p": self.controller.model_data[
                                "match_value_per_mod"
                            ][matches, 2].mean(),
                            "match_value_a": self.controller.model_data[
                                "match_value_per_mod"
                            ][matches, 3].mean(),
                            "goal_activation": goal_activation[
                                policy_ended
                            ].mean(),
                            "goal_activation_blue": goal_activation[
                                contexts == 1, :
                            ][policy_ended[contexts == 1, :]].mean(),
                            "goal_activation_red": goal_activation[
                                contexts == 2, :
                            ][policy_ended[contexts == 2, :]].mean(),
                            "goal_activation_green": goal_activation[
                                contexts == 3, :
                            ][policy_ended[contexts == 3, :]].mean(),
                            "touch_freq": touch_freq,
                        },
                        step=epoch,
                    )

            # diagnose
            if (epoch % self.params.epochs_to_test == 0) or epoch == (
                self.params.epochs - 1
            ):

                epoch_dir = f"{storage_dir}/{epoch:06d}"
                os.makedirs(epoch_dir, exist_ok=True)
                np.save(f"{epoch_dir}/main.dump", [self], allow_pickle=True)
                self.diagnose()
                self.evaluation_episodes(epoch=epoch)

                time_elapsed = time.perf_counter() - epoch_start
                print("---- TIME: %10.4f" % time_elapsed, flush=True)
                epoch_start = time.perf_counter()

            epoch += 1
            self.epoch = epoch
            sys.stdout.flush()

        df_final = pd.concat(
            [pd.DataFrame.from_dict(d) for d in internal_trajectory_data],
            axis=0,
            ignore_index=True,
        )
        df_final.to_csv("internal_trajectory_data.csv")

    def train_parasite(self, time_limits):

        if self.epoch == 0:
            print("Training", flush=True)
        else:
            if self.epoch >= self.params.epochs - 1:
                raise TimeLimitsException

        env = self.env
        agent = self.agent
        controller = self.controller
        logs = self.logs
        epoch = self.epoch
        epoch_start = time.perf_counter()
        contexts = (np.arange(self.params.batch_size) % 3) + 1
        gen = cycle(chain.from_iterable(repeat(x, 3) for x in range(len(self.obj_params_space))))
        params_ind = np.array([next(gen) for _ in range(self.params.batch_size)])

        self.controller_par = SMController(
            self.params,
            self.rng,
            load=self.params.load_weights,
            shuffle=self.params.shuffle_weights,
        )
        logs_par = np.zeros([self.params.epochs, 3])

        self.initialize_model_data(self.controller)
        self.initialize_model_data(self.controller_par)

        cum_match = None
        envs = [None] * self.params.batch_size
        states = [None] * self.params.batch_size

        cum_match_par = None
        envs_par = [None] * self.params.batch_size
        states_par = [None] * self.params.batch_size

        # Store internal trajectories
        internal_trajectory_data = []
        internal_trajectory_data_par = []

        while epoch < self.params.epochs:

            self.reset_model_data(self.controller)
            self.reset_model_data(self.controller_par)
            total_time_elapsed = time.perf_counter() - self.start
            if total_time_elapsed >= time_limits:
                if self.epoch > 0:
                    raise TimeLimitsException

            print(f"{epoch:6d}", end=" ", flush=True)

            # ----- prepare episodes
            for episode in range(self.params.batch_size):
                # Each environment in each epoch should have a different seed
                env = SMEnv(
                    self.seed + episode + epoch,
                    self.params,
                    self.params.action_steps,
                    store_observations=True,
                    rand_obj_params=self.obj_params_space[params_ind[episode]],
                )
                #env.b2d_env.prepare_world(contexts[episode])
                states[episode] = env.reset(contexts[episode])
                envs[episode] = env
                state = states[episode]
                self.controller.model_data["batch_v"][episode, 0, :] = state[
                    "VISUAL_SENSORS"
                ].ravel()
                self.controller.model_data["batch_ss"][episode, 0, :] = state[
                    "TOUCH_SENSORS"
                ]
                self.controller.model_data["batch_p"][episode, 0, :] = state[
                    "JOINT_POSITIONS"
                ][:5]

            (
                matches,
                max_match,
                cum_match,
                _,
                policy_changed,
                goal_activation,
            ) = self.run_episodes(
                agent,
                self.controller,
                contexts,
                envs,
                states,
            )

            # ----- prepare episodes
            for episode in range(self.params.batch_size):
                # Each environment in each epoch should have a different seed
                env = SMEnvParasite(
                    self.seed + episode + epoch,
                    self.params,
                    envs[episode].stored_observations,
                    rand_obj_params=self.obj_params_space[params_ind[episode]],
                )
                states_par[episode] = env.reset()
                envs_par[episode] = env
                state_par = states_par[episode]
                self.controller_par.model_data["batch_v"][episode, 0, :] = (
                    state_par["VISUAL_SENSORS"].ravel()
                )
                self.controller_par.model_data["batch_ss"][
                    episode, 0, :
                ] = state_par["TOUCH_SENSORS"]
                self.controller_par.model_data["batch_p"][
                    episode, 0, :
                ] = state_par["JOINT_POSITIONS"][:5]

            (
                matches_par,
                max_match_par,
                cum_match_par,
                _,
                policy_changed_par,
                goal_activation_par,
            ) = self.run_episodes(
                agent,
                self.controller_par,
                contexts,
                envs_par,
                states_par,
            )

            # Episode success rate: in how many episodes policy ever changes?
            episode_success_rate = (policy_changed.sum(axis=1) >= 2).mean()
            episode_success_rate_par = (
                policy_changed_par.sum(axis=1) >= 2
            ).mean()
        
            # How many timesteps involve touching the object?
            touch_freq = (self.controller.model_data["batch_ss"].sum(axis=-1) > 0).mean()
            touch_freq_par = (self.controller_par.model_data["batch_ss"].sum(axis=-1) > 0).mean()

            # Mark end of each policy
            policy_ended = np.zeros(policy_changed.shape, dtype=bool)
            policy_ended[:, -1] = 1
            policy_ended[:, :-1] = policy_changed[:, 1:]
            # Initial policy change does not count
            policy_ended[
                :,
                self.params.drop_first_n_steps
                + self.params.policy_selection_steps
                - 1,
            ] = 0

            policy_ended_par = np.zeros(policy_changed_par.shape, dtype=bool)
            policy_ended_par[:, -1] = 1
            policy_ended_par[:, :-1] = policy_changed_par[:, 1:]
            # Initial policy change does not count
            policy_ended_par[
                :,
                self.params.drop_first_n_steps
                + self.params.policy_selection_steps
                - 1,
            ] = 0

            # # Calculate within-episode match increase
            # episode_match_inc_p, episode_match_inc_ss = (
            #     self.calc_match_inc_within_goal(
            #         policy_ended,
            #         self.controller,
            #     )
            # )
            # episode_match_inc_p_par, episode_match_inc_ss_par = (
            #     self.calc_match_inc_within_goal(
            #         policy_ended_par,
            #         self.controller_par,
            #     )
            # )
            #
            # (mi_score_context_touch,
            #  mi_score_context_policy,
            #  mi_score_policy_touch,
            #  mi_score_context_gripper,
            #  mi_score_policy_gripper,
            # ) = self.calc_mi_metrics(contexts, params_ind,
            #                          self.controller,
            #                          policy_ended)
            #
            # (mi_score_context_touch_par,
            #  mi_score_context_policy_par,
            #  mi_score_policy_touch_par,
            #  mi_score_context_gripper_par,
            #  mi_score_policy_gripper_par,
            # ) = self.calc_mi_metrics(contexts, params_ind,
            #                          self.controller_par,
            #                          policy_ended_par)

            # Local competences based on predictor
            comp_dict = self.controller.get_global_local_competence(
                self.controller.model_data["batch_c"]
            )
            global_competence = comp_dict["global_competence"]
            global_incompetence = comp_dict["global_incompetence"]
            local_incompetences = comp_dict["local_incompetence"]

            # Local competences based on predictor
            comp_dict = self.controller_par.get_global_local_competence(
                self.controller_par.model_data["batch_c"]
            )
            global_competence_par = comp_dict["global_competence"]
            global_incompetence_par = comp_dict["global_incompetence"]
            local_incompetences_par = comp_dict["local_incompetence"]

            bsize = self.params.batch_size * self.params.stime
            local_incompetences = local_incompetences.reshape((bsize, -1))
            local_incompetences_par = local_incompetences_par.reshape(
                (bsize, -1)
            )

            def modulate_param(base, limit, prop):
                return base + (limit - base) * prop

            self.controller.match_sigma = modulate_param(
                self.params.base_match_sigma,
                self.params.match_sigma,
                global_incompetence,
            )

            self.controller_par.match_sigma = modulate_param(
                self.params.base_match_sigma,
                self.params.match_sigma,
                global_incompetence_par,
            )

            # No global modulation of LR and sigma
            # controller.curr_lr = modulate_param(
            #    self.params.base_lr,
            #    self.params.stm_lr,
            #    global_incompetence,
            # )
            self.controller.curr_sigma = modulate_param(
                self.params.base_internal_sigma,
                self.params.internal_sigma,
                global_incompetence,
            )
            self.controller_par.curr_sigma = modulate_param(
                self.params.base_internal_sigma,
                self.params.internal_sigma,
                global_incompetence_par,
            )

            # Local sigma is a vector of length batch_size * timesteps
            local_sigma = modulate_param(
                self.params.base_internal_sigma,
                self.params.internal_sigma,
                local_incompetences,
            )
            local_lr = modulate_param(
                self.params.base_lr,
                self.params.max_lr,
                local_incompetences,
            )
            self.controller.updateParams(
                self.controller.curr_sigma, self.controller.curr_lr
            )

            local_sigma_par = modulate_param(
                self.params.base_internal_sigma,
                self.params.internal_sigma,
                local_incompetences_par,
            )
            local_lr_par = modulate_param(
                self.params.base_lr,
                self.params.max_lr,
                local_incompetences_par,
            )
            self.controller_par.updateParams(
                self.controller_par.curr_sigma, self.controller_par.curr_lr
            )

            print(f"sigma: {local_sigma.mean()}")

            # ---- end of an epoch: controller update
            (update_items, update_episodes, curr_loss, mean_modulation) = (
                controller.update(
                    self.controller.model_data["batch_v"].reshape((bsize, -1)),
                    self.controller.model_data["batch_ss"].reshape(
                        (bsize, -1)
                    ),
                    self.controller.model_data["batch_p"].reshape((bsize, -1)),
                    self.controller.model_data["batch_a"].reshape((bsize, -1)),
                    self.controller.model_data["batch_g"].reshape((bsize, -1)),
                    self.controller.model_data["match_value"].reshape(-1),
                    matches.reshape(-1),
                    max_match.reshape(-1),
                    policy_changed.reshape(-1),
                    policy_ended.reshape(-1),
                    local_lr,
                    local_sigma,
                )
            )

            (
                update_items_par,
                update_episodes_par,
                curr_loss_par,
                mean_modulation_par,
            ) = self.controller_par.update(
                self.controller_par.model_data["batch_v"].reshape((bsize, -1)),
                self.controller_par.model_data["batch_ss"].reshape(
                    (bsize, -1)
                ),
                self.controller_par.model_data["batch_p"].reshape(
                    (bsize, -1)
                ),
                self.controller_par.model_data["batch_a"].reshape(
                    (bsize, -1)
                ),
                self.controller_par.model_data["batch_g"].reshape(
                    (bsize, -1)
                ),
                self.controller_par.model_data["match_value"].reshape(-1),
                matches_par.reshape(-1),
                max_match_par.reshape(-1),
                policy_changed_par.reshape(-1),
                policy_ended_par.reshape(-1),
                local_lr_par,
                local_sigma_par,
            )

            # Store trajectory data
            mvpm = self.controller.model_data["match_value_per_mod"].reshape(
                -1, 4
            )
            internal_trajectory_data.append(
                {
                    "epoch": [epoch]
                    * self.params.batch_size
                    * self.params.stime,
                    "episode": [
                        i
                        for i in range(self.params.batch_size)
                        for _ in range(self.params.stime)
                    ],
                    "context": [
                        c for c in contexts for _ in range(self.params.stime)
                    ],
                    "timestep": list(range(self.params.stime))
                    * self.params.batch_size,
                    "v_p": self.controller.model_data["v_p"].reshape((-1, 2))[
                        :, 0
                    ]
                    * 10
                    + self.controller.model_data["v_p"].reshape((-1, 2))[:, 1],
                    "ss_p": self.controller.model_data["ss_p"].reshape(
                        (-1, 2)
                    )[:, 0]
                    * 10
                    + self.controller.model_data["ss_p"].reshape((-1, 2))[
                        :, 1
                    ],
                    "p_p": self.controller.model_data["p_p"].reshape((-1, 2))[
                        :, 0
                    ]
                    * 10
                    + self.controller.model_data["p_p"].reshape((-1, 2))[:, 1],
                    "a_p": self.controller.model_data["a_p"].reshape((-1, 2))[
                        :, 0
                    ]
                    * 10
                    + self.controller.model_data["a_p"].reshape((-1, 2))[:, 1],
                    "g_p": self.controller.model_data["g_p"].reshape((-1, 2))[
                        :, 0
                    ]
                    * 10
                    + self.controller.model_data["g_p"].reshape((-1, 2))[:, 1],
                    "match_value_v": mvpm[:, 0].copy(),
                    "match_value_ss": mvpm[:, 1].copy(),
                    "match_value_p": mvpm[:, 2].copy(),
                    "match_value_a": mvpm[:, 3].copy(),
                    "sensory_change": matches.reshape(-1).copy(),
                }
            )

            # Store trajectory data
            mvpm_par = self.controller_par.model_data[
                "match_value_per_mod"
            ].reshape(-1, 4)
            internal_trajectory_data_par.append(
                {
                    "epoch": [epoch]
                    * self.params.batch_size
                    * self.params.stime,
                    "episode": [
                        i
                        for i in range(self.params.batch_size)
                        for _ in range(self.params.stime)
                    ],
                    "context": [
                        c for c in contexts for _ in range(self.params.stime)
                    ],
                    "timestep": list(range(self.params.stime))
                    * self.params.batch_size,
                    "v_p": self.controller_par.model_data["v_p"].reshape(
                        (-1, 2)
                    )[:, 0]
                    * 10
                    + self.controller_par.model_data["v_p"].reshape((-1, 2))[:, 1],
                    "ss_p": self.controller_par.model_data["ss_p"].reshape(
                        (-1, 2)
                    )[:, 0]
                    * 10
                    + self.controller_par.model_data["ss_p"].reshape((-1, 2))[
                        :, 1
                    ],
                    "p_p": self.controller_par.model_data["p_p"].reshape(
                        (-1, 2)
                    )[:, 0]
                    * 10
                    + self.controller_par.model_data["p_p"].reshape((-1, 2))[:, 1],
                    "a_p": self.controller_par.model_data["a_p"].reshape(
                        (-1, 2)
                    )[:, 0]
                    * 10
                    + self.controller_par.model_data["a_p"].reshape((-1, 2))[:, 1],
                    "g_p": self.controller_par.model_data["g_p"].reshape(
                        (-1, 2)
                    )[:, 0]
                    * 10
                    + self.controller_par.model_data["g_p"].reshape((-1, 2))[:, 1],
                    "match_value_v": mvpm_par[:, 0].copy(),
                    "match_value_ss": mvpm_par[:, 1].copy(),
                    "match_value_p": mvpm_par[:, 2].copy(),
                    "match_value_a": mvpm_par[:, 3].copy(),
                    "sensory_change": matches_par.reshape(-1).copy(),
                }
            )

            # ---- print
            c = np.outer(contexts, np.ones(self.params.stime)).ravel()
            items = [np.sum(update_episodes[c == k]) for k in range(1, 4)]
            items = "".join(
                list(
                    map(
                        lambda x: "{: 6d} {}".format(*x),
                        zip(items, ["f", "m", "c"]),
                    )
                )
            )

            print(f"{update_items:#7d} {items}", end=" ", flush=True)
            print(
                f"{self.controller.model_data['batch_ss'].sum():#10.2f}",
                end=" ",
                flush=True,
            )
            logs[epoch] = [
                self.controller.model_data["batch_log"][policy_ended].min(),
                self.controller.model_data["batch_log"][policy_ended].mean(),
                self.controller.model_data["batch_log"][policy_ended].max(),
            ]
            logs_par[epoch] = [
                self.controller_par.model_data["batch_log"][
                    policy_ended_par
                ].min(),
                self.controller_par.model_data["batch_log"][
                    policy_ended_par
                ].mean(),
                self.controller_par.model_data["batch_log"][
                    policy_ended_par
                ].max(),
            ]

            print(
                ("%8.7f " * 3)
                % (
                    self.controller.model_data["batch_log"][
                        policy_ended
                    ].min(),
                    self.controller.model_data["batch_log"][
                        policy_ended
                    ].mean(),
                    self.controller.model_data["batch_log"][
                        policy_ended
                    ].max(),
                ),
                end="",
            )

            # Do not update statistics if there are no successful timesteps
            if matches.sum() > 0: 
                print(f"  {np.mean(curr_loss):#8.7f}")
                if use_wandb:
                    wandb.log(
                        {
                            "min_comp": logs[epoch][0],
                            "mean_comp": logs[epoch][1],
                            "max_comp": logs[epoch][2],
                            "stm_loss": np.mean(curr_loss),
                            "stm_v_loss": curr_loss[0],
                            "stm_ss_loss": curr_loss[1],
                            "stm_p_loss": curr_loss[2],
                            "stm_a_loss": curr_loss[3],
                            "mean_sigma": local_sigma.mean(),
                            "mean_lr": local_lr.mean(),
                            "mean_cum_match": cum_match[policy_ended].mean()
                            / self.params.cum_match_stop_th,
                            "grid_comp_mean": global_competence,
                            "episode_success_rate": episode_success_rate,
                            "policy_weights_avg": np.abs(
                                controller.stm_a.get_weights()
                            ).mean(),
                            "policy_weights_norm": np.linalg.norm(
                                controller.stm_a.get_weights(), axis=-1
                            ).mean(),
                            "match_value_v": self.controller.model_data[
                                "match_value_per_mod"
                            ][matches, 0].mean(),
                            "match_value_ss": self.controller.model_data[
                                "match_value_per_mod"
                            ][matches, 1].mean(),
                            "match_value_p": self.controller.model_data[
                                "match_value_per_mod"
                            ][matches, 2].mean(),
                            "match_value_a": self.controller.model_data[
                                "match_value_per_mod"
                            ][matches, 3].mean(),
                            "touch_freq": touch_freq,
                        },
                        step=epoch
                    )

            # Do not update statistics if there are no successful timesteps
            if matches_par.sum() > 0: 
                if use_wandb:
                    wandb.log(
                        {                       
                            "min_comp_par": logs_par[epoch][0],
                            "mean_comp_par": logs_par[epoch][1],
                            "max_comp_par": logs_par[epoch][2],
                            "stm_loss_par": np.mean(curr_loss_par),
                            "stm_v_loss_par": curr_loss_par[0],
                            "stm_ss_loss_par": curr_loss_par[1],
                            "stm_p_loss_par": curr_loss_par[2],
                            "stm_a_loss_par": curr_loss_par[3],
                            "mean_sigma_par": local_sigma_par.mean(),
                            "mean_lr_par": local_lr_par.mean(),
                            "mean_cum_match_par": cum_match_par[
                                policy_ended_par
                            ].mean()
                            / self.params.cum_match_stop_th,
                            "grid_comp_mean_par": global_competence_par,
                            "episode_success_rate_par": episode_success_rate_par,
                            "policy_weights_norm_par": np.linalg.norm(
                                self.controller_par.stm_a.get_weights(), axis=-1
                            ).mean(),
                            "match_value_v_par": self.controller_par.model_data[
                                "match_value_per_mod"
                            ][matches_par, 0].mean(),
                            "match_value_ss_par": self.controller_par.model_data[
                                "match_value_per_mod"
                            ][matches_par, 1].mean(),
                            "match_value_p_par": self.controller_par.model_data[
                                "match_value_per_mod"
                            ][matches_par, 2].mean(),
                            "match_value_a_par": self.controller_par.model_data[
                                "match_value_per_mod"
                            ][matches_par, 3].mean(),
                            "goal_activation": goal_activation[
                                policy_ended
                            ].mean(),
                            "goal_activation_blue": goal_activation[
                                contexts == 1, :
                            ][policy_ended[contexts == 1, :]].mean(),
                            "goal_activation_red": goal_activation[
                                contexts == 2, :
                            ][policy_ended[contexts == 2, :]].mean(),
                            "goal_activation_green": goal_activation[
                                contexts == 3, :
                            ][policy_ended[contexts == 3, :]].mean(),
                            "goal_activation_par": goal_activation_par[
                                policy_ended_par
                            ].mean(),
                            "goal_activation_blue_par": goal_activation_par[
                                contexts == 1, :
                            ][policy_ended_par[contexts == 1, :]].mean(),
                            "goal_activation_red_par": goal_activation_par[
                                contexts == 2, :
                            ][policy_ended_par[contexts == 2, :]].mean(),
                            "goal_activation_green_par": goal_activation_par[
                                contexts == 3, :
                            ][policy_ended_par[contexts == 3, :]].mean(),
                            "touch_freq_par": touch_freq_par,
                        },
                        step=epoch,
                    )

            # diagnose
            if (epoch % self.params.epochs_to_test == 0) or epoch == (
                self.params.epochs - 1
            ):

                epoch_dir = f"{storage_dir}/{epoch:06d}"
                os.makedirs(epoch_dir, exist_ok=True)
                np.save(f"{epoch_dir}/main.dump", [self], allow_pickle=True)
                self.diagnose()
                self.evaluation_episodes(epoch=epoch)

                time_elapsed = time.perf_counter() - epoch_start
                print("---- TIME: %10.4f" % time_elapsed, flush=True)
                epoch_start = time.perf_counter()

                self.evaluation_episodes(
                    orig_controller=self.controller_par, epoch=epoch, suffix="_par"
                )

                self.controller_par.save(epoch, tag="parasite")
                visual_map(wfile=f"{site_dir}/visual_weights-parasite.npy")
                comp_map(wfile=f"{site_dir}/comp_grid-parasite.npy")

                if self.plots and os.path.isfile("PLOT_SIMS"):
                    print("----> Test Sims ...", end=" ", flush=True)
                    self.evaluation_episodes(
                        epoch=epoch,
                        n_episodes=self.params.tests,
                        suffix="_demo_par",
                        render="offline",
                        orig_controller=self.controller_par,
                        save_stats=False,
                    )

                if use_wandb:
                    log_data = {
                        "visual_map_par": wandb.Image("www/visual_map.png"),
                        "comp_map_par": wandb.Image("www/comp_map.png"),
                    }
                    wandb.log(log_data, step=epoch)

            epoch += 1
            self.epoch = epoch
            sys.stdout.flush()

        df_final = pd.concat(
            [pd.DataFrame.from_dict(d) for d in internal_trajectory_data],
            axis=0,
            ignore_index=True,
        )
        df_final.to_csv("internal_trajectory_data.csv")

        df_final_par = pd.concat(
            [pd.DataFrame.from_dict(d) for d in internal_trajectory_data_par],
            axis=0,
            ignore_index=True,
        )
        df_final_par.to_csv("internal_trajectory_data_parasite.csv")

    def diagnose(self):

        np.save("main.dump", [self], allow_pickle=True)

        controller = self.controller
        logs = self.logs
        epoch = self.epoch

        data = {}
        data["match_value"] = self.controller.model_data["match_value"]
        data["match_value_per_mod"] = self.controller.model_data[
            "match_value_per_mod"
        ]
        data["v_r"] = self.controller.model_data["v_r"]
        data["ss_r"] = self.controller.model_data["ss_r"]
        data["p_r"] = self.controller.model_data["p_r"]
        data["a_r"] = self.controller.model_data["a_r"]
        data["v"] = self.controller.model_data["batch_v"]
        data["ss"] = self.controller.model_data["batch_ss"]
        data["p"] = self.controller.model_data["batch_p"]
        data["a"] = self.controller.model_data["batch_a"]

        epoch_dir = f"{storage_dir}/{epoch:06d}"
        os.makedirs(epoch_dir, exist_ok=True)
        os.makedirs(site_dir, exist_ok=True)

        controller.save(epoch)
        np.save(f"{epoch_dir}/data", [data])
        np.save(f"{site_dir}/log", logs[: epoch + 1])
        np.save(f"{epoch_dir}/log", logs[: epoch + 1])

        if self.plots is False:
            return

        print("----> Graphs  ...", flush=True)
        remove_figs(epoch)
        visual_map()
        log()
        comp_map()

        if os.path.isfile("PLOT_SIMS"):
            print("----> Test Sims ...", end=" ", flush=True)
            self.evaluation_episodes(
                epoch=epoch,
                n_episodes=self.params.tests,
                render="offline",
                suffix="_demo",
                save_stats=False,
            )

        if use_wandb:
            log_data = {
                "visual_map": wandb.Image("www/visual_map.png"),
                "comp_map": wandb.Image("www/comp_map.png"),
            }
            wandb.log(log_data, step=epoch)

    def collect_sensory_states(self):
        pass

    def demo_episode(self, idx):
        pass

    def evaluation_episodes(
        self,
        orig_controller=None,
        epoch=0,
        suffix="",
        render=None,
        render_all_maps=False,
        env_states=None,
        save_stats=True,
        add_goal_suffix=False,
        n_episodes=None,
        zero_noise=False,
    ):

        print(f"------> {suffix}")

        n_episodes = n_episodes or self.params.evaluation_episodes
        agent = self.agent
        controller = SMController(self.params)
        if orig_controller is None:
            controller.__setstate__(self.controller.__getstate__())
        else:
            controller.__setstate__(orig_controller.__getstate__())

        if env_states is not None:
            n_episodes = len(env_states)

        gen = cycle(chain.from_iterable(repeat(x, 3) for x in range(len(self.obj_params_space))))
        params_ind = np.array([next(gen) for _ in range(n_episodes)])
       
        self.initialize_model_data(controller, n_episodes)
        if env_states is not None:
            contexts = [s["context"] for s in env_states]
        else:
            contexts = (np.arange(n_episodes) % 3) + 1

        envs = [None] * n_episodes
        states = [None] * n_episodes

        # ----- prepare episodes
        for episode in range(n_episodes):
            # Each environment in each epoch should have a different seed
            if env_states is not None:
                seed = env_states[episode]["seed"]
            else:
                seed = self.seed + episode
            env = SMEnv(
                seed,
                self.params,
                self.params.action_steps,
                rand_obj_params=self.obj_params_space[params_ind[episode]],
            )
            # env.b2d_env.renderer_figsize=(8, 8)
            # env.b2d_env.prepare_world(contexts[episode])
            states[episode] = env.reset(
                contexts[episode],
                plot=f"{site_dir}/episode_{episode}{suffix}",
                render=render,
            )
            if env_states is not None:
                env.set_b2d_state(env_states[episode])
            envs[episode] = env
            state = states[episode]
            controller.model_data["batch_v"][episode, 0, :] = state[
                "VISUAL_SENSORS"
            ].ravel()
            controller.model_data["batch_ss"][episode, 0, :] = state[
                "TOUCH_SENSORS"
            ]
            controller.model_data["batch_p"][episode, 0, :] = state[
                "JOINT_POSITIONS"
            ][:5]


        # print(pd.Series(contexts).value_counts() / n_episodes)
        # print(pd.Series(params_ind).value_counts() / n_episodes)
        # d = pd.DataFrame({"contexts": contexts, "params_ind": params_ind})
        # d["v"] = 1
        # print(d.pivot_table(values="v", index="params_ind", columns="contexts", aggfunc="sum") / n_episodes)
        # exit(1)

        # Notice: For some reason without noise eval results are much worse 
        # than with normal noise.
        if zero_noise:
            # Do not introduce noise to policy search
            controller.base_policy_noise = 0.0
            controller.max_policy_noise = 0.0

        print("Evaluation episodes")

        bsize = self.params.batch_size
        collected_res = defaultdict(list)
        controller_ = SMController(self.params)
        self.initialize_model_data(controller_, n_episodes=n_episodes)

        for i in range(0, n_episodes, bsize):
            self.initialize_model_data(controller, n_episodes=bsize)
            res = self.run_episodes(
                agent,
                controller,
                contexts[i:(i+bsize)],
                envs[i:(i+bsize)],
                states[i:(i+bsize)],
            )
            for j, r in enumerate(res):
                collected_res[j].append(r)
            for key in ["batch_v", "batch_ss", "batch_p", "batch_a",
                        "batch_c", "batch_log", "batch_g",
                        "v_r", "ss_r", "p_r", "a_r",
                        "v_p", "ss_p", "p_p", "a_p", "g_p",
                        "match_value", "match_value_per_mod"
                        ]:
                controller_.model_data[key][i:(i+bsize)] = controller.model_data[key]

        controller = controller_

        matches = np.concat(collected_res[0])
        max_match = np.concat(collected_res[1])
        cum_match = np.concat(collected_res[2])
        episodes_len = np.concat(collected_res[3])
        policy_changed = np.concat(collected_res[4])
        goal_activation = np.concat(collected_res[5])

        goal_counts = defaultdict(int)
        for goal in controller.model_data["g_p"][policy_changed]:
            goal_counts[(int(goal[0]), int(goal[1]))] += 1

        all_trajectories = []
        for i in range(n_episodes):
            # only trajectory of the i-th episode from batch is
            # collected
            trajectories = pd.DataFrame(
                controller.model_data["batch_p"][i, :, -2:]
            )
            trajectories.columns = ["d1", "d2"]
            trajectories["prototype_x"] = controller.model_data["g_p"][i, :, 0]
            trajectories["prototype_y"] = controller.model_data["g_p"][i, :, 1]
            trajectories["goal_id"] = np.cumsum(policy_changed[i])
            trajectories["tr_id"] = trajectories.goal_id + i * 100
            trajectories["episode_id"] = i
            if env_states is not None:
                trajectories["state"] = [
                    tuple(
                        np.hstack(
                            [
                                env_states[i]["verts"].reshape(-1).round(5),
                                env_states[i]["pos"].reshape(-1).round(5),
                                env_states[i]["color"].reshape(-1).round(5),
                                [env_states[i]["context"]],
                                [np.round(env_states[i]["rot"], 5)],
                            ]
                        )
                    )
                    for x in trajectories.index
                ]

            trajectories["ts"] = (
                np.hstack(
                    list(
                        map(
                            np.cumsum,
                            np.split(
                                np.ones(self.params.stime),
                                np.argwhere(policy_changed[i])[:, 0],
                            ),
                        )
                    )
                )
                - 1
            )
            trajectories = trajectories.iloc[
                self.params.drop_first_n_steps
                + self.params.policy_selection_steps : -1
            ]

            # TMP: remove empty records (when episode ends prematurely). Should be
            # solved better in the future
            trajectories.drop(
                trajectories[trajectories["d1"] == 0].index, inplace=True
            )

            all_trajectories.append(trajectories)

        trajectories = pd.concat(all_trajectories)

        # Episode success rate: in how many episodes policy ever changes?
        episode_success_rate = (policy_changed.sum(axis=1) >= 2).mean()
        print(f"eval success rate {episode_success_rate}")

        # How many timesteps involve touching the object?
        touch_freq = (controller.model_data["batch_ss"].sum(axis=-1) > 0).mean()

        # Mark end of each policy
        policy_ended = np.zeros(policy_changed.shape, dtype=np.bool)
        policy_ended[:, -1] = 1
        policy_ended[:, :-1] = policy_changed[:, 1:]
        # Initial policy change does not count
        policy_ended[
            :,
            self.params.drop_first_n_steps
            + self.params.policy_selection_steps
            - 1,
        ] = 0

        episode_match_inc_p, episode_match_inc_ss = (
            self.calc_match_inc_within_goal(policy_ended, controller)
        )

        mi_metrics = self.calc_mi_metrics(contexts, params_ind,
                                          controller,
                                          policy_ended)

        if render is not None:
            for i in range(n_episodes):
                episode_len = episodes_len[i]
                full_match_value = controller.model_data["match_value"][
                    i, :episode_len
                ]
                full_cum_match = (
                    cum_match[i, :episode_len] / self.params.cum_match_stop_th
                )
                full_max_match = max_match[i, :episode_len]
                f_vp = controller.model_data["v_p"][i, :episode_len]
                f_ssp = controller.model_data["ss_p"][i, :episode_len]
                f_pp = controller.model_data["p_p"][i, :episode_len]
                f_ap = controller.model_data["a_p"][i, :episode_len]
                f_gp = controller.model_data["g_p"][i, :episode_len]

                if render_all_maps:
                    envs[i].render_info_three_maps(
                        full_match_value,
                        full_max_match,
                        full_cum_match,
                        f_vp,
                        f_ssp,
                        f_pp,
                        f_ap,
                        f_gp,
                    )
                else:
                    envs[i].render_info(
                        full_match_value,
                        full_max_match,
                        full_cum_match,
                        f_vp,
                        f_ssp,
                        f_pp,
                        f_ap,
                        f_gp,
                    )
                envs[i].close()

                if add_goal_suffix:
                    first_g_p = controller.model_data["g_p"][
                        i,
                        self.params.drop_first_n_steps
                        + self.params.policy_selection_steps,
                    ]
                    shutil.copyfile(
                        f"{site_dir}/episode_{i}{suffix}.gif",
                        f"{site_dir}/episode_{i}{suffix}_"
                        f"{int(first_g_p[0])}_"
                        f"{int(first_g_p[1])}.gif",
                    )
                #TEST
                break

        if use_wandb:
            log_data = {}
            if save_stats:
                log_data = {
                    f"eval_mean_comp{suffix}": controller.model_data[
                        "batch_log"
                    ][policy_ended].mean(),
                    f"eval_mean_cum_match{suffix}": cum_match[
                        policy_ended
                    ].mean()
                    / self.params.cum_match_stop_th,
                    f"eval_episode_success_rate{suffix}": episode_success_rate,
                    f"eval_episode_match_inc_ss{suffix}": episode_match_inc_ss,
                    f"eval_episode_match_inc_p{suffix}": episode_match_inc_p,
                    f"eval_episode_match_inc{suffix}": (
                        episode_match_inc_ss + episode_match_inc_p
                    )
                    / 2,
                    f"eval_touch_freq{suffix}": touch_freq,
                }
                for key, val in mi_metrics.items():
                    log_data[f"{key}{suffix}"] = val
            for f in glob.glob(f"{site_dir}/episode_*{suffix}*.gif"):
                log_data[Path(f).stem] = wandb.Image(f)
            wandb.log(log_data, step=epoch)

        return goal_counts, trajectories

    def monte_carlo_episode_search(self, controller=None):
        n_trials = self.params.demo_episodes_max_trials
        agent = self.agent

        if controller is None:
            controller = SMController(self.params)
            controller.__setstate__(self.controller.__getstate__())
        else:
            copied = SMController(self.params)
            copied.__setstate__(controller.__getstate__())
            controller = copied

        controller.curr_sigma = 0.1

        self.initialize_model_data(controller, 1)

        goals_env_states = defaultdict(list)

        def choose_unique_policy(self, v_rt, ss_rt, p_rt, goal_activation, t):
            ret_val = self.choose_policy_(
                v_rt, ss_rt, p_rt, goal_activation, t
            )
            if ret_val[0].shape[0] < 1:
                return ret_val

            goal_p = (int(ret_val[0][0, 0]), int(ret_val[0][0, 1]))
            # Count frequency of individual goals
            goals_env_states[goal_p].append(init_b2d_state)
            raise RepeatedGoalPrototypeException(f"Goal prototype {goal_p}")

        controller.choose_policy_ = controller.choose_policy
        controller.choose_policy = types.MethodType(
            choose_unique_policy, controller
        )

        for i in range(n_trials):
            print(f"montecarlo_trial ---- {i: 4d}/{n_trials} ----")
            context = (i % 3) + 1
            seed = self.seed + i
            env = SMEnv(
                seed,
                self.params,
                self.params.action_steps,
                rand_obj_params=self.random_obj_params,
            )
            env.b2d_env.renderer_fig_size = (8, 8)
            env.b2d_env.prepare_world(context)
            state = env.reset(context)
            init_b2d_state = env.get_b2d_state()
            init_b2d_state["context"] = context
            init_b2d_state["seed"] = seed

            envs = [env]
            states = [state]
            contexts = [context]

            controller.model_data["batch_v"][0, 0, :] = state[
                "VISUAL_SENSORS"
            ].ravel()
            controller.model_data["batch_ss"][0, 0, :] = state["TOUCH_SENSORS"]
            controller.model_data["batch_p"][0, 0, :] = state[
                "JOINT_POSITIONS"
            ][:5]

            # Use minimal sigma for building internal representations
            controller.updateParams(
                self.params.base_internal_sigma, controller.curr_lr
            )

            # get Representations for initial states
            Rs, Rp = controller.spread(
                [
                    controller.model_data["batch_v"][:, 0, :],
                    controller.model_data["batch_ss"][:, 0, :],
                    controller.model_data["batch_p"][:, 0, :],
                    controller.model_data["batch_a"][:, 0, :],
                    controller.model_data["batch_g"][:, 0, :],
                ]
            )
            (
                controller.model_data["v_r"][:, 0, :],
                controller.model_data["ss_r"][:, 0, :],
                controller.model_data["p_r"][:, 0, :],
                controller.model_data["a_r"][:, 0, :],
                _,
            ) = Rs
            (
                controller.model_data["v_p"][:, 0, :],
                controller.model_data["ss_p"][:, 0, :],
                controller.model_data["p_p"][:, 0, :],
                controller.model_data["a_p"][:, 0, :],
                controller.model_data["g_p"][:, 0, :],
            ) = Rp

            # Do not introduce noise to policy search
            controller.base_policy_noise = 0.0
            controller.max_policy_noise = 0.0

            try:
                (
                    matches,
                    max_match,
                    cum_match,
                    episodes_len,
                    visual_goal_changed,
                    goal_activation,
                ) = self.run_episodes(
                    agent,
                    controller,
                    contexts,
                    envs,
                    states,
                )
            except RepeatedGoalPrototypeException as e:
                print(e)
                continue

            # Reset policy noise
            controller.base_policy_noise = self.params.base_policy_noise
            controller.max_policy_noise = self.params.max_policy_noise

        controller.choose_policy = controller.choose_policy_

        return goals_env_states

    def demo_episodes(self, epoch=0, render=None, render_all_maps=False):
        update_weight_data()
        visual_map()
        somatosensory_map()
        proprio_map()

        goal_counts, trajectories = self.evaluation_episodes(
            epoch=epoch,
            suffix="_goal",
            save_stats=False,
            render=render,
            render_all_maps=render_all_maps,
            n_episodes=self.params.tests,
        )

        action_onset = self.params.drop_first_n_steps + self.params.policy_selection_steps

        first_goal_counts = defaultdict(int)
        for _, row in trajectories[trajectories.index == action_onset].iterrows():
            first_goal_counts[(int(row["prototype_x"]), int(row["prototype_y"]))] += 1

        goal_frequency_map(first_goal_counts)
        shutil.copyfile(
            f"{site_dir}/goal_frequency_map.png",
            f"{site_dir}/first_goal_frequency_map.png",
        )

        goal_frequency_map(goal_counts)
        shutil.copyfile(
            f"{site_dir}/goal_frequency_map.png",
            f"{site_dir}/all_goal_frequency_map.png",
        )

        print("Plot grid graph of trajectories")
        tp = TPlotManager(
            plot_path=f"{site_dir}/trajectory_plots.png",
            n_prototypes=self.params.internal_size,
            max_ts=self.params.stime,
        )
        tp.plot_prototypes(trajectories)

        if use_wandb:
            log_data = {
                "first_goal_frequency_map": wandb.Image(
                    f"{site_dir}/first_goal_frequency_map.png"
                ),
                "all_goal_frequency_map": wandb.Image(
                    f"{site_dir}/all_goal_frequency_map.png"
                ),
                "trajectory_plots": wandb.Image(
                    f"{site_dir}/trajectory_plots.png"
                ),
            }
            wandb.log(log_data, step=epoch)

    def demo_episodes_monte_carlo(self, epoch=0, render=None):
        update_weight_data()
        visual_map()
        somatosensory_map()
        proprio_map()

        goals_env_states = main.monte_carlo_episode_search()
        goal_counts = {k: len(v) for k, v in goals_env_states.items()}
        goal_frequency_map(goal_counts)
        shutil.copyfile(
            f"{site_dir}/goal_frequency_map.png",
            f"{site_dir}/first_goal_frequency_map.png",
        )

        goal_counts = defaultdict(int)
        trajectories = []
        for i, (goal, states) in enumerate(goals_env_states.items()):
            if len(states) > 1:
                pass

            # evaluate evironments
            print(f"Evaluate {len(states)} environments for goal {goal}")
            gc, tr = self.evaluation_episodes(
                epoch=epoch,
                env_states=states,
                suffix="_goal",
                save_stats=False,
            )

            tr.loc[:, "episode_id"] = tr.episode_id + i * 100
            tr.loc[:, "tr_id"] = tr.tr_id + i * 10000
            trajectories.append(tr)

            # count goals
            for k1 in gc:
                goal_counts[k1] += gc[k1]

        trajectories = pd.concat(trajectories)

        # Find best state for goal
        groups = ["prototype_x", "prototype_y", "state"]
        state_freqs = (
            trajectories.query("ts == 0")
            .groupby(groups)
            .size()
            .reset_index(name="count")
        )

        groups = ["prototype_x", "prototype_y"]
        state_bests = (
            state_freqs.groupby(groups)
            .apply(
                lambda x: x.loc[x["count"].idxmax(), "state"],
                include_groups=False,
            )
            .reset_index(name="state")
        )
        trajectories = trajectories.reset_index()
        trajectories.loc[:, "idx"] = trajectories.index
        trajectories.loc[:, "best"] = False
        trajectories.loc[trajectories.merge(state_bests).idx, "best"] = True

        trajectories = trajectories.reset_index()
        print("Save trajectories dataset")
        trajectories.to_csv(f"{site_dir}/trajectories.csv")

        print("Select a dataset of  best trajectories for each prototype ")
        prototype_trajectories = trajectories.query("best==True")

        # Selects the first trajectory among the bests for each prototype
        prototype_trajectories["best_tr"] = False
        groups = ["prototype_x", "prototype_y"]
        for idx, data in prototype_trajectories.groupby(groups):
            groups = ["tr_id"]
            for counter, (idx1, data1) in enumerate(data.groupby(groups)):
                if counter == 0:
                    tr_idx = prototype_trajectories.tr_id == (
                        data1.tr_id.iloc[0]
                    )
                    prototype_trajectories.loc[tr_idx, "best_tr"] = True

        prototype_trajectories = prototype_trajectories.query(
            "best_tr == True"
        )

        prototype_trajectories.to_csv(f"{site_dir}/prototype_trajectories.csv")

        print("Render the simulation for each prototype")
        for prototype_idx, row in prototype_trajectories.groupby(
            ["prototype_x", "prototype_y"]
        ):

            head = row.iloc[0, :]

            state = np.array(head.state)

            state = {
                "verts": state[:-7].reshape(2, -1),
                "pos": state[-7:-5],
                "color": state[-5:-2],
                "context": int(state[-2]),
                "rot": state[-1],
                "seed": 0,
            }

            self.evaluation_episodes(
                epoch=0,
                env_states=[state],
                render=render,
                suffix=(
                    f"_{head.prototype_x:03.0f}_"
                    f"{head.prototype_y:03.0f}_demo"
                ),
                save_stats=False,
                add_goal_suffix=False,
                n_episodes=1,
            )

        goal_frequency_map(goal_counts)
        shutil.copyfile(
            f"{site_dir}/goal_frequency_map.png",
            f"{site_dir}/all_goal_frequency_map.png",
        )

        print("Plot grid graph of trajectories")
        tp = TPlotManager(
            plot_path=f"{site_dir}/trajectory_plots.png",
            n_prototypes=self.params.internal_size,
            max_ts=self.params.stime,
        )
        tp.plot_prototypes(prototype_trajectories)

        if use_wandb:
            log_data = {
                "first_goal_frequency_map": wandb.Image(
                    f"{site_dir}/first_goal_frequency_map.png"
                ),
                "all_goal_frequency_map": wandb.Image(
                    f"{site_dir}/all_goal_frequency_map.png"
                ),
                "trajectory_plots": wandb.Image(
                    f"{site_dir}/trajectory_plots.png"
                ),
            }
            wandb.log(log_data, step=epoch)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--time",
        help="The maximum time for the simulation (seconds)",
        action="store",
        default=1e99,
    )
    parser.add_argument("-g", "--gpu", help="Use gpu", action="store_true")
    parser.add_argument(
        "-s", "--seed", help="Simulation seed", action="store", default=1
    )
    parser.add_argument(
        "-w",
        "--wandb",
        help="Store simulations results to Weights" " and Biases",
        action="store_true",
    )
    parser.add_argument(
        "-x", "--plots", help="Plot graphs", action="store_true"
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Simulation name (to store results in" " named folders)",
        action="store",
        default=None,
    )
    parser.add_argument(
        "--group",
        help="Simulation group name (to organize wandb runs)",
        action="store",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--parasite",
        help="Train a parasite model alongside" " the normal one",
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--load_weights",
        help="Load controller weights from" " file",
        action="store",
        default=None,
    )
    parser.add_argument(
        "--wdb_project",
        help="the name of thw wandb project",
        action="store",
        default=None,
    )
    parser.add_argument(
        "--wdb_entity",
        help="the name of thw wandb entity",
        action="store",
        default=None,
    )
    parser.add_argument(
        "--demo", help="Generate demo episodes", action="store_true"
    )
    parser.add_argument(
        "--render", help="Render generated demo episodes", action="store_true"
    )
    parser.add_argument(
        "-o",
        "--opt",
        nargs=1,
        help="Additional simulation option"
        " in KEY=VALUE format (overrides params.py)",
        action=AppendParamsAction,
        metavar="KEY=VALUE",
    )
    return parser.parse_args()


class AppendParamsAction(argparse.Action):
    params_string = ""

    def __call__(self, parser, args, values, option_string=None):
        assert len(values) == 1
        try:
            k, v = values[0].split("=", 2)
            AppendParamsAction.params_string += f"{k}={v};"
        except ValueError:
            raise argparse.ArgumentError(
                self,
                f'could not parse argument "{values[0]}"' " as k=v format",
            )


if __name__ == "__main__":

    args = parse_arguments()
    timing = float(args.time)
    gpu = bool(args.gpu)
    seed = int(args.seed)
    plots = bool(args.plots)
    use_wandb = bool(args.wandb)
    train_parasite = bool(args.parasite)
    demo = bool(args.demo)
    render = bool(args.render)
    simulation_name = args.name
    wdb_project = args.wdb_project
    wdb_entity = args.wdb_entity

    params = Parameters()
    if os.path.isfile("params.json"):
        params.load("params.json", mode="json")

    device = "cuda" if torch.cuda.is_available() and gpu else "cpu"
    torch.set_default_device(device)

    if args.name is not None:
        named_dir = (Path(simulations_dir) / args.name).resolve()
        os.makedirs(named_dir, exist_ok=True)
        os.chdir(named_dir)
        if plots:
            Path("PLOT_SIMS").touch()
        else:
            Path("PLOT_SIMS").unlink(missing_ok=True)

    print(AppendParamsAction.params_string)
    params.update(AppendParamsAction.params_string)
    print(params)

    if use_wandb:
        config = {
            k: v for k, v in vars(params).items() if not k.startswith("_")
        }
        run = wandb.init(
            project=wdb_project or "kickstarting_concept",
            entity=wdb_entity or "hill_uw",
            name=args.name,
            group=args.group if args.group else None,
            config=config,
        )

    if os.path.isfile("main.dump.npy"):
        main = np.load("main.dump.npy", allow_pickle=True)[0]
        main.plots = plots
        main.params.update(AppendParamsAction.params_string)
    else:
        main = Main(seed=seed, params=params, plots=plots)

    if args.load_weights is not None:
        weights = np.load(args.load_weights, allow_pickle=True)[0]
        main.controller.load(weights=weights)

        epoch_dir = f"{storage_dir}/{main.epoch:06d}"
        os.makedirs(epoch_dir, exist_ok=True)
        os.makedirs(site_dir, exist_ok=True)

        main.controller.save(main.epoch)

    print(main.epoch)

    try:
        if demo:
            main.demo_episodes(
                render="offline" if render else None,
                render_all_maps=True
            )
        elif train_parasite:
            main.train_parasite(timing)
        else:
            main.train(timing)
        print("Done!!", flush=True)
    except TimeLimitsException:
        print(f"Epoch {main.epoch}. end", flush=True)
        try:
            main.diagnose()
        except AttributeError:
            pass
