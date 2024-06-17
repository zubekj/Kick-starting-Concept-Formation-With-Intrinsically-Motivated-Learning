import glob
import os, sys
import shutil
from pathlib import Path
import matplotlib
import torch

matplotlib.use("Agg")

import numpy as np
import time

import params
from SMController import SMController
from SMEnv import SMEnv
from SMAgent import SMAgent
from box2dsim.envs.Simulator import TestPlotterVisualSalience

from SMGraphs import (
    remove_figs,
    blank_video,
    visual_map,
    comp_map,
    trajectories_map,
    representations_movements,
    log,
)

import matplotlib.pyplot as plt

np.set_printoptions(formatter={"float": "{:6.4f}".format})

storage_dir = "storage"
site_dir = "www"
simulations_dir = "simulations"
os.makedirs(storage_dir, exist_ok=True)
os.makedirs(site_dir, exist_ok=True)
os.makedirs(simulations_dir, exist_ok=True)

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

def modulate_param(base, limit, prop):
    return base + (limit - base) * prop

def softmax(x, t=0.01):
    e = np.exp(x / t)
    return e / (e.sum() + 1e-100)


class Main:
    def __init__(self, seed=None, plots=False):

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

        self.env = SMEnv(seed, params.action_steps)
        self.agent = SMAgent(self.env)
        self.controller = SMController(
            self.rng,
            load=params.load_weights,
            shuffle=params.shuffle_weights,
        )
        self.logs = np.zeros([params.epochs, 3])
        self.epoch = 0

    def __getstate__(self):
        return {
            "controller": self.controller.__getstate__(),
            "env": self.env.__getstate__(),
            "rng": self.rng.__getstate__(),
            "seed": self.seed,
            "plots": self.plots,
            "logs": self.logs,
            "epoch": self.epoch,
        }

    def __setstate__(self, state):

        self.plots = state["plots"]
        self.logs = state["logs"]
        self.epoch = state["epoch"]
        self.seed = state["seed"]
        torch.manual_seed(self.seed)
        self.rng = np.random.RandomState()
        self.rng.__setstate__(state["rng"])

        nlogs = len(self.logs)
        if params.epochs > nlogs:
            tmp = np.zeros([params.epochs, 3])
            tmp[:nlogs, : ] = self.logs.copy()
            self.logs = tmp
            tmp = np.zeros([params.epochs, 2])

        self.env = SMEnv(self.seed, params.action_steps)
        self.controller = SMController(
            self.rng,
            load=params.load_weights,
            shuffle=params.shuffle_weights,
        )

        self.controller.__setstate__(state["controller"])
        self.env.__setstate__(state["env"])

        self.agent = SMAgent(self.env)
        self.start = time.perf_counter()
        if self.plots is True:
            remove_figs(self.epoch)

    def is_object_out_of_taskspace(self, state):
        obj_xy = state["OBJ_POSITION"][0, 0]
        xlim, ylim = params.task_space["xlim"], params.task_space["ylim"]
        return (obj_xy[0] < xlim[0] or obj_xy[0] > xlim[1]
                or obj_xy[1] < ylim[0] or obj_xy[1] > ylim[1])

    def run_episodes(self, batch_v, batch_ss, batch_p, batch_a, batch_g,
                     v_r, ss_r, p_r, a_r,
                     v_p, ss_p, p_p, a_p, g_p,
                     match_value_per_mod,
                     match_value,
                     match_increment_per_mod,
                     match_increment,
                     agent, controller,
                     contexts, envs, states,
                     ):
        batch_size = len(contexts)

        cum_match = np.zeros(batch_size, dtype=int)
        episode_len = np.zeros(batch_size, dtype=int)
        max_match = np.zeros(batch_size)
        matches = np.zeros((batch_size, params.stime), dtype=bool)

        # Main loop through time steps and episodes
        smcycles = [SensoryMotorCircle(params.action_steps)] * batch_size
        for t in range(1, params.stime+1):
            if t < params.stime:
                for episode in range(batch_size):
                    # Do not update the episode if it has ended
                    if states[episode] is None or cum_match[episode] >= params.cum_match_stop_th:
                        continue
                    episode_len[episode] = t

                    # set correct policy
                    agent.updatePolicy(batch_a[episode, 0, :])
                    state = smcycles[episode].step(envs[episode], agent, states[episode])

                    # End the episode if object moves too far away
                    if self.is_object_out_of_taskspace(state):
                        states[episode] = None
                    else:
                        states[episode] = state
                        batch_v[episode, t, :] = state["VISUAL_SENSORS"].ravel()
                        batch_ss[episode, t, :] = state["TOUCH_SENSORS"]
                        batch_p[episode, t, :] = state["JOINT_POSITIONS"][:5]

            if t % params.action_steps == 0 or t == params.stime:
                # get Representations for the last N = params.action_steps steps
                t0 = t - params.action_steps
                bsize = batch_size * params.action_steps
                sa = np.s_[:, t0:t, :]
                Rs, Rp = controller.spread(
                    [
                        batch_v[sa].reshape((bsize, -1)),
                        batch_ss[sa].reshape((bsize, -1)),
                        batch_p[sa].reshape((bsize, -1)),
                        batch_a[sa].reshape((bsize, -1)),
                        batch_g[sa].reshape((bsize, -1)),
                    ])
                v_r[sa].flat = Rs[0].flat
                ss_r[sa].flat = Rs[1].flat
                p_r[sa].flat = Rs[2].flat
                a_r[sa].flat = Rs[3].flat

                v_p[sa].flat = Rp[0].flat
                ss_p[sa].flat = Rp[1].flat
                p_p[sa].flat = Rp[2].flat
                a_p[sa].flat = Rp[3].flat
                g_p[sa].flat = Rp[4].flat

                # calculate match value
                match_value[:, t0:t], match_value_per_mod[sa] =\
                    controller.computeMatchSimple(v_p[sa], ss_p[sa], p_p[sa], a_p[sa], g_p[sa])
                if t > params.action_steps:
                    match_increment_per_mod[sa] = np.maximum(0, match_value_per_mod[sa] - match_value_per_mod[:, (t0-1):(t-1), :])
                    match_increment[:, t0:t] = np.mean(match_increment_per_mod[sa], axis=-1)
                    # update cumulative match
                    if t0 > params.drop_first_n_steps:
                        for i in range(t0, t):
                            mmask = (match_value[:, i] - max_match) > params.match_incr_th
                            matches[:, i] = mmask
                            cum_match += mmask
                            max_match[mmask] = match_value[mmask, i]
                        cum_match[cum_match > params.cum_match_stop_th] = params.cum_match_stop_th
        
        return matches, cum_match, episode_len

    def train(self, time_limits):

        if self.epoch == 0:
            print("Training", flush=True)
        else:
            if self.epoch >= params.epochs - 1:
                raise TimeLimitsException

        env = self.env
        agent = self.agent
        controller = self.controller
        logs = self.logs
        epoch = self.epoch
        epoch_start = time.perf_counter()
        contexts = (np.arange(params.batch_size) % 3) + 1

        batch_v = np.zeros([params.batch_size, params.stime, params.visual_size])
        batch_ss = np.zeros([params.batch_size, params.stime, params.somatosensory_size])
        batch_p = np.zeros([params.batch_size, params.stime, params.proprioception_size])
        batch_a = np.zeros([params.batch_size, params.stime, params.policy_size])
        batch_c = np.ones([params.batch_size, params.stime, 1])
        batch_log = np.ones([params.batch_size, params.stime, 1])
        batch_g = np.zeros([params.batch_size, params.stime, params.internal_size])
        v_r = np.zeros([params.batch_size, params.stime, params.internal_size])
        ss_r = np.zeros([params.batch_size, params.stime, params.internal_size])
        p_r = np.zeros([params.batch_size, params.stime, params.internal_size])
        a_r = np.zeros([params.batch_size, params.stime, params.internal_size])
        v_p = np.zeros([params.batch_size, params.stime, 2])
        ss_p = np.zeros([params.batch_size, params.stime, 2])
        p_p = np.zeros([params.batch_size, params.stime, 2])
        a_p = np.zeros([params.batch_size, params.stime, 2])
        g_p = np.zeros([params.batch_size, params.stime, 2])

        match_value = np.zeros([params.batch_size, params.stime])
        match_value_per_mod = np.zeros([params.batch_size, params.stime, 4])
        match_increment = np.zeros([params.batch_size, params.stime])
        match_increment_per_mod = np.zeros([params.batch_size, params.stime, 4])

        cum_match = None
        envs = [None] * params.batch_size
        states = [None] * params.batch_size
        reset_goal_mask = np.zeros(params.batch_size, dtype=bool)
        goals = np.zeros([params.batch_size, params.internal_size])

        while epoch < params.epochs:

            total_time_elapsed = time.perf_counter() - self.start
            if total_time_elapsed >= time_limits:
                if self.epoch > 0:
                    raise TimeLimitsException

            print(f"{epoch:6d}", end=" ", flush=True)

            controller.comp_grid = controller.getCompetenceGrid()
            comp = controller.comp_grid.mean()

            controller.match_sigma = modulate_param(
                params.base_match_sigma,
                params.match_sigma,
                1 - comp,
            )
            controller.curr_sigma = modulate_param(
                params.base_internal_sigma,
                params.internal_sigma,
                1 - comp,
            )
            controller.curr_lr = modulate_param(
                params.base_lr,
                params.stm_lr,
                1 - comp,
            )

            controller.explore_sigma = params.explore_sigma

            controller.updateParams(
                controller.curr_sigma, controller.curr_lr
            )

            print(f"{controller.curr_sigma}, {controller.curr_lr}")

            # ----- prepare episodes
            for episode in range(params.batch_size): 
                # Reset environment every env_reset_freq epochs or when the object is out of reach
                if states[episode] is None or epoch % params.env_reset_freq == 0:
                    # Each environment in each epoch should have a different seed
                    env = SMEnv(self.seed + episode + epoch, params.action_steps)
                    env.b2d_env.prepare_world(contexts[episode])
                    states[episode] = env.reset(contexts[episode])
                    envs[episode] = env
                    reset_goal_mask[episode] = True
                state = states[episode]
                batch_v[episode, 0, :] = state["VISUAL_SENSORS"].ravel()
                batch_ss[episode, 0, :] = state["TOUCH_SENSORS"]
                batch_p[episode, 0, :] = state["JOINT_POSITIONS"][:5]

            # get Representations for initial states
            Rs, Rp = controller.spread(
                    [
                        batch_v[:, 0, :],
                        batch_ss[:, 0, :],
                        batch_p[:, 0, :],
                        batch_a[:, 0, :],
                        batch_g[:, 0, :],
                    ])
            v_r[:, 0, :], ss_r[:, 0, :], p_r[:, 0, :], a_r[:, 0, :], _ = Rs
            v_p[:, 0, :], ss_p[:, 0, :], p_p[:, 0, :], a_p[:, 0, :], g_p[:, 0, :] = Rp

            # get policy at the first timestep
            goals[reset_goal_mask] = v_r[reset_goal_mask, 0, :]
            (policies,
             competences,
             rcompetences) = controller.getPoliciesFromRepresentationsWithNoise(goals)
            
            # fill all batches with policies, goals, and competences
            # (goal is different for each episode, but the same for each
            # time step within an episode)
            batch_a[::] = policies[:, None, :]
            batch_g[::] = goals[:, None, :]
            batch_c[::] = competences[:, None, :]
            batch_log[::] = rcompetences[:, None, :]

            # ---- run all episodes
            matches, cum_match, _ = self.run_episodes(
                batch_v, batch_ss, batch_p, batch_a, batch_g,
                v_r, ss_r, p_r, a_r,
                v_p, ss_p, p_p, a_p, g_p,
                match_value_per_mod,
                match_value,
                match_increment_per_mod,
                match_increment,
                agent, controller, contexts,
                envs, states)

            # ---- reset goals in the successful episodes
            reset_goal_mask = cum_match >= params.cum_match_stop_th

            # ---- end of an epoch: controller update
            bsize = params.batch_size * params.stime
            (update_items, update_episodes, curr_loss, mean_modulation) =\
            controller.update(
                batch_v.reshape((bsize, -1)),
                batch_ss.reshape((bsize, -1)),
                batch_p.reshape((bsize, -1)),
                batch_a.reshape((bsize, -1)),
                batch_g.reshape((bsize, -1)),
                match_value.reshape(-1),
                matches.reshape(-1),
                cum_match,
                competences=batch_c.reshape((bsize, -1))
            )

            # ---- print
            c = np.outer(contexts, np.ones(params.stime)).ravel()
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
            print(f"{batch_ss.sum():#10.2f}", end=" ", flush=True)
            logs[epoch] = [
                batch_log.min(),
                batch_log.mean(),
                batch_log.max(),
            ]
            print(
                ("%8.7f " * 3)
                % (
                    batch_log.min(),
                    batch_log.mean(),
                    batch_log.max(),
                ),
                end="",
            )
            print(f"  {np.mean(curr_loss):#8.7f}")
            print(logs[epoch][1])

            if use_wandb:
                wandb.log({'min_comp': logs[epoch][0],
                           'mean_comp': logs[epoch][1],
                           'max_comp': logs[epoch][2],
                           'stm_loss': np.mean(curr_loss),
                           'stm_v_loss': curr_loss[0],
                           'stm_ss_loss': curr_loss[1],
                           'stm_p_loss': curr_loss[2],
                           'stm_a_loss': curr_loss[3],
                           'stm_sigma': controller.curr_sigma,
                           'stm_lr': controller.curr_lr,
                           'mean_cum_match': cum_match.mean() / params.cum_match_stop_th,
                           'grid_comp_mean': comp,
                           }, step=epoch)

            self.match_value = match_value
            self.match_increment = match_increment
            self.match_value_per_mod = match_value_per_mod
            self.match_increment_per_mod = match_increment_per_mod
            self.v_r = v_r
            self.ss_r = ss_r
            self.p_r = p_r
            self.a_r = a_r
            self.batch_v = batch_v
            self.batch_ss = batch_ss
            self.batch_p = batch_p
            self.batch_a = batch_a

            # diagnose
            if (epoch > 0 and epoch % params.epochs_to_test == 0) or epoch == (
                params.epochs - 1
            ):

                epoch_dir = f"{storage_dir}/{epoch:06d}"
                os.makedirs(epoch_dir, exist_ok=True)
                np.save(f"{epoch_dir}/main.dump", [self], allow_pickle=True)
                self.diagnose()

                time_elapsed = time.perf_counter() - epoch_start
                print("---- TIME: %10.4f" % time_elapsed, flush=True)
                epoch_start = time.perf_counter()

            match_value[::] = 0
            match_increment[::] = 0
            match_value_per_mod[::] = 0
            match_increment_per_mod[::] = 0
            batch_v[::] = 0
            batch_ss[::] = 0
            batch_p[::] = 0
            batch_a[::] = 0
            v_r[::] = 0
            ss_r[::] = 0
            p_r[::] = 0
            a_r[::] = 0
            v_p[::] = 0
            ss_p[::] = 0
            p_p[::] = 0
            a_p[::] = 0

            epoch += 1
            self.epoch = epoch
            sys.stdout.flush()

    def diagnose(self):

        np.save("main.dump", [self], allow_pickle=True)

        env = self.env
        agent = self.agent
        controller = self.controller
        logs = self.logs
        epoch = self.epoch

        data = {}
        data["match_value"] = self.match_value
        data["match_increment"] = self.match_increment
        data["match_value_per_mod"] = self.match_value_per_mod
        data["match_increment_per_mod"] = self.match_increment_per_mod
        data["v_r"] = self.v_r
        data["ss_r"] = self.ss_r
        data["p_r"] = self.p_r
        data["a_r"] = self.a_r
        data["v"] = self.batch_v
        data["ss"] = self.batch_ss
        data["p"] = self.batch_p
        data["a"] = self.batch_a

        epoch_dir = f"{storage_dir}/{epoch:06d}"
        os.makedirs(epoch_dir, exist_ok=True)

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
            self.demo_episodes(n_episodes=params.tests, plot_prefix="episode")

        if os.path.isfile("COMPUTE_TRAJECTORIES"):
            print(
                "----> Compute Trajectories ...",
                end=" ",
                flush=True,
            )
            context = 4  # no object
            trj = np.zeros([params.internal_size, params.stime, 2])

            state = env.reset(context)
            agent.reset()
            for i, goal_r in enumerate(controller.goal_grid):
                policy = controller.getPoliciesFromRepresentations(np.array([goal_r]))
                agent.updatePolicy(policy)
                smcycle = SensoryMotorCircle()
                for t in range(params.stime):
                    state = smcycle.step(env, agent, state)
                    trj[i, t] = state["JOINT_POSITIONS"][-2:]
                if i % 10 == 0 or i == params.internal_size - 1:
                    print(
                        "{:d}% ".format(int(100 * (i / params.internal_size))),
                        end=" ",
                        flush=True,
                    )
            print(flush=True)
            np.save(f"{site_dir}/trajectories", trj)
            np.save(f"{epoch_dir}/trajectories", trj)
            trajectories_map()

        if use_wandb:
            log_data = {
                'visual_map': wandb.Image("www/visual_map.png"),
                'comp_map': wandb.Image("www/comp_map.png"),
            }
            for i in range(params.tests):
                log_data[f"episode{i}"] = wandb.Image(f"www/episode{i}.gif")
            wandb.log(log_data, step=epoch)

    def collect_sensory_states(self):
        pass

    def demo_episode(self, idx):
        pass

    def demo_episodes(self, n_episodes=params.internal_size, plot_prefix="demo"):
       
        if n_episodes > params.internal_size:
            n_episodes = params.internal_size

        side = int(np.sqrt(params.visual_size // 3))
        params.base_internal_sigma = 0.1

        env = self.env
        agent = self.agent
        controller = self.controller
        
        batch_v = np.zeros([1, params.stime, params.visual_size])
        batch_ss = np.zeros([1, params.stime, params.somatosensory_size])
        batch_p = np.zeros([1, params.stime, params.proprioception_size])
        batch_a = np.zeros([1, params.stime, params.policy_size])
        batch_g = np.zeros([1, params.stime, params.internal_size])
        batch_c = np.ones([1, params.stime, 1])
        batch_log = np.ones([1, params.stime, 1])

        v_r = np.zeros([1, params.stime, params.internal_size])
        ss_r = np.zeros([1, params.stime, params.internal_size])
        p_r = np.zeros([1, params.stime, params.internal_size])
        a_r = np.zeros([1, params.stime, params.internal_size])

        v_p = np.zeros([1, params.stime, 2])
        ss_p = np.zeros([1, params.stime, 2])
        p_p = np.zeros([1, params.stime, 2])
        a_p = np.zeros([1, params.stime, 2])
        g_p = np.zeros([1, params.stime, 2])
        
        match_value = np.zeros([1, params.stime])
        match_value_per_mod = np.zeros([1, params.stime, 4])
        match_increment = np.zeros([1, params.stime])
        match_increment_per_mod = np.zeros([1, params.stime, 4])

        v_p_set = set()
        i = 0

        env = self.env
        while len(v_p_set) < n_episodes:
            context = (i % 3) + 1
            env.b2d_env.prepare_world(context)
            state = env.reset(context,
                              plot=f"{site_dir}/{plot_prefix}",
                              render="offline")

            full_match_value = []
            full_matches = []
            envs = [env]
            states = [state]
            contexts = [context]
            goals = None
            for j in range(params.env_reset_freq):
                batch_v[0, 0, :] = state["VISUAL_SENSORS"].ravel()
                batch_ss[0, 0, :] = state["TOUCH_SENSORS"]
                batch_p[0, 0, :] = state["JOINT_POSITIONS"][:5]
               
                # get Representations for initial states
                Rs, Rp = controller.spread(
                        [
                            batch_v[:, 0, :],
                            batch_ss[:, 0, :],
                            batch_p[:, 0, :],
                            batch_a[:, 0, :],
                            batch_g[:, 0, :],
                        ])
                v_r[:, 0, :], ss_r[:, 0, :], p_r[:, 0, :], a_r[:, 0, :], _ = Rs
                v_p[:, 0, :], ss_p[:, 0, :], p_p[:, 0, :], a_p[:, 0, :], g_p[:, 0, :] = Rp

                # get policy at the first timestep
                if goals is None:
                    goals = v_r[:, 0, :]
                (policies,
                 competences,
                 rcompetences) = controller.getPoliciesFromRepresentationsWithNoise(goals)
                
                # fill all batches with policies, goals, and competences
                # (goal is different for each episode, but the same for each
                # time step within an episode)
                batch_a[::] = policies[:, None, :]
                batch_g[::] = goals[:, None, :]
                batch_c[::] = competences[:, None, :]
                batch_log[::] = rcompetences[:, None, :]
           
                if j == 0:
                    policy = tuple(v_p[0, 0, :])
                    if policy in v_p_set:
                        print(f"Skipping repeated prototype: {int(policy[0])}{int(policy[1])}")
                        break

                    print(f"{site_dir}/{plot_prefix}_00{int(policy[0])}{int(policy[1])}")
                    print(policy)
                    print(context)
                    v_p_set.add(policy)

                matches, cum_match, episodes_len = self.run_episodes(
                    batch_v, batch_ss, batch_p, batch_a, batch_g,
                    v_r, ss_r, p_r, a_r,
                    v_p, ss_p, p_p, a_p, g_p,
                    match_value_per_mod,
                    match_value,
                    match_increment_per_mod,
                    match_increment,
                    agent, controller, contexts,
                    envs, states)

                if cum_match > params.cum_match_stop_th:
                    goals = None

                full_match_value.append(match_value[0, :episodes_len[0]])
                full_matches.append(matches[0, :episodes_len[0]])
                if states[0] is None:
                    break

            i += 1
            
            full_match_value = np.concatenate(full_match_value)
            full_matches = np.concatenate(full_matches)
            
            env.render_info(full_match_value, full_matches)
            env.close()
            if plot_prefix == "demo":
                shutil.copyfile(f"{site_dir}/{plot_prefix}.gif", f"{site_dir}/{plot_prefix}_00{int(policy[0])}{int(policy[1])}.gif")
            else:
                shutil.copyfile(f"{site_dir}/{plot_prefix}.gif", f"{site_dir}/{plot_prefix}{len(v_p_set)-1}.gif")

        print("demo episodes: Done!!!")      

    def get_context_from_visual(self):
        pass


if __name__ == "__main__":

    import argparse

    override_params = {}

    class kvdictAppendAction(argparse.Action):
        """
        argparse action to split an argument into KEY=VALUE form
        on the first = and append to a dictionary.
        """
        def __call__(self, parser, args, values, option_string=None):
            assert(len(values) == 1)
            try:
                (k, v) = values[0].split("=", 2)
            except ValueError as ex:
                raise argparse.ArgumentError(self, f"could not parse argument \"{values[0]}\" as k=v format")

            if v == "False" or v == "True":
                v = bool(v)
            else:
                try:
                    v = int(v)
                except:
                    try:
                        v = float(v)
                    except:
                        pass

            override_params[k] = v

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
        "-s",
        "--seed",
        help="Simulation seed",
        action="store",
        default=1,
    )
    parser.add_argument(
        "-w",
        "--wandb",
        help="Store simulations results to Weights and Biases",
        action="store_true",
    )
    parser.add_argument(
        "-x",
        "--plots",
        help="Plot graphs",
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Simulation name (to store results in named folders)",
        action="store",
        default=None
    )
    parser.add_argument(
        "-o",
        "--opt",
        nargs=1,
        help="Additional simulation option in KEY=VALUE format (overrides params.py)",
        action=kvdictAppendAction,
        metavar="KEY=VALUE"
    )
    args = parser.parse_args()
    timing = float(args.time)
    gpu = bool(args.gpu)
    seed = int(args.seed)
    plots = bool(args.plots)
    use_wandb = bool(args.wandb)
    simulation_name = args.name

    if gpu:
        torch.set_default_device('cuda')

    if args.name is not None:
        named_dir = (Path(simulations_dir) / args.name).resolve() 
        os.makedirs(named_dir, exist_ok=True)
        os.chdir(named_dir)
        Path("PLOT_SIMS").touch()
        Path("COMPUTE_TRAJECTORIES").touch()

    # Override params with command-line options
    for k, v in override_params.items():
        vars(params)[k] = v

    if use_wandb:
        import wandb
        # Here only public fields of params module are selected
        config = {k: v for k, v in vars(params).items() if not k.startswith("_")}
        del config["np"] # This is an ugly way to remove numpy import from params
        run = wandb.init(
            project="kickstarting_concept",
            entity="hill_uw",
            name=args.name,
            config=config,
        )

    if os.path.isfile("main.dump.npy"):
        main = np.load("main.dump.npy", allow_pickle="True")[0]
        main.plots = plots
    else:
        main = Main(seed, plots)

    print(main.epoch)

    try:
        main.train(timing)
        print("Done!!", flush=True)
    except TimeLimitsException:
        print(f"Epoch {main.epoch}. end", flush=True)
        try:
            main.diagnose()
        except AttributeError:
            pass
