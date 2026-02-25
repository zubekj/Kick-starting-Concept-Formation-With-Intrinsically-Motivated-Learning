import copy
from pathlib import Path

import gymnasium as gym

from params import Parameters


class SMEnv:
    def __init__(
        self,
        seed,
        params,
        action_steps=5,
        store_observations=False,
        render_params=None,
        rand_obj_params=None,
    ):

        self.params = params
        self.action_steps = action_steps
        self.store_observations = store_observations

        self.rand_obj_params = rand_obj_params or {
            "stretch_conditions": params.obj_stretch_conditions,
            "rotation_conditions": params.obj_rotation_conditions,
            "pos": [params.obj_x, params.obj_y],
        }

        self.render_params = render_params or {
            "resolution_prop": 1,
            "duration": 100,
        }

        self.b2d_env = gym.make(
            "Box2DSimOneArmOneEye-v0",
            rand_obj_params=self.rand_obj_params,
            render_params=self.render_params,
        )
        self.b2d_env = self.b2d_env.unwrapped
        self.b2d_env.set_seed(seed)
        self.b2d_env.action_steps = action_steps

        self.b2d_env.set_taskspace(**self.params.task_space)
        self.render = None
        self.world = 0
        self.stored_observations = None
        if store_observations:
            self.stored_observations = []

    def __getstate__(self):
        return {
            "rng": self.b2d_env.rng,
            "action_steps": self.action_steps,
            "store_observations": self.store_observations,
            "rand_obj_params": self.rand_obj_params,
        }

    def __setstate__(self, state):
        self.__init__(
            seed=0,
            params=Parameters(),
            action_steps=state["action_steps"],
            store_observations=state["store_observations"],
            rand_obj_params=state["rand_obj_params"],
        )
        self.b2d_env.rng = state["rng"]

    def step(self, action):
        observation, *_ = self.b2d_env.step(action)
        if self.render is not None:
            self.b2d_env.render(self.render)
        if self.stored_observations is not None:
            self.stored_observations.append(copy.deepcopy(observation))
        return observation

    def get_b2d_state(self):
        return self.b2d_env.get_objects_params()

    def set_b2d_state(self, state):
        self.b2d_env.set_objects_params(state)

    def reset(
        self,
        world=None,
        world_dict=None,
        render=None,
        plot=None,
        plot_vision=None,
    ):
        self.render = render
        self.plot = plot
        if world is not None:
            self.world = world

        observation = self.b2d_env.reset(self.world, world_dict=world_dict)
        if self.render is not None:
            self.b2d_env.render_init(self.render)
        if self.stored_observations is not None:
            self.stored_observations = [observation]

        return observation

    def render_info(
        self,
        match_value,
        max_match,
        cum_match,
        f_vp,
        f_ssp,
        f_pp,
        f_ap,
        f_gp,
        map_path=None,
    ):
        if hasattr(self.params, "full_render"):
            assert self.render is not None

            path = map_path or Path("./www")
            visual_map_path = path / "visual_map.png"
            proprio_map_path = path / "proprio_map.png"
            touch_map_path = path / "ssensory_map.png"

            self.b2d_env.renderer.add_info_to_frames_three_maps(
                match_value,
                max_match,
                cum_match,
                f_vp,
                f_ssp,
                f_pp,
                f_ap,
                f_gp,
                visual_map_path=visual_map_path,
                proprio_map_path=proprio_map_path,
                touch_map_path=touch_map_path,
            )

    def render_info_three_maps(
        self, match_value, max_match, cum_match, f_vp, f_ssp, f_pp, f_ap, f_gp
    ):
        assert self.render is not None
        self.b2d_env.renderer.add_info_to_frames_three_maps(
            match_value,
            max_match,
            cum_match,
            f_vp,
            f_ssp,
            f_pp,
            f_ap,
            f_gp,
            visual_map_path="./www/visual_map.png",
            proprio_map_path="./www/proprio_map.png",
            touch_map_path="./www/ssensory_map.png",
        )


    def close(self):
        if self.plot is not None:
            self.b2d_env.renderer.close(self.plot)


class SMEnvParasite(SMEnv):

    def __init__(self, seed, params, observations, rand_obj_params=None):
        super(SMEnvParasite, self).__init__(seed, params, rand_obj_params=rand_obj_params)
        self.stored_observations = observations
        self.i = 0

    def step(self, action):
        self.i += 1
        return self.stored_observations[self.i % len(self.stored_observations)]

    def reset(self):
        self.i = 0
        return self.stored_observations[self.i]
