from logging import root
import torch

from isaacgym import gymapi
from isaacgym import gymtorch
import os

from .amp.a1_base import A1Base, dof_to_obs
from .a1_amp import A1AMP
from .a1_navigation import A1Navigation
from .amp.utils_amp import gym_util
from .amp.utils_amp.motion_lib import A1MotionLib

from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
import numpy as np

ADDITIONAL_OBS_NUM = 3 + 6 + 3 + 3 + 2  # local pos, 6d rot, linear vel, angular vel of the ball, goal pos
NUM_CURR_OBS = 18 + 3
BALL_RAD = 0.1
init_goal_dist = 7.0
OBJECT = ['sphere', 'box']


class A1Dribbling(A1AMP):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.obj = None
        self.randomize_object = cfg['task']['randomize_obj']
        if self.randomize_object:
            self.size_range = cfg['task']['obj_size_range']
            self.density_range = cfg['task']['obj_density_range']
        self.limit_space = cfg['env']['plane']['limit_space']
        self.space_width, self.space_length = cfg['env']['plane']['width'], cfg['env']['plane']['length']
        self.height = BALL_RAD
        self.ball_rad = cfg['task']['ball_size']
        super().__init__(cfg, sim_device, graphics_device_id, headless)
        # track goal progress
        self.goal_terminate = torch.randint(self.max_episode_length // 2, self.max_episode_length, (self.num_envs,), device=self.device, dtype=torch.int32)
        self.goal_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._pos_hist = torch.zeros((self.num_envs, 3, 3), device=self.device, dtype=torch.float)
        self._ball_hist = torch.zeros_like(self._pos_hist)
        self.add_ball_delay = self.cfg["task"]["noise"]["add_ball_delay"]
        self.push_ball = self.cfg['task']['domain_rand']['push_ball']
        self.delayed_ball_obs = torch.zeros((self.num_envs, 3, 3), dtype=torch.float, device=self.device)
        self.actor_vel_scale = self.cfg["task"]["reward"]["actor_vel_scale"]
        self.ball_vel_scale = self.cfg["task"]["reward"]["ball_vel_scale"]
        self.energy_scale = self.cfg["task"]["reward"]["energy_scale"]
        self.energy_weight = self.cfg["task"]["reward"]["energy_weight"]
        self.ab_dist_threshold = self.cfg["task"]["reward"]["ab_dist_threshold"]
        self.piecewise = self.cfg["task"]["reward"]["piecewise"]
        self.goal_reset = self.cfg["task"]["goal_reset_freq_inv"]
        self.goal_reset_upper_inv = self.cfg["task"]["goal_reset_freq_inv_upper"]
        self.rand_goal = self.cfg["task"]["randomize_goal"]

    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if self.history_steps is None:
            return obs_size + ADDITIONAL_OBS_NUM
        else:
            return obs_size + 2 + 3 * (self.history_steps+1)

    def _init_obs_tensors(self):
        if self.history_steps is not None:
            states_idx = NUM_CURR_OBS * (self.history_steps + 1)
            self._states_history = self.obs_buf[:, :states_idx].view(self.num_envs, self.history_steps + 1,
                                                                     -1)  # last entry is current obs
            self._goal_xy = self.obs_buf[:, states_idx:states_idx + 2]
            self._actions_history = self.obs_buf[:, states_idx + 2:].view(self.num_envs, self.history_steps, -1)
            if self.priv_obs:
                self.priv_obs_buf = torch.zeros_like(self.obs_buf)
                self._priv_states_history = self.priv_obs_buf[:, :states_idx].view(self.num_envs, self.history_steps + 1,
                                                                     -1)
                self._priv_goal_xy = self.priv_obs_buf[:, states_idx:states_idx + 2]
                self._priv_actions_history = self.priv_obs_buf[:, states_idx + 2:].view(self.num_envs, self.history_steps, -1)
        return

    def _get_rigid_body_states_from_tensor(self, states_tensor):
        if not self.headless:
            self._all_actor_rb_states = gymtorch.wrap_tensor(states_tensor)
            self._all_actor_rb_states = self._all_actor_rb_states.view(self.num_envs,
                                                                       self.num_bodies + self.num_markers, -1)
            self.rb_states = self._all_actor_rb_states[..., :self.num_bodies, :]
            self.ball_states = self._all_actor_rb_states[..., self.num_bodies:-1, :]
            self.marker_states = self._all_actor_rb_states[..., -1:, :]
        else:
            self._all_actor_rb_states = gymtorch.wrap_tensor(states_tensor)
            self._all_actor_rb_states = self._all_actor_rb_states.view(self.num_envs,
                                                                       self.num_bodies + self.num_markers, -1)
            self.rb_states = self._all_actor_rb_states[..., :self.num_bodies, :]
            self.ball_states = self._all_actor_rb_states[..., self.num_bodies:, :]
        return

    def _push_robots(self):
        """ Randomly pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """

        forces = torch.zeros((self.num_envs, self._all_actor_rb_states.shape[1], 3), device=self.device, dtype=torch.float)
        torques = torch.zeros((self.num_envs, self._all_actor_rb_states.shape[1], 3), device=self.device, dtype=torch.float)

        actor_forces = forces[:, :self.rb_states.shape[1], :]
        actor_torques = torques[:, :self.rb_states.shape[1], :]

        mask = self.perturbation_time <= 0
        apply_mask = torch.rand(self.num_envs, device=self.device) < self.dt  # apply to only a percentage of envs
        self.perturbation_time[self.perturbation_enabled_episodes & mask & apply_mask] = self.perturbation_time[
            self.perturbation_enabled_episodes & mask & apply_mask].uniform_(0.1, 3)
        self.perturbation_forces[self.perturbation_enabled_episodes & mask & apply_mask] = self.perturbation_forces[
            self.perturbation_enabled_episodes & mask & apply_mask].uniform_(-20, 20)
        self.perturbation_torques[self.perturbation_enabled_episodes & mask & apply_mask] = self.perturbation_torques[
            self.perturbation_enabled_episodes & mask & apply_mask].uniform_(-5, 5)

        if self.push_ball:
            ball_forces = forces[:, self.rb_states.shape[1]:self.ball_states.shape[1], :]
            ball_torques = torques[:, self.rb_states.shape[1]:self.ball_states.shape[1], :]
            ball_forces[~mask] = self.perturbation_forces[~mask].expand(-1, self.ball_states.shape[1], -1)
            ball_torques[~mask] = self.perturbation_torques[~mask].expand(-1, self.ball_states.shape[1], -1)

        actor_forces[~mask] = self.perturbation_forces[~mask].expand(-1, self.rb_states.shape[1], -1)
        actor_torques[~mask] = self.perturbation_torques[~mask].expand(-1, self.rb_states.shape[1], -1)
        self.perturbation_time[~mask] -= self.dt

        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces),
                                                gymtorch.unwrap_tensor(torques), gymapi.GLOBAL_SPACE)
        return

    def _get_root_states_from_tensor(self, state_tensor):
        self.all_actor_indices = torch.arange(self.num_envs * (self.num_markers + 1), dtype=torch.int32,
                                              device=self.device).reshape(
            (self.num_envs, (self.num_markers + 1))
        )
        self._all_actor_root_states = gymtorch.wrap_tensor(state_tensor)
        self._root_states = self._all_actor_root_states.view(self.num_envs, self.num_markers + 1, self.num_dof + 1)[
                            ..., 0, :]
        self._ball_root_states = self._all_actor_root_states.view(self.num_envs, self.num_markers + 1,
                                                                  self.num_dof + 1)[..., 1, :]
        self._ball_root_states[:, :7] = self.initial_ball_pos
        self._ball_root_states[:, 7:] = 0
        self._prev_ball_states = self._ball_root_states.clone()
        if not self.headless:
            self._goal_root_states = self._all_actor_root_states.view(self.num_envs, self.num_markers + 1, self.num_dof + 1)[..., 2, :]
        return

    # def process_ball_shape_props(self, props, i):
    #     for p in range(len(props)):
    #         props[p].restitution = 1.0
    #     return props

    def _process_ball_body_props(self, props, i):
        if self.randomize_object:
            mass_ratio = self.domain_rand["mass_ratio_range"]
            inertia_ratio = self.domain_rand["inertia_ratio_range"]
            for k in range(len(props)):
                props[k].mass *= np.random.uniform(1 - mass_ratio, 1 + mass_ratio)
                props[k].inertia.x *= np.random.uniform(1 - inertia_ratio, 1 + inertia_ratio)
                props[k].inertia.y *= np.random.uniform(1 - inertia_ratio, 1 + inertia_ratio)
                props[k].inertia.z *= np.random.uniform(1 - inertia_ratio, 1 + inertia_ratio)

        return props

    def _create_marker_actors(self, env_ptr, marker_asset, init_goal_pos, i):
        if self.randomize_object:
            ball_asset, ball_init_pos = marker_asset[i], init_goal_pos[0]
        else:
            ball_asset, ball_init_pos = marker_asset[0], init_goal_pos[0]
        ball_init_pos.p.x = self.initial_ball_pos[i, 0]
        ball_init_pos.p.y = self.initial_ball_pos[i, 1]
        ball_init_pos.p.z = self.initial_ball_pos[i, 2]

        # ball_shape_props = self.gym.get_asset_rigid_shape_properties(ball_asset)
        # ball_shape_props = self.process_ball_shape_props(ball_shape_props, i)
        # self.gym.set_asset_rigid_shape_properties(ball_asset, ball_shape_props)

        ball_handle = self.gym.create_actor(env_ptr, ball_asset, ball_init_pos, "ball", i, 0, 0)
        self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                      gymapi.Vec3(1, 1, 0))
        ball_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, ball_handle)
        ball_body_props = self._process_ball_body_props(ball_body_props, i)
        self.gym.set_actor_rigid_body_properties(env_ptr, ball_handle, ball_body_props)

        if not self.headless:
            if not self.limit_space:
                goal_asset, goal_init_pos = marker_asset[-1], init_goal_pos[1]
            else:
                goal_asset, goal_init_pos = marker_asset[-5], init_goal_pos[1]
                bound_left_asset, bound_right_asset, bound_top_asset, bound_bottom_asset = marker_asset[-4], \
                marker_asset[-3], marker_asset[-2], marker_asset[-1]
                bound_left_pos, bound_right_pos, bound_top_pos, bound_bottom_pos = init_goal_pos[2], init_goal_pos[3], \
                init_goal_pos[4], init_goal_pos[5]
                bound_left_pos.p.x, bound_left_pos.p.y, bound_left_pos.p.z = self.space_length / 2, 0, 0.03
                bound_right_pos.p.x, bound_right_pos.p.y, bound_right_pos.p.z = -self.space_length / 2, 0, 0.03
                bound_top_pos.p.x, bound_top_pos.p.y, bound_top_pos.p.z = 0, self.space_width // 2, 0.03
                bound_bottom_pos.p.x, bound_bottom_pos.p.y, bound_bottom_pos.p.z = 0, -self.space_width // 2, 0.03
                bound_left_handle = self.gym.create_actor(env_ptr, bound_left_asset, bound_left_pos, "bound_left",
                                                          i + self.num_envs, 1, 0)
                bound_right_handle = self.gym.create_actor(env_ptr, bound_right_asset, bound_right_pos, "bound_right",
                                                           i + self.num_envs, 1, 0)
                bound_top_handle = self.gym.create_actor(env_ptr, bound_top_asset, bound_top_pos, "bound_top",
                                                         i + self.num_envs, 1, 0)
                bound_bottom_handle = self.gym.create_actor(env_ptr, bound_bottom_asset, bound_bottom_pos,
                                                            "bound_bottom",
                                                            i + self.num_envs, 1, 0)
                self.gym.set_rigid_body_color(env_ptr, bound_left_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              gymapi.Vec3(1, 1, 1))
                self.gym.set_rigid_body_color(env_ptr, bound_right_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              gymapi.Vec3(1, 1, 1))
                self.gym.set_rigid_body_color(env_ptr, bound_top_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              gymapi.Vec3(1, 1, 1))
                self.gym.set_rigid_body_color(env_ptr, bound_bottom_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                              gymapi.Vec3(1, 1, 1))
            goal_init_pos.p.x = self._goal_pos[i, 0]
            goal_init_pos.p.y = self._goal_pos[i, 1]
            goal_init_pos.p.z = 0.2
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_init_pos, "goal", i+self.num_envs, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(1, 0, 0))

        return

    def _create_marker_envs(self, asset_opts):
        asset, pos = [], []
        # set initial ball pos
        self.initial_ball_pos = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        self.initial_ball_pos[..., 6] = 1.
        if self.limit_space:
            self._reset_limited_ball_pos()
        else:
            _ball_dist = torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device) * 3.0
            _ball_angle = torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device) * torch.pi * 2
            self.initial_ball_pos[..., 0] = torch.flatten(_ball_dist * torch.cos(_ball_angle))
            self.initial_ball_pos[..., 1] = torch.flatten(_ball_dist * torch.sin(_ball_angle))
            self.initial_ball_pos[..., 2] = self.height

        ball_asset_opts = gymapi.AssetOptions()
        ball_asset_opts.fix_base_link = False
        ball_asset_opts.override_inertia = True
        # ball_asset_opts.use_mesh_materials = True

        if self.randomize_object:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'base/' + self.size_range)
            object_assets = os.listdir(asset_root)
            offset = int(np.random.uniform(0, self.num_envs, 1))
            for i in range(self.num_envs):
                asset_file = object_assets[(i + offset) % self.num_envs]
                ball_rad = float(asset_file[0] + '.' + asset_file[1:5])
                ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, ball_asset_opts)
                # ball_asset_opts.angular_damping = np.random.uniform(0.5, 3.)
                # ball_asset_opts.linear_damping = np.random.uniform(0.5, 3.)
                # ball_asset_opts.density = np.random.uniform(self.density_range[0], self.density_range[1])
                # ball_rad = np.random.uniform(self.size_range[0], self.size_range[1])
                # ball_asset = self.gym.create_sphere(self.sim, ball_rad, ball_asset_opts)
                self.height = ball_rad + 1e-4
                asset.append(ball_asset)
                self.initial_ball_pos[i, 2] = self.height
        else:
            if "asset" in self.cfg["env"]:
                asset_file = self.cfg["env"]["asset"]["ballAsset"]
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
            ball_asset = self.gym.load_asset(self.sim, asset_root, asset_file, ball_asset_opts)
            # ball_asset_opts.angular_damping = 1.
            # ball_asset_opts.linear_damping = 1.
            # ball_asset = self.gym.create_sphere(self.sim, self.ball_rad, ball_asset_opts)
            # self.height = self.ball_rad + 1e-4
            asset.append(ball_asset)
        init_ball_pos = gymapi.Transform()
        self.num_markers = 1

        pos.append(init_ball_pos)
        # create goal pos near the ball
        self._goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        if self.limit_space:
            self._reset_limited_goal_pos()
        else:
            _goal_dist = torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device) * init_goal_dist
            _goal_rot = torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device) * torch.pi * 2
            self._goal_pos[..., 0] = torch.flatten(_goal_dist * torch.cos(_goal_rot)) + self.initial_ball_pos[..., 0]
            self._goal_pos[..., 1] = torch.flatten(_goal_dist * torch.sin(_goal_rot)) + self.initial_ball_pos[..., 1]
        if not self.headless:
            goal_asset_opts = gymapi.AssetOptions()
            goal_asset_opts.fix_base_link = True
            goal_asset = self.gym.create_sphere(self.sim, 0.03, goal_asset_opts)
            init_goal_pos = gymapi.Transform()
            self.num_markers += 1
            asset.append(goal_asset)
            pos.append(init_goal_pos)
            if self.limit_space:

                for i in range(4):
                    bound_asset_opts = gymapi.AssetOptions()
                    bound_asset_opts.fix_base_link = True
                    bound_asset = self.gym.create_sphere(self.sim, 0.03, bound_asset_opts)
                    asset.append(bound_asset)
                    init_bound_pos = gymapi.Transform()
                    pos.append(init_bound_pos)
                    self.num_markers += 1
        return asset, pos

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._root_states[env_ids, 0:3] = root_pos
        self._root_states[env_ids, 3:7] = root_rot
        self._root_states[env_ids, 7:10] = root_vel
        self._root_states[env_ids, 10:13] = root_ang_vel
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        return

    def _set_actors_tensors(self, env_ids):
        actor_indices = self.all_actor_indices[env_ids, 0].flatten()
        reset_indices = self.all_actor_indices[env_ids].flatten()

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._all_actor_root_states),
                                                     gymtorch.unwrap_tensor(reset_indices), len(reset_indices))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
        return

    def _reset_default(self, env_ids):
        if self.dr_init_dof:
            self._dof_pos[env_ids] = self._initial_dof_pos[env_ids] * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        else:
            self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        self._root_states[env_ids] = self._initial_root_states[env_ids]
        self._reset_default_env_ids = env_ids
        return

    # remove this to use all motions
    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_refs(num_envs)
        
        if (self._state_init == A1AMP.StateInit.Random
            or self._state_init == A1AMP.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == A1AMP.StateInit.Start):
            motion_times = np.zeros(num_envs)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _compute_reward(self, actions):
        is_additive = self.cfg['task']['is_additive']
        rew_dict = compute_a1_reward(self._root_states,
                                            self._prev_root_states[:, :2],
                                            self._goal_pos,
                                            self._ball_root_states[:, :2],
                                            self._prev_ball_states[:, :2],
                                            self.dt,
                                            self.torques,
                                            self._dof_vel,
                                            self.actor_vel_scale,
                                            self.ball_vel_scale,
                                            self.energy_scale,
                                            self.energy_weight,
                                            self.ab_dist_threshold,
                                            self.piecewise,
                                            is_additive,
                                            )
        self.rew_buf[:] = rew_dict["total_rew"]
        return rew_dict

    def _compute_observations(self, env_ids=None):
        super()._compute_observations(env_ids)
        self._goal_xy += torch.rand_like(self._goal_xy) * self.noise_scale_vec[0]
        return

    def _compute_a1_obs_full_states(self, env_ids=None):
        if (env_ids is None):
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            goal_pos = self._goal_pos
            ball_states = self._ball_root_states
        else:
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
            goal_pos = self._goal_pos[env_ids]
            ball_states = self._ball_root_states[env_ids]

        obs = compute_a1_observations(root_states, dof_pos, dof_vel,
                                      key_body_pos, self._local_root_obs, goal_pos, ball_states)
        return obs

    def _compute_a1_obs_reduced_states(self, env_ids=None):
        if (env_ids is None):
            if self.add_delay:
                root_states = self.delayed_states[:, :7]
                root_quat = self.delayed_states[:, 3:7]
                dof_pos = self.delayed_states[:, 13:25]
            else:
                root_states = self._root_states
                root_quat = root_states[:, 3:7]
                dof_pos = self._dof_pos
            if self._local_root_obs:
                root_quat = compute_local_root_quat(root_quat)
            # root_quat = quat_to_tan_norm(root_quat)
            # goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            # goal_pos[..., 2] = 0.2
            goal_pos = self._goal_pos
            if self.add_ball_delay:
                ball_pos = self.delayed_ball_obs[:, 0, :]
                self.delayed_ball_obs[:] = self.delayed_ball_obs.roll(-1, 1)
                self.delayed_ball_obs[:, -1, :] = self._ball_root_states[:, :3]
            else:
                ball_pos = self._ball_root_states[:, :3]
            goal_xy, local_ball_pos = compute_goal_observations(root_states, goal_pos, ball_pos)
            # local_ball_pos[...] = 0.0
            # local_ball_pos[:, 0] = 3.0
            self._goal_xy[:] = goal_xy
            if self.priv_obs:
                self._priv_goal_xy[:] = goal_xy

        else:
            if self.add_delay:
                root_states = self.delayed_states[env_ids, :7]
                root_quat = self.delayed_states[env_ids, 3:7]
                dof_pos = self.delayed_states[env_ids, 13:25]
            else:
                root_states = self._root_states[env_ids]
                root_quat = root_states[:, 3:7]
                dof_pos = self._dof_pos[env_ids]
            if self._local_root_obs:
                root_quat = compute_local_root_quat(root_quat)
            # root_quat = quat_to_tan_norm(root_quat)
            # goal_pos = torch.zeros((len(env_ids), 3), dtype=torch.float, device=self.device)
            # goal_pos[..., 2] = 0.2
            goal_pos = self._goal_pos[env_ids]
            if self.add_ball_delay:
                ball_pos = self.delayed_ball_obs[env_ids, 0, :]
                self.delayed_ball_obs[env_ids] = self.delayed_ball_obs[env_ids].roll(-1, 1)
                self.delayed_ball_obs[env_ids, -1, :] = self._ball_root_states[env_ids, :3]
            else:
                ball_pos = self._ball_root_states[env_ids, :3]
            goal_xy, local_ball_pos = compute_goal_observations(root_states, goal_pos, ball_pos)
            # local_ball_pos[env_ids, :] = 0.0
            # local_ball_pos[env_ids, 0] = 3.0
            self._goal_xy[env_ids] = goal_xy
            if self.priv_obs:
                self._priv_goal_xy[env_ids] = goal_xy

        # TODO hardcode goal position
        # self._goal_xy[:, 0] = 3.0
        # self._goal_xy[:, 1] = 0.0
        # print('goal:', self._goal_xy[0])
        # print('ball:', local_ball_pos[0])
        ob_curr = torch.cat([root_quat, dof_pos, local_ball_pos], dim=-1)
        return ob_curr

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._reset_ball_pos(env_ids)
        self._reset_goal_pos(env_ids)
        self._set_actors_tensors(env_ids)
        if self.domain_rand and self.dr_pd:
            self._reset_pd_gains(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        if self.history_steps is not None:
            self._reset_obs(env_ids)
        self._reset_robot(env_ids)
        if self.domain_rand and self.dr_push_robot:
            self._reset_push(env_ids)

        self._init_amp_obs(env_ids)
        return

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure
        Args:
            cfg (Dict): Environment config file
        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_scales = self.cfg["task"]["noise"]["noise_scales"]
        noise_level = self.cfg["task"]["noise"]["noise_level"]
        # TODO change for actual obs
        noise_vec = torch.zeros(NUM_CURR_OBS, dtype=torch.float, device=self.device)
        noise_vec[:6] = noise_scales["base_quat"] * noise_level
        noise_vec[6:18] = noise_scales["dof_pos"] * noise_level
        noise_vec[18:21] = noise_scales["ball_pos"] * noise_level
        self.noise_scale_dof_vel = noise_scales["dof_vel"] * noise_level
        return noise_vec

    def post_physics_step(self):
        self._pos_hist[:, 0, :] = self._prev_root_states[:, :3]
        self._pos_hist[:, 1, :] = self._root_states[:, :3]
        self.goal_step += 1
        self._prev_root_states[:] = self._root_states[:]
        self._prev_ball_states[:] = self._ball_root_states[:]
        self._ball_hist[:, 0, :] = self._prev_ball_states[:, :3]
        self._ball_hist[:, 1, :] = self._ball_root_states[:, :3]
        rew_dict = super().post_physics_step()
        self._pos_hist[:, 2, :] = self._root_states[:, :3]
        self._ball_hist[:, 2, :] = self._ball_root_states[:, :3]
        # states_idx = NUM_CURR_OBS * (self.history_steps + 1)
        # print('goal: ', self.obs_buf[:, states_idx:states_idx + 2])
        # print('xy: ', self._goal_xy)
        # self.extras["success"] = self._success_buf

        # reset goal pos
        if self.rand_goal:
            goal_reset_envs = (self.goal_step >= self.goal_terminate).nonzero(as_tuple=False).flatten()
            if len(goal_reset_envs) > 0:
                # print('reset encountered!')
                self._reset_goal_pos(goal_reset_envs, set_goal=True)
        return rew_dict

    def _reset_ball_pos(self, env_ids):
        if self.limit_space:
            self._reset_limited_ball_pos(env_ids)
        else:
            ball_dist = torch.rand((len(env_ids), 1), dtype=torch.float, device=self.device) * 5.0
            ball_rot = torch.rand((len(env_ids), 1), dtype=torch.float, device=self.device) * torch.pi * 2
            self.initial_ball_pos[env_ids, 0] = torch.flatten(ball_dist * torch.cos(ball_rot)) + self._root_states[env_ids, 0]
            self.initial_ball_pos[env_ids, 1] = torch.flatten(ball_dist * torch.sin(ball_rot)) + self._root_states[env_ids, 1]
            self.initial_ball_pos[env_ids, 2] = self.height
        # self.initial_ball_pos[env_ids, 0] = 2.4
        # self.initial_ball_pos[env_ids, 1] = 0.6
        # u = torch.rand((len(env_ids), 1), dtype=torch.float, device=self.device)
        # v = torch.rand((len(env_ids), 1), dtype=torch.float, device=self.device)
        # w = torch.rand((len(env_ids), 1), dtype=torch.float, device=self.device)
        # self.initial_ball_pos[env_ids, 3] = torch.flatten(torch.sqrt_(1. - u) * torch.sin(v * torch.pi * 2))
        # self.initial_ball_pos[env_ids, 4] = torch.flatten(torch.sqrt_(1. - u) * torch.cos(v * torch.pi * 2))
        # self.initial_ball_pos[env_ids, 5] = torch.flatten(torch.sqrt_(u) * torch.sin(w * torch.pi * 2))
        # self.initial_ball_pos[env_ids, 6] = torch.flatten(torch.sqrt_(u) * torch.cos(w * torch.pi * 2))
        self._ball_root_states[env_ids, :7] = self.initial_ball_pos[env_ids, :]
        self._ball_root_states[env_ids, 7:] = 0.
        if self.add_ball_delay:
            self.delayed_ball_obs[env_ids, ...] = self.initial_ball_pos[env_ids, :3].unsqueeze(1)
        return

    def _reset_goal_pos(self, goal_reset_envs, set_goal=False):

        self.goal_terminate[goal_reset_envs] = torch.randint(int(self.max_episode_length//self.goal_reset), int(self.max_episode_length//self.goal_reset_upper_inv), (len(goal_reset_envs),), device=self.device,
                                                                dtype=torch.int32)
        self.goal_step[goal_reset_envs] = 0
        if self.limit_space:
            self._reset_limited_goal_pos(goal_reset_envs)
        else:
            goal_dist = torch.rand((len(goal_reset_envs), 1), dtype=torch.float, device=self.device) * init_goal_dist
            goal_rot = torch.rand((len(goal_reset_envs), 1), dtype=torch.float, device=self.device) * torch.pi * 2
            self._goal_pos[goal_reset_envs, 0] = torch.flatten(goal_dist * torch.cos(goal_rot)) + self.initial_ball_pos[goal_reset_envs, 0]
            self._goal_pos[goal_reset_envs, 1] = torch.flatten(goal_dist * torch.sin(goal_rot)) + self.initial_ball_pos[goal_reset_envs, 1]
        # self._goal_pos[goal_reset_envs, 0] = 3.6
        # self._goal_pos[goal_reset_envs, 1] = 0.0
        if not self.headless:
            self._goal_root_states[goal_reset_envs, :3] = self._goal_pos[goal_reset_envs]
            if set_goal:
                actor_indices = self.all_actor_indices[goal_reset_envs, 2].flatten()

                self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._all_actor_root_states),
                                                             gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
        return
    
    def _reset_limited_ball_pos(self, env_ids=None):
        if env_ids == None:
            self.initial_ball_pos[..., 0] = torch.flatten(torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device) * 2 - 1) * (self.space_length / 2 - 0.7)
            self.initial_ball_pos[..., 1] = torch.flatten(torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device) * 2 - 1) * (self.space_width / 2 - 0.7)
            self.initial_ball_pos[..., 2] = self.height
        else:
            self.initial_ball_pos[env_ids, 0] = torch.flatten(torch.rand((len(env_ids), 1), dtype=torch.float, device=self.device) * 2 - 1) * (self.space_length / 2 - 0.7)
            self.initial_ball_pos[env_ids, 1] = torch.flatten(torch.rand((len(env_ids), 1), dtype=torch.float, device=self.device) * 2 - 1) * (self.space_width / 2 - 0.7)
            self.initial_ball_pos[env_ids, 2] = self.height
        return

    def _reset_limited_goal_pos(self, env_ids=None):
        if env_ids == None:
            self._goal_pos[..., 0] = torch.flatten(torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device) * 2 - 1) * (self.space_length / 2 - 0.7)
            self._goal_pos[..., 1] = torch.flatten(torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device) * 2 - 1) * (self.space_width / 2 - 0.7)
        else:
            self._goal_pos[env_ids, 0] = torch.flatten(torch.rand((len(env_ids), 1), dtype=torch.float, device=self.device) * 2 - 1) * (self.space_length / 2 - 0.7)
            self._goal_pos[env_ids, 1] = torch.flatten(torch.rand((len(env_ids), 1), dtype=torch.float, device=self.device) * 2 - 1) * (self.space_width / 2 - 0.7)
        return

    def _compute_observations(self, env_ids=None):
        super()._compute_observations(env_ids)
        if env_ids is None:
            self._goal_xy += torch.rand_like(self._goal_xy) * 0.05
        else:
            self._goal_xy[env_ids] += torch.rand_like(self._goal_xy[env_ids]) * 0.05

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_a1_reset(self.reset_buf, self.progress_buf,
                                                                     self._contact_forces, self._contact_body_ids,
                                                                     self._rigid_body_pos, self.max_episode_length,
                                                                     self._enable_early_termination,
                                                                     self._termination_height,
                                                                     self._goal_pos[:, :2], self._ball_root_states[:, :3], init_goal_dist, 
                                                                     self._pos_hist, self._ball_hist,
                                                                     self._root_states[:, :2],
                                                                     self.limit_space, self.space_width, self.space_length)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    dof_obs_size = 12
    dof_offsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs


@torch.jit.script
def compute_a1_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs, goal_pos, ball_states):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]
    ball_pos = ball_states[:, 0:3]
    ball_rot = ball_states[:, 3:7]
    ball_vel = ball_states[:, 7:10]
    ball_ang_vel = ball_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand_for_key_body = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
                                           local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand_for_key_body.view(heading_rot_expand_for_key_body.shape[0] * heading_rot_expand_for_key_body.shape[1],
                                               heading_rot_expand_for_key_body.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0],
                                            local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    local_goal_pos = goal_pos - root_pos
    local_target_pos = my_quat_rotate(heading_rot, local_goal_pos)
    flat_local_goal_xy = local_target_pos[:, :2]

    # ground truth obs for the ball
    local_ball_pos = ball_pos - root_pos
    local_ball_pos = my_quat_rotate(heading_rot, local_ball_pos)

    local_ball_rot = quat_mul(heading_rot, ball_rot)
    ball_rot_obs = quat_to_tan_norm(local_ball_rot)

    local_ball_vel = my_quat_rotate(heading_rot, ball_vel)

    local_ball_ang_vel = my_quat_rotate(heading_rot, ball_ang_vel)

    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos,
                     flat_local_goal_xy, local_ball_pos[:, :2], ball_pos[:, 2:3], ball_rot_obs, local_ball_vel, local_ball_ang_vel),
                    dim=-1)
    return obs


@torch.jit.script
def compute_a1_reward(
    root_states, prev_root_xy, goal_xy, ball_xy, prev_ball_xy, dt, torque, 
    dof_vel, actor_vel_scale, ball_vel_scale, energy_scale, energy_weight, ab_dist_threshold, piecewise, additive):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, float, float, bool, bool) -> Dict[str, Tensor]
    
    root_xy = root_states[:, :2]
    # energy saving reward
    energy_sum = torch.sum(torch.square(torque * dof_vel), dim=1)
    energy_reward = torch.exp(- energy_scale * energy_sum)

    v_char = torch.zeros_like(root_states[:, :3])
    v1_char = (root_xy[:, 0] - prev_root_xy[:, 0]) / dt
    v2_char = (root_xy[:, 1] - prev_root_xy[:, 1]) / dt
    v1_ball = (ball_xy[:, 0] - prev_ball_xy[:, 0]) / dt
    v2_ball = (ball_xy[:, 1] - prev_ball_xy[:, 1]) / dt

    diff_b = ball_xy - root_xy
    dist_b = diff_b[:, 0] ** 2 + diff_b[:, 1] ** 2
    d_ball = diff_b / torch.sqrt_(dist_b).reshape(-1, 1)

    diff = goal_xy[:, :2] - ball_xy
    dist = diff[:, 0] ** 2 + diff[:, 1] ** 2
    d = diff / torch.sqrt_(dist).reshape(-1, 1)

    ball_vel = d[:, 0] * v1_ball + d[:, 1] * v2_ball
    actor_vel = d_ball[:, 0] * v1_char + d_ball[:, 1] * v2_char

    ball_vel_static = tolerance(ball_vel, 0., 0., 0.05)
    ball_vel_move = tolerance(ball_vel, 0.5, 1., 0.2)
    actor_vel_static = tolerance(actor_vel, 0., 0., 0.1)
    actor_vel_move = tolerance(actor_vel, 0.5, 1., 1.)

    if additive:
        ball_vel_static = torch.where(dist<0.2, ball_vel_static, torch.zeros_like(ball_vel_static))
        ball_vel_move = torch.where(dist<0.2, torch.ones_like(ball_vel_move), ball_vel_move)
        actor_vel_static = torch.where(dist<0.2, torch.ones_like(actor_vel_static), torch.where(dist_b<0.2, actor_vel_static, torch.zeros_like(actor_vel_static)))
        actor_vel_move = torch.where(dist<0.2, torch.ones_like(actor_vel_move), torch.where(dist_b<0.2, torch.ones_like(actor_vel_move), actor_vel_move))
        reward = 0.1 * actor_vel_static + 0.1 * actor_vel_move + 0.4 * ball_vel_static + 0.4 * ball_vel_move
        total_reward = (1 - energy_weight) * reward + energy_weight * energy_reward
        rewards = {
        "actor_dist/static_rew": actor_vel_static,
        "actor_move_rew": actor_vel_move,
        "ball_dist/static_rew": ball_vel_static,
        "ball_move_rew": ball_vel_move,
        "total_rew": total_reward,
        "energy_rew": energy_reward
        }

    else:
        # using distance and velocity reward
        actor_dist_reward = torch.exp(- dist_b ** 2 * 0.5)

        actor_vel_reward = torch.exp(
            - actor_vel_scale * torch.maximum(torch.zeros_like(v1_char),
                            0.5 - actor_vel) ** 2)

        dist_reward = torch.exp(- dist ** 2 * 0.5)

        ball_vel_reward = torch.exp(
            - ball_vel_scale * torch.maximum(torch.zeros_like(v1_ball),
                            0.5 - ball_vel) ** 2)

        if piecewise:
            ball_vel_reward = torch.where(dist_b > ab_dist_threshold, torch.zeros_like(ball_vel_reward), ball_vel_reward)
            dist_reward = torch.where(dist_b > ab_dist_threshold, torch.zeros_like(dist_reward), dist_reward)

        reward = 0.1 * actor_vel_reward + 0.1 * actor_dist_reward + 0.3 * ball_vel_reward + 0.5 * dist_reward
        total_reward = (1 - energy_weight) * reward + energy_weight * energy_reward
        total_reward = torch.where(dist < 0.2, torch.ones_like(total_reward), total_reward)
        rewards = {
        "actor_dist/static_rew": actor_dist_reward,
        "actor_move_rew": actor_vel_reward,
        "ball_dist/static_rew": dist_reward,
        "ball_move_rew": ball_vel_reward,
        "total_rew": total_reward,
        "energy_rew": energy_reward
        }

    # # override the reward to be the max if ball is close enough to ball
    # reward = torch.where(dist < 0.2, torch.ones_like(reward), reward)
    # total_reward = (1 - energy_weight) * reward + energy_weight * energy_reward

    # test printouts
    # print('total_rew: {}, rew:{}, dist_rew:{}, actor_dist_rew:{}, energy:{}, actor_vel:{}, ball_vel:{}'.format(
    #     total_reward, reward, dist_reward, actor_dist_reward, energy_reward, actor_vel_reward, ball_vel_reward))



    return rewards


@torch.jit.script
def compute_a1_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                     max_episode_length, enable_early_termination, termination_height, goal, ball_pos, goal_dist,
                     pos_hist, ball_hist, robot_xy,
                     limit_space, space_w, space_l):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float, Tensor, Tensor, float, Tensor, Tensor, Tensor, bool, float, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    # ball = ball_pos[:, :2]
    # h = ball_pos[:, 2]
    # bg_dist = (goal[:, 0] - ball[:, 0]) ** 2 + (goal[:, 1] - ball[:, 1]) ** 2
    # success = torch.zeros_like(reset_buf)
    # success = torch.where((bg_dist < 0.05) & (h <= 0.3), torch.ones_like(reset_buf), success)
    # p1 = pos_hist[:, 0, :].squeeze(1)
    # p2 = pos_hist[:, 1, :].squeeze(1)
    # p3 = pos_hist[:, 2, :].squeeze(1)
    # b1 = ball_hist[:, 0, :].squeeze(1)
    # b2 = ball_hist[:, 1, :].squeeze(1)
    # b3 = ball_hist[:, 2, :].squeeze(1)
    # d1 = (p1[:, 0] - b1[:, 0]) ** 2 + (p1[:, 1] - b1[:, 1]) ** 2 + (p1[:, 2] - b1[:, 2]) ** 2
    # d2 = (p2[:, 0] - b2[:, 0]) ** 2 + (p2[:, 1] - b2[:, 1]) ** 2 + (p2[:, 2] - b2[:, 2]) ** 2
    # d3 = (p3[:, 0] - b3[:, 0]) ** 2 + (p3[:, 1] - b3[:, 1]) ** 2 + (p3[:, 2] - b3[:, 2]) ** 2
    # v1 = (p2[:, 0] - p1[:, 0]) ** 2 + (p2[:, 1] - p1[:, 1]) ** 2 + (p2[:, 2] - p1[:, 2]) ** 2
    # v2 = (p3[:, 0] - p2[:, 0]) ** 2 + (p3[:, 1] - p2[:, 1]) ** 2 + (p3[:, 2] - p2[:, 2]) ** 2
    fail = torch.zeros_like(reset_buf)
    # fail = torch.where((bg_dist > goal_dist ** 2 + 2.0), torch.ones_like(reset_buf), fail)
    # fail = torch.where(((v1 < 1e-5) & (v2 < 1e-5)), torch.ones_like(reset_buf), fail)
    # fail = torch.where(
    #     ((0.25 <= d1) & 
    #      (d1 <= d2) & 
    #      (d2 <= d3) & 
    #      (b1[:, 0] == b2[:, 0]) & 
    #      (b1[:, 1] == b2[:, 1]) & 
    #      (b1[:, 2] == b2[:, 2]) & 
    #      (b2[:, 0] == b3[:, 0]) & 
    #      (b2[:, 1] == b3[:, 1]) &
    #      (b2[:, 2] == b3[:, 2])), torch.ones_like(reset_buf), fail)

    ######### constrain space #########
    if limit_space:
        limit = torch.zeros_like(ball_pos[0, :2])
        limit[0], limit[1] = space_l, space_w
        limit /= 2
        limit -= 0.7
        out_of_space = (
            (robot_xy[:, 0] > limit[0]) | (robot_xy[:, 1] < - limit[0]) |
            (robot_xy[:, 1] > limit[1]) | (robot_xy[:, 1] < - limit[1]) |
            (ball_pos[:, 0] > limit[0]) | (ball_pos[:, 0] < - limit[0]) |
            (ball_pos[:, 1] > limit[1]) | (ball_pos[:, 1] < - limit[1])
            ) 
        fail = torch.where(out_of_space, torch.ones_like(reset_buf), fail)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
        # terminated = torch.where(success == 1., success, terminated)
        terminated = torch.where(fail == 1., fail, terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    # return reset, terminated, success
    return reset, terminated

@torch.jit.script
def compute_goal_observations(root_states, goal_pos, ball_pos):
    # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = calc_heading_quat_inv(root_rot)

    local_goal_pos = goal_pos - root_pos
    local_target_pos = my_quat_rotate(heading_rot, local_goal_pos)
    flat_local_goal_xy = local_target_pos[:, :2]

    local_ball_pos = ball_pos - root_pos
    local_ball_pos = my_quat_rotate(heading_rot, local_ball_pos)
    local_ball_pos[:, 2] = ball_pos[:, 2]

    return flat_local_goal_xy, local_ball_pos

@torch.jit.script
def compute_local_root_quat(root_rot):
    # type: (Tensor) -> Tensor
    heading_rot = calc_heading_quat_inv(root_rot)
    root_rot_obs = quat_mul(heading_rot, root_rot)
    root_rot_obs = quat_to_tan_norm(root_rot_obs)
    return root_rot_obs


