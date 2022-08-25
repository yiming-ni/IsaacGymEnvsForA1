from enum import Enum
import numpy as np
import torch
import os

from gym import spaces

from isaacgym import gymapi
from isaacgym import gymtorch

from .amp.a1_base import A1Base, dof_to_obs
from .a1_amp import A1AMP
from .amp.utils_amp import gym_util
from .amp.utils_amp.motion_lib import A1MotionLib

from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *


NUM_OBS = 1 + 6 + 3 + 3 + 12 + 12 + 4*3 + 2
NUM_CURR_OBS = 18

class A1Navigation(A1AMP):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        super().__init__(cfg, sim_device, graphics_device_id, headless)
        # track goal progress
        self.goal_terminate = torch.randint(self.max_episode_length//2, self.max_episode_length, (self.num_envs,), device=self.device, dtype=torch.int32)
        self.goal_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    def get_obs_size(self):
        ob_size = super().get_obs_size()
        return ob_size + 2

    def _init_obs_tensors(self):
        if self.history_steps is not None:
            states_idx = NUM_CURR_OBS * (self.history_steps + 1)
            self._states_history = self.obs_buf[:, :states_idx].view(self.num_envs, self.history_steps + 1,
                                                                     -1)  # last entry is current obs
            self._goal_xy = self.obs_buf[:, states_idx:states_idx+2]
            self._actions_history = self.obs_buf[:, states_idx+2:].view(self.num_envs, self.history_steps, -1)
        return

    def _get_rigid_body_states_from_tensor(self, states_tensor):
        if not self.headless:
            self._all_actor_rb_states = gymtorch.wrap_tensor(states_tensor)
            self._all_actor_rb_states = self._all_actor_rb_states.view(self.num_envs, self.num_bodies+self.num_markers, -1)
            self.rb_states = self._all_actor_rb_states[..., :self.num_bodies, :]
            self.marker_states = self._all_actor_rb_states[..., self.num_bodies:, :]
        else:
            super()._get_rigid_body_states_from_tensor(states_tensor)
        return

    def _push_robots(self):
        """ Randomly pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        if self.headless:
            super()._push_robots()
        else:
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

            actor_forces[~mask] = self.perturbation_forces[~mask].expand(-1, self.rb_states.shape[1], -1)
            actor_torques[~mask] = self.perturbation_torques[~mask].expand(-1, self.rb_states.shape[1], -1)
            self.perturbation_time[~mask] -= self.dt

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces),
                                                    gymtorch.unwrap_tensor(torques), gymapi.GLOBAL_SPACE)
        return

    def _get_root_states_from_tensor(self, state_tensor):
        if not self.headless:
            self.all_actor_indices = torch.arange(self.num_envs * (self.num_markers + 1), dtype=torch.int32,
                                                  device=self.device).reshape(
                (self.num_envs, (self.num_markers + 1))
            )
            self._all_actor_root_states = gymtorch.wrap_tensor(state_tensor)
            self._root_states = self._all_actor_root_states.view(self.num_envs, self.num_markers + 1, self.num_dof + 1)[
                                ..., 0, :]
            self._goal_root_states = self._all_actor_root_states.view(self.num_envs, self.num_markers + 1,
                                                                      self.num_dof + 1)[..., 1, :]
        else:
            super()._get_root_states_from_tensor(state_tensor)

        return

    def _create_marker_actors(self, env_ptr, marker_asset, init_goal_pos, i):
        if not self.headless:

            init_goal_pos.p.x = self._goal_pos[i, 0]
            init_goal_pos.p.y = self._goal_pos[i, 1]
            init_goal_pos.p.z = 0.2
            goal_handle = self.gym.create_actor(env_ptr, marker_asset, init_goal_pos, "goal", i, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(1, 0, 0))
        else:
            super()._create_marker_actors(env_ptr, marker_asset, init_goal_pos, i)
        return

    def _create_marker_envs(self, asset_opts):
        _goal_dist = torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device) * 5.0
        _goal_rot = torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device) * torch.pi * 2
        self._goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self._goal_pos[..., 0] = torch.flatten(_goal_dist * torch.cos(_goal_rot))
        self._goal_pos[..., 1] = torch.flatten(_goal_dist * torch.sin(_goal_rot))
        # self._goal_pos[..., 2] = 0.1
        if not self.headless:
            goal_asset_opts = gymapi.AssetOptions()
            goal_asset_opts.fix_base_link = True
            goal_asset = self.gym.create_sphere(self.sim, 0.03, goal_asset_opts)

            # if "asset" in self.cfg["env"]:
            #     asset_file = self.cfg["env"]["asset"]["ballAsset"]
            # asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
            # goal_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_opts)

            init_goal_pos = gymapi.Transform()
            self.num_markers = 1
            return goal_asset, init_goal_pos
        else:

            super()._create_marker_envs(asset_opts)
            return 0, 0


    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._root_states[env_ids, 0:3] = root_pos
        self._root_states[env_ids, 3:7] = root_rot
        self._root_states[env_ids, 7:10] = root_vel
        self._root_states[env_ids, 10:13] = root_ang_vel

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        if not self.headless:
            actor_indices = self.all_actor_indices[env_ids, 0].flatten()
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._all_actor_root_states),
                                                         gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
            self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                  gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
        else:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                         gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_a1_reward(self._root_states[:, :2], self._prev_root_states[:, :2], self._goal_pos, self.dt, self.device)
        return

    def _compute_a1_obs_full_states(self, env_ids=None):
        if (env_ids is None):
            # goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            # goal_pos[..., 2] = 0.2
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            goal_pos = self._goal_pos
        else:
            # goal_pos = torch.zeros((len(env_ids), 3), dtype=torch.float, device=self.device)
            # goal_pos[..., 2] = 0.2
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
            goal_pos = self._goal_pos[env_ids]

        obs = compute_a1_observations(root_states, dof_pos, dof_vel,
                                      key_body_pos, self._local_root_obs, goal_pos)
        return obs

    def _compute_a1_obs_reduced_states(self, env_ids=None):
        if (env_ids is None):
            root_states = self._root_states
            root_quat = root_states[:, 3:7]
            dof_pos = self._dof_pos
            if self._local_root_obs:
                root_quat = compute_local_root_quat(root_quat)
            root_rot_obs = quat_to_tan_norm(root_quat)
            # goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            # goal_pos[..., 2] = 0.2
            goal_pos = self._goal_pos
            goal_xy = compute_goal_observations(root_states, goal_pos)
            self._goal_xy[:] = goal_xy

        else:
            root_states = self._root_states[env_ids]
            root_quat = root_states[:, 3:7]
            dof_pos = self._dof_pos[env_ids]
            if self._local_root_obs:
                root_quat = compute_local_root_quat(root_quat)
            root_rot_obs = quat_to_tan_norm(root_quat)
            # goal_pos = torch.zeros((len(env_ids), 3), dtype=torch.float, device=self.device)
            # goal_pos[..., 2] = 0.2
            goal_pos = self._goal_pos[env_ids]
            goal_xy = compute_goal_observations(root_states, goal_pos)
            self._goal_xy[env_ids] = goal_xy

        ob_curr = torch.cat([root_rot_obs, dof_pos], dim=-1)
        return ob_curr

    def post_physics_step(self):
        self.goal_step += 1
        self._prev_root_states[:] = self._root_states[:]
        super().post_physics_step()
        # reset goal pos
        goal_reset_envs = (self.goal_step >= self.goal_terminate).nonzero(as_tuple=False).flatten()
        if len(goal_reset_envs) > 0:
            # print('reset encountered!')
            self._reset_goal_pos(goal_reset_envs)
        return

    def _reset_goal_pos(self, goal_reset_envs):
        self.goal_terminate[goal_reset_envs] = torch.randint(self.max_episode_length//2, self.max_episode_length, (len(goal_reset_envs),), device=self.device,
                                                             dtype=torch.int32)
        self.goal_step[goal_reset_envs] = 0
        goal_dist = torch.rand((len(goal_reset_envs), 1), dtype=torch.float, device=self.device) * 5.0
        goal_rot = torch.rand((len(goal_reset_envs), 1), dtype=torch.float, device=self.device) * torch.pi * 2
        self._goal_pos[goal_reset_envs, 0] = torch.flatten(goal_dist * torch.cos(goal_rot))
        self._goal_pos[goal_reset_envs, 1] = torch.flatten(goal_dist * torch.sin(goal_rot))
        if not self.headless:
            # goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            # goal_pos[goal_reset_envs, :2] = self._goal_pos[goal_reset_envs]
            # goal_pos[goal_reset_envs, 2] = 0.2
            actor_indices = self.all_actor_indices[goal_reset_envs, 1].flatten()
            self._goal_root_states[goal_reset_envs, :3] = self._goal_pos[goal_reset_envs]
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._all_actor_root_states),
                                                         gymtorch.unwrap_tensor(actor_indices), len(actor_indices))

    def reset_idx(self, env_ids):
        self._reset_goal_pos(env_ids)
        super().reset_idx(env_ids)
        return

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)
        if not self.headless:
            body_ids.append(self.num_bodies)  # the last rigid body

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids



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
def compute_a1_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs, goal_pos):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

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

    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos,
                     flat_local_goal_xy),
                    dim=-1)
    return obs


@torch.jit.script
def compute_goal_observations(root_states, goal_pos):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = calc_heading_quat_inv(root_rot)

    local_goal_pos = goal_pos - root_pos
    local_target_pos = my_quat_rotate(heading_rot, local_goal_pos)
    flat_local_goal_xy = local_target_pos[:, :2]

    return flat_local_goal_xy

@torch.jit.script
def compute_local_root_quat(root_rot):
    # type: (Tensor) -> Tensor
    heading_rot = calc_heading_quat_inv(root_rot)
    root_rot_obs = quat_mul(heading_rot, root_rot)
    root_rot_obs = quat_to_tan_norm(root_rot_obs)
    return root_rot_obs


@torch.jit.script
def compute_a1_reward(root_xy, prev_root_xy, goal_xy, dt, device):
    # type: (Tensor, Tensor, Tensor, float, Optional[Device]) -> Tensor

    x_diff = goal_xy[:, 0] - root_xy[:, 0]
    y_diff = goal_xy[:, 1] - root_xy[:, 1]
    dist = x_diff * x_diff + y_diff * y_diff
    dist_reward = torch.exp(- dist * 0.5)

    v1 = (root_xy[:, 0] - prev_root_xy[:, 0]) / dt
    v2 = (root_xy[:, 1] - prev_root_xy[:, 1]) / dt
    d_len = torch.sqrt_(x_diff * x_diff + y_diff * y_diff)
    d1 = x_diff / d_len
    d2 = y_diff / d_len
    vel_reward = torch.exp(- torch.maximum(torch.zeros_like(v1, dtype=torch.float, device=device), 1.0 - (d1 * v1 + d2 * v2)) ** 2)
    vel_reward = torch.where(dist > 0.04, vel_reward, torch.ones_like(vel_reward, dtype=torch.float, device=device))
    reward = 0.7 * dist_reward + 0.3 * vel_reward
    # print("reward: ", reward)
    # print("dist_reward: {}, \nvel_reward: {}".format(dist_reward, vel_reward))
    return reward


