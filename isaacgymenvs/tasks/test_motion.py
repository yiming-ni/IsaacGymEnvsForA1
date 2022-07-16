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


class TestMotion(A1AMP):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.add_markers = self.cfg["env"].get("addMarkers", False)

        super().__init__(cfg, sim_device, graphics_device_id, headless)

    def _get_root_states_from_tensor(self, state_tensor):
        if self.add_markers:
            self.all_actor_indices = torch.arange(self.num_envs * (self.num_markers+1), dtype=torch.int32, device=self.device).reshape(
                (self.num_envs, (self.num_markers+1)))  # TODO root_states
            self._all_actor_root_states = gymtorch.wrap_tensor(state_tensor)
            self._root_states = self._all_actor_root_states.view(self.num_envs, self.num_markers+1, self.num_dof+1)[..., 0, :]
            self._marker_root_states = self._all_actor_root_states.view(self.num_envs, self.num_markers+1, self.num_dof+1)[..., 1:, :]
        else:
            super()._get_root_states_from_tensor(state_tensor)

        return

    def _create_marker_actors(self, env_ptr, marker_asset, init_marker_pos, i):
        if self.add_markers:
            marker_handle_0 = self.gym.create_actor(env_ptr, marker_asset, init_marker_pos, "marker-0", i, 1, 0)
            marker_handle_1 = self.gym.create_actor(env_ptr, marker_asset, init_marker_pos, "marker-1", i, 1, 0)
            marker_handle_2 = self.gym.create_actor(env_ptr, marker_asset, init_marker_pos, "marker-2", i, 1, 0)
            marker_handle_3 = self.gym.create_actor(env_ptr, marker_asset, init_marker_pos, "marker-3", i, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, marker_handle_0, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(1, 0, 0))  # red
            self.gym.set_rigid_body_color(env_ptr, marker_handle_1, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(0, 1, 0))  # green
            self.gym.set_rigid_body_color(env_ptr, marker_handle_2, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(0, 0, 1))  # blue
            self.gym.set_rigid_body_color(env_ptr, marker_handle_3, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          gymapi.Vec3(1, 1, 1))  # white
        else:
            super()._create_marker_actors(env_ptr, marker_asset, init_marker_pos, i)
        return

    def _create_marker_envs(self):
        if self.add_markers:
            marker_asset_options = gymapi.AssetOptions()
            marker_asset_options.angular_damping = 0.0
            marker_asset_options.max_angular_velocity = 4 * np.pi
            marker_asset_options.slices_per_cylinder = 16

            marker_asset_options.fix_base_link = True
            marker_asset = self.gym.create_sphere(self.sim, 0.02, marker_asset_options)
            init_marker_pos = gymapi.Transform()
            init_marker_pos.p.z = 0.3
            self.num_markers = 4
            return marker_asset, init_marker_pos
        else:
            super()._create_marker_envs()
            return 0, 0


    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._root_states[env_ids, 0:3] = root_pos
        self._root_states[env_ids, 3:7] = root_rot
        self._root_states[env_ids, 7:10] = root_vel
        self._root_states[env_ids, 10:13] = root_ang_vel

        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        if self.add_markers:
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

    def step(self, actions):

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.actions = action_tensor.to(self.device).clone()
        # step physics and render each frame
        self.render()



        curr_time = self.progress_buf[0].cpu().numpy() * self.dt % self._motion_lib.get_motion_length(0)
        # try:
        #     self.extras["prev_time"] = self.extras["curr_time"]
        # except Exception as e:
        #     self.extras['prev_time'] = 0
        # self.extras["curr_time"] = curr_time
        motion_id = np.array([0])
        # self.extras['current_trans'] = self.fetch_amp_obs_demo_time(motion_id, self.extras['curr_time'])
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = self._motion_lib.get_motion_state(motion_id, curr_time)
        # key_pos[..., 2] += 2

        self._root_states[..., :3] = root_pos
        # self._root_states[..., 2] += 2
        self._root_states[..., 3:7] = root_rot
        self._root_states[..., 7:10] = root_vel
        self._root_states[..., 10:13] = root_ang_vel

        self._dof_pos[..., :] = dof_pos
        self._dof_vel[..., :] = dof_vel
        if 'obs' in self.extras.keys():
            hist_obs = self.extras['obs']
        else:

            hist_obs = None

        self.extras['obs'] = build_amp_observations(self._root_states, self._dof_pos, self._dof_vel, key_pos, self._local_root_obs)
        if hist_obs is not None:
            self.extras['hist_obs'] = hist_obs
        else:
            self.extras['hist_obs'] = self.extras['obs']

        if self.add_markers:
            self._marker_root_states[..., 0, :3] = key_pos[0][0]
            self._marker_root_states[..., 1, :3] = key_pos[0][1]
            self._marker_root_states[..., 2, :3] = key_pos[0][2]
            self._marker_root_states[..., 3, :3] = key_pos[0][3]

            env_ids = torch.arange(0, self.num_envs, device=self.device)

            actor_indices = self.all_actor_indices[env_ids, 0]
            all_actor_ids = self.all_actor_indices[:, :].flatten()
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._all_actor_root_states),
                                                 gymtorch.unwrap_tensor(all_actor_ids), len(all_actor_ids))
            self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                 gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
        else:
            env_ids = torch.arange(0, self.num_envs, device=self.device, dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                         gymtorch.unwrap_tensor(env_ids), len(env_ids))
            self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                  gymtorch.unwrap_tensor(env_ids), len(env_ids))

        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        # self.gym.refresh_dof_state_tensor(self.sim)

        self.timeout_buf = torch.where(self.progress_buf >= self.max_episode_length - 1,
                                       torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))

        self.post_physics_step()

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_amp_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
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
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
                                           local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0],
                                            local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos),
                    dim=-1)
    return obs
