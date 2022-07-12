# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import os
import torch
import random

from isaacgym import gymtorch
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from ..base.vec_task import VecTask
from ..base.base_utils.action_filter import ActionFilterButter
from ..base.observation_buffer import ObservationBuffer
# from tensorboardX import SummaryWriter

DOF_BODY_IDS = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
DOF_OFFSETS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
NUM_OBS = 1 + 6 + 3 + 3 + 12 + 12 + 4*3 + 2  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, local_goal_xy]
# base_height, base_orientation=4, base_angular_vel=3, joint_pos=12, joint_velocity=12
# orientation, joint_pos, 4+12+history
NUM_ACTIONS = 12

# action 30hz
# for i in range(int(2000/30)):
#    torque = pgain*(desied_jointpos - actual_jointpois) + dgain*(0-actual jointvel)
#    sim.step(torque) # check hz
#    actual joint pos, actual joint vel = update_data
# update obs

KEY_BODY_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

class A1Base(VecTask):

    def __init__(self, config, sim_device, graphics_device_id, headless):

        self.cfg = config

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless)

        self.history_steps = self.cfg["env"].get("historySteps", None)
        if self.history_steps is not None:
            self.obs_buf_history = ObservationBuffer(self.num_envs, self.num_obs, self.history_steps, self.device)

        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 4
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.add_markers:
            self.all_actor_indices = torch.arange(self.num_envs * (self.num_markers+1), dtype=torch.int32, device=self.device).reshape(
                (self.num_envs, (self.num_markers+1)))  # TODO root_states
            self._all_actor_root_states = gymtorch.wrap_tensor(actor_root_state)
            self._root_states = self._all_actor_root_states.view(self.num_envs, self.num_markers+1, self.num_dof+1)[..., 0, :]
            self._marker_root_states = self._all_actor_root_states.view(self.num_envs, self.num_markers+1, self.num_dof+1)[..., 1:, :]
        elif not self.headless:
            self.all_actor_indices = torch.arange(self.num_envs * (self.num_markers+1), dtype=torch.int32, device=self.device).reshape(
                (self.num_envs, (self.num_markers+1))
            )
            self._all_actor_root_states = gymtorch.wrap_tensor(actor_root_state)
            self._root_states = self._all_actor_root_states.view(self.num_envs, self.num_markers+1, self.num_dof+1)[..., 0, :]
            self._goal_root_states = self._all_actor_root_states.view(self.num_envs, self.num_markers+1, self.num_dof+1)[..., 1, :]
        else:
            self._root_states = gymtorch.wrap_tensor(actor_root_state)

        # self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._initial_root_states = self._root_states.clone()
        self._initial_root_states[:] = 0
        self._initial_root_states[..., 2] = 0.35
        self._initial_root_states[..., 6] = 1
        self._prev_root_states = self._root_states.clone()

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self._dof_vel)

        for i in range(self.num_dof):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg["control"]["stiffness"].keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg["control"]["stiffness"][dof_name]
                    self.d_gains[i] = self.cfg["control"]["damping"][dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg["control"]["control_type"] in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        initial_dof = np.array([0, 0.9, -1.8] * 4)
        initial_dof = torch.tensor(initial_dof, device=self.device, dtype=torch.float)
        self._initial_dof_pos[..., :] = initial_dof

        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies+self.num_markers, 13)[..., 0:3]   #TODO remove num_markers
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies+self.num_markers, 13)[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies+self.num_markers, 13)[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies+self.num_markers, 13)[..., 10:13]
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.num_bodies+self.num_markers, 3)

        self.action_filter = ActionFilterButter(lowcut=None, highcut=[4], sampling_rate=1. / self.dt, order=2,
                                                num_joints=12, device=self.device, num_envs=self.num_envs)

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        # track goal progress
        self.goal_terminate = torch.randint(100, 200, (self.num_envs,), device=self.device, dtype=torch.int32)
        self.goal_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        if self.viewer != None:
            self._init_camera()

        if self._pd_control:
            self._build_pd_action_offset_scale()

        # env_ids = torch.arange(0, self.num_envs, device=self.device)
        # self.action_filter.reset(env_ids, self._pd_target_to_action(initial_dof.expand(self.num_envs, -1)))

        return

    def _get_root_states_from_tensor(self, states_tensor):
        self._root_states = gymtorch.wrap_tensor(states_tensor)
        return

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

    def create_sim(self):
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        return

    def _process_dof_props(self, props, env_id):
        """store, change, randomize the DOF properties of each env.
           called during env creation.
           stores position, velocity and torques limits defined in the urdf

           returns: [numpy.array]: modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                props["friction"][i] = 0.2
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]

                self.dof_pos_limits[i, 0] = m - 0.5 * r * \
                                            self.cfg["task"]["randomization_params"]["actor_params"]["a1"][
                                                "dof_properties"]["soft_dof_pos_limit"]
                self.dof_pos_limits[i, 1] = m + 0.5 * r * \
                                            self.cfg["task"]["randomization_params"]["actor_params"]["a1"][
                                                "dof_properties"]["soft_dof_pos_limit"]
                self.torque_limits[i] *= \
                    self.cfg["task"]["randomization_params"]["actor_params"]["a1"]["dof_properties"][
                        "soft_dof_torque_limit"]
        return props

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._reset_goal_pos(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        # self._reset_robot(env_ids)
        return

    def set_char_color(self, col):
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            handle = self.a1_handles[i]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                              gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        # plane_params.distance = 2  #TODO only for testing ref
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        asset_file = "urdf/a1_ig.urdf"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()

        asset_options.default_dof_drive_mode = self.cfg["asset"]["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = self.cfg["asset"]["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = self.cfg["asset"]["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = self.cfg["asset"]["flip_visual_attachments"]
        asset_options.fix_base_link = self.cfg["asset"]["fix_base_link"]
        asset_options.density = self.cfg["asset"]["density"]
        asset_options.angular_damping = self.cfg["asset"]["angular_damping"]
        asset_options.linear_damping = self.cfg["asset"]["linear_damping"]
        asset_options.max_angular_velocity = self.cfg["asset"]["max_angular_velocity"]
        asset_options.max_linear_velocity = self.cfg["asset"]["max_linear_velocity"]
        asset_options.armature = self.cfg["asset"]["armature"]
        asset_options.thickness = self.cfg["asset"]["thickness"]
        asset_options.disable_gravity = self.cfg["asset"]["disable_gravity"]

        _goal_dist = torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device) * 4.0 + 1.0
        _goal_rot = torch.rand((self.num_envs, 1), dtype=torch.float, device=self.device) * torch.pi * 2
        self._goal_pos = torch.zeros((self.num_envs, 2), dtype=torch.float, device=self.device)
        self._goal_pos[..., 0] = torch.flatten(_goal_dist * torch.cos(_goal_rot))
        self._goal_pos[..., 1] = torch.flatten(_goal_dist * torch.sin(_goal_rot))
        # create marker options if add_markers is true
        if self.add_markers:
            marker_asset_options = gymapi.AssetOptions()
            marker_asset_options.angular_damping = 0.0
            marker_asset_options.max_angular_velocity = 4 * np.pi
            marker_asset_options.slices_per_cylinder = 16

            marker_asset_options.fix_base_link = True
            marker_asset = self.gym.create_sphere(self.sim, 0.03, marker_asset_options)
            init_marker_pos = gymapi.Transform()
            init_marker_pos.p.z = 0.3
            self.num_markers = 4
        elif not self.headless:
            goal_asset_opts = gymapi.AssetOptions()
            goal_asset_opts.fix_base_link = True
            goal_asset = self.gym.create_sphere(self.sim, 0.03, goal_asset_opts)
            init_goal_pos = gymapi.Transform()
            self.num_markers = 1  # TODO for goal pos
        else:
            self.num_markers = 0

        a1_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # actuator_props = self.gym.get_asset_actuator_properties(a1_asset)
        # motor_efforts = [prop.motor_effort for prop in actuator_props]
        # TODO above for actuator
        dof_props_asset = self.gym.get_asset_dof_properties(a1_asset)
        motor_efforts = [p.item() for p in dof_props_asset["effort"]]

        # create force sensors at the feet
        rr_foot_idx = self.gym.find_asset_rigid_body_index(a1_asset, "RR_foot")
        rl_foot_idx = self.gym.find_asset_rigid_body_index(a1_asset, "RL_foot")
        fr_foot_idx = self.gym.find_asset_rigid_body_index(a1_asset, "FR_foot")
        fl_foot_idx = self.gym.find_asset_rigid_body_index(a1_asset, "FL_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(a1_asset, rr_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(a1_asset, rl_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(a1_asset, fr_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(a1_asset, fl_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.body_dict = self.gym.get_asset_rigid_body_names(a1_asset)
        # ['base',
        # 'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot',
        # 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot',
        # 'RR_hip', 'RR_thigh', 'RR_calf', 'RR_foot',
        # 'RL_hip', 'RL_thigh', 'RL_calf', 'RL_foot']

        self.dof_names = self.gym.get_asset_dof_names(a1_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(a1_asset)
        self.num_dof = self.gym.get_asset_dof_count(a1_asset)
        self.num_joints = self.gym.get_asset_joint_count(a1_asset)

        # Below for humanoid
        # start_pose = gymapi.Transform()
        # start_pose.p = gymapi.Vec3(*get_axis_params(0.89, self.up_axis_idx))
        # start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        pos = [0.0, 0.0, 0.35]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        base_init_state_list = pos + rot + lin_vel + ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # self.get_env_origins()
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg["env"]["envSpacing"]
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w],
                                           device=self.device)

        self.a1_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # pos = self.env_origins[i].clone()
            # pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            # start_pose.p = gymapi.Vec3(*pos)

            contact_filter = 1
            handle = self.gym.create_actor(env_ptr, a1_asset, start_pose, "a1", i, contact_filter, 0)

            if not self.headless:

                init_goal_pos.p.x = self._goal_pos[i, 0]
                init_goal_pos.p.y = self._goal_pos[i, 1]
                init_goal_pos.p.z = 0.2
                goal_handle = self.gym.create_actor(env_ptr, goal_asset, init_goal_pos, "goal", i, 1, 0)
                self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))

            # create markers
            self._create_marker_actors(env_ptr, marker_asset, marker_pos, i)

            # TODO below for computing torque limits
            self._process_dof_props(dof_props_asset, i)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.549, 0.6863))

            self.envs.append(env_ptr)
            self.a1_handles.append(handle)

            if (self._pd_control):
                dof_prop = self.gym.get_asset_dof_properties(a1_asset)
                dof_prop["driveMode"] = gymapi.DOF_MODE_EFFORT  # TODO DOF_MODE_POS for actuator, EFFORT for dof
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)

        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        # [4, 8, 12, 16]
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)

        # if self._pd_control:
        #     self._build_pd_action_offset_scale()

        return

    def _create_marker_actors(self, env_ptr, marker_asset, init_marker_pos, i):
        """overridden by subclasses"""
        return

    def _create_marker_envs(self):
        self.num_markers = 0
        return self.num_markers, self.num_markers

    def _build_pd_action_offset_scale(self):
        num_joints = len(DOF_OFFSETS) - 1

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = DOF_OFFSETS[j]
            dof_size = DOF_OFFSETS[j + 1] - DOF_OFFSETS[j]

            if (dof_size == 3):
                lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] = curr_high

        self._pd_action_offset = self._initial_dof_pos[0, ...]
        lim_high = torch.tensor(lim_high, dtype=torch.float, device=self.device)
        lim_low = torch.tensor(lim_low, dtype=torch.float, device=self.device)
        self._pd_action_scale = torch.maximum(torch.abs(lim_high - self._pd_action_offset), torch.abs(lim_low - self._pd_action_offset))
        self._pd_action_scale = torch.clamp_max(self._pd_action_scale, torch.pi)

        return

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_a1_reward(self._root_states[:, :2], self._goal_pos)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_a1_reset(self.reset_buf, self.progress_buf,
                                                                     self._contact_forces, self._contact_body_ids,
                                                                     self._rigid_body_pos, self.max_episode_length,
                                                                     self._enable_early_termination,
                                                                     self._termination_height)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_a1_obs(env_ids)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_a1_obs(self, env_ids=None):
        if (env_ids is None):
            goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            goal_pos[..., 2] = 0.2
            root_states = self._root_states
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
            goal_pos[..., :2] = self._goal_pos
        else:
            goal_pos = torch.zeros((len(env_ids), 3), dtype=torch.float, device=self.device)
            goal_pos[..., 2] = 0.2
            root_states = self._root_states[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
            goal_pos[..., :2] = self._goal_pos[env_ids]

        obs = compute_a1_observations(root_states, dof_pos, dof_vel,
                                      key_body_pos, self._local_root_obs, goal_pos)

        return obs

    def _reset_actors(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def _reset_robot(self, env_ids):
        ref = torch.tensor([0.0, 0.90, -1.80, 0.0, 0.90, -1.80,
                            0.0, 0.90, -1.80, 0.0, 0.90, -1.80]).to(self.device)
        self.action_filter.reset(env_ids, self._pd_target_to_action(ref.expand(len(env_ids), -1)))

    def _compute_torques(self, pd_tar):
        """compute torques from actions.
           actions can be position or velocity targets given to a PD controller, or scaled torques.
           torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        returns: torques sent to the simulation
        """
        control_type = self.cfg["control"].get("control_type", "P")
        p_gains = self.p_gains
        d_gains = self.d_gains
        if control_type == "P":
            torques = p_gains * (pd_tar - self._dof_pos) - d_gains * self._dof_vel
        elif control_type == "V":
            torques = p_gains * (pd_tar - self._dof_vel) - d_gains * (
                    self._dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type == "T":
            torques = pd_tar
        else:
            raise NameError(f'Unknown controller type: {control_type}')
        # return torques
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def step(self, actions):
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.actions = action_tensor.to(self.device).clone()
        # reset action filter if it is the first step
        reset_env_ids = (self.progress_buf == 0).nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self.action_filter.reset(reset_env_ids, self.actions[reset_env_ids])
        self.actions = self.action_filter.filter(self.actions)
        # step physics and render each frame
        self.render()
        pd_tar = self._action_to_pd_targets(self.actions)
        for _ in range(self.control_freq_inv):
            self.torques = self._compute_torques(pd_tar).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        self.timeout_buf = torch.where(self.progress_buf >= self.max_episode_length - 1,
                                       torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        if self.history_steps is not None:
            obs_buf = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
            done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            self.obs_buf_history.reset(done_env_ids, obs_buf[done_env_ids])
            self.obs_buf_history.insert(obs_buf)
            self.obs_dict["obs"] = self.obs_buf_history.get_obs_vec(np.arange(self.history_steps))
        else:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()

        if (self._pd_control):
            pd_tar = self._action_to_pd_targets(self.actions)
            pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
            self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)

        else:
            forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            force_tensor = gymtorch.unwrap_tensor(forces)
            self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        self.goal_step += 1
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        # reset goal pos
        goal_reset_envs = (self.goal_step >= self.goal_terminate).nonzero(as_tuple=False).flatten()
        if len(goal_reset_envs) > 0:
            # print('reset encountered!')
            self._reset_goal_pos(goal_reset_envs)

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def _reset_goal_pos(self, goal_reset_envs):
        self.goal_terminate[goal_reset_envs] = torch.randint(100, 200, (len(goal_reset_envs),), device=self.device,
                                                             dtype=torch.int32)
        self.goal_step[goal_reset_envs] = 0
        goal_dist = torch.rand((len(goal_reset_envs), 1), dtype=torch.float, device=self.device) * 4.0 + 1.0
        goal_rot = torch.rand((len(goal_reset_envs), 1), dtype=torch.float, device=self.device) * torch.pi * 2
        self._goal_pos[goal_reset_envs, 0] = torch.flatten(goal_dist * torch.cos(goal_rot))
        self._goal_pos[goal_reset_envs, 1] = torch.flatten(goal_dist * torch.sin(goal_rot))
        if not self.headless:
            # goal_pos = torch.zeros((self.num_envs, 3), device=self.device)
            # goal_pos[goal_reset_envs, :2] = self._goal_pos[goal_reset_envs]
            # goal_pos[goal_reset_envs, 2] = 0.2
            actor_indices = self.all_actor_indices[goal_reset_envs, 1].flatten()
            self._goal_root_states[goal_reset_envs, :2] = self._goal_pos[goal_reset_envs]
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._all_actor_root_states),
                                                         gymtorch.unwrap_tensor(actor_indices), len(actor_indices))


    def render(self):
        if self.viewer and self.camera_follow:
            self._update_camera()

        super().render()
        return

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _pd_target_to_action(self, pd_tar):
        action = (pd_tar - self._pd_action_offset) / self._pd_action_scale
        return action

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0],
                              self._cam_prev_char_pos[1] - 3.0,
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0],
                                  char_root_pos[1] + cam_delta[1],
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
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
def compute_a1_reward(root_xy, goal_xy):
    # type: (Tensor, Tensor) -> Tensor
    # without task reward
    # reward = torch.ones_like(obs_buf[:, 0])

    x_diff = root_xy[:, 0] - goal_xy[:, 0]
    y_diff = root_xy[:, 1] - goal_xy[:, 1]
    reward = torch.exp(- x_diff * x_diff * 0.5 - y_diff * y_diff / 0.5)
    return reward


@torch.jit.script
def compute_a1_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                     max_episode_length, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

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

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated


class A1BaseTrial(A1Base):
    def __int__(self):
        super.__init__()

    def post_physics_step(self):
        self._prev_root_states[..., :] = self._root_states
        super().post_physics_step()

        return

    def _compute_reward(self, actions):
        tar_speed = 2
        prev_root_pos = self._prev_root_states[:, 0:3]

        self.rew_buf[:] = compute_trial_reward(self._root_states, prev_root_pos, tar_speed, self.dt)
        # print('reward: ', self.rew_buf)
        return

@torch.jit.script
def compute_trial_reward(root_states, prev_root_pos, tar_speed, dt):
    # type: (Tensor, Tensor, float, float) -> Tensor
    vel_err_scale = 0.5
    tangent_err_w = 0.1
    roll_err_scale = 0.1
    pitch_err_scale = 0.3
    yaw_err_scale = 0.1
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    # root speed reward
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = root_vel[..., 0]
    y_speed = root_vel[..., 1]
    z_speed = root_vel[..., 2]

    # root rotation reward
    roll, pitch, yaw = get_euler_xyz(root_rot)

    tar_vel_err = tar_speed - tar_dir_speed
    dir_reward = torch.exp(- vel_err_scale * tar_vel_err * tar_vel_err -
                           tangent_err_w * y_speed * y_speed -
                           tangent_err_w * z_speed * z_speed -
                           roll_err_scale * roll * roll -
                           pitch_err_scale * pitch * pitch -
                           yaw_err_scale * yaw * yaw)
    return dir_reward
