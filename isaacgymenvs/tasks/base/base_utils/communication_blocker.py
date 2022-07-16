import torch
import copy

from dataclasses import dataclass, fields, asdict


class CommunicationBlocker():
    def __init__(self, num_env, freq, delay_bound, struct, prob=0.1, device=None):
        '''
        If action, input sim_freq, if obs, input pol_freq
        '''
        self.device = device
        t_delay = torch.zeros(num_env, device=self.device).uniform_(delay_bound[0], delay_bound[
            1])  # delay from 2200 hz to 40 hz, udp will be running at 2000Hz realtime
        self.max_buffer_size = torch.round(t_delay * freq).long()
        self.prob = prob
        self.num_env = num_env
        self.hold_flag = torch.zeros(self.num_env, dtype=torch.bool, device=self.device)
        self.prev_hold_flag = torch.zeros(self.num_env, dtype=torch.bool, device=self.device)
        # self.held_msg = copy.deepcopy(struct)
        # self.blank_msg = copy.deepcopy(struct)
        self.held_msg = torch.zeros_like(struct, dtype=torch.float, device=self.device)
        self.blank_msg = torch.zeros_like(struct, dtype=torch.float, device=self.device)
        self.held_msg[:] = struct[:]
        self.blank_msg[:] = struct[:]
        self.held_count = torch.zeros(self.num_env, dtype=torch.int32, device=self.device)
        self.held_buffer_size = torch.zeros(self.num_env, dtype=torch.int32, device=self.device)

    def reset(self, env_ids, start_msg=None):
        self.hold_flag[env_ids] = False
        self.prev_hold_flag[env_ids] = False
        if start_msg is not None:
            self.blank_msg[env_ids] = start_msg
        self.held_msg[env_ids] = self.blank_msg[env_ids]
        self.held_count[env_ids] = 0
        self.held_buffer_size[env_ids] = 0

    def send_msg(self, msg_to_send):
        self.hold_flag = self.hold_flag | (torch.rand(self.num_env, device=self.device) < self.prob)
        msg_sent = msg_to_send

        # ------------- start hold ---------------
        mask = self.hold_flag & ~self.prev_hold_flag

        self.held_msg[mask] = msg_to_send[mask]

        self.held_count[mask] = 0
        self.held_buffer_size[mask] = (
                    torch.round(torch.rand(torch.sum(mask), device=self.device) * self.max_buffer_size[mask]) + 10).to(
            torch.int32)

        # ------------- hold ---------------
        mask = self.hold_flag
        msg_sent[mask] = self.held_msg[mask]
        msg_sent[~mask] = msg_to_send[~mask]
        self.held_count[mask] = self.held_count[mask] + 1
        self.prev_hold_flag[:] = self.hold_flag[:]

        # ------------- reset hold ---------------
        mask = self.hold_flag & (self.held_count > self.held_buffer_size)

        self.reset(mask)

        return msg_sent