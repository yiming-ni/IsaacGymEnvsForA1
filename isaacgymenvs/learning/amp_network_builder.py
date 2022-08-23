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

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0


class AMPBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)
                    
            amp_input_shape = kwargs.get('amp_input_shape')
            self._build_disc(amp_input_shape)

            return

        def load(self, params):
            super().load(params)

            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']
            return

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

        def eval_disc(self, amp_obs):
            disc_mlp_out = self._disc_mlp(amp_obs)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

        def get_disc_logit_weights(self):
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits.weight))
            return weights

        def _build_disc(self, input_shape):
            self._disc_mlp = nn.Sequential()

            mlp_args = {
                'input_size' : input_shape[0], 
                'units' : self._disc_units, 
                'activation' : self._disc_activation, 
                'dense_func' : torch.nn.Linear
            }
            self._disc_mlp = self._build_mlp(**mlp_args)
            
            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias) 

            return

        def forward(self, obs_dict):
            if ('c_obs' not in obs_dict.keys()) or (obs_dict['c_obs'] is None):
                return super().forward(obs_dict)
            else:
                obs = obs_dict['obs']
                c_obs = obs_dict['c_obs']
                states = obs_dict.get('rnn_states', None)
                seq_length = obs_dict.get('seq_length', 1)
                if self.has_cnn:
                    # for obs shape 4
                    # input expected shape (B, W, H, C)
                    # convert to (B, C, W, H)
                    if len(obs.shape) == 4:
                        obs = obs.permute((0, 3, 1, 2))

                if self.separate:
                    a_out = obs
                    c_out = c_obs
                    a_out = self.actor_cnn(a_out)
                    a_out = a_out.contiguous().view(a_out.size(0), -1)

                    c_out = self.critic_cnn(c_out)
                    c_out = c_out.contiguous().view(c_out.size(0), -1)

                    if self.has_rnn:
                        if not self.is_rnn_before_mlp:
                            a_out_in = a_out
                            c_out_in = c_out
                            a_out = self.actor_mlp(a_out_in)
                            c_out = self.critic_mlp(c_out_in)

                            if self.rnn_concat_input:
                                a_out = torch.cat([a_out, a_out_in], dim=1)
                                c_out = torch.cat([c_out, c_out_in], dim=1)

                        batch_size = a_out.size()[0]
                        num_seqs = batch_size // seq_length
                        a_out = a_out.reshape(num_seqs, seq_length, -1)
                        c_out = c_out.reshape(num_seqs, seq_length, -1)

                        if self.rnn_name == 'sru':
                            a_out = a_out.transpose(0, 1)
                            c_out = c_out.transpose(0, 1)

                        if len(states) == 2:
                            a_states = states[0]
                            c_states = states[1]
                        else:
                            a_states = states[:2]
                            c_states = states[2:]
                        a_out, a_states = self.a_rnn(a_out, a_states)
                        c_out, c_states = self.c_rnn(c_out, c_states)

                        if self.rnn_name == 'sru':
                            a_out = a_out.transpose(0, 1)
                            c_out = c_out.transpose(0, 1)
                        else:
                            if self.rnn_ln:
                                a_out = self.a_layer_norm(a_out)
                                c_out = self.c_layer_norm(c_out)
                        a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
                        c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                        if type(a_states) is not tuple:
                            a_states = (a_states,)
                            c_states = (c_states,)
                        states = a_states + c_states

                        if self.is_rnn_before_mlp:
                            a_out = self.actor_mlp(a_out)
                            c_out = self.critic_mlp(c_out)
                    else:
                        a_out = self.actor_mlp(a_out)
                        c_out = self.critic_mlp(c_out)

                    value = self.value_act(self.value(c_out))

                    if self.is_discrete:
                        logits = self.logits(a_out)
                        return logits, value, states

                    if self.is_multi_discrete:
                        logits = [logit(a_out) for logit in self.logits]
                        return logits, value, states

                    if self.is_continuous:
                        mu = self.mu_act(self.mu(a_out))
                        if self.space_config['fixed_sigma']:
                            sigma = mu * 0.0 + self.sigma_act(self.sigma)
                        else:
                            sigma = self.sigma_act(self.sigma(a_out))

                        return mu, sigma, value, states
                else:
                    return super().forward(obs_dict)

    def build(self, name, **kwargs):
        net = AMPBuilder.Network(self.params, **kwargs)
        return net