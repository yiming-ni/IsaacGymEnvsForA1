from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn


class BaseBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __int__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            return

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value

    def build(self, name, **kwargs):
        net = BaseBuilder.Network(self.params, **kwargs)
        return net
