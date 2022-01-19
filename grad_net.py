import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GradSineLayer(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, is_first: bool = False, omega_0: int = 30
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, input):
        prior_act, prior_deriv_act = input[0], input[1]

        pre_act = self.omega_0 * self.linear(prior_act)
        act = torch.sin(pre_act)

        deriv_pre_act = self.omega_0 * F.linear(prior_deriv_act, self.linear.weight, None)
        deriv_act = torch.cos(pre_act).unsqueeze(-2) * deriv_pre_act

        return act, deriv_act


class GradSiren(nn.Module):
    def __init__(
        self, in_features: int = 2, out_features: int = 2, hidden_features: int = 256, n_layers: int = 6, w0: int = 30
    ) -> None:
        super(GradSiren, self).__init__()

        self.net_layers = nn.ModuleList()
        for idx in range(n_layers):
            if idx == 0:
                self.net_layers.append(GradSineLayer(in_features, hidden_features, bias=True, is_first=idx == 0))
            elif idx > 0 and idx != n_layers - 1:
                self.net_layers.append(GradSineLayer(hidden_features, hidden_features, bias=True, is_first=idx == 0))
            else:
                final_linear = nn.Linear(hidden_features, out_features)
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / hidden_features), np.sqrt(6 / hidden_features))
                self.net_layers.append(final_linear)

    def forward(self, input):
        out = input
        deriv_shape = list(input.shape) + [2]
        deriv_out = torch.zeros(deriv_shape, dtype=input.dtype, device=input.device)
        deriv_out[:, torch.arange(2), torch.arange(2)] = 1
        for ndx, net_layer in enumerate(self.net_layers):
            if ndx < len(self.net_layers) - 1:
                out, deriv_out = net_layer([out, deriv_out])
            else:
                deriv_out = F.linear(deriv_out, net_layer.weight, None)
        return deriv_out.squeeze(-1)
