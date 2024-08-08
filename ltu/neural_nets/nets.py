"""Neural nets."""

import torch
from torch import nn


class FFN(nn.Module):
    """Fully connected neural net."""
    def __init__(
        self,
        in_dim,
        out_dim: int,
        hidden_dim: list,
        activation: list,
        bias: bool = True,
        drop_out: float = 0.0,
        seed: int = None
    ):
        assert len(hidden_dim) == len(activation)
        self.activation = [_.lower() for _ in activation]
        self.bias = bias
        self.drop_out = drop_out
        if seed is not None:
            torch.manual_seed(seed)
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self._update()

    def _update(self):
        return self.update()

    def update(self):
        """Update method."""
        if not self.hidden_dim:
            module_dict = {'net_input': {'size': [self.in_dim, self.out_dim],
                                         'activation': ''}}
        else:
            module_dict = {'net_input': {'size': [self.in_dim, self.hidden_dim[0]],
                                         'activation': self.activation[0]}}
            for i in range(1, len(self.hidden_dim)):
                module_dict.update(
                    {'net_hidden' + str(i): {'size': [self.hidden_dim[i-1], self.hidden_dim[i]],
                                                            'activation': self.activation[i]}}
                )
            module_dict.update({'net_output': {'size': [self.hidden_dim[-1], self.out_dim],
                                               'activation': ''}})

        for module_name in module_dict.keys(): # pylint: disable=C0201, C0206
            module_size = module_dict[module_name]['size']
            module_activation = module_dict[module_name]['activation']
            self.add_module(module_name, nn.Sequential(nn.Linear(in_features=module_size[0],
                                                                 out_features=module_size[1],
                                                                 bias=self.bias))
                            )
            if module_activation != '':
                if module_activation.lower() == 'tanh':
                    self.add_module(module_name + '_activation', nn.Sequential(
                                                nn.Tanh(),
                                                nn.Dropout(self.drop_out),
                                                nn.BatchNorm1d(module_size[1], affine=False))
                                    )
                elif module_activation.lower() == 'selu':
                    self.add_module(module_name + '_activation', nn.Sequential(
                                                nn.SELU(True),
                                                nn.Dropout(self.drop_out),
                                                nn.BatchNorm1d(module_size[1], affine=False))
                                    )
                elif module_activation.lower() == 'relu':
                    self.add_module(module_name + '_activation', nn.Sequential(
                                                nn.ReLU(True),
                                                nn.Dropout(self.drop_out),
                                                nn.BatchNorm1d(module_size[1], affine=False))
                                    )
                else:
                    raise NotImplementedError

    def forward(self, x):
        """Forward method."""
        for module_name in list(self._modules.keys()):
            x = self._modules[module_name](x)
        return x
