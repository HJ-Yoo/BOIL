import torch.nn as nn
import torch.nn.functional as F
# import torch_geometric.nn as gnn

from collections import OrderedDict
from torch.nn.modules.utils import _single, _pair, _triple
from torchmeta.modules.module import MetaModule

class MetaGCNConv(nn.Conv1d, MetaModule):
    __doc__ = nn.Conv1d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(input, expanded_padding, mode='circular'),
                            params['weight'], bias, self.stride,
                            _single(0), self.dilation, self.groups)

        return F.conv1d(input, params['weight'], bias, self.stride,
                        self.padding, self.dilation, self.groups)