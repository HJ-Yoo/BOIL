import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from collections import OrderedDict
from torch.nn.modules.utils import _single, _pair, _triple
from torchmeta.modules.module import MetaModule

class MetaGCNConv(gnn.GCNConv, MetaModule):
    __doc__ = gnn.GCNConv.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        return F.conv1d(input, params['weight'], bias, self.stride,
                        self.padding, self.dilation, self.groups)