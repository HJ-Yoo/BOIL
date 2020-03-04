import torch
import torch.nn.init as init
import torch_geometric.nn as gnn

from collections import OrderedDict
from torchmeta.modules.module import MetaModule

class MetaGCNConv(gnn.GCNConv, MetaModule):
    __doc__ = gnn.GCNConv.__doc__
    
    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.zeros_(self.bias)
        self.cached_result = None
        self.cached_num_edges = None
        
    def forward(self, x, edge_index, edge_weight=None, params=None):
        self.normalize = False
        
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        
        x = torch.matmul(x, params['weight'])
        
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(self.node_dim),
                                             edge_weight, self.improved,
                                             x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm
        
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)