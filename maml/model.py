import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaGCNConv, MetaBatchNorm2d, MetaLinear)
from torchmeta.modules.utils import get_subdict

def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ConvNet(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size, model_size, wh_size):
        super(ConvNet, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        
        if model_size == 'smallconv':
            self.features = MetaSequential(
                conv3x3(in_channels, hidden_size),
                conv3x3(hidden_size, hidden_size),
                conv3x3(hidden_size, hidden_size),
                conv3x3(hidden_size, hidden_size)
            )
            self.classifier = MetaLinear(hidden_size*wh_size*wh_size, out_features)
        
        if model_size == 'largeconv':
            self.features = MetaSequential(
                conv3x3(in_channels, hidden_size),
                conv3x3(hidden_size, 2*hidden_size),
                conv3x3(2*hidden_size, 4*hidden_size),
                conv3x3(4*hidden_size, 8*hidden_size)
            )
            self.classifier = MetaLinear(8*hidden_size*wh_size*wh_size, out_features)
        
    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        
        return features, logits
    
class GraphInput():
    def __init__(self, edge_generation_method):
        self.edge_generation_method = edge_generation_method
        if self.edge_generation_method == 'max_normalization':
            self.max_norm = 0.
        elif self.edge_generation_method == 'weighted_max_normalization':
            self.weighted_max_norm = 0.
            self.task_num = 0
        
    def get_graph_inputs(self, features):
        euclidean_matrix = torch.cdist(features, features)
        if self.edge_generation_method == 'max_normalization':
            current_max_norm = torch.max(euclidean_matrix)
            if self.max_norm < current_max_norm:
                self.max_norm = current_max_norm
            euclidean_matrix = euclidean_matrix / self.max_norm
        elif self.edge_generation_method == 'weighted_max_normalization':
            self.weighted_max_norm = (self.weighted_max_norm*self.task_num + torch.max(euclidean_matrix).detach().cpu()) / (self.task_num+1)
            euclidean_matrix = euclidean_matrix / self.weighted_max_norm
            self.task_num += 1
        elif self.edge_generation_method == 'unit_normalization':
            euclidean_matrix = euclidean_matrix / torch.max(euclidean_matrix)
        
        edge_index = torch.transpose(torch.tensor([[i,j] for i in range(len(features)) for j in range(len(features))]), 0, 1)
        row, col = edge_index
        edge_weight = euclidean_matrix[row, col].view(-1, 1)
        
        edge_num = int(math.sqrt(edge_weight.shape[0]))
        self_idx = [(e * edge_num) + e for e in range(edge_num)]
        edge_weight[self_idx,] = 1
        return edge_index.to(features.device), edge_weight.detach().to(features.device)