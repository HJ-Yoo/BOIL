import math
import torch
import torch.nn as nn
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

class OmniglotNet(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(OmniglotNet, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )
        
        self.classifier = MetaLinear(hidden_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        return features, logits

class MiniimagenetNet(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(MiniimagenetNet, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # self.gcn1 = MetaGCNConv(hidden_size*5*5, hidden_size*5*5 // 2)
        # self.gcn2 = MetaGCNConv(hidden_size*5*5 // 2, hidden_size*5*5 // 4)
        
        self.classifier = MetaLinear(hidden_size + hidden_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = self.pool(features)
        features = features.view((features.size(0), -1))        
#         edge_index, edge_weight = get_graph_inputs(features)
        
#         task_embedding = self.gcn1(x=features,
#                                    edge_index=edge_index,
#                                    edge_weight=edge_weight,
#                                    params=get_subdict(params, 'gcn1'))
        
#         task_embedding = torch.mean(task_embedding, dim=0)
        task_embedding = torch.mean(features, dim=0)
        features = torch.cat([features, torch.stack([task_embedding]*len(features))], dim=1)
        
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        return features, logits
    

def get_graph_inputs(features, gamma=0.001):
    euclidean_matrix = torch.cdist(features, features)
    
    edge_index = torch.transpose(torch.tensor([[i,j] for i in range(len(features)) for j in range(len(features))]), 0, 1)
    row, col = edge_index
    edge_weight = euclidean_matrix[row, col].view(-1, 1)
    edge_weight = torch.exp(-gamma * edge_weight * edge_weight)
    
    edge_num = int(math.sqrt(edge_weight.shape[0]))
    self_idx = [(e * edge_num) + e for e in range(edge_num)]
    edge_weight[self_idx,] = 1
    
    return edge_index.to(features.device), edge_weight.detach().to(feature.device)