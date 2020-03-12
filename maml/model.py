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
        nn.LeakyReLU(),
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
    def __init__(self, in_channels, out_features, hidden_size):
        super(MiniimagenetNet, self).__init__()
        torch.manual_seed(2020)
        
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.features = MetaSequential(
            conv3x3(in_channels, 64),
            conv3x3(64, 64),
            conv3x3(64, 64),
            conv3x3(64, 64)
        )
        self.classifier = MetaLinear(64*5*5, out_features)

    def forward(self, inputs, task_idx=None, update_mode=None, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        
        return features, logits

class ScaleNet(MetaModule):
    """Graph Construction Module"""
    def __init__(self):
        super(ScaleNet, self).__init__()

        self.layer1 = MetaSequential(
            MetaConv2d(64, 64, kernel_size=3, padding=1),
            MetaBatchNorm2d(64, momentum=1., track_running_stats=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, padding=1)
        )
        self.layer2 = MetaSequential(
            MetaConv2d(64, 1, kernel_size=3, padding=1),
            MetaBatchNorm2d(1, momentum=1., track_running_stats=False),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, padding=1)
        )
        self.fc3 = MetaLinear(2*2, 8)
        self.fc4 = MetaLinear(8, 1)

    def forward(self, inputs):
        inputs = inputs.view(-1,64,5,5)
        scale = self.layer1(inputs)
        scale = self.layer2(scale)
        # flatten
        scale = scale.view((scale.size(0),-1))
        scale = F.relu(self.fc3(scale))
        scale = self.fc4(scale) # no relu
        scale = scale.view(scale.size(0),-1) # bs*1
        
        return scale    
    
    

    
    
    
    
class GraphInput():
    def __init__(self, edge_generation_method):
        self.edge_generation_method = edge_generation_method
        if self.edge_generation_method == 'max_normalization':
            self.max_norm = 0.
        elif self.edge_generation_method == 'weighted_max_normalization':
            self.weighted_max_norm = 0.
            self.task_num = 0
        elif self.edge_generation_method == 'max_normalization':
            self.max_norm = 0.
    def get_graph_inputs(self, features, gamma=2.0):
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
        elif self.edge_generation_method == 'max_normalization':
            if self.max_norm <= torch.max(euclidean_matrix):
                self.max_norm = troch.max(euclidean_matrix)
            euclidean_matrix = euclidean_matrix / self.max_norm
        elif self.edge_generation_method == 'mean_cosine_similarity' :
            featrues = features - torch.mean(features, dim=0, keepdim=True)
            euclidean_matrix = torch.zeros(euclidean_matrix.shape)
            pairwise_distance = nn.CosineSimilarity(dim=-1)
            for i in range(len(features)):
                euclidean_matrix[:,i] = pairwise_distance(features[i].view(1, -1), features)
        
        edge_index = torch.transpose(torch.tensor([[i,j] for i in range(len(features)) for j in range(len(features))]), 0, 1)
        row, col = edge_index
        edge_weight = euclidean_matrix[row, col].view(-1, 1)
        edge_weight = torch.exp(-gamma * edge_weight * edge_weight).view(-1,1)
        
        edge_num = int(math.sqrt(edge_weight.shape[0]))
        self_idx = [(e * edge_num) + e for e in range(edge_num)]
        edge_weight[self_idx,] = 1
        return edge_index.to(features.device), edge_weight.detach().to(features.device)