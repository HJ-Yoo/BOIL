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
    def __init__(self, in_channels, out_features, hidden_size, task_embedding_method, edge_generation_method):
        super(MiniimagenetNet, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.task_embedding_method = task_embedding_method

        self.features1 = MetaSequential(
            conv3x3(in_channels, 64),
            conv3x3(64, 64)
        )
        self.features2 = MetaSequential(
            conv3x3(64, 64),
            conv3x3(64, 64)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        if self.task_embedding_method == 'gcn':
            self.graph_input = GraphInput(edge_generation_method = edge_generation_method)
            self.gcn1 = MetaGCNConv(64, 64 // 2)
            self.classifier = MetaLinear(64 + 64 // 2, out_features)
        
        elif self.task_embedding_method == 'gcn_2layers':
            self.graph_input = GraphInput(edge_generation_method = edge_generation_method)
            self.gcn1 = MetaGCNConv(64, 64 // 2)
            self.gcn_relu = nn.ReLU()
            self.gcn2 = MetaGCNConv(64 // 2, 64 // 4)
            self.classifier = MetaLinear(64 + 64 // 4, out_features)
        
        elif self.task_embedding_method == 'avgpool':
            self.classifier = MetaLinear(64 + 64, out_features)
        
        else:
            self.classifier = MetaLinear(64, out_features)
            
    def forward(self, inputs, params=None):
        features1 = self.features1(inputs, params=get_subdict(params, 'features1'))
        features1_pool = self.pool(features1)
        features1_pool = features1_pool.view((features1_pool.size(0), -1))
        features2 = self.features2(features1, params=get_subdict(params, 'features2'))
        features2 = self.pool(features2)
        features2 = features2.view((features2.size(0), -1))
        
        if self.task_embedding_method == 'gcn':
            edge_index, edge_weight = self.graph_input.get_graph_inputs(features2)
            task_embeddings = self.gcn1(x=features2,
                                       edge_index=edge_index,
                                       edge_weight=edge_weight,
                                       params=get_subdict(params, 'gcn1'))
            task_embedding = torch.mean(task_embeddings, dim=0)
            features2 = torch.cat([features2, torch.stack([task_embedding]*len(features2))], dim=1)
        
        if self.task_embedding_method == 'gcn_2layers':
            edge_index, edge_weight = self.graph_input.get_graph_inputs(features)
            task_embedding = self.gcn1(x=features,
                                       edge_index=edge_index,
                                       edge_weight=edge_weight,
                                       params=get_subdict(params, 'gcn1'))
            task_embedding = self.gcn_relu(task_embedding)
            task_embedding = self.gcn2(x=task_embedding,
                                       edge_index=edge_index,
                                       edge_weight=edge_weight,
                                       params=get_subdict(params, 'gcn2'))
            task_embedding = torch.mean(task_embedding, dim=0)
            features = torch.cat([features, torch.stack([task_embedding]*len(features))], dim=1)
            
        elif self.task_embedding_method == 'avgpool':
            task_embedding = torch.mean(features, dim=0) # for average pooling embedding
            features = torch.cat([features, torch.stack([task_embedding]*len(features))], dim=1)
        
        logits = self.classifier(features2, params=get_subdict(params, 'classifier'))
        
        return features1_pool, features2, task_embeddings, logits

class GraphInput():
    def __init__(self, edge_generation_method):
        self.edge_generation_method = edge_generation_method
        if self.edge_generation_method == 'weighted_max_normalization':
            self.weighted_max_norm = 0.
            self.task_num = 0
        
    def get_graph_inputs(self, features):
        euclidean_matrix = torch.cdist(features, features)
        if self.edge_generation_method == 'weighted_max_normalization':
            self.weighted_max_norm = (self.weighted_max_norm*self.task_num + torch.max(euclidean_matrix).detach().cpu()) / (self.task_num+1)
            euclidean_matrix = euclidean_matrix / self.weighted_max_norm
            self.task_num += 1
        elif self.edge_generation_method == 'unit_normalization':
            euclidean_matrix = euclidean_matrix / torch.max(euclidean_matrix)
        elif self.edge_generation_method == 'mean_cosine_similarity' :
            featrues = features - torch.mean(features, dim=0, keepdim=True)
            euclidean_matrix = torch.zeros(euclidean_matrix.shape)
            pairwise_distance = nn.CosineSimilarity(dim=-1)
            for i in range(len(features)):
                euclidean_matrix[:,i] = pairwise_distance(features[i].view(1, -1), features)
        elif self.edge_generation_method == 'cosine_similarity' :
            euclidean_matrix = torch.zeros(euclidean_matrix.shape)
            pairwise_distance = nn.CosineSimilarity(dim=-1)
            for i in range(len(features)):
                euclidean_matrix[:,i] = pairwise_distance(features[i].view(1, -1), features)
        
        edge_index = torch.transpose(torch.tensor([[i,j] for i in range(len(features)) for j in range(len(features))]), 0, 1)
        row, col = edge_index
        edge_weight = euclidean_matrix[row, col].view(-1, 1)
        
        edge_num = int(math.sqrt(edge_weight.shape[0]))
        self_idx = [(e * edge_num) + e for e in range(edge_num)]
        edge_weight[self_idx,] = 1
        return edge_index.to(features.device), edge_weight.detach().to(features.device)