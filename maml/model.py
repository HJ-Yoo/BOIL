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
    def __init__(self, in_channels, out_features, hidden_size, task_embedding_method, edge_generation_method):
        super(MiniimagenetNet, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.task_embedding_method = task_embedding_method

        self.features = MetaSequential(
            conv3x3(in_channels, 64),
            conv3x3(64, 64),
            conv3x3(64, 64),
            conv3x3(64, 64)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        if self.task_embedding_method == 'gcn_concat':
            self.graph_input = GraphInput(edge_generation_method = edge_generation_method)
            self.gcn1 = MetaGCNConv(64, 64 // 2)
            self.classifier = MetaLinear(64 + 64 // 2, out_features)
        
        elif self.task_embedding_method == 'gcn_scaling':
            self.graph_input = GraphInput(edge_generation_method = edge_generation_method)
            self.gcn1 = MetaGCNConv(64, 64)
            self.gcn_sigmoid = nn.Sigmoid()
            self.classifier = MetaLinear(64, out_features)
        elif self.task_embedding_method == 'gcn_fc_layer':
            self.graph_input = GraphInput(edge_generation_method = edge_generation_method)
            self.gcn1 = MetaGCNConv(64, 64 // 2)
            self.gcn_leakyrelu1 = nn.LeakyReLU()
            self.gcn_fc = nn.Linear(64//2, 64//4)
            self.gcn_leakyrelu2 = nn.LeakyReLU()
            self.classifier = MetaLinear(64 + 64 // 4, out_features)
        
        elif self.task_embedding_method == 'avgpool':
            self.classifier = MetaLinear(64 + 64, out_features)
        
        else:
            self.classifier = MetaLinear(64, out_features)
            
    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = self.pool(features)
        features = features.view((features.size(0), -1))

        if self.task_embedding_method == 'gcn_concat':
            edge_index, edge_weight = self.graph_input.get_graph_inputs(features)
            task_embeddings = self.gcn1(x=features,
                                       edge_index=edge_index,
                                       edge_weight=edge_weight,
                                       params=get_subdict(params, 'gcn1'))
            task_embedding = torch.mean(task_embeddings, dim=0)
            features_cat = torch.cat([features, torch.stack([task_embedding]*len(features2))], dim=1)
            logits = self.classifier(features_cat, params=get_subdict(params, 'classifier'))
            return features, task_embedding, logits
        
        elif self.task_embedding_method == 'gcn_scaling':
            edge_index, edge_weight = self.graph_input.get_graph_inputs(features)
            task_embedding = self.gcn1(x=features,
                                       edge_index=edge_index,
                                       edge_weight=edge_weight,
                                       params=get_subdict(params, 'gcn1'))
            task_embedding = torch.mean(task_embedding, dim=0)
            task_scaling = self.gcn_sigmoid(task_embedding)
            features_cat = features * task_scaling
            logits = self.classifier(features_cat, params=get_subdict(params, 'classifier'))
            return features, task_embedding, logits
        
        elif self.task_embedding_method == 'gcn_fc_layer':
            edge_index, edge_weight = self.graph_input.get_graph_inputs(features)
            task_embedding = self.gcn1(x=features,
                                       edge_index=edge_index,
                                       edge_weight=edge_weight,
                                       params=get_subdict(params, 'gcn1'))
            task_embedding = self.gcn_leakyrelu1(task_embedding)
            task_embedding = self.gcn_fc(task_embedding)
            task_embedding = self.gcn_leakyrelu2(task_embedding)
            task_embedding = torch.mean(task_embedding, dim=0)
            features_cat = torch.cat([features, torch.stack([task_embedding]*len(features))], dim=1)    
            logits = self.classifier(features_cat, params=get_subdict(params, 'classifier'))
            return features, task_embedding, logits
        
        elif self.task_embedding_method == 'avgpool':
            task_embedding = torch.mean(features, dim=0) # for average pooling embedding
            features_cat = torch.cat([features, torch.stack([task_embedding]*len(features))], dim=1)
            logits = self.classifier(features_cat, params=get_subdict(params, 'classifier'))
            return features, task_embedding, logits
        
        else:
            logits = self.classifier(features, params=get_subdict(params, 'classifier'))
            return features, None, logits
        
class GraphInput():
    def __init__(self, edge_generation_method):
        self.edge_generation_method = edge_generation_method
        if self.edge_generation_method == 'weighted_max_normalization':
            self.weighted_max_norm = 0.
            self.task_num = 0
        elif self.edge_generation_method == 'max_normalization':
            self.max_norm = 0.
    def get_graph_inputs(self, features, gamma=2.0):
        euclidean_matrix = torch.cdist(features, features)
        if self.edge_generation_method == 'weighted_max_normalization':
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