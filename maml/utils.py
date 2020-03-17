import torch
import torch.nn as nn
import numpy as np

from maml.model import OmniglotNet, MiniimagenetNet, ScaleNet, LearningRateNet # TieredimagenetNet, Cifar_fsNet, CubNet, DoublemnistNet, TriplemnistNet
from torchmeta.datasets.helpers import omniglot, miniimagenet, tieredimagenet, cifar_fs, cub, doublemnist, triplemnist
from collections import OrderedDict

def load_dataset(args, mode):
    folder = args.folder
    ways = args.num_ways
    shots = args.num_shots
    test_shots = 15
    download = args.download
    shuffle = True
    
    if mode == 'meta_train':
        args.meta_train = True
        args.meta_val = False
        args.meta_test = False
    elif mode == 'meta_valid':
        args.meta_train = False
        args.meta_val = True
        args.meta_test = False
    elif mode == 'meta_test':
        args.meta_train = False
        args.meta_val = False
        args.meta_test = True
    
    if args.dataset == 'omniglot':
        dataset = omniglot(folder=folder,
                           shots=shots,
                           ways=ways,
                           shuffle=shuffle,
                           test_shots=test_shots,
                           meta_train=args.meta_train,
                           meta_val=args.meta_val,
                           meta_test=args.meta_test,
                           download=download)
    elif args.dataset == 'miniimagenet':
        dataset = miniimagenet(folder=folder,
                               shots=shots,
                               ways=ways,
                               shuffle=shuffle,
                               test_shots=test_shots,
                               meta_train=args.meta_train,
                               meta_val=args.meta_val,
                               meta_test=args.meta_test,
                               download=download)
    elif args.dataset == 'tieredimagenet':
        dataset = tieredimagenet(folder=folder,
                                 shots=shots,
                                 ways=ways,
                                 shuffle=shuffle,
                                 test_shots=test_shots,
                                 meta_train=args.meta_train,
                                 meta_val=args.meta_val,
                                 meta_test=args.meta_test,
                                 download=download)
    elif args.dataset == 'cifar_fs':
        dataset = cifar_fs(folder=folder,
                           shots=shots,
                           ways=ways,
                           shuffle=shuffle,
                           test_shots=test_shots,
                           meta_train=args.meta_train,
                           meta_val=args.meta_val,
                           meta_test=args.meta_test,
                           download=download)
    elif args.dataset == 'cub':
        dataset = cub(folder=folder,
                      shots=shots,
                      ways=ways,
                      shuffle=shuffle,
                      test_shots=test_shots,
                      meta_train=args.meta_train,
                      meta_val=args.meta_val,
                      meta_test=args.meta_test,
                      download=download)
    elif args.dataset == 'doublemnist':
        dataset = doublemnist(folder=folder,
                              shots=shots,
                              ways=ways,
                              shuffle=shuffle,
                              test_shots=test_shots,
                              meta_train=args.meta_train,
                              meta_val=args.meta_val,
                              meta_test=args.meta_test,
                              download=download)
    elif args.dataset == 'triplemnist':
        dataset = triplemnist(folder=folder,
                              shots=shots,
                              ways=ways,
                              shuffle=shuffle,
                              test_shots=test_shots,
                              meta_train=args.meta_train,
                              meta_val=args.meta_val,
                              meta_test=args.meta_test,
                              download=download)
    
    return dataset

def load_model(args):
    if args.dataset == 'omniglot':
        model = OmniglotNet(1, args.num_ways, hidden_size=args.hidden_size)
    elif args.dataset == 'miniimagenet':
        model = MiniimagenetNet(3, args.num_ways, hidden_size=args.hidden_size)
        scale_model = ScaleNet()
        lr_model = LearningRateNet()
        
    elif args.dataset == 'tieredimagenet':
        pass
    elif args.dataset == 'cifar_fs':
        pass
    elif args.dataset == 'cub':
        pass
    elif args.dataset == 'doublemnist':
        pass
    elif args.dataset == 'triplemnist':
        pass
    
    return model, scale_model, lr_model

def update_parameters(model, loss, step_size=0.5, first_order=False):
    """Update the parameters of the model, with one step of gradient descent.

    Parameters
    ----------
    model : `MetaModule` instance
        Model.
    loss : `torch.FloatTensor` instance
        Loss function on which the gradient are computed for the descent step.
    step_size : float (default: `0.5`)
        Step-size of the gradient descent step.
    first_order : bool (default: `False`)
        If `True`, use the first-order approximation of MAML.

    Returns
    -------
    params : OrderedDict
        Dictionary containing the parameters after one step of adaptation.
    """
    grads = torch.autograd.grad(loss,
                                model.meta_parameters(),
                                create_graph=not first_order)

    params = OrderedDict()
    for (name, param), grad in zip(model.meta_named_parameters(), grads):
        if 'classifier' in name: # To control inner update parameter
            params[name] = param - step_size * grad
        else:
            params[name] = param - step_size * grad # params[name] = param

    return params

def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points

    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(num_examples,)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())



def get_graph_regularizer(features, labels=None, model=None, args=None):
    eps = np.finfo(float).eps
    # scale = model(features)
    # features = features / (scale+eps)
    
    features1 = torch.unsqueeze(features, 1)
    features2 = torch.unsqueeze(features, 0)
    features_distance = torch.mean((features1-features2)**2, dim=2)
    features_distance = torch.exp(-features_distance/2)

    if args.graph_edge_generation == 'ss_sq_qq':
        edge_weight_LL = torch.zeros([len(labels), len(labels)]).to(args.device)
        for i, class_i in enumerate(labels):
            for j, class_j in enumerate(labels):
                if class_i == class_j:
                    edge_weight_LL[i][j] = 0.5
        edge_weight_LU = torch.ones([25, 75]).to(args.device)/4
        edge_weight_UU = torch.ones([75, 75]).to(args.device)/4
    elif args.graph_edge_generation == 'ss_sq':
        edge_weight_LL = torch.zeros([len(labels), len(labels)]).to(args.device)
        for i, class_i in enumerate(labels):
            for j, class_j in enumerate(labels):
                if class_i == class_j:
                    edge_weight_LL[i][j] = 0.5
        edge_weight_LU = torch.ones([25, 75]).to(args.device)/4
        edge_weight_UU = torch.zeros([75, 75]).to(args.device)
    elif args.graph_edge_generation == 'sq':
        edge_weight_LL = torch.zeros([25, 25]).to(args.device)
        edge_weight_LU = torch.ones([25, 75]).to(args.device)/2
        edge_weight_UU = torch.zeros([75, 75]).to(args.device)
    elif args.graph_edge_generation == 'sq_qq':
        edge_weight_LL = torch.zeros([25, 25]).to(args.device)
        edge_weight_LU = torch.ones([25, 75]).to(args.device)/2
        edge_weight_UU = torch.zeros([75, 75]).to(args.device)/4
        
    if args.graph_edge_generation == 'no_edges':
        edge_weight = torch.ones([100,100]).to(args.device)/2
    else:
        edge_weight = torch.cat((torch.cat((edge_weight_LL/(25*25), edge_weight_LU/(25*75)), dim=1),torch.cat((edge_weight_LU.t()/(25*75), edge_weight_UU/(75*75)), dim=1)), dim=0).to(args.device)
   
    graph_loss = torch.sum(features_distance*edge_weight)
    
    if args.graph_edge_generation == 'sq_single_element':
        sq_distance = features_distance[:25,25:]
        sq_distance = torch.min(sq_distance, dim=0)[0]
        graph_loss = torch.sum(sq_distance)

#==============================================================================================================================    
#     pairwise_distance = nn.PairwiseDistance(p=2)
#     graph_distance = torch.zeros([len(features), len(features)]).to(args.device)
#     for i in range(len(features)):
#         graph_distance[:,i] = pairwise_distance(features[i].view(1, -1), features)
#     edge_weight_LL = torch.zeros([len(labels), len(labels)]).to(args.device)
#     for i, class_i in enumerate(labels):
#         for j, class_j in enumerate(labels):
#             if class_i == class_j:
#                 edge_weight_LL[i][j] = 0.5
#     edge_weight_LU = torch.ones([25, 75]).to(args.device)/4
#     edge_weight_UU = torch.ones([75, 75]).to(args.device)/4
    
#     edge_weight = torch.cat((torch.cat((edge_weight_LL/(25*25), edge_weight_LU/(25*75)), dim=1),torch.cat((edge_weight_LU.t()/(25*75), edge_weight_UU/(75*75)), dim=1)), dim=0).to(args.device)
    
#     graph_loss = torch.sum(graph_distance * edge_weight)
#================================================================================================================================
#     pairwise_distance = nn.PairwiseDistance(p=2).to(args.device)
#     features_dist_matrix = torch.zeros([len(features), len(features)]).to(args.device)
#     for i in range(len(features)):
#         features_dist_matrix[:,i] = pairwise_distance(features[i].view(1, -1), features)
    
#     centroid = torch.zeros([5, features.shape[1]])
#     for i in range(5):
#         centroid[i] = torch.mean(features[torch.where(labels==i)[0],:], dim=0)
#     centroid_dist_matrix = torch.cdist(centroid, centroid).detach()
#     rank_centroid_matrix = 1 - (centroid_dist_matrix / torch.sum(torch.unique(centroid_dist_matrix))) * args.graph_gamma
#     edge_matrix = torch.zeros(features_dist_matrix.shape).to(args.device)

#     for i in range(args.num_ways):
#         for j in range(args.num_ways):
#             for k in range(args.num_shots):
#                 for l in range(args.num_shots):
#                     edge_matrix[(5*i)+k][(5*j)+l] = rank_centroid_matrix[i][j]

#     penalty = torch.sum(features_dist_matrix*edge_matrix).to(args.device)*args.graph_beta
    return graph_loss

def get_adaptive_lr(features, model=None):
    adaptive_lr = model(features)
    adaptive_lr = torch.clamp(adaptive_lr, 0.1, 1.)
    return adaptive_lr