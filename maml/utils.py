import torch
import torch.nn as nn

from maml.model import ConvNet, BasicBlock, ResNet
from torchmeta.datasets.helpers import miniimagenet, tieredimagenet, cifar_fs, fc100
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
    
    if args.dataset == 'miniimagenet':
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
    elif args.dataset == 'fc100':
        dataset = fc100(folder=folder,
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
    if args.dataset == 'miniimagenet' or args.dataset == 'tieredimagenet':
        wh_size = 5
    elif args.dataset == 'cifar_fs' or args.dataset == 'fc100':
        wh_size = 2
        
    if args.model == 'smallconv' or args.model == 'largeconv':
        model = ConvNet(in_channels=3, out_features=args.num_ways, hidden_size=args.hidden_size, model_size=args.model, wh_size=wh_size)
    elif args.model == 'resnet':
        model = ResNet(block=BasicBlock, keep_prob=1.0, avg_pool=True)
    return model

def update_parameters(model, loss, extractor_step_size, classifier_step_size, first_order=False):
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
            params[name] = param - classifier_step_size * grad
        else:
            params[name] = param - extractor_step_size * grad # params[name] = param
    
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


def get_graph_regularizer(features, labels=None, args=None):
    support_features = features[:args.num_ways*args.num_shots,:]
    query_features = features[args.num_ways*args.num_shots:,:]
    support_labels = labels[0]
    query_labels = labels[1]
    
    distance = torch.zeros([len(support_labels), len(query_labels)]).to(args.device)
    if args.graph_type=='single':
        for i, class_i in enumerate(support_labels):
            dist = []
            for j, class_j in enumerate(query_labels):
                if class_i == class_j:
                    dist.append(torch.dist(support_features[i], query_features[j]))
            distance[i][dist.index(min(dist))] = min(dist)
    elif args.graph_type=='all':
        for i, class_i in enumerate(support_labels):
            for j, class_j in enumerate(query_labels):
                distance[i][j] = torch.dist(support_features[i], query_features[j])
    
    graph_loss = torch.sum(distance)
    
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


#     graph_distance_uu = torch.zeros([len(query_features), len(query_features)]).to(args.device)
#     for i in range(len(query_features)):
#         graph_distance_uu[:,i] = pairwise_distance(query_features[i].view(1, -1), query_features)
# #     graph_distance_uu = torch.cdist(query_features, query_features).to(args.device)
#     edge_weight_uu = torch.ones_like(graph_distance_uu).to(args.device)
#     graph_loss_uu = torch.mean(edge_weight_uu * graph_distance_uu*0.5*0.5).to(args.device)

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