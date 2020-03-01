import torch

from maml.model import OmniglotNet, MiniimagenetNet # TieredimagenetNet, Cifar_fsNet, CubNet, DoublemnistNet, TriplemnistNet
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
        model = MiniimagenetNet(3, args.num_ways, hidden_size=args.hidden_size, task_embedding_method=args.task_embedding_method, edge_generation_method=args.edge_generation_method)
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
    
    return model

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
        params[name] = param - step_size * grad

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
    features_dist_matrix = torch.cdist(features, features).detach()
    centroid = torch.zeros([5, features.shape[1]])
    
    for i in range(5):
        centroid[i] = torch.mean(features[torch.where(labels==i)[0],:], dim=0)

    centroid_dist_matrix = torch.cdist(centroid, centroid).detach()
    rank_centroid_matrix = 1 - (centroid_dist_matrix / torch.sum(torch.unique(centroid_dist_matrix))) * args.graph_gamma

    edge_matrix = torch.zeros(features_dist_matrix.shape).to(args.device)

    for i in range(args.num_ways):
        for j in range(args.num_ways):
            for k in range(args.num_shots):
                for l in range(args.num_shots):
                    edge_matrix[(5*i)+k][(5*j)+l] = rank_centroid_matrix[i][j]

    penalty = torch.sum(features_dist_matrix*edge_matrix)*args.graph_beta

    return torch.sum(penalty)
