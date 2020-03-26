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
        model = ResNet(block=BasicBlock, keep_prob=1.0, avg_pool=True, out_features=args.num_ways, wh_size=1)
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
            params[name] = param - extractor_step_size * grad
    
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
        for j, class_j in enumerate(query_labels):
            dist = []
            for i, class_i in enumerate(support_labels):
                if class_i == class_j:
                    dist.append(torch.dist(support_features[i], query_features[j]))
            distance[dist.index(min(dist))][j] = min(dist)
    elif args.graph_type=='all':
        for i, class_i in enumerate(support_labels):
            for j, class_j in enumerate(query_labels):
                distance[i][j] = torch.dist(support_features[i], query_features[j])
                distance = distance/15
    
    graph_loss = torch.sum(distance)
    return graph_loss