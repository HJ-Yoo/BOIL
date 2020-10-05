import torch
import torch.nn as nn

from maml.model import ConvNet, BasicBlock, BasicBlockWithoutResidual, ResNet
from torchmeta.datasets.helpers import (miniimagenet, tieredimagenet, cifar_fs, fc100,
                                        cub, vgg_flower, aircraft, traffic_sign, svhn, cars)
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
    elif args.dataset == 'vgg_flower':
        dataset = vgg_flower(folder=folder,
                             shots=shots,
                             ways=ways,
                             shuffle=shuffle,
                             test_shots=test_shots,
                             meta_train=args.meta_train,
                             meta_val=args.meta_val,
                             meta_test=args.meta_test,
                             download=download)
    elif args.dataset == 'aircraft':
        dataset = aircraft(folder=folder,
                           shots=shots,
                           ways=ways,
                           shuffle=shuffle,
                           test_shots=test_shots,
                           meta_train=args.meta_train,
                           meta_val=args.meta_val,
                           meta_test=args.meta_test,
                           download=download)
    elif args.dataset == 'traffic_sign':
        dataset = traffic_sign(folder=folder,
                               shots=shots,
                               ways=ways,
                               shuffle=shuffle,
                               test_shots=test_shots,
                               meta_train=args.meta_train,
                               meta_val=args.meta_val,
                               meta_test=args.meta_test,
                               download=download)
    elif args.dataset == 'svhn':
        dataset = svhn(folder=folder,
                               shots=shots,
                               ways=ways,
                               shuffle=shuffle,
                               test_shots=test_shots,
                               meta_train=args.meta_train,
                               meta_val=args.meta_val,
                               meta_test=args.meta_test,
                               download=download)
    elif args.dataset == 'cars':
        dataset = cars(folder=folder,
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
    if args.dataset == 'miniimagenet' or args.dataset == 'tieredimagenet' or args.dataset == 'cub' or args.dataset == 'cars':
        wh_size = 5
    elif args.dataset == 'cifar_fs' or args.dataset == 'fc100' or args.dataset == 'vgg_flower' or args.dataset == 'aircraft' or args.dataset == 'traffic_sign' or args.dataset == 'svhn':
        wh_size = 2
        
    if args.model == '4conv':
        model = ConvNet(in_channels=3, out_features=args.num_ways, hidden_size=args.hidden_size, wh_size=wh_size)
    elif args.model == 'resnet':
        if args.blocks_type == 'a':
            blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlock]
        elif args.blocks_type == 'b':
            blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlockWithoutResidual]
        elif args.blocks_type == 'c':
            blocks = [BasicBlock, BasicBlock, BasicBlockWithoutResidual, BasicBlockWithoutResidual]
        elif args.blocks_type == 'd':
            blocks = [BasicBlock, BasicBlockWithoutResidual, BasicBlockWithoutResidual, BasicBlockWithoutResidual]
        elif args.blocks_type == 'e':
            blocks = [BasicBlockWithoutResidual, BasicBlockWithoutResidual, BasicBlockWithoutResidual, BasicBlockWithoutResidual]
        
        model = ResNet(blocks=blocks, keep_prob=1.0, avg_pool=True, drop_rate=0.0, out_features=args.num_ways, wh_size=1)
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