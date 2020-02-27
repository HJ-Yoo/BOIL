import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

from maml.model import OmniglotNet, MiniimagenetNet # TieredimagenetNet, Cifar_fsNet, CubNet, DoublemnistNet, TriplemnistNet
from torchmeta.datasets.helpers import omniglot, miniimagenet, tieredimagenet, cifar_fs, cub, doublemnist, triplemnist

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

def load_model(args, maml=True):
    if args.dataset == 'omniglot':
        model = OmniglotNet(1, args.num_ways, hidden_size=args.hidden_size)
    elif args.dataset == 'miniimagenet':
        model = MiniimagenetNet(3, args.num_ways if maml else 64, hidden_size=args.hidden_size)
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

def load_pretrained_model(args):
    pretrained_model = load_model(args, maml=False)
    pretrained_model_filename = os.path.join(args.output_folder, 'pretrained', '{}.pt'.format(args.save_name))

    if not os.path.exists(pretrained_model_filename):
        num_classes = 64
        pretrained_model.to(device=args.device)
        pretrained_model.train()
        pretrained_model_optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=0.01)

        dataset = load_dataset(args, 'meta_train')
        data = torch.cat([torch.from_numpy(np.array(dataset.dataset.__getitem__(i).data))/255. for i in range(num_classes)], dim=0).permute(0, 3, 1, 2).float()
        labels = torch.tensor(sum([[i]*600 for i in range(num_classes)], []))

        dataloader = DataLoader(dataset=list(zip(data,labels)), batch_size=128, shuffle=True)
        for _ in tqdm(range(args.pretrain_epochs)):
            for input, target in dataloader:
                input = input.type(torch.FloatTensor).to(device=args.device)
                target = target.type(torch.LongTensor).to(device=args.device)

                features, logit = pretrained_model(input)
                loss = F.cross_entropy(logit, target)

                pretrained_model_optimizer.zero_grad()
                loss.backward()
                pretrained_model_optimizer.step()

        with open(pretrained_model_filename, 'wb') as f:
            state_dict = pretrained_model.state_dict()
            torch.save(state_dict, f)

    # Load pretrained model
    pretrained_model.load_state_dict(torch.load(pretrained_model_filename), strict=True)
    return pretrained_model.state_dict()

def load_best_valid_model(args):
    filename = os.path.join(args.output_folder, args.dataset+'_'+args.save_name, 'logs', 'logs.csv')
    logs = pd.read_csv(filename)
    if args.best_valid_error_test:
        valid_logs = list(logs[logs['valid_error']!=0]['valid_error'])
        best_valid_epoch = (valid_logs.index(min(valid_logs))+1)*50
    else:
        valid_logs = list(logs[logs['valid_accuracy']!=0]['valid_accuracy'])
        best_valid_epoch = (valid_logs.index(max(valid_logs))+1)*50
        
    return torch.load('./output/miniimagenet_{}/models/epochs_{}.pt'.format(args.save_name, best_valid_epoch))

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