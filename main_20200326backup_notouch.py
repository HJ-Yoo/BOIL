import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torchmeta.utils.data import BatchMetaDataLoader
from maml.utils import load_dataset, load_model, update_parameters, get_accuracy, get_graph_regularizer
    
def main(args, mode, iteration=None):
    dataset = load_dataset(args, mode)
    dataloader = BatchMetaDataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    model.to(device=args.device)
    model.train()
    
    # To control outer update parameter
    # If you want to control inner update parameter, please see update_parameters function in ./maml/utils.py
    freeze_params = [p for name, p in model.named_parameters() if 'classifier' not in name]
    learnable_params = [p for name, p in model.named_parameters() if 'classifier' in name]
    meta_optimizer = torch.optim.Adam([{'params': freeze_params, 'lr': args.meta_lr},
                                       {'params': learnable_params, 'lr': args.meta_lr}]) 
    
    if args.meta_train:
        total = args.train_batches
    elif args.meta_val:
        total = args.valid_batches
    elif args.meta_test:
        total = args.test_batches
        
    loss_logs, accuracy_logs = [], []
    
    # Training loop
    with tqdm(dataloader, total=total, leave=False) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()
            
            support_inputs, support_targets = batch['train']
            support_inputs = support_inputs.to(device=args.device)
            support_targets = support_targets.to(device=args.device)

            query_inputs, query_targets = batch['test']
            query_inputs = query_inputs.to(device=args.device)
            query_targets = query_targets.to(device=args.device)

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)
            
            for task_idx, (support_input, support_target, query_input, query_target) in enumerate(zip(support_inputs, support_targets, query_inputs, query_targets)):
                model.train()
                
                support_features, support_logit = model(support_input)
                inner_loss = F.cross_entropy(support_logit, support_target)
                if args.graph_regularizer:
                    _query_features, _query_logit = model(query_input)
                    features = torch.cat((support_features, _query_features), dim=0)
                    labels = [support_target, query_target]
                    graph_loss = get_graph_regularizer(features=features, labels=labels, args=args)
                    inner_loss += graph_loss * args.graph_beta
                    
                model.zero_grad()
                params = update_parameters(model, inner_loss, extractor_step_size=args.extractor_step_size, classifier_step_size=args.classifier_step_size, first_order=args.first_order)
                
                if args.meta_val or args.meta_test:
                    model.eval()
                
                query_features, query_logit = model(query_input, params=params)
                outer_loss += F.cross_entropy(query_logit, query_target)
                
                with torch.no_grad():
                    accuracy += get_accuracy(query_logit, query_target)
                    
            outer_loss.div_(args.batch_size)
            accuracy.div_(args.batch_size)
            loss_logs.append(outer_loss.item())
            accuracy_logs.append(accuracy.item())
            
            if args.meta_train:
                outer_loss.backward()
                meta_optimizer.step()

            postfix = {'mode': mode, 'iter': iteration, 'acc': round(accuracy.item(), 5)}
            pbar.set_postfix(postfix)
            if batch_idx+1 == total:
                break

    # Save model
    if args.meta_train:
        filename = os.path.join(args.output_folder, args.dataset+'_'+args.save_name, 'models', 'epochs_{}.pt'.format((iteration+1)*total))
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)
    
    return loss_logs, accuracy_logs

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')
    
    parser.add_argument('--folder', type=str, help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str, help='Dataset: miniimagenet, tieredimagenet, cifar_fs, fc100')
    parser.add_argument('--model', type=str, help='Model: smallconv, largeconv, resnet')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
    parser.add_argument('--download', action='store_true', help='Download the dataset in the data folder.')
    parser.add_argument('--num-shots', type=int, default=5, help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate of meta optimizer.')

    parser.add_argument('--first-order', action='store_true', help='Use the first-order approximation of MAML.')
    parser.add_argument('--extractor-step-size', type=float, default=0.5, help='Extractor step-size for the gradient step for adaptation (default: 0.5).')
    parser.add_argument('--classifier-step-size', type=float, default=0.5, help='Classifier step-size for the gradient step for adaptation (default: 0.5).')
    parser.add_argument('--hidden-size', type=int, default=64, help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default='./output/', help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--save-name', type=str, default=None, help='Name of model (optional).')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--batch-iter', type=int, default=600, help='Number of times to repeat train batches (i.e., total epochs = batch_iter * train_batches) (default: 600).')
    parser.add_argument('--train-batches', type=int, default=100, help='Number of batches the model is trained over (i.e., validation save steps) (default: 100).')
    parser.add_argument('--valid-batches', type=int, default=25, help='Number of batches the model is validated over (default: 25).')
    parser.add_argument('--test-batches', type=int, default=2500, help='Number of batches the model is tested over (default: 2500).')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for data loading (default: 1).')
        
    parser.add_argument('--graph-regularizer', action='store_true', help='classwise graph regularizer init')
    parser.add_argument('--graph-beta', type=float, default=1e-2, help='Hyperparameter for controlling graph loss.')
    parser.add_argument('--graph-type', type=str, default='single', help='How to calculate graph loss')
    
    parser.add_argument('--init', action='store_true', help='extractor init')
        
    args = parser.parse_args()  
    os.makedirs(os.path.join(args.output_folder, args.dataset+'_'+args.save_name, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, args.dataset+'_'+args.save_name, 'models'), exist_ok=True)
    
    arguments_txt = "" 
    for k, v in args.__dict__.items():
        arguments_txt += "{}: {}\n".format(str(k), str(v))
    filename = os.path.join(args.output_folder, args.dataset+'_'+args.save_name, 'logs', 'arguments.txt')
    with open(filename, 'w') as f:
        f.write(arguments_txt[:-1])
    
    args.device = torch.device(args.device)  
    model = load_model(args)
    
    if args.init:
        args.num_ways = 64
        pretrained_model = load_model(args)
        filename = './pretrained.pt'
        checkpoint = torch.load(filename)
        pretrained_model.load_state_dict(checkpoint, strict=True)

        for pre_p, p in list(zip(pretrained_model.parameters(), model.parameters())):
            if pre_p.shape == p.shape:
                p.data = copy.deepcopy(pre_p.data)
        
        args.num_ways = 5
        u, sigma, v = torch.svd(pretrained_model.classifier.weight.data)
        classifier_pca = torch.mm(torch.mm(u[:2,:], torch.diag(sigma)), torch.t(v))
        model.classifier.weight = torch.nn.Parameter(torch.cat([torch.mean(classifier_pca, dim=0, keepdims=True)] * args.num_ways, dim=0))
    
    log_pd = pd.DataFrame(np.zeros([args.batch_iter*args.train_batches, 6]),
                          columns=['train_error', 'train_accuracy', 'valid_error', 'valid_accuracy', 'test_error', 'test_accuracy'])

    for iteration in tqdm(range(args.batch_iter)):
        meta_train_loss_logs, meta_train_accuracy_logs = main(args=args, mode='meta_train', iteration=iteration)
        meta_valid_loss_logs, meta_valid_accuracy_logs = main(args=args, mode='meta_valid', iteration=iteration)
        log_pd['train_error'][iteration*args.train_batches:(iteration+1)*args.train_batches] = meta_train_loss_logs
        log_pd['train_accuracy'][iteration*args.train_batches:(iteration+1)*args.train_batches] = meta_train_accuracy_logs
        log_pd['valid_error'][(iteration+1)*args.train_batches-1] = np.mean(meta_valid_loss_logs)
        log_pd['valid_accuracy'][(iteration+1)*args.train_batches-1] = np.mean(meta_valid_accuracy_logs)
        filename = os.path.join(args.output_folder, args.dataset+'_'+args.save_name, 'logs', 'logs.csv')
        log_pd.to_csv(filename, index=False)
    meta_test_loss_logs, meta_test_accuracy_logs = main(args=args, mode='meta_test')
    log_pd['test_error'][args.batch_iter*args.train_batches-1] = np.mean(meta_test_loss_logs)
    log_pd['test_accuracy'][args.batch_iter*args.train_batches-1] = np.mean(meta_test_accuracy_logs)
    filename = os.path.join(args.output_folder, args.dataset+'_'+args.save_name, 'logs', 'logs.csv')
    log_pd.to_csv(filename, index=False)