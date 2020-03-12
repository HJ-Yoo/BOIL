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
    scale_model.to(device=args.device)
    model.train()
    scale_model.train()
    
    # To control outer update parameter
    # If you want to control inner update parameter, please see update_parameters function in ./maml/utils.py
    freeze_params = [p for name, p in model.named_parameters() if 'classifier' not in name]
    learnable_params = [p for name, p in model.named_parameters() if 'classifier' in name]
    scale_params = [p for name, p in scale_model.named_parameters()]
    meta_optimizer = torch.optim.Adam([{'params': freeze_params, 'lr': args.meta_lr},
                                       {'params': learnable_params, 'lr': args.meta_lr},
                                       {'params': scale_params, 'lr': args.meta_lr}]) 
    
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
                # inner loop
                model.train()
                if args.adaptive_lr:
                    initial_params=model.state_dict()
                
                support_features, support_logit = model(support_input)
                if args.adaptive_lr:
                    query_features_, query_logit_ = model(query_input)
                inner_loss = F.cross_entropy(support_logit, support_target)
                
                model.zero_grad()
                params = update_parameters(model, inner_loss, step_size=args.step_size, first_order=args.first_order)
                
                if args.adaptive_lr_double_inner_loop:
                    support_features_, support_logit_ = model(support_input, params=params) # get features from task specific parameters
                    query_features_, query_logit_ = model(query_input, params=params)
                    second_inner_loss = F.cross_entropy(support_logit_, support_target)
                    
                    distance = torch.norm(torch.mean(support_features_, dim=0) - torch.mean(query_features_, dim=0))
                    adaptive_lr = torch.exp(-0.1 * distance * distance)
                    model.zero_grad()
                    adaptive_params = update_parameters(model, second_inner_loss, step_size=adaptive_lr, first_order=args.first_order) # not from initial parameter
                    params = adaptive_params
                
                elif args.double_inner_loop:
                    support_features_, support_logit_ = model(support_input, params=params)
                    second_inner_loss = F.cross_entropy(support_logit_, support_target)
                    model.zero_grad()
                    second_params = update_parameters(model, second_inner_loss, step_size=args.step_size, irst_order=args.first_order)
                    params = second_params
                
                elif args.adaptive_lr:
                    distance = torch.norm(torch.mean(support_features, dim=0) - torch.mean(query_features_, dim=0))
                    adaptive_lr = torch.exp(-0.1 * distance * distance)

                    model.load_state_dict(initial_params)
                    model.zero_grad()
                    adaptive_params = update_parameters(model, inner_loss, step_size=adaptive_lr, first_order=args.first_order)
                    params=adaptive_params
                
                # outer loop
                if args.meta_val or args.meta_test:
                    model.eval()
                    
                query_features, query_logit = model(query_input, params=params)
                outer_loss += F.cross_entropy(query_logit, query_target)
                if args.graph_regularizer:
                    support_features_, support_logit_ = model(support_input, params=params)
                    graph_loss = get_graph_regularizer(features=torch.cat((support_features_, query_features), dim=0), labels=support_target, model=scale_model, args=args)
                    outer_loss += args.graph_beta * graph_loss
                
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
    parser.add_argument('--dataset', type=str, help='Dataset: omniglot, miniimagenet, tieredimagenet, cifar_fs, cub, doublemnist, triplemnist')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
    parser.add_argument('--download', action='store_true', help='Download the dataset in the data folder.')
    parser.add_argument('--num-shots', type=int, default=5, help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate of meta optimizer.')

    parser.add_argument('--first-order', action='store_true', help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.7, help='Step-size for the gradient step for adaptation (default: 0.5).')
    parser.add_argument('--hidden-size', type=int, default=64, help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default='./output/', help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--save-name', type=str, default=None, help='Name of model (optional).')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--batch-iter', type=int, default=1200, help='Number of times to repeat train batches (i.e., total epochs = batch_iter * train_batches) (default: 1200).')
    parser.add_argument('--train-batches', type=int, default=50, help='Number of batches the model is trained over (i.e., validation save steps) (default: 50).')
    parser.add_argument('--valid-batches', type=int, default=25, help='Number of batches the model is validated over (default: 25).')
    parser.add_argument('--test-batches', type=int, default=2500, help='Number of batches the model is tested over (default: 2500).')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for data loading (default: 1).')
    
    parser.add_argument('--graph-gamma', type=float, default=5.0, help='classwise difference magnitude in making graph edges')
    parser.add_argument('--graph-beta', type=float, default=1e-5, help='hyperparameter for graph regularizer')
    parser.add_argument('--graph-edge-generation', type=str, default=None, help='where to get the features to make the graph')
    
    parser.add_argument('--adaptive-lr', action='store_true', help='adaptive learning rate in inner loop')
    parser.add_argument('--adaptive-lr-double-inner-loop', action='store_true', help='adaptive learning rate in the second inner loop')
    parser.add_argument('--double-inner-loop', action='store_true', help='maml with twice inner loop for comparison')

    parser.add_argument('--graph-regularizer', action='store_true', help='graph regularizer')
    parser.add_argument('--fc-regularizer', action='store_true', help='fully connected layer regularizer')
    parser.add_argument('--distance-regularizer', action='store_true', help='distance regularizer')
    parser.add_argument('--distance-lambda', type=float, default=1.0, help='modulate the magnitude of distance regularizer')
    parser.add_argument('--norm-regularizer', action='store_true', help='norm regularizer')
    parser.add_argument('--task-embedding-method', type=str, default=None, help='task embedding method')
    parser.add_argument('--edge-generation-method', type=str, default=None, help='edge generation method')
    
    parser.add_argument('--best-valid-error-test', action='store_true', help='Test using the best valid error model')
    parser.add_argument('--best-valid-accuracy-test', action='store_true', help='Test using the best valid accuracy model')
    
    parser.add_argument('--init', action='store_true', help='model initialization')

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
    model, scale_model = load_model(args)
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
        classifier_pca = torch.mm(torch.mm(u[:5,:], torch.diag(sigma)), torch.t(v))
        model.classifier.weight = torch.nn.Parameter(torch.cat([torch.mean(classifier_pca, dim=0, keepdims=True)] * args.num_ways, dim=0))
    
    log_pd = pd.DataFrame(np.zeros([args.batch_iter*args.train_batches, 6]),
                          columns=['train_error', 'train_accuracy', 'valid_error', 'valid_accuracy', 'test_error', 'test_accuracy'])
    
    if args.best_valid_error_test or args.best_valid_accuracy_test:
        filename = './output/miniimagenet_{}/logs/logs.csv'.format(args.save_name)
        logs = pd.read_csv(filename)

        if args.best_valid_error_test:
            valid_logs = list(logs[logs['valid_error']!=0]['valid_error'])
            best_valid_epochs = (valid_logs.index(min(valid_logs))+1)*50
        else:
            valid_logs = list(logs[logs['valid_accuracy']!=0]['valid_accuracy'])
            best_valid_epochs = (valid_logs.index(max(valid_logs))+1)*50

        best_valid_model = torch.load('./output/miniimagenet_{}/models/epochs_{}.pt'.format(args.save_name, best_valid_epochs))
        model.load_state_dict(best_valid_model)

        meta_test_loss_logs, meta_test_accuracy_logs = main(args=args, mode='meta_test')
        print ('loss: {}, accuracy: {}'.format(np.mean(meta_test_loss_logs), np.mean(meta_test_accuracy_logs)))
    else:
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
