import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torchmeta.utils.data import BatchMetaDataLoader
from maml.utils import (load_dataset, load_model, load_pretrained_model, load_best_valid_model,
                        update_parameters, get_accuracy)

def main(args, mode, iteration=None):
    dataset = load_dataset(args, mode)
    
    if args.meta_train:
        total = args.train_batches
    elif args.meta_val:
        total = args.valid_batches
    elif args.meta_test:
        total = args.test_batches
        
    dataloader = BatchMetaDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model.to(device=args.device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    
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

                model.zero_grad()
                params = update_parameters(model, inner_loss, step_size=args.step_size, first_order=args.first_order)
                
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
    parser.add_argument('--dataset', type=str, help='Dataset: omniglot, miniimagenet, tieredimagenet, cifar_fs, cub, doublemnist, triplemnist')
    parser.add_argument('--device', type=str, default='cuda:0', help='gpu device')
    parser.add_argument('--download', action='store_true', help='Download the dataset in the data folder.')
    parser.add_argument('--num-shots', type=int, default=5, help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5, help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate of meta optimizer.')

    parser.add_argument('--first-order', action='store_true', help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.5, help='Step-size for the gradient step for adaptation (default: 0.5).')
    parser.add_argument('--hidden-size', type=int, default=64, help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default='./output/', help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--save-name', type=str, default=None, help='Name of model (optional).')
    parser.add_argument('--batch-size', type=int, default=4, help='Number of tasks in a mini-batch of tasks (default: 4).')
    parser.add_argument('--batch-iter', type=int, default=1200, help='Number of times to repeat train batches (i.e., total epochs = batch_iter * train_batches) (default: 1200).')
    parser.add_argument('--train-batches', type=int, default=50, help='Number of batches the model is trained over (i.e., validation save steps) (default: 50).')
    parser.add_argument('--valid-batches', type=int, default=25, help='Number of batches the model is validated over (default: 25).')
    parser.add_argument('--test-batches', type=int, default=250, help='Number of batches the model is tested over (default: 250).')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for data loading (default: 1).')
    
    parser.add_argument('--init', action='store_true', help='Initialization with pre-trained model.')
    parser.add_argument('--pretrain-epochs', type=int, default=20, help='Number of epochs for training pretrained model.')
    parser.add_argument('--best-valid-error-test', action='store_true', help='Test using the best valid error model')
    parser.add_argument('--best-valid-accuracy-test', action='store_true', help='Test using the best valid accuracy model')

    args = parser.parse_args()
    args.device = torch.device(args.device)
    os.makedirs(os.path.join(args.output_folder, 'pretrained'), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, args.dataset+'_'+args.save_name, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, args.dataset+'_'+args.save_name, 'models'), exist_ok=True)
    
    model = load_model(args, maml=True)
    
    if args.init:
        pretrained_model_dict = load_pretrained_model(args)
        # model.classifier.weight.data = pretrained_model_dict['classifier.weight'][random.sample(range(64), 5),:]
        # u, _, _ = torch.svd(torch.t(pretrained_model_dict['classifier.weight']))
        # model.classifier.weight = torch.nn.Parameter(torch.cat([torch.mean(torch.t(u)[:5, :], dim=0, keepdims=True)] * args.num_ways, dim=0))
        # model.classifier.weight.data += 0.01 * torch.randn(model.classifier.weight.shape)
        load_layer_name = ['features', 'gcn1', 'gcn2']
        pretrained_model_dict = {k: v for k, v in pretrained_model_dict.items() if k.split(".")[0] in load_layer_name}
        model.load_state_dict(pretrained_model_dict, strict=False)
        
    if not args.best_valid_error_test and not args.best_valid_accuracy_test:
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
    else:
        best_valid_model = load_best_valid_model(args)
        model.load_state_dict(best_valid_model)
        meta_test_loss_logs, meta_test_accuracy_logs = main(args=args, mode='meta_test')
        print ('test loss: {}, test accuracy: {}'.format(np.mean(meta_test_loss_logs), np.mean(meta_test_accuracy_logs)))