import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import outline1
import argparse
import torch
import losses
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def arg_parse():
    parser = argparse.ArgumentParser(description='Visualization')

    parser.add_argument('--prior', dest='prior', action='store_const',
                        const=True, default=False)

    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3)
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--log-interval', default=1, type=int)
    parser.add_argument('--aug', nargs=2, default=['dnodes', 'dnodes'], type=str)

    parser.add_argument('--lam_div', nargs='*', default=0.5, type=float)
    parser.add_argument('--embedding_dim', type=int, nargs='*', default=160)
    parser.add_argument('--num_group', nargs='*', default=4)
    parser.add_argument('--start', default=1, type=int)
    parser.add_argument('--add_global_group', action='store_true')
    parser.add_argument('--lam_glb', default=0.01, type=float)
    parser.add_argument('--pool', type=str, default='mean')
    parser.add_argument('--att-norm', type=str, default='softmax')

    parser.add_argument('--loss_emb', type=str, default='binomial_deviance')
    parser.add_argument('--loss_div', type=str, default='div_bd')
    parser.add_argument('--learning_rate', nargs='*', default=0.001, type=float)
    parser.add_argument('--weight_decay', nargs='*', default=[0] * 5,
                        type=float, help='conv, bn, que, key, val')
    parser.add_argument('--folder_name', type=str, default='visualization')

    parser.add_argument('--DS', default='PROTEINS', help='dataset')
    parser.add_argument('--rs', default=3, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--bias', type=str, default='true')
    return parser.parse_args()


def main(args):
    outline1.setup_seed(args.rs)
    train_loader, eval_loader = outline1.pre_data(args)

    device = torch.device('cuda')
    model = outline1.simclr(args).to(device)
    to_optim = outline1.set_lr_wd(model, args)

    Cost_emb, to_optim = losses.select_loss(args.loss_emb, args, to_optim)
    Cost_div, to_optim = losses.select_loss(args.loss_div, args, to_optim)

    optimizer = torch.optim.Adam(to_optim)
    outline1.print_info(args)

    for epoch in range(1, 1+args.epochs):
        loss_all = outline1.train(model, Cost_emb, Cost_div, train_loader, optimizer, device, args)

        if epoch % args.log_interval == 0:
            acc_val, acc = outline1.eval(model, eval_loader)

    query = model.encoder.grouping.w_q.weight.squeeze()
    query = query.detach().cpu().numpy()
    compute_correlation(query)


def compute_correlation(query):
    x = query / np.linalg.norm(query, axis=-1)[:, np.newaxis]
    cor = np.matmul(x, x.T)

    fig, ax = plt.subplots(figsize=(3, 3))

    target = pd.DataFrame(np.round(cor, 2), columns=range(1, 5), index=range(1, 5))
    sns.heatmap(target, linewidths=.5,
                annot=True, annot_kws={"size": 12},
                vmax=1, vmin=-0.2, square=True,
                cmap="YlGnBu", cbar_kws={"shrink": 1.0})


if __name__=='__main__':
    args = arg_parse()
    args.device = torch.device('cuda')
    main(args)
