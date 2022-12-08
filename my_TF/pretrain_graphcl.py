import os
import argparse
from loader import MoleculeDataset_aug
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform

import torch

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN

from copy import deepcopy
import sys
sys.path.append('/mnt/dive/shared/xyxu/projects/GraphCL-master')
sys.path.append('/mnt/data/shared/xyxu/projects/GraphCL-master')
from termcolor import cprint
import losses


def parser():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default='zinc_standard_agent',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type=str, default='', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default='dropN')
    parser.add_argument('--aug_ratio1', type=float, default=0.2)
    parser.add_argument('--aug2', type=str, default='dropN')
    parser.add_argument('--aug_ratio2', type=float, default=0.2)

    parser.add_argument('--num-group', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=160)
    parser.add_argument('--lam_div', type=float, default=0.02)
    parser.add_argument('--PH', type=str, default='true')
    parser.add_argument('--folder', type=str, default='recover')

    # attention
    parser.add_argument('--att_norm', type=str, default='softmax')
    parser.add_argument('--bias', type=str, default='false')
    parser.add_argument('--loss_emb', type=str, default='binomial_deviance')
    parser.add_argument('--loss_div', type=str, default='div_bd')
    parser.add_argument('--add_global_group', type=str, default='false')
    parser.add_argument('--lam_glb', type=float, default=0.02)
    parser.add_argument('--save_every', type=int, default=1)


    args = parser.parse_args()
    return args


args = parser()


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x * h, dim=1)


class graphcl(nn.Module):
    def __init__(self, gnn, args):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.in_dim = args.dim_per_group
        if args.PH == 'true':
            self.projection_head = nn.Sequential(nn.Linear(self.in_dim, self.in_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(self.in_dim, self.in_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        x, _ = self.gnn(x, edge_index, edge_attr, batch)
        if args.PH == 'true':
            return self.projection_head(x)
        else:
            return x


def train(args, model, device, dataset, optimizer):
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model.train()

    train_acc_accum = 0
    train_loss_accum = 0

    for step, batch in enumerate(tqdm(zip(loader1, loader2), desc="Iteration")):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        x1 = model(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)

        loss_emb, _ = losses.select_loss(args.loss_emb, args)
        loss_div, _ = losses.select_loss(args.loss_div, args)
        loss = loss_emb(x1, x2) + args.lam_div * (loss_div(x1) + loss_div(x2))
        # loss = mul_group_loss(x1, x2, args.lam_div)
        if isinstance(loss, tuple):
            loss = loss[0]

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        # acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        acc = torch.tensor(0)
        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum / (step + 1), train_loss_accum / (step + 1)


def main():
    # Training settings
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # set up dataset
    dataset = MoleculeDataset_aug("../dataset/" + args.dataset, dataset=args.dataset)
    print(dataset)

    # set up model (Choose one from GIN, GCN, GAT, GraphSAGE), in which each layer is followed by BN
    gnn = GNN(args.num_layer, args.emb_dim, args, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)

    model = graphcl(gnn, args)

    model.to(device)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    save_path = f"./models_group_graphcl/{args.folder}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(1, args.epochs):
        print("====epoch " + str(epoch))

        train_acc, train_loss = train(args, model, device, dataset, optimizer)

        print(train_acc)
        print(train_loss)

        if epoch % args.save_every == 0:
            cprint('saving model...', 'red')
            file_name = '{}l-{}-{}_{}-{}_ld{}_{}.pth'.format(args.num_layer,
                                                             args.aug1, args.aug_ratio1,
                                                             args.aug2, args.aug_ratio2,
                                                             args.lam_div, epoch)
            torch.save(gnn.state_dict(), "./models_group_graphcl/{}/{}".format(args.folder, file_name))


if __name__ == "__main__":
    main()
