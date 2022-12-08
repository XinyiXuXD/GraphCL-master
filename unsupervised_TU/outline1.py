import torch
import numpy as np
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import torch.nn as nn
from gin import Encoder
from evaluate_embedding import evaluate_embedding

import datetime
import pytz
import os
from funcs import gimme_save_string
from termcolor import cprint


class simclr(nn.Module):
    def __init__(self, args, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.k = args.num_group
        self.hidden_dim = args.hidden_dim
        self.num_gc_layers = args.num_gc_layers
        self.start = args.start
        self.add_global_group = args.add_global_group

        self.encoder = Encoder(args.dataset_num_features, args)

        self.embedding_dim = mi_units = args.hidden_dim * args.num_gc_layers
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):

        x_group, x_pool = self.encoder(x, edge_index, batch)
        return x_group


def pre_data(args):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', args.DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    train_data, eval_data = TUDataset(path, name=args.DS, aug=args.aug).shuffle(), \
                            TUDataset(path, name=args.DS, aug=['none', 'none']).shuffle()

    try:
        args.dataset_num_features = train_data.get_num_feature()
    except:
        args.dataset_num_features = 1

    print(len(train_data))
    print(args.dataset_num_features)
    args.num_train = len(train_data)

    train_loader, eval_loader = DataLoader(train_data, batch_size=args.batch_size), \
                                DataLoader(eval_data, batch_size=args.batch_size)

    args.n_classes = len(train_loader.dataset)

    return train_loader, eval_loader


def train(model, Cost_emb, Cost_div, train_loader, optimizer, device, args):
    model.train()
    loss_all = 0
    for data in train_loader:
        data, data_aug = data
        optimizer.zero_grad()

        node_num, _ = data.x.size()
        data = data.to(device)
        x = model(data.x, data.edge_index, data.batch)

        if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' \
                or args.aug == 'random4':
            # node_num_aug, _ = data_aug.x.size()
            edge_idx = data_aug.edge_index.numpy()
            _, edge_num = edge_idx.shape
            idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

            node_num_aug = len(idx_not_missing)
            data_aug.x = data_aug.x[idx_not_missing]

            data_aug.batch = data.batch[idx_not_missing]
            idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
            edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                        not edge_idx[0, n] == edge_idx[1, n]]
            data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

        data_aug = data_aug.to(device)
        x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch)

        loss_emb = Cost_emb(x, x_aug, data.ind)
        loss_div = Cost_div(x, x_aug)
        loss = loss_emb + args.lam_div * loss_div

        print("loss: {:.3f}; loss_emb: {:.3f};  loss_div: {:.3f}".format(
            loss.item(), loss_emb.item(), loss_div.item()))

        loss_all += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return loss_all


def train_club(model, Cost_emb, Cost_div, train_data, optimizer, device, args):
    Cost_div.clubs.train()
    # loader1 = DataLoader(train_data, args.batch_size, shuffle=False)
    # args.n_classes = len(loader1.dataset)
    # args.num_train = len(loader1)

    loss_all = 0
    for data, _ in train_data:
        optimizer.zero_grad()

        node_num, _ = data.x.size()
        data = data.to(device)
        x = model(data.x, data.edge_index, data.batch)

        loss_club = Cost_div.club_params_loss(x)

        print("loss_club: {:.3f}".format(loss_club.item()))

        loss_all += loss_club.item() * data.num_graphs
        loss_club.backward()
        optimizer.step()
    return loss_all


def set_lr_wd(model, args):
    conv_param, bn_param = [], []
    que_wei, key_wei, val_wei = [], [], []
    for name, param in model.named_parameters():
        print(name, param.shape)
        if 'convs' in name:
            conv_param.append(param)
        elif 'bns' in name:
            bn_param.append(param)
        elif 'w_q' in name:
            que_wei.append(param)
        elif 'w_k' in name:
            key_wei.append(param)
        elif 'w_v' in name:
            val_wei.append(param)
    to_optim = [{'params': conv_param, 'weight_decay': args.weight_decay[0], 'lr': args.learning_rate},
                                 {'params': bn_param, 'weight_decay': args.weight_decay[1], 'lr': args.learning_rate},
                                 {'params': que_wei, 'weight_decay': args.weight_decay[2], 'lr': args.learning_rate},
                                 {'params': key_wei, 'weight_decay': args.weight_decay[3], 'lr': args.learning_rate},
                                 {'params': val_wei, 'weight_decay': args.weight_decay[4], 'lr': args.learning_rate}]
    return to_optim


def eval(model, eval_loader):
    model.eval()
    emb, y = model.encoder.get_embeddings(eval_loader)
    acc_val, acc = evaluate_embedding(emb, y)
    return acc_val, acc


def setup_seed(seed):
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def print_info(args):
    cprint('================')
    cprint('Dataset: {}'.format(args.DS), 'green')
    cprint('Learning the effect of {}'.format(args.folder_name), 'green')
    cprint('num_group: {}'.format(args.num_group), 'green')
    cprint('lambda_div: {}'.format(args.lam_div), 'green')
    cprint('lr: {}'.format(args.learning_rate), 'green')
    cprint('random_seed: {}'.format(args.rs), 'green')
    cprint('num_features: {}'.format(args.dataset_num_features), 'green')
    cprint('hidden_dim: {}'.format(args.hidden_dim), 'green')
    cprint('num_gc_layers: {}'.format(args.num_gc_layers), 'green')
    cprint('================')


def set_logging(args, acc):
    project_path = os.getcwd()
    args.save_path = os.path.join(project_path, 'training_results_r1', args.DS, args.folder_name)
    if args.folder_name != 'test':
        date = datetime.datetime.now(tz=pytz.timezone('US/Central'))
        time_string = '{}-{}-{}'.format(date.month, date.day, date.hour)

        param_string = 'k{}-ld{}-lr{}-rr{}-{}-{}'.format(args.num_group, args.lam_div, args.learning_rate,
                                                         args.reduction_ratio, args.aug[0], args.aug[1])
        args.save_path += '/{}_{}'.format(param_string, time_string)

    counter = 1
    while os.path.exists(args.save_path):
        args.save_path += '_' + str(counter)
        counter += 1
    try:
        os.makedirs(args.save_path)
    except:
        args.save_path = args.save_path.replace('/dive', '/data')
        os.makedirs(args.save_path)

    with open(args.save_path + '/Paramter_Info.tex', 'w') as f:
        f.write(gimme_save_string(args))

    val_accs = acc['val']
    test_accs = acc['test']
    val_avg = np.mean(val_accs, axis=0)
    val_std = np.std(val_accs, axis=0)
    test_avg = np.mean(test_accs, axis=0)
    test_std = np.std(test_accs, axis=0)
    best_epoch = np.argmax(test_avg)
    cprint('best_test_acc is {} at epoch {}'.format(test_avg[best_epoch], best_epoch), 'red')
    with open(args.save_path + '/results.tex', 'w') as f:
        for i in range(len(test_accs)):
            f.write('random_seed: {}\n'.format(i))
            f.write('val: {}\n'.format(val_accs[i]))
            f.write('test: {}\n'.format(test_accs[i]))
        f.write('*******\n')
        f.write('val_avg: {}, \nval_std: {}\n'.format(val_avg, val_std))
        f.write('test_avg: {}, \ntest_std: {}\n'.format(test_avg, test_std))
        f.write('best_test_avg: ({}, {}), best_epoch: {}\n'.format(test_avg[best_epoch],
                                                                   test_std[best_epoch], best_epoch))
        f.write('*******\n')
        f.write('\n')