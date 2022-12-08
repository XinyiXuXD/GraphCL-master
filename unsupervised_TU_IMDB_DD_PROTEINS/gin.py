import os.path as osp
from typing import Tuple
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

import sys
sys.path.append('/data/xxy/GraphCL-master/0utils')
sys.path.append('/mnt/data/shared/xyxu/projects/GraphCL-master/0utils')
import multiple_embedding


class Encoder(torch.nn.Module):
    def __init__(self, num_features, args):
        super(Encoder, self).__init__()

        self.in_dim = num_features
        self.hidden_dim = args.hidden_dim
        self.num_gc_layers = args.num_gc_layers
        self.pool = args.pool
        self.start = args.start

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.global_fea = args.global_fea

        for i in range(self.num_gc_layers):

            if i:
                nn = Sequential(Linear(self.hidden_dim, self.hidden_dim), ReLU(),
                                Linear(self.hidden_dim, self.hidden_dim))
            else:
                nn = Sequential(Linear(self.in_dim, self.hidden_dim), ReLU(),
                                Linear(self.hidden_dim, self.hidden_dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(self.hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)
            in_emb_dim = (self.num_gc_layers - self.start) * self.hidden_dim
            if args.grouping_layer == 'mul_linear':
                self.grouping = multiple_embedding.MLGrouping(args, in_emb_dim)
            else:
                self.grouping = multiple_embedding.QGrouping(args, in_emb_dim)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        x_group = self.grouping(torch.cat(xs[self.start:], dim=-1), batch)
        if isinstance(x_group, Tuple): 
            x_group, _ = x_group
        if self.pool == 'add':
            x_pool = [global_add_pool(x, batch) for x in xs]
        elif self.pool == 'mean':
            x_pool = [global_mean_pool(x, batch) for x in xs]
        else:
            x_pool = [global_add_pool(x, batch) for x in xs]
        # x_pool = torch.cat(x_pool[self.start:], 1)

        if self.global_fea == 'pool0':
            x_pool = x_pool[0]
        elif self.global_fea == 'pool1':
            x_pool = x_pool[1]
        elif self.global_fea == 'pool2':
            x_pool = x_pool[2]
        elif self.global_fea == 'pool12':
            x_pool = torch.cat(x_pool[1:], 1)
        elif self.global_fea == 'none':
            x_pool = []
        elif self.global_fea == 'all':
            x_pool = torch.cat(x_pool, 1)
        return x_group, x_pool

    def get_embeddings(self, loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret, ret_g = [], []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, x_global = self.forward(x, edge_index, batch)
                x = F.normalize(x, dim=-1)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())

                # global
                if self.global_fea != 'none':
                    x_global = F.normalize(x_global, dim=-1)
                    ret_g.append(x_global.cpu().numpy())
        ret = np.concatenate(ret, 0)
        ret = np.reshape(ret, [ret.shape[0], -1])

        # global
        if self.global_fea != 'none':
            ret_g = np.concatenate(ret_g, 0)
            ret = np.concatenate([ret, ret_g], 1)
        y = np.concatenate(y, 0)
        return ret, y

    def group_correlation(self, fea):
        fea = np.reshape(fea, [fea.shape[0], self.k, -1])
        cor = np.eye(self.k, dtype=np.float32)
        for i in range(self.k):
            for j in range(i+1, self.k):
                this_s = np.sum(fea[:, i, :] * fea[:, j, :], axis=1)
                cor[i, j] = np.mean(this_s)
                cor[j, i] = np.mean(this_s)
        return cor

    def get_embeddings_v(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x_g, x = self.forward(x, edge_index, batch)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                   break

        return x_g, ret, y


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        try:
            num_features = dataset.num_features
        except:
            num_features = 1
        dim = 32

        self.encoder = Encoder(num_features, dim)

        self.fc1 = Linear(dim*5, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        x, _ = self.encoder(x, edge_index, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # print(data.x.shape)
        # [ num_nodes x num_node_labels ]
        # print(data.edge_index.shape)
        #  [2 x num_edges ]
        # print(data.batch.shape)
        # [ num_nodes ]
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_dataset)

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

if __name__ == '__main__':

    for percentage in [ 1.]:
        for DS in [sys.argv[1]]:
            if 'REDDIT' in DS:
                epochs = 200
            else:
                epochs = 100
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
            accuracies = [[] for i in range(epochs)]
            #kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
            dataset = TUDataset(path, name=DS) #.shuffle()
            num_graphs = len(dataset)
            print('Number of graphs', len(dataset))
            dataset = dataset[:int(num_graphs * percentage)]
            dataset = dataset.shuffle()

            kf = KFold(n_splits=10, shuffle=True, random_state=None)
            for train_index, test_index in kf.split(dataset):

                # x_train, x_test = x[train_index], x[test_index]
                # y_train, y_test = y[train_index], y[test_index]
                train_dataset = [dataset[int(i)] for i in list(train_index)]
                test_dataset = [dataset[int(i)] for i in list(test_index)]
                print('len(train_dataset)', len(train_dataset))
                print('len(test_dataset)', len(test_dataset))

                train_loader = DataLoader(train_dataset, batch_size=128)
                test_loader = DataLoader(test_dataset, batch_size=128)
                # print('train', len(train_loader))
                # print('test', len(test_loader))

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = Net().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(1, epochs+1):
                    train_loss = train(epoch)
                    train_acc = test(train_loader)
                    test_acc = test(test_loader)
                    accuracies[epoch-1].append(test_acc)
                    tqdm.write('Epoch: {:03d}, Train Loss: {:.7f}, '
                          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                                       train_acc, test_acc))
            tmp = np.mean(accuracies, axis=1)
            print(percentage, DS, np.argmax(tmp), np.max(tmp), np.std(accuracies[np.argmax(tmp)]))
            input()