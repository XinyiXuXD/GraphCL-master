import torch
from arguments import arg_parse
import outline1
import os
import losses
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def main(args):
    acc_all_rs = {'val': [], 'test': []}
    for rs in args.random_seed:
        args.rs = rs
        outline1.setup_seed(args.rs)
        train_loader, eval_loader = outline1.pre_data(args)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = outline1.simclr(args).to(device)
        to_optim = outline1.set_lr_wd(model, args)

        Cost_emb, to_optim = losses.select_loss(args.loss_emb, args, to_optim)
        Cost_div, to_optim = losses.select_loss(args.loss_div, args, to_optim)

        optimizer = torch.optim.Adam(to_optim)
        outline1.print_info(args)

        acc_per_rs = {'val': [], 'test': []}
        for epoch in range(1, args.epochs + 1):
            # train
            loss_all = outline1.train(model, Cost_emb, Cost_div, train_loader, optimizer, device, args)
            print('Epoch {}, Loss {:.3f}'.format(epoch, loss_all / len(train_loader)))
            # eval
            if epoch % args.log_interval == 0:
                acc_val, acc = outline1.eval(model, eval_loader)
                acc_per_rs['val'].append(acc_val)
                acc_per_rs['test'].append(acc)

        acc_all_rs['val'].append(acc_per_rs['val'])
        acc_all_rs['test'].append(acc_per_rs['test'])

    outline1.set_logging(args, acc_all_rs)


if __name__ == '__main__':

    from itertools import product
    args = arg_parse()
    args.device = torch.device('cuda')
    lam_div, learning_rate, num_group, reduction_ratio, embedding_dim = \
        args.lam_div, args.learning_rate, args.num_group, args.reduction_ratio, args.embedding_dim

    for param in product(reduction_ratio, num_group, lam_div, learning_rate, embedding_dim):
        args.reduction_ratio, args.num_group, args.lam_div, args.learning_rate, args.embedding_dim = param
        main(args)
        

