import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--local', dest='local', action='store_const',
                        const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const',
                        const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const',
                        const=True, default=False)

    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3)
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32)

    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--log-interval', default=1, type=int)

    parser.add_argument('--att-norm', type=str, default='softmax')
    parser.add_argument('--pool', type=str, default='mean')
    parser.add_argument('--start', default=1, type=int)
    parser.add_argument('--add_global_group', action='store_true')
    parser.add_argument('--lam_glb', default=0.01, type=float)
    parser.add_argument('--random_seed', nargs='*', default=[0, 1, 2, 3, 4], type=int)
    # parser.add_argument('--random_seed', nargs='*', default=[1], type=int)

    parser.add_argument('--DS', default='IMDB-BINARY', dest='DS', help='Dataset')
    parser.add_argument('--learning_rate', nargs='*', default=[0.001], type=float)
    parser.add_argument('--weight_decay', nargs='*', default=[0.00] * 5,
                        type=float, help='conv, bn, que, key, val')
    parser.add_argument('--reduction_ratio', nargs='*', default=[1],
                        type=int, help='reduction ratio of key')

    parser.add_argument('--project_path', type=str,
                        default='/mnt/dive/shared/xyxu/projects/GraphCL-master/unsupervised_TU')
    parser.add_argument('--folder_name', type=str, default='test')
    parser.add_argument('--embedding_dim', type=int, nargs='*', default=[192])
    parser.add_argument('--aug', nargs=2, default=['dnodes', 'dnodes'], type=str)
    parser.add_argument('--aug_ratio', nargs=2, default=[0.1, 0.1], type=float)

    parser.add_argument('--loss_emb', type=str, default='IG_binomial_deviance')
    parser.add_argument('--loss_div', type=str, default='div_bd')
    parser.add_argument('--lam_div', nargs='*', default=[0.5], type=float)
    parser.add_argument('--num_group', nargs='*', default=[4],
                        type=int, help='reduction ratio of key')

    parser.add_argument('--top_k', default=11, type=int)

    parser.add_argument('--feat_str', type=str, default='deg+odeg100')
    parser.add_argument('--pre_transform', type=str, default='true')

    parser.add_argument('--loss_margin_margin', nargs='*', default=0.1, type=float)
    # parser.add_argument('--loss_margin_beta', nargs='*', default=[0.01, 0.02, 0.03, 0.04, 0.05], type=float)
    parser.add_argument('--loss_margin_beta', nargs='*', default=0.8, type=float)

    parser.add_argument('--loss_margin_beta_constant', action='store_true')
    parser.add_argument('--loss_margin_beta_lr', nargs='*', default=0.5, type=float)

    parser.add_argument('--miner_distance_lower_cutoff', default=0.5, type=float)
    parser.add_argument('--miner_distance_upper_cutoff', default=1.4, type=float)

    parser.add_argument('--loss_soft_margin', nargs='*', default=1.0, type=float)
    parser.add_argument('--loss_soft_beta', nargs='*', default=1.0, type=float)
    parser.add_argument('--loss_soft_beta_lr', nargs='*', default=0.0, type=float)
    parser.add_argument('--club_hidden', nargs='*', default=4, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    print(args.lam_div)
    print(args.lr)
