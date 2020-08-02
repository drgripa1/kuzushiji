import argparse


def parse_args_base(parser):
    parser.add_argument('--n', help='network complexity', type=int, default=3)
    parser.add_argument('--batch', help='batch size', type=int, default=256)
    parser.add_argument('--loss_type', help='loss function in {crossentropy, softmaxnorm, softtriple}')
    parser.add_argument('--la', default=2.0, type=float, help='lambda')
    parser.add_argument('--gamma', default=0.1, type=float, help='gamma')
    parser.add_argument('--tau', default=0.2, type=float, help='tau')
    parser.add_argument('--margin', default=0.01, type=float, help='margin')
    parser.add_argument('-K', default=10, type=int, help='K')
    return parser


def parse_args_train():
    parser = argparse.ArgumentParser()
    parser = parse_args_base(parser)
    parser.add_argument('--checkpoint_dir', default='./checkpoint')
    parser.add_argument('--print_freq', help='print loss freq', type=int, default=10)
    parser.add_argument('--save_params_freq', help='parameters saving freq', type=int, default=1000)
    parser.add_argument('--modellr', help='initial learning rate of network', type=float, default=0.05)
    parser.add_argument('--centerlr', help='initial learning rate of centers', type=float, default=0.02)
    parser.add_argument('--eps', help='Adam optimizer epsilon', type=float, default=0.01)
    parser.add_argument('--weight_decay', help='optimizer weight decay (L2 reg.)', type=float, default=0.0001)
    parser.add_argument('--decay_lr_1', help='iteration at which lr decays 1st', type=int, default=16000)
    parser.add_argument('--decay_lr_2', help='iteration at which lr decays 2nd', type=int, default=24000)
    parser.add_argument('--lr_decay_rate', help='lr *= lr_decay_rate at drop_lr_i-th iteration', type=float, default=0.1)
    parser.add_argument('--n_iter', help='learning iterations', type=int, default=32000)
    args = parser.parse_args()

    return args


def parse_args_test():
    parser = argparse.ArgumentParser()
    parser = parse_args_base(parser)
    parser.add_argument('--params_path_m', help='path to saved model weights')
    parser.add_argument('--params_path_c', help='path to saved criterion weights')
    args = parser.parse_args()

    return args
