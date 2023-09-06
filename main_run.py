import argparse
import torch
import numpy as np
import os

from server_PMAFL import server_PMAFL_run
from server_FedAvg import server_FedAvg_run
from server_FedRep import server_FedRep_run


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: n")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.5, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
    parser.add_argument('--local_updates', type=int, default=1000000, help="maximum number of local updates")
    parser.add_argument('--m_tr', type=int, default=500, help="maximum number of samples/user to use for training")
    parser.add_argument('--m_ft', type=int, default=500, help="maximum number of samples/user to use for fine-tuning")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')  # cnn, mlp
    parser.add_argument('--alg', type=str, default='fedavg', help='FL algorithm to use')  # fedrep, fedavg, fedsim
    parser.add_argument('--local_rep_ep', type=int, default=1, help="the number of local epochs for the representation for FedRep")
    parser.add_argument('--mu', type=float, default='0.1', help='FedProx parameter mu')
    parser.add_argument('--gmf', type=float, default='0', help='FedProx parameter gmf')

    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")  # cifar10, mnist
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--result_path', type=str, default='results_mnist5', help='define fed results save folder')

    args = parser.parse_args()
    return args



def main():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    algorithms = ['fedavg', 'fedrep', 'PMAFL']
    fracs = [0.1, 0.5]

    for alg in algorithms:
        args.alg = alg

        for frac in fracs:
            args.frac = frac

            if alg == 'fedavg':
                accs, loss, loss_t, var_accs = server_FedAvg_run(args)
            elif alg == 'PMAFL':
                accs, loss, loss_t, var_accs = server_PMAFL_run(args)
            elif alg == 'fedrep':
                accs, loss, loss_t, var_accs = server_FedRep_run(args)
            else:
                print('The algorithm is not implemented')









if __name__ == '__main__':
    main()

