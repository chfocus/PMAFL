import copy
import itertools
import numpy as np
import torch

from get_dataset import get_data
from get_models import get_model
from model_train import LocalUpdate
from model_test import test_img_local_all

def get_shared_net_keys(dataset, net_glob):
    if 'mnist' in dataset:
        w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2]]
        return w_glob_keys
    elif 'cifar' in dataset:
        w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 3, 4]]
        return w_glob_keys
    else:
        exit('Error: the shared layers are not defined')


def server_FedRep_run(args):
    user_data_samples = np.ones(args.num_users)
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    for idx in dict_users_train.keys():
        np.random.shuffle(dict_users_train[idx])

    for u_id in range(args.num_users):
        user_data_samples[u_id] = len(dict_users_train[u_id])

    net_glob = get_model(args)
    net_glob.train()

    total_num_layers = len(net_glob.state_dict().keys())  #
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    w_glob_keys = get_shared_net_keys(args.dataset, net_glob)
    w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    print('total_num_layers=', total_num_layers)
    print('w_glob_keys=', w_glob_keys)
    print('net_keys=', net_keys)

    w_locals = {}  # local models for all users
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict

    indd = None
    loss_train = []
    accs = []
    loss_round = []
    var_accs = []

    for iter in range(args.epochs + 1):
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)  # number of selected users
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # select users

        total_len = 0
        for ind, idx in enumerate(idxs_users):
            if args.epochs == iter:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            if args.alg != 'fedavg':
                for k in w_locals[idx].keys():
                    if k not in w_glob_keys:
                        w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)

            last = iter == args.epochs
            w_local, loss = local.train(net=net_local.to(args.device), w_glob_keys=w_glob_keys, lr=args.lr, last=last)

            loss_locals.append(copy.deepcopy(loss))
            total_len += user_data_samples[idx]

            # copy parameters
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key] * user_data_samples[idx]  # add to global model
                    w_locals[idx][key] = w_local[key]  # save to local model
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        w_glob[key] += w_local[key] * user_data_samples[idx]
                    else:
                        w_glob[key] += w_local[key] * user_data_samples[idx]
                    w_locals[idx][key] = w_local[key]  # save to local model

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)
        net_glob.load_state_dict(w_glob)

        # ---------- test ---------------
        if iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 10:
            if args.alg == 'fedavg':
                acc_test, loss_test, var_acc = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                         w_locals=None, indd=indd, dataset_train=dataset_train,
                                                         dict_users_train=dict_users_train, return_all=False)
            else:
                acc_test, loss_test, var_acc = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                         w_glob_keys=w_glob_keys, w_locals=w_locals, indd=indd,
                                                         dataset_train=dataset_train, dict_users_train=dict_users_train,
                                                         return_all=False)
            accs.append(acc_test)
            loss_round.append(loss_test)
            var_accs.append(var_acc)

            if iter != args.epochs:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    iter, loss_avg, loss_test, acc_test))
            else:
                print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    loss_avg, loss_test, acc_test))

    return accs, loss_round, loss_train, var_accs



