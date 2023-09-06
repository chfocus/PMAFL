import copy
import numpy as np
import torch

from get_dataset import get_data
from get_models import get_model
from model_train import LocalUpdate
from model_test import test_img_local_all


def server_FedAvg_run(args):
    user_data_samples = np.ones(args.num_users)
    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    for idx in dict_users_train.keys():
        np.random.shuffle(dict_users_train[idx])

    for u_id in range(args.num_users):
        user_data_samples[u_id] = len(dict_users_train[u_id])

    # ------------ get model --------------------
    net_glob = get_model(args)
    net_glob.train()
    print(net_glob.state_dict().keys())
    # -------------------------initialize the local models---------------------------------
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

    for iter in range(args.epochs + 1):  # global rounds
        w_glob = {} # store the global parameters
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)  # number of selected users
        if iter == args.epochs:
            m = args.num_users

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)  # select users

        total_len = 0
        for ind, idx in enumerate(idxs_users):
            if args.epochs == iter:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])
            else:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            net_local.load_state_dict(w_local)

            last = iter == args.epochs
            w_local, loss = local.train(net=net_local.to(args.device), w_glob_keys=[], lr=args.lr, last=last)

            loss_locals.append(copy.deepcopy(loss))
            total_len += user_data_samples[idx]

            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key] * user_data_samples[idx]
                    w_locals[idx][key] = w_local[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] += w_local[key] * user_data_samples[idx]
                    w_locals[idx][key] = w_local[key]

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        net_glob.load_state_dict(w_glob)

        # test
        if iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 10:
            acc_test, loss_test, var_acc = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                         w_locals=None, indd=indd, dataset_train=dataset_train,
                                                         dict_users_train=dict_users_train, return_all=False)
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


