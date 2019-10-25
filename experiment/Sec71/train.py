import os, sys
import argparse
import copy
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.externals import joblib
import torch
import torch.nn as nn
from DataModule import MnistModule, NewsModule, AdultModule
from MyNet import LogReg, DNN, NetList

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def settings_logreg(key):
    assert key in ['mnist', '20news', 'adult']
    if key == 'mnist':
        module = MnistModule()
        module.append_one = False
        n_tr, n_val, n_test = 200, 200, 200
        lr, decay, num_epoch, batch_size = 0.1, True, 5, 5
        return module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size)
    elif key == '20news':
        module = NewsModule()
        module.append_one = False
        n_tr, n_val, n_test = 200, 200, 200
        lr, decay, num_epoch, batch_size = 0.01, True, 10, 5
        return module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size)
    elif key == 'adult':
        module = AdultModule(csv_path='./data')
        module.append_one = False
        n_tr, n_val, n_test = 200, 200, 200
        lr, decay, num_epoch, batch_size = 0.1, True, 20, 5
        return module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size)

def settings_dnn(key):
    assert key in ['mnist', '20news', 'adult']
    if key == 'mnist':
        module = MnistModule()
        module.append_one = False
        n_tr, n_val, n_test = 200, 200, 200
        m = [8, 8]
        alpha = 0.001
        lr, decay, num_epoch, batch_size = 0.1, False, 10, 20
        return module, (n_tr, n_val, n_test), m, alpha, (lr, decay, num_epoch, batch_size)
    elif key == '20news':
        module = NewsModule()
        module.append_one = False
        n_tr, n_val, n_test = 200, 200, 200
        m = [8, 8]
        alpha = 0.001
        lr, decay, num_epoch, batch_size = 0.1, False, 10, 20
        return module, (n_tr, n_val, n_test), m, alpha, (lr, decay, num_epoch, batch_size)
    elif key == 'adult':
        module = AdultModule(csv_path='./data')
        module.append_one = False
        n_tr, n_val, n_test = 200, 200, 200
        m = [8, 8]
        alpha = 0.001
        lr, decay, num_epoch, batch_size = 0.1, False, 10, 20
        return module, (n_tr, n_val, n_test), m, alpha, (lr, decay, num_epoch, batch_size)

def test(key, model_type, seed=0, gpu=0):
    dn = './%s_%s' % (key, model_type)
    fn = '%s/sgd%03d.dat' % (dn, seed)
    if not os.path.exists(dn):
        os.mkdir(dn)
    device = 'cuda:%d' % (gpu,)
    
    # fetch data
    if model_type == 'logreg':
        module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size) = settings_logreg(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        
        # selection of alpha
        model = LogisticRegressionCV(random_state=seed, fit_intercept=False, cv=5)
        model.fit(x_tr, y_tr)
        alpha = 1 / (model.C_[0] * n_tr)
        
        # model
        net_func = lambda : LogReg(x_tr.shape[1]).to(device)
    elif model_type == 'dnn':
        module, (n_tr, n_val, n_test), m, alpha, (lr, decay, num_epoch, batch_size) = settings_dnn(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : DNN(x_tr.shape[1]).to(device)
    
    # to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(np.expand_dims(y_tr, axis=1)).to(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)
    
    # fit
    num_steps = int(np.ceil(n_tr / batch_size))
    list_of_sgd_models = []
    list_of_counterfactual_models = []
    list_of_losses = []
    for n in range(-1, n_tr):
        torch.manual_seed(seed)
        model = net_func()
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.0)
        lr_n = lr
        skip = [n]
        info = []
        c = 0
        for epoch in range(num_epoch):
            np.random.seed(epoch)
            idx_list = np.array_split(np.random.permutation(n_tr), num_steps)
            for i in range(num_steps):
                info.append({'idx':idx_list[i], 'lr':lr_n})
                c += 1

                # store model
                if n < 0:
                    m = net_func()
                    m.load_state_dict(copy.deepcopy(model.state_dict()))
                    list_of_sgd_models.append(m)

                # sgd
                idx = idx_list[i]
                b = idx.size
                idx = np.setdiff1d(idx, skip)
                z = model(x_tr[idx])
                loss = loss_fn(z, y_tr[idx])
                for p in model.parameters():
                    loss += 0.5 * alpha * (p * p).sum()
                optimizer.zero_grad()
                loss.backward()
                for p in model.parameters():
                    p.grad.data *= idx.size / b
                optimizer.step()

                # decay
                if decay:
                    lr_n *= np.sqrt(c / (c + 1))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_n

        # save
        if n < 0:
            m = net_func()
            m.load_state_dict(copy.deepcopy(model.state_dict()))
            list_of_sgd_models.append(m)
        else:
            m = net_func()
            m.load_state_dict(copy.deepcopy(model.state_dict()))
            list_of_counterfactual_models.append(m)

        # eval
        z = model(x_val)
        list_of_losses.append(loss_fn(z, y_val).item())
    list_of_losses = np.array(list_of_losses)
    
    # save
    models = NetList(list_of_sgd_models)
    counterfactual = NetList(list_of_counterfactual_models)
    joblib.dump({'models':models, 'info':info, 'counterfactual':counterfactual, 'alpha':alpha}, fn)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Models & Save')
    parser.add_argument('--target', default='adult', type=str, help='target data')
    parser.add_argument('--model', default='logreg', type=str, help='model type')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')
    args = parser.parse_args()
    assert args.target in ['mnist', '20news', 'adult']
    assert args.model in ['logreg', 'dnn']
    if args.seed >= 0:
        test(args.target, args.model, args.seed, args.gpu)
    else:
        for seed in range(100):
            test(args.target, args.model, seed, args.gpu)
    
