import os, sys
import argparse
import numpy as np
from sklearn.externals import joblib
import torch
from DataModule import MnistModule, NewsModule, AdultModule
from MyNet import LogReg, DNN, NetList
import train

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

### parameters ###
# for training in icml
batch_size = 200
lr = 0.01
momentum = 0.9
num_epochs = 100
### parameters ###


def compute_gradient(x, y, model, loss_fn):
    z = model(x)
    loss = loss_fn(z, y)
    model.zero_grad()
    loss.backward()
    u = [param.grad.data.clone() for param in model.parameters()]
    for uu in u:
        uu.requires_grad = False
    return u

def infl_true(key, model_type, seed=0, gpu=0):
    dn = './%s_%s' % (key, model_type)
    fn = '%s/sgd%03d.dat' % (dn, seed)
    gn = '%s/infl_true%03d.dat' % (dn, seed)
    device = 'cuda:%d' % (gpu,)
    
    # setup
    if model_type == 'logreg':
        module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size) = train.settings_logreg(key)
        _, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
        (x_val, y_val) = z_val
        net_func = lambda : LogReg(x_tr.shape[1]).to(device)
    elif model_type == 'dnn':
        module, (n_tr, n_val, n_test), m, alpha, (lr, decay, num_epoch, batch_size) = train.settings_dnn(key)
        _, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
        (x_val, y_val) = z_val
        net_func = lambda : DNN(x_tr.shape[1]).to(device)
    
    # to tensor
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)
    
    # model setup
    res = joblib.load(fn)
    model = res['models'].models[-1].to(device)
    #loss_fn = torch.nn.functional.nll_loss
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()
    
    # influence
    z = model(x_val)
    loss = loss_fn(z, y_val)
    #acc = ((z > 0.5).to(torch.float) - y_val).abs().mean().item()
    #print(acc, loss.item())
    infl = np.zeros(n_tr)
    for i in range(n_tr):
        m = res['counterfactual'].models[i]
        m.eval()
        zi = m(x_val)
        lossi = loss_fn(zi, y_val)
        infl[i] = lossi.item() - loss.item()
        #acc = ((zi > 0.5).to(torch.float) - y_val).abs().mean().item()
        #print(i, acc, lossi.item())
    
    # save
    joblib.dump(infl, gn, compress=9)


def infl_sgd(key, model_type, seed=0, gpu=0):
    dn = './%s_%s' % (key, model_type)
    fn = '%s/sgd%03d.dat' % (dn, seed)
    gn = '%s/infl_sgd%03d.dat' % (dn, seed)
    device = 'cuda:%d' % (gpu,)
    
    # setup
    if model_type == 'logreg':
        module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size) = train.settings_logreg(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : LogReg(x_tr.shape[1]).to(device)
    elif model_type == 'dnn':
        module, (n_tr, n_val, n_test), m, alpha, (lr, decay, num_epoch, batch_size) = train.settings_dnn(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : DNN(x_tr.shape[1]).to(device)
    
    # to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(np.expand_dims(y_tr, axis=1)).to(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)
    
    # model setup
    res = joblib.load(fn)
    model = res['models'].models[-1].to(device)
    #loss_fn = torch.nn.functional.nll_loss
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()
    
    # gradient
    u = compute_gradient(x_val, y_val, model, loss_fn)
    u = [uu.to(device) for uu in u]
    
    # model list
    models = res['models'].models[:-1]
    
    # influence
    alpha = res['alpha']
    info = res['info']
    infl = np.zeros(n_tr)
    for t in range(len(models)-1, -1, -1):
        m = models[t]
        m.eval()
        idx, lr = info[t]['idx'], info[t]['lr']
        for i in idx:
            z = m(x_tr[[i]])
            loss = loss_fn(z, y_tr[[i]])
            for p in m.parameters():
                loss += 0.5 * alpha * (p * p).sum()
            m.zero_grad()
            loss.backward()
            for j, param in enumerate(m.parameters()):
                infl[i] += lr * (u[j].data * param.grad.data).sum().item() / idx.size
        
        # update u
        z = m(x_tr[idx])
        loss = loss_fn(z, y_tr[idx])
        for p in m.parameters():
            loss += 0.5 * alpha * (p * p).sum()
        grad_params = torch.autograd.grad(loss, m.parameters(), create_graph=True)
        ug = 0
        for uu, g in zip(u, grad_params):
            ug += (uu * g).sum()
        m.zero_grad()
        ug.backward()
        for j, param in enumerate(m.parameters()):
            #u[j] -= lr * param.grad.data / idx.size
            u[j] -= lr * param.grad.data
        
    # save
    joblib.dump(infl, gn, compress=9)

def infl_nohess(key, model_type, seed=0, gpu=0):
    dn = './%s_%s' % (key, model_type)
    fn = '%s/sgd%03d.dat' % (dn, seed)
    gn = '%s/infl_nohess%03d.dat' % (dn, seed)
    device = 'cuda:%d' % (gpu,)
    
    # setup
    if model_type == 'logreg':
        module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size) = train.settings_logreg(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : LogReg(x_tr.shape[1]).to(device)
    elif model_type == 'dnn':
        module, (n_tr, n_val, n_test), m, alpha, (lr, decay, num_epoch, batch_size) = train.settings_dnn(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : DNN(x_tr.shape[1]).to(device)
    
    # to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(np.expand_dims(y_tr, axis=1)).to(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)
    
    # model setup
    res = joblib.load(fn)
    model = res['models'].models[-1].to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()
    
    # gradient
    u = compute_gradient(x_val, y_val, model, loss_fn)
    u = [uu.to(device) for uu in u]
    
    # model list
    models = res['models'].models[:-1]
    
    # influence
    alpha = res['alpha']
    info = res['info']
    infl = np.zeros(n_tr)
    for t in range(len(models)-1, -1, -1):
        m = models[t]
        m.eval()
        idx, lr = info[t]['idx'], info[t]['lr']
        for i in idx:
            z = m(x_tr[[i]])
            loss = loss_fn(z, y_tr[[i]])
            for p in m.parameters():
                loss += 0.5 * alpha * (p * p).sum()
            m.zero_grad()
            loss.backward()
            for j, param in enumerate(m.parameters()):
                infl[i] += lr * (u[j].data * param.grad.data).sum().item() / idx.size
        
    # save
    joblib.dump(infl, gn, compress=9)
    

def infl_icml(key, model_type, seed=0, gpu=0):
    dn = './%s_%s' % (key, model_type)
    fn = '%s/sgd%03d.dat' % (dn, seed)
    gn = '%s/infl_icml%03d.dat' % (dn, seed)
    hn = '%s/loss_icml%03d.dat' % (dn, seed)
    device = 'cuda:%d' % (gpu,)
    
    # setup
    if model_type == 'logreg':
        #module, (n_tr, n_val, n_test), (_, _, _, batch_size) = train.settings_logreg(key)
        module, (n_tr, n_val, n_test), _ = train.settings_logreg(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : LogReg(x_tr.shape[1]).to(device)
    elif model_type == 'dnn':
        #module, (n_tr, n_val, n_test), m, _, (_, _, _, batch_size) = train.settings_dnn(key)
        module, (n_tr, n_val, n_test), m, _, _ = train.settings_dnn(key)
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda : DNN(x_tr.shape[1]).to(device)
    
    # to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(np.expand_dims(y_tr, axis=1)).to(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)
    
    # model setup
    res = joblib.load(fn)
    model = res['models'].models[-1].to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()
    
    # gradient
    u = compute_gradient(x_val, y_val, model, loss_fn)
    u = [uu.to(device) for uu in u]
    
    # Hinv * u with SGD
    if model_type == 'logreg':
        alpha = res['alpha']
    elif model_type == 'dnn':
        alpha = 1.0
    num_steps = int(np.ceil(n_tr / batch_size))
    #v = [torch.zeros(*param.shape, requires_grad=True, device=device) for param in model.parameters()]
    v = []
    for uu in u:
        v.append(uu.clone())
        v[-1].to(device)
        v[-1].requires_grad = True
    optimizer = torch.optim.SGD(v, lr=lr, momentum=momentum)
    #optimizer = torch.optim.Adam(v, lr=lr)
    loss_train = []
    for epoch in range(num_epochs):
        model.eval()
        
        # training
        np.random.seed(epoch)
        idx_list = np.array_split(np.random.permutation(n_tr), num_steps)
        for i in range(num_steps):
            idx = idx_list[i]
            z = model(x_tr[idx])
            loss = loss_fn(z, y_tr[idx])
            model.zero_grad()
            grad_params = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            vg = 0
            for vv, g in zip(v, grad_params):
                vg += (vv * g).sum()
            model.zero_grad()
            vgrad_params = torch.autograd.grad(vg, model.parameters(), create_graph=True)
            loss_i = 0
            for vgp, vv, uu in zip(vgrad_params, v, u):
                loss_i += 0.5 * (vgp * vv + alpha * vv * vv).sum() - (uu * vv).sum()
            optimizer.zero_grad()
            loss_i.backward()
            optimizer.step()
            loss_train.append(loss_i.item())
            #print(loss_i.item())

    # save
    joblib.dump(np.array(loss_train), hn, compress=9)
    
    # influence
    infl = np.zeros(n_tr)
    for i in range(n_tr):
        z = model(x_tr[[i]])
        loss = loss_fn(z, y_tr[[i]])
        model.zero_grad()
        loss.backward()
        infl_i = 0
        for j, param in enumerate(model.parameters()):
            infl_i += (param.grad.data.cpu().numpy() * v[j].data.cpu().numpy()).sum()
        infl[i] = infl_i / n_tr
        
    # save
    joblib.dump(infl, gn, compress=9)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Models & Save')
    parser.add_argument('--target', default='adult', type=str, help='target data')
    parser.add_argument('--model', default='logreg', type=str, help='model type')
    parser.add_argument('--type', default='true', type=str, help='influence type')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--gpu', default=0, type=int, help='gpu index')
    args = parser.parse_args()
    assert args.target in ['mnist', '20news', 'adult']
    assert args.model in ['logreg', 'dnn']
    assert args.type in ['true', 'sgd', 'nohess', 'icml']
    if args.type == 'true':
        if args.seed >= 0:
            infl_true(args.target, args.model, args.seed, args.gpu)
        else:
            for seed in range(100):
                infl_true(args.target, args.model, seed, args.gpu)
    elif args.type == 'sgd':
        if args.seed >= 0:
            infl_sgd(args.target, args.model, args.seed, args.gpu)
        else:
            for seed in range(100):
                infl_sgd(args.target, args.model, seed, args.gpu)
    elif args.type == 'nohess':
        if args.seed >= 0:
            infl_nohess(args.target, args.model, args.seed, args.gpu)
        else:
            for seed in range(100):
                infl_nohess(args.target, args.model, seed, args.gpu)
    elif args.type == 'icml':
        if args.seed >= 0:
            infl_icml(args.target, args.model, args.seed, args.gpu)
        else:
            for seed in range(100):
                infl_icml(args.target, args.model, seed, args.gpu)
    