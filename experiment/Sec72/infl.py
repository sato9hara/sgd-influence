import os, sys
import copy
import numpy as np
from sklearn.externals import joblib
import torch
from torchvision import transforms
import MyNet
import MyMNIST, MyCIFAR10
import train

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

### parameters ###
# for data
nval = 10000

# for training in icml
batch_size = 1000
lr = 0.005
momentum = 0.9
num_epochs = 2

# for evaluation
test_batch_size = 1000

# for storing
bundle_size = 200
### parameters ###

def compute_gradient(model, loss_fn, device, dataset, idx):
    n = idx.size
    grad_idx = np.array_split(np.arange(n), test_batch_size)
    u = [torch.zeros(*param.shape, requires_grad=False).to(device) for param in model.parameters()]
    model.eval()
    for i in grad_idx:
        X = []
        y = []
        for ii in i:
            d = dataset[idx[ii]]
            X.append(d[0])
            y.append(d[1])
        X = torch.stack(X).to(device)
        y = torch.from_numpy(np.array(y)).to(device)
        z = model(X)
        loss = loss_fn(z, y, reduction='sum')
        model.zero_grad()
        loss.backward()
        for j, param in enumerate(model.parameters()):
            u[j] += param.grad.data / n
    return u

def infl_sgd(setup, output_path, target_epoch, seed=0, gpu=0):
    device = 'cuda:%d' % (gpu,)
    
    # setup
    net_func, trainset, valset, _ = setup()
    n = len(trainset)
    
    # data split
    np.random.seed(seed)
    idx_val = np.random.permutation(n)[:nval]
    idx_train = np.setdiff1d(np.arange(n), idx_val)
    ntr = idx_train.size
    
    # model setup
    torch.manual_seed(seed)
    model = net_func().to(device)
    loss_fn = torch.nn.functional.nll_loss
    fn = '%s/epoch%02d_final_model.dat' % (output_path, target_epoch-1)
    model.load_state_dict(torch.load(fn))
    model.to(device)
    model.eval()
    
    # gradient
    u = compute_gradient(model, loss_fn, device, valset, idx_val)
    
    # model list
    list_of_models = [net_func().to(device) for _ in range(bundle_size)]
    models = MyNet.NetList(list_of_models)
    
    # influence
    infl = torch.zeros(ntr, target_epoch, requires_grad=False).to(device)
    for epoch in range(target_epoch-1, -1, -1):
        fn = '%s/epoch%02d_info.dat' % (output_path, epoch)
        info = joblib.load(fn)
        T = len(info)
        B = [bundle_size] * 3
        B.append(T % (bundle_size * 3))
        c = -1
        for k in range(3, -1, -1):
            fn = '%s/epoch%02d_bundled_models%02d.dat' % (output_path, epoch, k)
            models.load_state_dict(torch.load(fn))
            for t in range(B[k]-1, -1, -1):
                m = models.models[t].to(device)
                idx, seeds, lr = info[c]['idx'], info[c]['seeds'], info[c]['lr']
                c -= 1

                # influence
                Xtr = []
                ytr = []
                for i, s in zip(idx, seeds):
                    trainset.seed = s
                    d = trainset[idx_train[i]]
                    Xtr.append(d[0])
                    ytr.append(d[1])
                    z = m(torch.stack([d[0]]).to(device))
                    loss = loss_fn(z, torch.from_numpy(np.array([d[1]])).to(device))
                    m.zero_grad()
                    loss.backward()
                    for j, param in enumerate(m.parameters()):
                        infl[i, epoch] += lr * (u[j].data * param.grad.data).sum() / idx.size
                
                # update u
                Xtr = torch.stack(Xtr).to(device)
                ytr = torch.from_numpy(np.array(ytr)).to(device)
                z = m(Xtr)
                loss = loss_fn(z, ytr)
                grad_params = torch.autograd.grad(loss, m.parameters(), create_graph=True)
                ug = 0
                for uu, g in zip(u, grad_params):
                    ug += (uu * g).sum()
                m.zero_grad()
                ug.backward()
                for j, param in enumerate(m.parameters()):
                    u[j] -= lr * param.grad.data / idx.size
                
        # save
        fn = '%s/infl_sgd_at_epoch%02d.dat' % (output_path, target_epoch)
        joblib.dump(infl.cpu().numpy(), fn, compress=9)
        if epoch > 0:
            infl[:, epoch-1] = infl[:, epoch].clone()

def infl_icml(setup, output_path, target_epoch, alpha, seed=0, gpu=0):
    device = 'cuda:%d' % (gpu,)
    
    # setup
    net_func, trainset, valset, _ = setup()
    n = len(trainset)
    
    # data split
    np.random.seed(seed)
    idx_val = np.random.permutation(n)[:nval]
    idx_train = np.setdiff1d(np.arange(n), idx_val)
    ntr = idx_train.size
    
    # model setup
    torch.manual_seed(seed)
    model = net_func().to(device)
    loss_fn = torch.nn.functional.nll_loss
    fn = '%s/epoch%02d_final_model.dat' % (output_path, target_epoch-1)
    model.load_state_dict(torch.load(fn))
    model.to(device)
    model.eval()
    
    # gradient
    u = compute_gradient(model, loss_fn, device, valset, idx_val)
    
    # Hinv * u with SGD
    seed_train = 0
    num_steps = int(np.ceil(ntr / batch_size))
    v = [torch.zeros(*param.shape, requires_grad=True, device=device) for param in model.parameters()]
    optimizer = torch.optim.SGD(v, lr=lr, momentum=momentum)
    loss_train = []
    for epoch in range(num_epochs):
        model.eval()
        
        # training
        np.random.seed(epoch)
        idx = np.array_split(np.random.permutation(ntr), num_steps)
        for j, i in enumerate(idx):
            Xtr = []
            ytr = []
            for ii in i:
                trainset.seed = seed_train
                d = trainset[idx_train[ii]]
                Xtr.append(d[0])
                ytr.append(d[1])
                seed_train += 1
            Xtr = torch.stack(Xtr).to(device)
            ytr = torch.from_numpy(np.array(ytr)).to(device)
            z = model(Xtr)
            loss = loss_fn(z, ytr)
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
            print(loss_i.item())

    # save
    fn = '%s/loss_icml_at_epoch%02d.dat' % (output_path, target_epoch)
    joblib.dump(np.array(loss_train), fn, compress=9)
    
    # influence
    infl = np.zeros(ntr)
    for i in range(ntr):
        d = valset[idx_train[i]]
        Xtr = torch.stack([d[0]]).to(device)
        ytr = torch.from_numpy(np.array([d[1]])).to(device)
        z = model(Xtr)
        loss = loss_fn(z, ytr)
        model.zero_grad()
        loss.backward()
        infl_i = 0
        for j, param in enumerate(model.parameters()):
            infl_i += (param.grad.data.cpu().numpy() * v[j].data.cpu().numpy()).sum()
        infl[i] = - infl_i / ntr
        
    # save
    fn = '%s/infl_icml_at_epoch%02d.dat' % (output_path, target_epoch)
    joblib.dump(infl, fn, compress=9)
    

if __name__ == '__main__':
    target = sys.argv[1]
    infl_type = sys.argv[2]
    target_epoch = int(sys.argv[3])
    start_seed = int(sys.argv[4])
    end_seed = int(sys.argv[5])
    gpu = int(sys.argv[6])
    assert target in ['mnist', 'cifar10']
    assert infl_type in ['sgd', 'icml']
    if target == 'mnist':
        setup = train.mnist
        alpha = 0.1
    elif target == 'cifar10':
        setup = train.cifar10
        alpha = 10.0
    for seed in range(start_seed, end_seed):
        output_path = './%s/%s_%02d' % (target, target, seed)
        if infl_type == 'sgd':
            infl_sgd(setup, output_path, target_epoch, seed=seed, gpu=gpu)
        elif infl_type == 'icml':
            infl_icml(setup, output_path, target_epoch, alpha, seed=seed, gpu=gpu)
