import os, sys
import copy
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
from torchvision import transforms
import MyNet
import MyMNIST, MyCIFAR10
import train

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

### parameters ###
# for data
nval = 10000

# for training AE
batch_size = 128
lr = 1e-3
num_epochs = 20

# for evaluation
test_batch_size = 1000
### parameters ###

def mnist():
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    trainset = MyMNIST.MNIST(root='./data', train=True, download=True, transform=transform_train, seed=0)
    valset = MyMNIST.MNIST(root='./data', train=True, download=True, transform=transform_test, seed=0)
    testset = MyMNIST.MNIST(root='./data', train=False, download=True, transform=transform_test, seed=0)
    net_func = MyNet.MnistAE
    return net_func, trainset, valset, testset

def cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = MyCIFAR10.CIFAR10(root='./data', train=True, download=True, transform=transform_train, seed=0)
    valset = MyCIFAR10.CIFAR10(root='./data', train=True, download=True, transform=transform_test, seed=0)
    testset = MyCIFAR10.CIFAR10(root='./data', train=False, download=True, transform=transform_test, seed=0)
    net_func = MyNet.CifarAE
    return net_func, trainset, valset, testset

def eval_model(model, loss_fn, device, dataset, idx):
    model.eval()
    n = idx.size
    with torch.no_grad():
        loss = 0
        eval_idx = np.array_split(np.arange(n), test_batch_size)
        for i in eval_idx:
            x = []
            for ii in i:
                d = dataset[idx[ii]]
                x.append(d[0])
            x = torch.stack(x).to(device)
            y = model(x)
            loss += loss_fn(y, x).item() * i.size
        loss /= n
    return loss

def outlier_ae(setup, output_path, seed=0, gpu=0):
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
    model = net_func(device).to(device)
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # train AE
    info = []
    num_steps = int(np.ceil(nval / batch_size))
    for epoch in range(num_epochs):
        np.random.seed(epoch)
        idx = np.array_split(np.random.permutation(nval), num_steps)
        for i in idx:
            x = []
            for ii in i:
                d = trainset[idx_val[ii]]
                x.append(d[0])
            x = torch.stack(x).to(device)
            y = model(x)
            loss = loss_fn(y, x)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        loss_val = eval_model(model, loss_fn, device, valset, idx_val)
        loss_tr = eval_model(model, loss_fn, device, valset, idx_train)
        info.append((loss_val, loss_tr))
        #print(epoch, loss_val, loss_tr)
        
    fn = '%s/outlier_ae_loss.dat' % (output_path,)
    joblib.dump(np.array(info), fn, compress=9)
    
    # outlierness
    infl = np.zeros(ntr)
    for i in range(ntr):
        d = valset[idx_train[i]]
        xtr = torch.stack([d[0]]).to(device)
        ytr = model(xtr)
        loss = loss_fn(ytr, xtr)
        infl[i] = loss.item()
    
    # save
    fn = '%s/outlier_ae.dat' % (output_path,)
    joblib.dump(infl, fn, compress=9)

def outlier_iso(setup, output_path, seed=0, gpu=0):
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
    fn = '%s/epoch%02d_final_model.dat' % (output_path, 19)
    model.load_state_dict(torch.load(fn))
    model.to(device)
    model.eval()
    
    # flatten - val
    z_val = []
    num_steps = int(np.ceil(nval / test_batch_size))
    with torch.no_grad():
        for epoch in range(num_epochs):
            np.random.seed(epoch)
            idx = np.array_split(np.random.permutation(nval), num_steps)
            for i in idx:
                x = []
                for ii in i:
                    d = trainset[idx_val[ii]]
                    x.append(d[0])
                x = torch.stack(x).to(device)
                z_val.append(model.flatten(x))
    z_val = torch.cat(z_val, dim=0).data.cpu().numpy()
    
    # flatten - train
    z_tr = []
    num_steps = int(np.ceil(ntr / test_batch_size))
    with torch.no_grad():
        idx = np.array_split(np.arange(ntr), num_steps)
        for i in idx:
            x = []
            for ii in i:
                d = valset[idx_train[ii]]
                x.append(d[0])
            x = torch.stack(x).to(device)
            z_tr.append(model.flatten(x))
    z_tr = torch.cat(z_tr, dim=0).data.cpu().numpy()
    
    # fit & evaluate isolation forest
    forest = IsolationForest()
    forest.fit(z_val)
    infl = forest.score_samples(z_tr)
    
    # save
    fn = '%s/outlier_iso.dat' % (output_path,)
    joblib.dump(infl, fn, compress=9)
    
if __name__ == '__main__':
    target = sys.argv[1]
    outlier_type = sys.argv[2]
    start_seed = int(sys.argv[3])
    end_seed = int(sys.argv[4])
    gpu = int(sys.argv[5])
    assert target in ['mnist', 'cifar10']
    assert outlier_type in ['ae', 'iso']
    if target == 'mnist':
        if outlier_type == 'ae':
            setup = mnist
        else:
            setup = train.mnist
    elif target == 'cifar10':
        if outlier_type == 'ae':
            setup = cifar10
        else:
            setup = train.cifar10
    for seed in range(start_seed, end_seed):
        output_path = './%s/%s_%02d' % (target, target, seed)
        if outlier_type == 'ae':
            outlier_ae(setup, output_path, seed=seed, gpu=gpu)
        elif outlier_type == 'iso':
            outlier_iso(setup, output_path, seed=seed, gpu=gpu)
