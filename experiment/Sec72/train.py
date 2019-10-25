import os, sys
import copy
import numpy as np
from sklearn.externals import joblib
import torch
from torchvision import transforms
import MyNet
import MyMNIST, MyCIFAR10

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

### parameters ###
# for data
nval = 10000

# for training
batch_size = 64
lr = 0.05
momentum = 0.0
num_epochs = 20

# for evaluation
test_batch_size = 1000

# for storing
bundle_size = 200
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
    net_func = MyNet.MnistNet
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
    net_func = MyNet.CifarNet
    return net_func, trainset, valset, testset

def eval_model(model, loss_fn, device, dataset, idx):
    model.eval()
    n = idx.size
    with torch.no_grad():
        loss = 0
        acc = 0
        eval_idx = np.array_split(np.arange(n), test_batch_size)
        for i in eval_idx:
            X = []
            y = []
            for ii in i:
                d = dataset[idx[ii]]
                X.append(d[0])
                y.append(d[1])
            X = torch.stack(X).to(device)
            y = torch.from_numpy(np.array(y)).to(device)
            z = model(X)
            loss += loss_fn(z, y, reduction='sum').item()
            pred = z.argmax(dim=1, keepdim=True)
            acc += pred.eq(y.view_as(pred)).sum().item()
        loss /= n
        acc /= n
    return loss, acc

def train(setup, output_path, seed=0, gpu=0):
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
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    # model list
    list_of_models = [net_func().to(device) for _ in range(bundle_size)]
    
    # training
    seed_train = 0
    score = []
    torch.manual_seed(seed)
    num_steps = int(np.ceil(ntr / batch_size))
    for epoch in range(num_epochs):
        
        # training
        model.train()
        np.random.seed(epoch)
        idx = np.array_split(np.random.permutation(ntr), num_steps)
        info = []
        c = 0
        k = 0
        for j, i in enumerate(idx):
            
            # save
            list_of_models[c].load_state_dict(copy.deepcopy(model.state_dict()))
            seeds = list(range(seed_train, seed_train+i.size))
            info.append({'idx':i, 'lr':lr, 'seeds':seeds})
            c += 1
            if c == bundle_size or j == len(idx) - 1:
                fn = '%s/epoch%02d_bundled_models%02d.dat' % (output_path, epoch, k)
                models = MyNet.NetList(list_of_models)
                torch.save(models.state_dict(), fn)
                k += 1
                c = 0
            
            # sgd
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        fn = '%s/epoch%02d_final_model.dat' % (output_path, epoch)
        torch.save(model.state_dict(), fn)
        fn = '%s/epoch%02d_final_optimizer.dat' % (output_path, epoch)
        torch.save(optimizer.state_dict(), fn)
        fn = '%s/epoch%02d_info.dat' % (output_path, epoch)
        joblib.dump(info, fn, compress=9)
        
        # evaluation
        loss_tr, acc_tr = eval_model(model, loss_fn, device, valset, idx_train)
        loss_val, acc_val = eval_model(model, loss_fn, device, valset, idx_val)
        print(epoch, acc_tr, acc_val)
        score.append((loss_tr, loss_val, acc_tr, acc_val))
    
    # save
    fn = '%s/score.dat' % (output_path,)
    joblib.dump(np.array(score), fn, compress=9)
    
if __name__ == '__main__':
    target = sys.argv[1]
    start_seed = int(sys.argv[2])
    end_seed = int(sys.argv[3])
    gpu = int(sys.argv[4])
    assert target in ['mnist', 'cifar10']
    if target == 'mnist':
        setup = mnist
    elif target == 'cifar10':
        setup = cifar10
    for seed in range(start_seed, end_seed):
        output_path = './%s/%s_%02d' % (target, target, seed)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        train(setup, output_path, seed=seed, gpu=gpu)
    
