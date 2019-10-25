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

# for training
batch_size = 64
lr = 0.05
momentum = 0.0
num_epochs = 20

# for evaluation
test_batch_size = 1000
k_list = [1, 3, 6, 10, 30, 60, 100, 300, 600, 1000, 3000, 6000, 10000]

# for storing
bundle_size = 200
### parameters ###

def load_model(net_func, device, output_path, start_epoch):
    if start_epoch == 0:
        list_of_models = [net_func().to(device) for _ in range(bundle_size)]
        models = MyNet.NetList(list_of_models)
        fn = '%s/epoch%02d_bundled_models%02d.dat' % (output_path, 0, 0)
        models.load_state_dict(torch.load(fn))
        model = models.models[0]
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        fn = '%s/epoch%02d_final_model.dat' % (output_path, start_epoch-1)
        model = net_func().to(device)
        model.load_state_dict(torch.load(fn))
        fn = '%s/epoch%02d_final_optimizer.dat' % (output_path, start_epoch-1)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        optimizer.load_state_dict(torch.load(fn))
    return model, optimizer

def sgd_with_skip(info, device, model, optimizer, loss_fn, trainset, idx_train, skip_idx):
    n = idx_train.size
    num_steps = int(np.ceil(n / batch_size))
    for info_i in info:
        idx, seeds = info_i['idx'], info_i['seeds']
        Xtr = []
        ytr = []
        t = 0
        for i, s in zip(idx, seeds):
            if i in skip_idx:
                continue
            t += 1
            trainset.seed = s
            d = trainset[idx_train[i]]
            Xtr.append(d[0])
            ytr.append(d[1])
        Xtr = torch.stack(Xtr).to(device)
        ytr = torch.from_numpy(np.array(ytr)).to(device)
        z = model(Xtr)
        loss = loss_fn(z, ytr)
        optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data *= t / idx.size
        optimizer.step()
    return model, optimizer

def eval_infl(setup, output_path, target_epoch, start_epoch, end_epoch, seed=0, gpu=0):
    device = 'cuda:%d' % (gpu,)
    
    # setup
    net_func, trainset, valset, testset = setup()
    n = len(trainset)
    
    # data split
    np.random.seed(seed)
    idx_val = np.random.permutation(n)[:nval]
    idx_train = np.setdiff1d(np.arange(n), idx_val)
    ntr, nte = idx_train.size, len(testset)
    idx_test = np.arange(nte)
    
    # model, optimizer, and lss
    model_init, optimizer_init = load_model(net_func, device, output_path, start_epoch)
    model = net_func().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = torch.nn.functional.nll_loss
    
    # infl
    infl_sgd_last = joblib.load('%s/infl_sgd_at_epoch%02d.dat' % (output_path, target_epoch))[:, -1]
    infl_sgd_all = joblib.load('%s/infl_sgd_at_epoch%02d.dat' % (output_path, target_epoch))[:, 0]
    infl_icml = joblib.load('%s/infl_icml_at_epoch%02d.dat' % (output_path, target_epoch))
    np.random.seed(seed)
    infls = {'baseline':[], 'icml':infl_icml, 'sgd_last':infl_sgd_last, 'sgd_all':infl_sgd_all, 'random':np.random.rand(ntr)}
    
    # eval
    score = {}
    for k in k_list:
        for key in infls.keys():
            if key in score.keys():
                continue
            if key == 'baseline':
                skip_idx = []
            else:
                skip_idx = np.argsort(infls[key])[:k]
            
            # sgd
            torch.manual_seed(seed)
            model.load_state_dict(copy.deepcopy(model_init.state_dict()))
            optimizer.load_state_dict(copy.deepcopy(optimizer_init.state_dict()))
            model.train()
            for epoch in range(start_epoch, end_epoch):
                fn = '%s/epoch%02d_info.dat' % (output_path, epoch)
                info = joblib.load(fn)
                np.random.seed(epoch)
                model, optimzier = sgd_with_skip(info, device, model, optimizer, loss_fn, trainset, idx_train, skip_idx)
            
            # evaluation
            loss_tr, acc_tr = train.eval_model(model, loss_fn, device, valset, idx_train)
            loss_val, acc_val = train.eval_model(model, loss_fn, device, valset, idx_val)
            loss_te, acc_te = train.eval_model(model, loss_fn, device, testset, idx_test)
            if key == 'baseline':
                score[key] = (loss_tr, loss_val, loss_te, acc_tr, acc_val, acc_te)
            else:
                score[(key, k)] = (loss_tr, loss_val, loss_te, acc_tr, acc_val, acc_te)
            #print((key, k), acc_tr, acc_val, acc_te)
    
        # save
        fn = '%s/eval_epoch_%02d_to_%02d.dat' % (output_path, start_epoch, end_epoch)
        joblib.dump(score, fn, compress=9)
        

if __name__ == '__main__':
    target = sys.argv[1]
    start_epoch = int(sys.argv[2])
    end_epoch = int(sys.argv[3])
    start_seed = int(sys.argv[4])
    end_seed = int(sys.argv[5])
    gpu = int(sys.argv[6])
    assert target in ['mnist', 'cifar10']
    if target == 'mnist':
        setup = train.mnist
    elif target == 'cifar10':
        setup = train.cifar10
    target_epoch = end_epoch
    for seed in range(start_seed, end_seed):
        output_path = './%s/%s_%02d' % (target, target, seed)
        eval_infl(setup, output_path, target_epoch, start_epoch, end_epoch, seed=seed, gpu=gpu)
    