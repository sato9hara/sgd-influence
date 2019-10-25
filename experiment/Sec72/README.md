## Description
The files used for the experiments in Section7.2.
To run the codes, you will need Python 3 and PyTorch.

## To Run Codes
### Step1. Train Models

To train models, run

```
python train.py [target] [start_seed] [end_seed] [gpu]
```

Here, `target` is one of {mnist, cifar10}, and `gpu` is the index of GPU used.
The program trains models for each seed in between `start_seed` and `end_seed`.
For example, to train models for MNIST with seed in between 0 and 5, with gpu_index=2, run

```
python train.py mnist 0 5 2
```

### Step2. Compute LIEs

To compute linear influences, run

```
python infl.py [target] [type] [epoch] [start_seed] [end_seed] [gpu_index]
```

where `type` is one of {sgd, icml}, where sgd denotes the proposed method, and icml denotes the method of [Koh and Liang, ICML'17].
`epoch` is the target epoch of the model where we computes the influence.
In the experiments, `epoch` is fixed to 20.
To compute influences with the proposed method, run

```
python infl.py mnist sgd 20 0 5 2
```

### Step3. Train Outlier Detection Models

To train outlier detection models, run

```
python outlier.py [target] [type] [start_seed] [end_seed] [gpu_index]
```

where `type` is one of {ae, iso}, where ae denotes Autoencocer and iso denotes Isolation Forest.
For example, to train Autoencoder for MNIST, run

```
python outlier.py mnist ae 0 5 2
```

### Step4. Retrain Models

To retrain models, run

```
python eval.py [target] [start_epoch] [start_seed] [end_seed] [gpu_index]
python eval_outlier.py [target] [start_epoch] [start_seed] [end_seed] [gpu_index]
```

where `start_epoch` is the step we start the retarining.
For "Retrain Last", set `start_epoch` to 19, and for "Retarin All" set `start_epoch` to 20.
To retrain the models for the proposed method with "Retrain Last", run

```
python eval.py mnist 19 0 5 2
```

### Step5. See Results

To see the results, check `ViewResults.ipynb`.
