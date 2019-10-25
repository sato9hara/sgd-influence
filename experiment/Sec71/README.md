## Description
The files used for the experiments in Section7.1.
To run the codes, you will need Python 3 and PyTorch.

## To Run Codes
### Step1. Train Models

To train models, run

```
python train.py --target [target] --model [model] --seed [seed] --gpu [gpu]
```

Here, `[target]` is one of {adult, 20news, mnist}, `[model]` is one of {logreg, dnn}, and `[gpu]` is the index of GPU used.
The program trains models for the assigned random seed `[seed]`.
If `[seed]` is negative, the program trains models for the random seeds in between `0` and `99`.
For example, to train a logistic regression model for adult with seed 0 on the second gpu, run

```
python train.py --target adult --model logreg --seed 0 --gpu 2
```

### Step2. Compute LIEs

To compute linear influences, run

```
python infl.py --target [target] --model [model] --type [type] --seed [seed] --gpu [gpu]
```

where `type` is one of {true, sgd, icml}.

* `true`: Compute the true linear influence by running counterfactual SGDs.
* `sgd`: Estiamate the linear influence using the proposed algorithm in the paper.
* `icml`: Estiamate the linear influence using the algorithm of [Koh & Liang, ICML'17].

To compute influences with the proposed algorithm, run

```
python infl.py --target adult --model logreg --type sgd --seed 0 --gpu 2
```

### Step3. See Results

To see the results, check `ViewResults.ipynb`.
