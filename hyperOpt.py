import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from flags import get_default_args 
from hyperopt import fmiin, tpe, hp, STATUS_OK, Trials, space_eval, SparkTrials

def main():
    FLAGS = get_default_args
    hyperopt_space = {
    'dataset':hp.choice('dataset', ['mnist', 'cifar']),
    'ensemble_num':hp.quniform('ensemble_num',1,64,1),
    'train_epoch':hp.quniform('train_epoch',1,64,1)
}


    


if __name__ == '__main__':
    main()
