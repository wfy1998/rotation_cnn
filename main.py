import argparse
from RotEqCnn import RotEqCNN
from flags import get_default_args
# from RotEqCnn_notrain import RotEqCNN

# I. Improve accuracy (99%)
# II. Add one more dataset (CIFAR 10)

# 1. Change ensemble_num and train_epoch (nothing to implement)



# 2. Change CNN (implment but not hard)
# CNN < VGG (2012) < ResNet (2016)



# 3. Change dataset
# MNIST, F-MNIST, KMNIST

# 3 channel (bonus)
# (R, G, B)
# CIFAR10 (ResNet)

def main():
    opt = get_default_args()

    dataset = opt.dataset
    e_num = int(opt.ensemble_num)
    t_epoch = opt.train_epoch
    #get_data_only = opt.get_data_only
    get_data_only = False
    recnn = RotEqCNN(e_num, dataset, t_epoch)
    #if(get_data_only):
    #    recnn.get_dataset()
    #    recnn.init_models()
    #    recnn.train()       
    #else:
    #    recnn.init_models()
    #    recnn.train_es_clf()
    #    recnn.getTestAccuracy()
    recnn.get_dataset()
    recnn.init_models()
    #recnn.train()
    recnn.train_es_clf()
    recnn.getTestAccuracy()
        

    
    #recnn.get_dataset()
    #recnn.init_models()
    #recnn.train()
    #recnn.show_test_result()
    #recnn.getTestAccuracy()
if __name__ == '__main__':
    #freeze_support() #here if program needs to be frozen
    main()
