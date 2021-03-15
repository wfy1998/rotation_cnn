from typing import List, Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from resnet_real import MnistResNet
from resnet_cifar import make_resnet18k
from es_clf import ES_clf
import torch.nn.functional as F
import numpy as np
import csv

# define all parameters
model = []
mlp = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ensemble_num = 0
dataset = None



def main():
        # read data from file
        X_train = []
        X_test = []
        y_train = []
        y_test = []

        # set parameters
        ensemble_num = 4
        dataset = 'mnist'
        mlp = ES_clf(k = ensemble_num * 10).to(device)

        # begin train and test
        train_es_clf(X_train, y_train)
        train()
        getTestAccuracy()

def init_models():
        # define models
        for i in range(ensemble_num):
            if dataset == 'mnist':
                model.append(MnistResNet().to(device))
            if dataset == 'cifar':
                model.append(make_resnet18k(k = 64).to(device))


def train_es_clf(X_train, y_train):
        from torch.optim import Adam
        torch_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        
        loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=64, shuffle=True, num_workers=2)
        criterion = nn.CrossEntropyLoss()
        EPOCHS = 200
        optm = Adam(mlp.parameters(), lr = 0.0001)
        for t in range(50):
            for step, (batch_x, batch_y) in enumerate(loader):
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                out = mlp(batch_x)
                loss = criterion(out, batch_y)

                optm.zero_grad()
                loss.backward()
                optm.step()
        print("End of training es_clf")

def show_test_result(X_test, y_test):
        pred = mlp(X_test)
        total = 0
        correct = 1
        for i in range(len(y_test)):
            total += 1
            if y_test[i] == pred[i]:
                correct += 1
        print("The correction rate under MLPClassifier is:")
        print(correct * 1.0 / total * 1.0)

def predict(image, label):
        label = label.cpu().numpy()
        test_encoded_set = []
        with torch.no_grad():
            image = image.to(device)
            for i in range(ensemble_num):
                test_encoded_set.append(model[i](image))
                test_encoded_set[i] = test_encoded_set[i].cpu().numpy()
        for i in range(len(test_encoded_set[0])):
            op=[]
            for j in range(ensemble_num):
                d=np.concatenate((test_encoded_set[j][i],op),axis=None)
                op=d
        d = np.array(d)
        d = d.reshape(1, ensemble_num*10)
        d = torch.FloatTensor(d).cuda()
        mlp.eval()
        pred = mlp(d)
        prediction = torch.max(F.softmax(pred), 1)[1]
        pred = prediction.data.cpu().numpy().squeeze()
        print("pred")
        print(pred)
        print(pred.shape)
        print("label:")
        print(label)
        print(label.shape)
        if pred == label[0]:
            return 1
        else:
            return 0

def getTestAccuracy():
        if dataset == 'cifar':
            test_transforms = transforms.Compose([transforms.RandomRotation([-180, 180]),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transforms)
        if dataset == 'mnist':
            means = deviations = [0.5]
            test_transforms = transforms.Compose([transforms.RandomRotation([-180, 180]),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, deviations)])

            testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=test_transforms)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)
        # get test accuracy on test dataset
        for i in range(ensemble_num):
            model[i].eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                if_correct = predict(images, labels)
                total += 1
                correct += if_correct

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct*1.0 / total*1.0))

if __name__=='__main__':
	main()


