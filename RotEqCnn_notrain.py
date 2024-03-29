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


class RotEqCNN():
    def __init__(self, ensemble_num=9, dataset='mnist', train_epoch=6):
        self.ensemble_num = ensemble_num
        self.dataset = dataset
        self.rotate_angle = 360.0 / (self.ensemble_num * 1.0)
        self.start_angle = -270 # -270
        self.end_angle = 0
        self.train_transform = []
        self.trainset = []
        self.trainloader = []
        self.model = []
        self.optimizer = []
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoded_train_data = []
        self.encoded_train_label = []
        self.mlp = None
        self.X_test = None
        self.y_test = None
        self.train_epoch = train_epoch

    def get_dataset(self):
        if self.dataset == 'mnist':
            means = deviations = [0.5]
            for i in range(self.ensemble_num):
                self.train_transform.append(transforms.Compose(
                    [transforms.RandomRotation([self.start_angle, self.end_angle]), transforms.ToTensor(),
                     transforms.Normalize(means, deviations)]))
                start_angle = -270 + self.rotate_angle * (i + 1)
                end_angle = self.rotate_angle * (i + 1)
        if self.dataset == 'cifar':
            for i in range(self.ensemble_num):
                self.train_transform.append(transforms.Compose(
                    [transforms.RandomRotation([self.start_angle, self.end_angle]), transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
                start_angle = -270 + self.rotate_angle * (i + 1)
                end_angle = self.rotate_angle * (i + 1)
        # add trainset
        for i in range(self.ensemble_num):
            if self.dataset == 'mnist':
                self.trainset.append(torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=self.train_transform[i]))
            if self.dataset == 'cifar':
                self.trainset.append(torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.train_transform[i]))
        # add trainloader
        for i in range(self.ensemble_num):
            self.trainloader.append(
                torch.utils.data.DataLoader(self.trainset[i], batch_size=128, shuffle=True, num_workers=2))

    def init_models(self):
        # define models
        for i in range(self.ensemble_num):
            if self.dataset == 'mnist':
                self.model.append(MnistResNet().to(self.device))
            if self.dataset == 'cifar':
                self.model.append(make_resnet18k(k = 64).to(self.device))
        # define optimizers
        for i in range(self.ensemble_num):
            self.optimizer.append(optim.SGD(self.model[i].parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4))

    def train(self):
        epoch_range = self.train_epoch
        # training loop + eval loop
        for ensemble_id in range(self.ensemble_num):
            running_loss = 0.0
            print("Loss of ensemble model", ensemble_id)
            for epoch in range(epoch_range):
                for i, data in enumerate(self.trainloader[ensemble_id], 0):
                    # get the inputs
                    inputs, labels = data
                    #         print(labels.numpy().shape)

                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer[ensemble_id].zero_grad()

                    # forward + backward + optimize
                    outputs = self.model[ensemble_id](inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer[ensemble_id].step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 20 == 19:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.6f' % (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0

        """## Step 6: Form the encoded sets"""

        correct = 0
        for i in range(self.ensemble_num):
            with torch.no_grad():
                for data in self.trainloader[i]:
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = []
                    for j in range(self.ensemble_num):
                        outputs.append(self.model[j](images))
                        outputs[j] = outputs[j].cpu().numpy()

                    labels = labels.cpu().numpy()
                    for j in range(len(outputs[0])):
                        op=[]
                        self.encoded_train_label.append(labels[j])
                        for k in range(self.ensemble_num):
                            d=np.concatenate((outputs[k][j],op),axis=None)
                            op=d

                        self.encoded_train_data.append(d)


        X_train, self.X_test, y_train, self.y_test = train_test_split(self.encoded_train_data, self.encoded_train_label,
                                                                      test_size=0.01, random_state=0)

        
    def train_es_clf(self, X_train, y_train):
        from torch.optim import Adam
        torch_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        
        loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=64, shuffle=True, num_workers=2)
        criterion = nn.CrossEntropyLoss()
        EPOCHS = 200
        optm = Adam(self.mlp.parameters(), lr = 0.1)
        for t in range(30):
            for step, (batch_x, batch_y) in enumerate(loader):
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                out = self.mlp(batch_x)
                loss = criterion(out, batch_y)

                optm.zero_grad()
                loss.backward()
                optm.step()
        print("End of training es_clf")
        
    def show_test_result(self):
        pred = self.mlp(self.X_test)
        total = 0
        correct = 1
        for i in range(len(self.y_test)):
            total += 1
            if self.y_test[i] == pred[i]:
                correct += 1
        print("The correction rate under MLPClassifier is:")
        print(correct * 1.0 / total * 1.0)
    
    def predict(self, image, label):
        label = label.cpu().numpy()
        test_encoded_set = []
        image = image.to(self.device)
        for i in range(self.ensemble_num):
            test_encoded_set.append(self.model[i](image))
            test_encoded_set[i] = test_encoded_set[i].cpu().numpy()
        averaged_ensemble = np.zeros([10])
        for i in range(self.ensemble_num):
            averaged_ensemble += test_encoded_set[i][0]
        pred = np.argmax(averaged_ensemble)
        
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

    def getTestAccuracy(self):
        if self.dataset == 'cifar':
            test_transforms = transforms.Compose([transforms.RandomRotation([-180, 180]),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transforms)
        if self.dataset == 'mnist':
            means = deviations = [0.5]
            test_transforms = transforms.Compose([transforms.RandomRotation([-180, 180]),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, deviations)])

            testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=test_transforms)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=2)
        # get test accuracy on test dataset
        for i in range(self.ensemble_num):
            self.model[i].eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                if_correct = self.predict(images, labels)
                total += 1
                correct += if_correct

        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct*1.0 / total*1.0))
