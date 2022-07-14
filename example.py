import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim
from torch import nn, relu
from torchvision import datasets, transforms
from pt import NNP
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

hp = {
    'lr' : 0.001,
    'batch_size' : 5,
    'epochs' : 300,
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu'
}

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader =  DataLoader(trainset, batch_size=5, shuffle=True)
testloader = DataLoader(testset, batch_size=1)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, padding_mode='replicate'),
            nn.MaxPool2d(2, 2)
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, stride=1, padding=1, padding_mode='replicate'),
            nn.MaxPool2d(2, 2)
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, padding_mode='replicate'),
            nn.MaxPool2d(2, 2)
        )
        self.l4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, padding_mode='replicate'),
            nn.MaxPool2d(2, 2)
        )
        self.l5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, padding_mode='replicate'),
            nn.MaxPool2d(2, 2)
        )
        self.lf = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = torch.flatten(x, 1)
        x = self.lf(x)
        return x

model = NN()
criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(),lr=hp['lr'])
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lambda epoch:0.95**epoch)

model = NNP(model=model, trainloader=trainloader, testloader=testloader, criterion=criterion, opt=opt, hp=hp, sch=scheduler, checkpoint=50)
model.train()
model.infer()