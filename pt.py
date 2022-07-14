import pandas as pd
import numpy as np
import torch
from tqdm import tqdm, trange

class NNP:
    def __init__(self, model, trainloader, testloader, criterion, opt, hp, sch, checkpoint=50):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.opt = opt
        self.hp = hp
        self.sch = sch
        self.checkpoint = checkpoint
        self.epochs = hp['epochs']
        self.batch_size = hp['batch_size']
        self.device = hp['device']
        
    def train(self, epoch = 0, loss = 0.):
        self.model.to(self.device)
        self.model.train()
        for epoch in tqdm(range(epoch, self.epochs), unit='epoch'):
            for _, (x, y) in enumerate(tqdm(self.trainloader, postfix={'Loss':loss})):
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                curloss = self.criterion(pred, y)
                self.opt.zero_grad()
                curloss.backward()
                self.opt.step()
                loss = curloss.item()
            if epoch == self.checkpoint - 1:
                self.save(epoch, self.model.state_dict(), self.opt.state_dict(), loss)
            #Visualization

    def infer(self):
        self.model.eval()
        acc = 0.
        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                print("DEBUG : Device Check : Inference Output : {0}".format(output.device()))
                # need to be modified for generalization
                # current code is for classfier
                _, inferred = torch.max(output, 1)
                acc += torch.sum(inferred == y).item()
        acc /= self.testloader.__len__
        print(f"Accuracy : {acc}")

    
    def save(self, epoch, parm, opt, loss, fname='checkpoint_'):
        torch.save({
            'epoch': epoch,
            'parm': parm,
            'opt': opt,
            'loss':loss
        }, fname+str(epoch))

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['parm'])
        self.opt.load_state_dict(checkpoint['opt'])
        return (checkpoint['epoch'], checkpoint['loss'])


