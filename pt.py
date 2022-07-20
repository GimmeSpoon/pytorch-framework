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
        self.device = torch.device(hp['device'])
        
    def train(self, epoch = 0, loss = 0.):
        last_epoch = epoch
        self.model.to(self.device)
        self.model.train()
        for epoch in tqdm(range(epoch, self.epochs), unit='epoch', position=0):
            for i, (x, y) in enumerate(tqdm(self.trainloader, postfix={'Loss':'%.5F'%(loss / self.batch_size)}, leave=False)):
                if i == 0: 
                    loss = 0
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                curloss = self.criterion(pred, y)
                self.opt.zero_grad()
                curloss.backward()
                self.opt.step()
                loss += curloss.item()
            if epoch % self.checkpoint == self.checkpoint - 1:
                self.save(epoch+1, self.model.state_dict(), self.opt.state_dict(), loss)
            self.sch.step()
        if last_epoch != epoch:
            self.save(epoch+1, self.model.state_dict(), self.opt.state_dict(), loss)

    def infer(self):
        self.model.eval()
        self.model.to(self.device)
        acc = 0.
        with torch.no_grad():
            for x, y in tqdm(self.testloader):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                # need to be modified for generalization
                # current code is for classfier
                _, inferred = torch.max(output, 1)
                acc += torch.sum(inferred == y).item()
        acc /= len(self.testloader.dataset)
        print(f"Accuracy : {acc}")

    
    def save(self, epoch, parm, opt, loss, fname='checkpoint_'):
        torch.save({
            'epoch': epoch,
            'parm': parm,
            'opt': opt,
            'loss':loss
        }, fname+str(epoch)+".pt")

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['parm'])
        self.opt.load_state_dict(checkpoint['opt'])
        return (checkpoint['epoch'], checkpoint['loss'])


