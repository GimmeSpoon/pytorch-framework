from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch, torch.utils.data
from tqdm import tqdm, trange

class HP:
    def __init__(self, dict:dict):
        for k, v in dict.items():
            setattr(self, k, v)

class Eval(ABC):
    @abstractmethod
    def eval(pred, y):
        pass
    @abstractmethod
    def total_eval():
        pass

class NNP:
    def __init__(self, model:torch.nn.Module=None, trainloader:torch.utils.data.DataLoader=None, testloader:torch.utils.data.DataLoader=None, criterion:torch.nn.Module=None, opt:torch.optim.Optimizer=None, hp:HP=None, sch:torch.optim.lr_scheduler._LRScheduler=None, checkpoint=50, save_last=False, eval:Eval=None, device='cuda'):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion # Loss Function
        self.opt = opt
        self.hp = hp
        self.sch = sch
        self.checkpoint = checkpoint
        self.save_last = save_last
        self.device = torch.device(device)
        self.eval = eval
        self.parm = None
        
    def train(self, epoch = 0, loss = 0.):
        last_epoch = epoch
        self.model.to(self.device)
        self.model.train()
        for epoch in tqdm(range(epoch, self.hp.epochs), desc='Total', unit='epoch', position=0):
            for i, (x, y) in enumerate(tqdm(self.trainloader, desc='Batch', postfix={'Loss':'%.5F'%(loss / self.hp.batch_size)}, leave=False)):
                if i == 0: 
                    loss = 0
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                curloss = self.criterion(pred, y)
                self.opt.zero_grad()
                curloss.backward()
                self.opt.step()
                loss += curloss.item()
            if self.save_last:
                last_epoch = epoch
                self.save(epoch+1, self.model.state_dict(), loss)
            elif epoch % self.checkpoint == self.checkpoint - 1:
                last_epoch = epoch
                self.save(epoch+1, self.model.state_dict(), loss, f'checkpoint_{epoch+1}.pt')
            self.sch.step()
        if last_epoch != epoch:
            self.save(epoch+1, self.model.state_dict(), loss, f'checkpoint_{epoch+1}.pt')

    def infer(self):
        assert eval, "No Evaluation Function" 
        self.model.eval()
        self.model.to(self.device)
        acc = 0.
        with torch.no_grad():
            for x, y in tqdm(self.testloader):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                self.eval.eval(output, y)
                #_, inferred = torch.max(output, 1)
                #acc += torch.sum(inferred == y).item()
        #acc /= len(self.testloader.dataset)
        self.eval.total_eval()

    def validate(self, mode:str)->bool:
        if mode == 'train':
            if not self.model or not self.trainloader or not self.criterion or not self.opt or not self.sch or not self.hp or not self.hp.epochs:
                return False
            return True
        elif mode == 'infer':
            if not self.model or not self.testloader or not self.eval:
                return False
            return True
        else:
            print(f"mode : {mode} is not defined.")

    def save(self, epoch, parm, loss, fname='last_checkpoint.pt'):
        torch.save({
            'epoch': epoch,
            'parm': parm,
            'opt': self.opt.state_dict(),
            'loss': loss,
            'sch': self.sch.state_dict()
        }, fname)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['parm'])
        self.opt.load_state_dict(checkpoint['opt'])
        self.sch.load_state_dict(checkpoint['sch'])
        return (checkpoint['epoch'], checkpoint['loss'])
