import os
import torch
import torch.nn as nn
import torch.optim as optim

import nets
import SoftTriple as ST


def init_weights(net, gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class ResNetModel:
    def __init__(self, opt, train=True):
        self.net = nets.ResNetCifar(opt.n)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.net = self.net.to('cuda')
        else:
            self.device = torch.device('cpu')

        init_weights(self.net)

        if opt.loss_type == 'crossentropy':
            self.criterion = ST.CELoss(64, 10)
        elif opt.loss_type == 'softmaxnorm':
            self.criterion = ST.SoftTriple(opt.la, opt.gamma, 0.0, 0.0, 64, 10, 1)
        elif opt.loss_type == 'softtriple':
            self.criterion = ST.SoftTriple(opt.la, opt.gamma, opt.tau, opt.margin, 64, 10, opt.K)
        else:
            raise NotImplementedError('loss_type must be chosen from [crossentropy, softmaxnorm, softtriple]')

        if train:
            self.checkpoint_dir = opt.checkpoint_dir

            self.optimizer = optim.Adam(
                [{"params": self.net.parameters(), "lr": opt.modellr},
                 {"params": self.criterion.parameters(), "lr": opt.centerlr}],
                eps=opt.eps, weight_decay=opt.weight_decay)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[opt.decay_lr_1, opt.decay_lr_2],
                gamma=opt.lr_decay_rate
                )
            self.loss = 0.0
            self.net.train()
            self.criterion.train()
        else:
            self.net.eval()
            self.criterion.eval()

    def optimize_params(self, x, label):
        x = x.to(self.device)
        label = label.to(self.device)
        emb = self._forward(x)
        self._update_params(emb, label)

    def _forward(self, x):
        return self.net(x)

    def _backward(self, emb, label):
        self.loss = self.criterion(emb, label)
        self.loss.backward()

    def _update_params(self, emb, label):
        self.optimizer.zero_grad()
        self._backward(emb, label)
        self.optimizer.step()
        self.scheduler.step()  # scheduler step in each iteration

    def test(self, x, label):
        with torch.no_grad():
            x = x.to(self.device)
            label = label.to(self.device)
            emb = self._forward(x)
            predicted = self.criterion.infer(emb)
            total = label.size(0)
            correct = (predicted == label).sum().item()
            return correct, total, predicted

    def val(self, x, label):
        with torch.no_grad():
            x = x.to(self.device)
            label = label.to(self.device)
            emb = self._forward(x)
            return self.criterion(emb, label).item()

    def save_model(self, name):
        path_m = os.path.join(self.checkpoint_dir, f'model_{name}.pth')
        path_c = os.path.join(self.checkpoint_dir, f'criterion_{name}.pth')
        torch.save(self.net.state_dict(), path_m)
        torch.save(self.criterion.state_dict(), path_c)
        print(f'model saved to {path_m}, {path_c}')

    def load_model(self, path_m, path_c):
        self.net.load_state_dict(torch.load(path_m))
        self.criterion.load_state_dict(torch.load(path_c))
        print(f'model loaded from {path_m}, {path_c}')

    def get_current_loss(self):
        return self.loss.item()
