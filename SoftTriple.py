# https://github.com/idstcv/SoftTriple
# Implementation of SoftTriple Loss
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class SoftTriple(nn.Module):
    def __init__(self, la, gamma, tau, margin, dim, cN, K, device):
        super(SoftTriple, self).__init__()
        self.device = device
        self.la = la
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.fc = Parameter(torch.Tensor(dim, cN*K).to(self.device))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).to(self.device)
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))

    def forward(self, input, target):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        marginM = torch.zeros(simClass.shape).to(self.device)
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin
        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify

    def infer(self, input):
        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)
        return torch.max(simClass.data, 1)[1]  # return class indices


class CELoss(nn.Module):
    def __init__(self, dim, cN, device):
        super().__init__()
        self.device = device
        self.dim = dim
        self.cN = cN
        self.fc = Parameter(torch.Tensor(dim, cN).to(self.device))
        self.ce = nn.CrossEntropyLoss()
        init.kaiming_uniform_(self.fc, a=math.sqrt(5))

    def forward(self, input, target):
        emb = input.matmul(self.fc)
        return self.ce(emb, target)

    def infer(self, input):
        emb = input.matmul(self.fc)
        return torch.max(emb.data, 1)[1]  # return class indices
