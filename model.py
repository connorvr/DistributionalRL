from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init, unif_col_init
import math
import numpy as np


class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space, quantile_embedding_dim=64, num_quantiles=32):
        super(A3Clstm, self).__init__()
        #Input Shape = [1,1,80,80]
        self.quantile_embedding_dim = quantile_embedding_dim
        self.num_quantiles = num_quantiles
        
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2) #Shape = [1,32,80,80]
        self.maxp1 = nn.MaxPool2d(2, 2) #Shape = [1,32,40,40]
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1) #Shape = [1,32,38,38]
        self.maxp2 = nn.MaxPool2d(2, 2) #Shape = [1,32,19,19]
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2) #Shape = [1,64,9,9]
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2) #Shape = [1,64,4,4]

        #self.lstm = nn.LSTMCell(1024, 512) #Shape = [1,512]
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(512, num_outputs)
#        self.actor_linear = nn.Linear(512, num_outputs)
        
        self.quantile_linear = nn.Linear(64, 1024) 
        self.middle_linear = nn.Linear(1024, 512)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
#        self.actor_linear.weight.data = norm_col_init(
#            self.actor_linear.weight.data, 0.01)
#        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 0.01)
        self.critic_linear.bias.data.fill_(0)
        self.quantile_linear.weight.data = unif_col_init(
            self.quantile_linear.weight.data, std=1.0/np.sqrt(3.0))
        self.quantile_linear.bias.data.fill_(0)
        self.middle_linear.weight.data = unif_col_init(
            self.middle_linear.weight.data, std=1.0/np.sqrt(3.0))
        self.middle_linear.bias.data.fill_(0)

#        self.lstm.bias_ih.data.fill_(0)
#        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)
        
#        hx, cx = self.lstm(x, (hx, cx))
#
#        x = hx
        
        
        batch_size = x.size(0)
        x_tiled = x.repeat(self.num_quantiles * batch_size, 1) #x_tile = [32,512]
        
        
        quantiles = torch.FloatTensor(self.num_quantiles * batch_size, 1).uniform_(0, 1) #Shape = [32,1]
        quantile_net = quantiles.repeat(1, self.quantile_embedding_dim)  #Shape = [32,64]
        quantile_net = torch.arange(1, self.quantile_embedding_dim + 1, 1, dtype=torch.float32) * \
            math.pi * quantile_net #Shape = [32,64]
        quantile_net = torch.cos(quantile_net)
#        quantile_net = quantile_net.view(1, -1) #Shape = [1, 2048]
        
        
        if 1 >= 0:
            with torch.cuda.device(0):
                quantile_net = quantile_net.cuda()
        quantile_net = self.quantile_linear(quantile_net) #Shape = [32,512]
        quantile_net = F.relu(quantile_net)
        
        
        x = x_tiled * quantile_net #shape = [32,512]
        
        x = self.middle_linear(x)
        x = F.relu(x)
        x = self.critic_linear(x)
        #print("weights :", self.critic_linear.weight.data)
        return x, quantiles, (hx, cx)
