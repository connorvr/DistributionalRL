from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.hx_prev = None
        self.cx_prev = None
        self.eps_len = 0
        self.args = args
        self.logits_array = []
        self.rewards = []
        self.done = True
        self.info = None
        self.reward = 0
        self.gpu_id = -1
        self.quantiles_array = []

    def action_train(self):
        self.hx_prev = self.hx
        self.cx_prev = self.cx
        logit, _, (self.hx, self.cx) = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))
        q_vals = torch.mean(logit,0)
        prob = F.softmax(q_vals, dim=0)
        action = prob.multinomial(1).data

#        _, action = torch.max(q_vals,0)
        logit, quantiles, _ = self.model((
            Variable(self.state.unsqueeze(0)), (self.hx_prev, self.cx_prev)))
        self.quantiles_array.append(quantiles)
        logit = logit[:,Variable(action)]
        self.logits_array.append(logit)
        

        state, self.reward, self.done, self.info = self.env.step(
            action.cpu().numpy())
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.reward = max(min(self.reward, 1), -1)
        self.rewards.append(self.reward)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = Variable(
                            torch.zeros(1, 512).cuda())
                        self.hx = Variable(
                            torch.zeros(1, 512).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, 512))
                    self.hx = Variable(torch.zeros(1, 512))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
            logit, _, (self.hx, self.cx) = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))
        
        q_vals = torch.mean(logit,0)
        _, action = torch.max(q_vals,0)
        
        state, self.reward, self.done, self.info = self.env.step(action.data.cpu().numpy())
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        self.eps_len += 1
        return self

    def clear_actions(self):
        self.logits_array = []
        self.rewards = []
        self.quantiles_array = []
        return self
