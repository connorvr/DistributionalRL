from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import atari_env
from utils import ensure_shared_grads
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import torch.nn.functional as F


def train(rank, args, shared_model, optimizer, env_conf, num_tau_samples=32, num_tau_prime_samples=32, kappa=1.0, num_quantiles=32):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    env = atari_env(args.env, env_conf, args)
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Clstm(player.env.observation_space.shape[0],
                           player.env.action_space)

    player.state = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
    player.model.train()
    player.eps_len += 2
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, 512).cuda())
                    player.hx = Variable(torch.zeros(1, 512).cuda())
            else:
                player.cx = Variable(torch.zeros(1, 512))
                player.hx = Variable(torch.zeros(1, 512))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)

        for step in range(args.num_steps):
            player.action_train()
            if player.done:
                break

        if player.done:
            state = player.env.reset()
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        
        R = torch.zeros(1,num_tau_prime_samples)
        if not player.done:
            logit, _, _ = player.model((Variable(
                    player.state.unsqueeze(0)), (player.hx, player.cx)))
        
            q_vals = torch.mean(logit,0)
            _, action = torch.max(q_vals,0)
            logit, _, _ = player.model((Variable(player.state.unsqueeze(0)),
                    (player.hx, player.cx)))
            
            R = logit[:,action]

        
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()       
        #R = R.detach()
        R = Variable(R)
        
        value_loss = 0
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i]

            advantage = R.repeat(num_tau_samples,1) - player.logits_array[i].repeat(1, num_tau_prime_samples)
            #print("Ad: ",advantage)
            loss = (torch.abs(advantage) <= kappa).float() * 0.5 * advantage ** 2
            #print("loss: ",loss.sum(0).sum(0), loss)
            loss += (torch.abs(advantage) > kappa).float() * kappa * (torch.abs(advantage) - 0.5 * kappa)
            #print("loss: ",loss.sum(0).sum(0), loss)
            step_loss = torch.abs(player.quantiles_array[i].cuda() - (advantage.detach()<0).float()) * loss/kappa                 
            value_loss += step_loss.sum(0).mean(0)

        
        player.model.zero_grad()
        value_loss.backward()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()
