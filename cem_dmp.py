###### Description : This file optimizes the parameters of a DMP using the CEM algorithm ######

import numpy as np
import torch
import matplotlib.pyplot as plt

from dmps import DMP


###### Import trajectory data ######
data = np.load('data/2.npz')
traj = data['arr_0']
traj = torch.from_numpy(traj).float()
dtraj = traj[1:] - traj[:-1]
ddtraj = dtraj[1:] - dtraj[:-1]

###### Initialize DMP ######
N = 8
dt = 0.01
a_x = 1
a_z = 20
dmp = DMP(N, a_x, a_z, dt=dt)

###### Set up CEM hyperparameters######
num_samples = 1024
num_elites = 100
num_iters = 400

###### Set up initial weights ######
w_mean = torch.zeros(2*N)
w_std = torch.ones(2*N)*100

###### CEM optimization ######
i = 0
while i < num_iters:
    losses = []
    samples = torch.randn(num_samples, 2*N) * w_std + w_mean
    w = torch.clip(samples.view(num_samples, 2, N), -1000, 1000)
    start = traj[0].view(1, 2).repeat(num_samples, 1)
    goal = traj[-1].view(1, 2).repeat(num_samples, 1)
    dstart = torch.zeros(num_samples, 2)
    gen_traj, dgen_traj, ddgen_traj = dmp.integrate(start, dstart, goal, w, traj.shape[0])


    loss = torch.pow(gen_traj - traj.T.view(1, traj.shape[1], traj.shape[0]), 2)
    loss = loss.sum(2).sum(1)
    elites = samples[loss.argsort()[:num_elites]]
    w_mean = elites.mean(0)
    w_std = elites.std(0)
    #### print the minimum loss
    print('loss : ', loss[loss.argsort()[0]].item())

    if i == num_iters-1:
        print('max dmp', torch.max(w_mean))
        print('min dmp', torch.min(w_mean))
        ###### Plot the results ######
        gen_traj = dmp.integrate(traj[0].view(1, 2), torch.zeros(2).view(1, 2), traj[-1].view(1, 2), w_mean.view(1, 2, N), traj.shape[0])
        gen_traj = gen_traj[0].detach().numpy().T
        plt.plot(traj[:,0], traj[:,1], 'r')
        plt.plot(gen_traj[:, 0], gen_traj[:, 1], 'b')
        plt.savefig('cem_dmp.png')

    i+=1
