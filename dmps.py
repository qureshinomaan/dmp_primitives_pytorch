###### File for the DMPs class ######
import numpy as np
import torch
import matplotlib.pyplot as plt

class DMP:
    def __init__(self, N, a_x, a_z, rbf='gaussian', dt=0.01, tau=1, device='cpu') -> None:
        """
        Initialize the DMP class
        N: number of kernels
        a_x: scaling factor for the phase variable
        a_z: scaling factor for the forcing term
        rbf: type of RBF to use
        dt: time step
        tau: time constant
        """
        self.device = device

        self.N = N
        self.a_x = a_x
        self.a_z = a_z
        self.c = torch.Tensor([np.exp(-i*self.a_x/N) for i in range(N)]).to(self.device)
        self.h = torch.Tensor([N**1.5/self.c[i]/self.a_x for i in range(N)]).to(self.device)
        self.rbf = rbf
        self.dt = dt
        self.tau = tau

    def integrate(self, start, dstart, goal, w, steps, az=False, alpha_z=None):
        """
        Integrate the DMP
        w: weights of the DMP
        start: starting position
        dstart: starting velocity
        goal: goal position
        steps: number of steps to integrate
        az: if True, use alpha_z as the scaling factor for the forcing term
        alpha_z: scaling factor for the forcing term
        """
        y = start
        z = dstart * self.tau
        x = 1
        Y = torch.zeros((w.shape[0], w.shape[1], int(steps)))
        dY = torch.zeros((w.shape[0], w.shape[1], int(steps)))
        ddY = torch.zeros((w.shape[0], w.shape[1], int(steps)))

        Y[:, :, 0] = y
        dY[:, :, 0] = dstart
        ddY[:, :, 0] = z
        if az:
            self.a_z = alpha_z
            self.a_z = torch.clamp(a_z, 0.5, 30)
        b_z = self.a_z / 4
        for i in range(0, steps-1):
            dx = (-self.a_x * x) / self.tau
            x = x + dx * self.dt
            eps = torch.pow((x - self.c), 2)
            if self.rbf == 'gaussian':
                psi = torch.exp(-self.h * eps)
            if self.rbf == 'multiquadric':
                psi = torch.sqrt(1 + self.h * eps)
            if self.rbf == 'inverse_quadric':
                psi = 1/(1 + self.h*eps)
            if self.rbf == 'inverse_multiquadric':
                psi = 1/torch.sqrt(1 + self.h * eps)
            if self.rbf == 'linear':
                psi = self.h * eps

            fx = torch.matmul(w, psi)*x*(goal-start) / torch.sum(psi)
            dz = self.a_z * (b_z * (goal - y) - z) + fx
            dy = z 
            dz = dz / self.tau
            dy = dy / self.tau 
            y = y + dy * self.dt 
            z = z + dz * self.dt 
            Y[:, :, i+1] = y
            dY[:, :, i+1] = dy
            ddY[:, :, i+1] = dz
        return Y, dY, ddY


if __name__ == '__main__':
    N = 64
    dt = 0.01
    a_x = 2
    a_z = 20
    dmp = DMP(N=N, a_x=a_x, a_z=a_z, dt=dt)
    w = torch.randn((1, 2, N))
    start = torch.tensor([0, 0]).view(1, 2)
    dstart = torch.tensor([0, 0]).view(1, 2)
    goal = torch.randn(1, 2)
    steps = 1000
    Y, dY, ddY = dmp.integrate(start, dstart, goal, w, steps)