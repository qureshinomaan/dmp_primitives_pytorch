"""
    File for the differentiable Linear Dynamical System class
"""
import numpy as np
import torch

class dLDS:
    def __init__(self, N, traj_dim, dt=0.01) -> None:
        """
        Initialize the dLDS class
        N: number of LDS
        traj_dim: dimension of each element of the trajectory
        dt: time step
        """
        self.N = N
        self.traj_dim = traj_dim
        self.dt = dt

    def create_lds(self, params):
        """
        params : parameters of the LDS >> eigenvalues and eigenvectors
        """
        eigen_values = params[0:self.traj_dim]
        eigen_vectors = params[self.traj_dim:].view(self.traj_dim, self.traj_dim)

        ##### shrink the eigen values between -1 and 0 #####
        eigen_values = -0.8 * torch.sigmoid(eigen_values) - 0.1

        ##### create the LDS matrix #####
        A = torch.diag(eigen_values)
        lds = torch.matmul(torch.matmul(eigen_vectors, A), torch.inverse(eigen_vectors))

        return lds

    def integrate(self, start, goal, params, steps=100):
        """
        Integrate the dLDS
        start: starting position
        goal: goal position for the LDS
        steps: number of steps to integrate
        """
        Y = torch.zeros((self.traj_dim, self.N))
        Y[:, 0] = start
        lds = self.create_lds(params)
        for i in range(1, steps):
            action = torch.matmul(lds, Y[:, i-1] - goal)
            Y[:, i] = Y[:, i-1] + action*self.dt
        return Y
