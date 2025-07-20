import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import configs.config_setting as cfg


class nmODE(nn.Module):
    def __init__(self, gamma):
        super(nmODE, self).__init__()
        self.gamma = gamma

    def forward(self, t, y):
        dydt = -y + torch.pow(torch.sin(y + self.gamma), 2)
        return dydt


class nmNet_head_link_d1d2_2(nn.Module):
    def __init__(self, delta, input_size, output_size, con_size):
        super().__init__()
        self.con_size = con_size
        self.delta = delta

        self.conv1 = nn.Conv2d(input_size, con_size, kernel_size=3, stride=1, padding=1)
        self.t = torch.tensor([0.0, self.delta], requires_grad=False)

        self.running_mean1 = nn.Parameter(torch.zeros(cfg.y0_channel).cuda(), requires_grad=False)
        self.running_var1 = nn.Parameter(torch.ones(cfg.y0_channel).cuda(), requires_grad=False)
        self.running_mean2 = nn.Parameter(torch.zeros(cfg.y0_channel).cuda(), requires_grad=False)
        self.running_var2 = nn.Parameter(torch.ones(cfg.y0_channel).cuda(), requires_grad=False)

        self.momentun = 0.8

    def forward(self, x_shallow, y_deep):
        B = y_deep.shape[0]
        C = y_deep.shape[1]
        h = y_deep.shape[2]
        w = y_deep.shape[3]


        gamma = self.conv1(x_shallow).reshape(B, -1)
        nmODE1 = nmODE(gamma)
        # get the solution of all time steps
        all_solutions = odeint(nmODE1, y_deep.reshape(B, -1), self.t)

        dydt_solution1 = nmODE1(self.t, all_solutions[0]).reshape(B, C, h, w)
        dydt_solution2 = nmODE1(self.t,all_solutions[1]).reshape(B, C, h, w)

        if self.training and cfg.DUSE == False:
            with torch.no_grad():
                # Calculate the mean and variance along the batch and spatial dimensions.
                batch_mean1 = dydt_solution1.mean(dim=(0, 2, 3))  # (1, C, 1, 1)
                batch_var1 = dydt_solution1.var(dim=(0, 2, 3))
                batch_mean2 = dydt_solution2.mean(dim=(0, 2, 3))
                batch_var2 = dydt_solution2.var(dim=(0, 2, 3))

                # update running_mean and running_var
                self.running_mean1.data = (1 - self.momentun) * self.running_mean1.data + self.momentun * batch_mean1
                self.running_var1.data = (1 - self.momentun) * self.running_var1.data + self.momentun * batch_var1
                # update running_mean and running_var
                self.running_mean2.data = (1 - self.momentun) * self.running_mean2.data + self.momentun * batch_mean2
                self.running_var2.data = (1 - self.momentun) * self.running_var2.data + self.momentun * batch_var2
        else:

            new_mu_1, new_var_1 = self.running_mean1, self.running_var1
            new_mu_2, new_var_2 = self.running_mean2, self.running_var2


            cur_mu1 = dydt_solution1.mean((2, 3), keepdims=True) # (B,c,1,1)
            cur_std1 = dydt_solution1.std((2, 3), keepdims=True)
            cur_mu2 = dydt_solution2.mean((2, 3), keepdims=True)
            cur_std2 = dydt_solution2.std((2, 3), keepdims=True)
            self.dydt_loss = 0.5 * (
                    (new_mu_1 - cur_mu1).abs().mean() + (new_var_1.sqrt() - cur_std1).abs().mean()
                    + (new_mu_2 - cur_mu2).abs().mean() + (new_var_2.sqrt() - cur_std2).abs().mean()
            )

        y = all_solutions[-1]
        y = y.reshape(B, C, h, w)
        return y

