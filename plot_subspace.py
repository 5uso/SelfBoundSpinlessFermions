import sys
sys.path.append("./src/")

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.nn.functional import pad
from utils import load_model, trans_invar_rot
from Models import vLogHarmonicNet
from Samplers import MetropolisHastings
from matplotlib import rcParams

rcParams.update({'font.size': 35})

def plot_1d(net):
    n_mesh = 1000
    plot_bounds = (-4, 4)
    trans = torch.tensor(np.linalg.inv(trans_invar_rot(2)), dtype=torch.float32)
    relative = torch.linspace(plot_bounds[0], plot_bounds[1], n_mesh)
    space = pad(relative.unsqueeze(-1), (1,0,0,0))
    space = torch.matmul(trans, space.unsqueeze(-1)).squeeze(-1)
    sign_psi, log_abs_psi = net(space)
    density = 2**log_abs_psi.cpu().detach().numpy() * sign_psi.cpu().detach().numpy()
    plt.plot(relative, density)
    plt.grid(which='both')
    plt.xlim(plot_bounds)
    plt.ylabel('$\Psi$')
    plt.xlabel('Relative position, $r$')
    plt.title('$N=2, V=-13.262, \sigma=0.375$')
    plt.show()

def plot_2d(net):
    n_mesh = 512
    plot_bounds = (-4, 4)
    trans = torch.tensor(np.linalg.inv(trans_invar_rot(3)), dtype=torch.float32)
    relative = torch.linspace(plot_bounds[0], plot_bounds[1], n_mesh)
    relative_grid = torch.cartesian_prod(relative, relative)
    space = pad(relative_grid, (1,0,0,0))
    space = torch.matmul(trans, space.unsqueeze(-1)).squeeze(-1)
    sign_psi, log_abs_psi = net(space)
    density = (2**log_abs_psi.cpu().detach().numpy() * sign_psi.cpu().detach().numpy()).reshape(n_mesh,n_mesh)
    fcont = plt.contourf(relative, relative, density, cmap='coolwarm')
    plt.contour(relative, relative, density, colors='k')
    plt.colorbar(fcont)
    plt.axis("square")
    plt.xlabel('Relative position, $r_0$')
    plt.ylabel('Relative position, $r_1$')
    plt.title('$N=3, V=-13.262, \sigma=0.375$')
    plt.show()

#"""
model_path = "results/notrap/checkpoints/A02_H064_L02_D01_Tanh_W4096_P001000_V-1.33e+01_S3.75e-01_RMSprop_PT_False_device_cuda_dtype_float32_chkp.pt"

nfermions = 2
num_hidden = 64
num_layers = 2
num_dets = 1
device = torch.device('cuda')
net = vLogHarmonicNet(num_input=nfermions, num_hidden=num_hidden, num_layers=num_layers, num_dets=num_dets, func=torch.nn.Tanh(), pretrain=False)
optim = torch.optim.RMSprop(params=net.parameters())
sampler = MetropolisHastings(network=net, dof=nfermions, nwalkers=1, target_acceptance=0.5)

output_dict = load_model(model_path=model_path, device=device, net=net, optim=optim, sampler=sampler)
net=output_dict['net']

plot_1d(net)
#"""

"""
model_path = "results/notrap/checkpoints/A03_H064_L02_D01_Tanh_W4096_P001000_V-1.33e+01_S3.75e-01_RMSprop_PT_False_device_cuda_dtype_float32_chkp.pt"

nfermions = 3
num_hidden = 64
num_layers = 2
num_dets = 1
device = torch.device('cuda')
net = vLogHarmonicNet(num_input=nfermions, num_hidden=num_hidden, num_layers=num_layers, num_dets=num_dets, func=torch.nn.Tanh(), pretrain=False)
optim = torch.optim.RMSprop(params=net.parameters())
sampler = MetropolisHastings(network=net, dof=nfermions, nwalkers=1, target_acceptance=0.5)

output_dict = load_model(model_path=model_path, device=device, net=net, optim=optim, sampler=sampler)
net=output_dict['net']

plot_2d(net)
#"""
