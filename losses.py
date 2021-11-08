import torch
from torch import nn
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from layers import *
from utils import show_result
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
torch.manual_seed(0) # Set for testing purposes, please do not change!

def get_noise(n_samples, z_dim, arch_type ,device='cuda'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    if arch_type == '1D':
        return torch.randn(n_samples,z_dim,device=device)
    else:
        noise_ = torch.randn((n_samples, z_dim)).view(-1, z_dim, 1, 1)
        noise_ = Variable(noise_.cuda())
        return noise_

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device, arch_type):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    if arch_type == '2D':
        y_real_samples = torch.ones(num_images)
        y_fake_samples = torch.zeros(num_images)
        real , y_real_samples, y_fake_samples = Variable(real.cuda()), Variable(y_real_samples.cuda()), Variable(y_fake_samples.cuda())
        fake_noise = get_noise(num_images, z_dim, arch_type, device=device)
        disc_output = disc(real).squeeze()
        disc_real_loss = criterion(disc_output, y_real_samples)
        fake_noise = Variable(fake_noise.cuda())
        gen_output = gen(fake_noise)
        disc_output = disc(gen_output).squeeze()
        disc_fake_loss = criterion(disc_output, y_fake_samples)
        disc_fake_score = disc_output.data.mean()
        return disc_fake_loss, disc_fake_score, disc_real_loss, y_real_samples
    else:
        fake_noise = get_noise(num_images, z_dim, arch_type, device=device)
        fake = gen(fake_noise)
        disc_fake_pred = disc(fake.detach())
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc(real)
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        return disc_loss

def get_gen_loss(gen, disc, criterion, num_images, z_dim, device, y_real_samples, arch_type):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        gen_loss: a torch scalar loss value for the current batch
    '''
    if arch_type == '2D':
        fake_noise = get_noise(num_images, z_dim, arch_type, device=device)
        gen_output = gen(fake_noise)
        disc_output = disc(gen_output).squeeze()
        gen_train_loss = criterion(disc_output, y_real_samples)
        return gen_train_loss
    else:
        fake_noise = get_noise(num_images, z_dim, arch_type, device=device)
        fake = gen(fake_noise)
        disc_fake_pred = disc(fake)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        return gen_loss