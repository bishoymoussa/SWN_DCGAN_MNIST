import torch
from torch import nn
from torch.autograd import Variable
from normalizers import *
torch.manual_seed(0)

class GatedConv2dWithActivation(torch.nn.Module):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x

def get_linear_generator_block(input_dim, output_dim):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(inplace=True),
    )

def get_bn_generator_block(input_dim, output_dim):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True),
    )

def get_conv2d_bn2d_generator_block(input_dim, output_dim, kernel_s, stride, pad, batch_n):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    if batch_n:
      return nn.Sequential(
          nn.ConvTranspose2d(input_dim, output_dim, kernel_s, stride, pad),
          nn.BatchNorm2d(output_dim),
          nn.ReLU(inplace=True)
      )
    return nn.Sequential(
          nn.ConvTranspose2d(input_dim, output_dim, kernel_s, stride, pad),
          nn.ReLU(inplace=True)
      )

def get_swn_conv2d_bn2d_generator_block(input_dim, output_dim, kernel_s, stride, pad, batch_n):
    '''
    Function for returning a block of the generator's neural network
    given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a generator neural network layer, with a linear transformation 
          followed by a batch normalization and then a relu activation
    '''
    if batch_n:
      return nn.Sequential(
          nn.ConvTranspose2d(input_dim, output_dim, kernel_s, stride, pad),
          SpectralNorm(output_dim),
          nn.ReLU(inplace=True)
      )
    return nn.Sequential(
          nn.ConvTranspose2d(input_dim, output_dim, kernel_s, stride, pad),
          nn.ReLU(inplace=True)
      )
    

def get_gated_conv_generator_block(input_dim, output_dim, kernel_s, stride, pad):
    '''
    Gated Conv Discriminator Block with activation
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
    '''
    return nn.Sequential(
        nn.ConvTranspose2d(input_dim, output_dim, kernel_s, stride, pad),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(inplace=True)
    )

def get_swn_gated_conv_generator_block(input_dim, output_dim, kernel_s, stride, pad):
    '''
    Gated Conv Discriminator Block with activation
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
    '''
    return nn.Sequential(
        nn.ConvTranspose2d(input_dim, output_dim, kernel_s, stride, pad),
        SpectralNorm(output_dim),
        nn.ReLU(inplace=True)
    )
    
def get_linear_discriminator_block(input_dim, output_dim):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
          followed by an nn.LeakyReLU activation with negative slope of 0.2 
          (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    '''
    return nn.Sequential(
         nn.Linear(input_dim, output_dim),
         nn.LeakyReLU(0.2, inplace=True)
    )

def get_conv2d_bn2d_discriminator_block(input_dim, output_dim, kernel_size, pad):
    '''
    Discriminator Block
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
          followed by an nn.LeakyReLU activation with negative slope of 0.2 
          (https://pytorch.org/docs/master/generated/torch.nn.LeakyReLU.html)
    '''
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size, 2, pad),
        nn.BatchNorm2d(output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )

def get_gated_conv_discriminator_block(input_dim, output_dim, kernel_size, stride, pad):
    '''
    Gated Conv Discriminator Block with activation
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
    '''
    return nn.Sequential(
        GatedConv2dWithActivation(input_dim, output_dim, kernel_size, stride, pad),
        nn.BatchNorm2d(output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )

def get_swn_gated_conv_discriminator_block(input_dim, output_dim, kernel_size, stride, pad):
    '''
    Gated Conv Discriminator Block with activation
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation 
    '''
    return nn.Sequential(
        GatedConv2dWithActivation(input_dim, output_dim, kernel_size, stride, pad),
        SpectralNorm(output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )

def get_swn_conv2d_bn2d_discriminator_block(input_dim, output_dim, kernel_size, pad):
    '''
    Specteral Weight Normalization Conv Discriminator Block with activation
    Function for returning a neural network of the discriminator given input and output dimensions.
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a discriminator neural network layer, with a linear transformation and speteral normalization 
    '''
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size, 2, pad),
        SpectralNorm(output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )

