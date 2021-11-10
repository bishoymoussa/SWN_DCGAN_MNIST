from threading import Semaphore
import torch
from torch import nn
from torch.nn.functional import leaky_relu
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from layers import *
from utils import show_result, show_train_hist, show_tensor_images
from losses import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
import json
torch.manual_seed(0) # Set for testing purposes, please do not change!

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

def normal_init(m, mean, std):
    '''
    Funtion used to initiate the Conv2d and TransposeConv2d Layers Weights
    '''
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
        nn_type = Used to determine the Network type from ['linear', 'bn', 'convbn2d', 'swnconvbn2d', 'gconv2d', 'swngconv2d']
    '''
    def __init__(self, z_dim=100, im_dim=784, hidden_dim=128, nn_type='linear'):
        super(Generator, self).__init__()
        if nn_type == 'linear':
            self.gen = nn.Sequential(
                get_linear_generator_block(z_dim, hidden_dim),
                get_linear_generator_block(hidden_dim, hidden_dim * 2),
                get_linear_generator_block(hidden_dim * 2, hidden_dim * 4),
                get_linear_generator_block(hidden_dim * 4, hidden_dim * 8),
                nn.Linear(hidden_dim * 8, im_dim),
                nn.Sigmoid()
            )
        elif nn_type == 'bn':
          self.gen = nn.Sequential(
              get_bn_generator_block(z_dim, hidden_dim),
              get_bn_generator_block(hidden_dim, hidden_dim * 2),
              get_bn_generator_block(hidden_dim * 2, hidden_dim * 4),
              get_bn_generator_block(hidden_dim * 4, hidden_dim * 8),
              nn.Linear(hidden_dim * 8, im_dim),
              nn.Sigmoid()
          )
        
        elif nn_type == 'convbn2d':
          self.gen = nn.Sequential(
              get_conv2d_bn2d_generator_block(z_dim, hidden_dim * 8, 4, 1, 0, True),
              get_conv2d_bn2d_generator_block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, True),
              get_conv2d_bn2d_generator_block(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, True),
              get_conv2d_bn2d_generator_block(hidden_dim * 2, hidden_dim, 4, 2 ,1, True),
              get_conv2d_bn2d_generator_block(hidden_dim, 1, 4, 2, 1, False),
              nn.Tanh()
          )
        
        elif nn_type == 'swnconvbn2d':
          self.gen = nn.Sequential(
              get_conv2d_bn2d_generator_block(z_dim, hidden_dim * 8, 4, 1, 0, True),
              get_conv2d_bn2d_generator_block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, True),
              get_conv2d_bn2d_generator_block(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, True),
              get_conv2d_bn2d_generator_block(hidden_dim * 2, hidden_dim, 4, 2 ,1, True),
              get_conv2d_bn2d_generator_block(hidden_dim, 1, 4, 2, 1, False),
              nn.Tanh()
          )
        elif nn_type == 'gconv2d':
            self.gen = nn.Sequential(
              get_gated_conv_generator_block(z_dim, hidden_dim * 8, 4, 1, 0),
              get_gated_conv_generator_block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
              get_gated_conv_generator_block(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
              get_gated_conv_generator_block(hidden_dim * 2, hidden_dim, 4, 2 ,1),
              get_gated_conv_generator_block(hidden_dim, 1, 4, 2, 1),
              nn.Tanh()
            )
        elif nn_type == 'swngconv2d':
            self.gen = nn.Sequential(
              get_swn_gated_conv_generator_block(z_dim, hidden_dim * 8, 4, 1, 0),
              get_swn_gated_conv_generator_block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
              get_swn_gated_conv_generator_block(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
              get_swn_gated_conv_generator_block(hidden_dim * 2, hidden_dim, 4, 2 ,1),
              get_swn_gated_conv_generator_block(hidden_dim, 1, 4, 2, 1),
              nn.Tanh()
            )
    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor, 
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        return self.gen(noise)

    # weight initialization needed for stable testing in the conv2d and bn2d
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    # Needed for grading
    def get_gen(self):
        '''
        Returns:
            the sequential model
        '''
        return self.gen


class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
            (MNIST images are 28x28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
        nn_type = Used to determine the Network type from ['linear', 'bn', 'convbn2d', 'swnconvbn2d', 'gconv2d', 'swngconv2d']
    '''
    def __init__(self, z_dim=100, im_dim=784, hidden_dim=128, nn_type='linear'):
        super(Discriminator, self).__init__()
        if nn_type == 'linear':
            self.disc = nn.Sequential(
                get_linear_discriminator_block(im_dim, hidden_dim * 4),
                get_linear_discriminator_block(hidden_dim * 4, hidden_dim * 2),
                get_linear_discriminator_block(hidden_dim * 2, hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
        elif nn_type == 'bn':
            self.disc = nn.Sequential(
                get_linear_discriminator_block(im_dim, hidden_dim * 4),
                get_linear_discriminator_block(hidden_dim * 4, hidden_dim * 2),
                get_linear_discriminator_block(hidden_dim * 2, hidden_dim),
                nn.Linear(hidden_dim, 1)
            )
        elif nn_type == 'convbn2d':
            self.disc = nn.Sequential(
                nn.Conv2d(1, hidden_dim, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                get_conv2d_bn2d_discriminator_block(hidden_dim, hidden_dim * 2, 4, 1),
                get_conv2d_bn2d_discriminator_block(hidden_dim * 2, hidden_dim * 4, 4, 1),
                get_conv2d_bn2d_discriminator_block(hidden_dim * 4, hidden_dim * 8, 3, 1),
                nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0), 
                nn.Sigmoid()
            )
        elif nn_type == 'swnconvbn2d':
            self.disc = nn.Sequential(
                nn.Conv2d(1, hidden_dim, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                get_swn_conv2d_bn2d_discriminator_block(hidden_dim, hidden_dim * 2, 4, 1),
                get_swn_conv2d_bn2d_discriminator_block(hidden_dim * 2, hidden_dim * 4, 4, 1),
                get_swn_conv2d_bn2d_discriminator_block(hidden_dim * 4, hidden_dim * 8, 3, 1),
                nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0), 
                nn.Sigmoid()
            )
        elif nn_type == 'gconv2d':
            self.disc = nn.Sequential(
                GatedConv2dWithActivation(1, hidden_dim, 4, 1, 0),
                get_gated_conv_discriminator_block(hidden_dim, hidden_dim * 2, 4, 1, 0),
                get_gated_conv_discriminator_block(hidden_dim * 2, hidden_dim * 4, 4, 1, 1),
                get_gated_conv_discriminator_block(hidden_dim * 4, hidden_dim * 8, 3, 1, 1),
                GatedConv2dWithActivation(hidden_dim * 8, 1, 4, 1, 0), 
                nn.Sigmoid()
            )
        elif nn_type == 'swngconv2d':
            self.disc = nn.Sequential(
                GatedConv2dWithActivation(1, hidden_dim, 4, 1, 0),
                get_swn_gated_conv_discriminator_block(hidden_dim, hidden_dim * 2, 4, 1, 0),
                get_swn_gated_conv_discriminator_block(hidden_dim * 2, hidden_dim * 4, 4, 1, 1),
                get_swn_gated_conv_discriminator_block(hidden_dim * 4, hidden_dim * 8, 3, 1, 1),
                GatedConv2dWithActivation(hidden_dim * 8, 1, 4, 1, 0), 
                nn.Sigmoid()
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        return self.disc(image)

    # weight initialization needed for stable testing
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def get_disc(self):
        '''
        Returns:
            the sequential model
        '''
        return self.disc



def train(arch_type, nn_type, batch_size, n_epochs, display_step, device, visualize):
    n_epochs = int(n_epochs)
    arch_type = arch_type
    batch_size = int(batch_size)
    nn_type = nn_type
    device = device
    display_step = int(display_step)
    
    if arch_type == '2D':
        criterion = nn.BCEWithLogitsLoss()#nn.BCELoss()    
        lr = 0.0002
        # Load the MNIST data in Grayscale Format
        img_size = 64
        z_dim = 100
        transform = transforms.Compose([
                transforms.Scale(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    else:
        criterion = nn.BCEWithLogitsLoss()
        lr = 0.00001
        z_dim = 64
        # Load MNIST dataset as tensors
        dataloader = DataLoader(
            MNIST('.', download=True, transform=transforms.ToTensor()),
            batch_size=batch_size,
            shuffle=True)




    # Initialize The Generator and Discriminator Networks
    gen = Generator(nn_type=nn_type, z_dim=z_dim).to(device)
    disc = Discriminator(nn_type=nn_type, z_dim=z_dim).to(device)
    
    
    if arch_type == "2D":
        gen.weight_init(mean=0.0, std=0.02)
        disc.weight_init(mean=0.0, std=0.02)
    if nn_type == "swnconvbn2d":
        disc_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, disc.parameters()), lr=lr, betas=(0.0,0.9))
        gen_opt  = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.0,0.9))
    else:
        gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
        disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))


    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    test_generator = False # Whether the generator should be tested
    gen_loss = False
    error = False
    gen_loss_values = []
    disc_loss_values = []
    if arch_type == '2D':
        for epoch in range(n_epochs):
            # Dataloader returns the batches
            for real, _ in tqdm(train_loader):
                D_losses = []
                G_losses = []
                cur_batch_size = real.size()[0]
                ### Update discriminator ###
                # Zero out the gradients before backpropagation
                disc_opt.zero_grad()
                # Calculate discriminator loss
                disc_train_loss, disc_fake_score, disc_real_loss, y_real_samples = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device, arch_type)
                # Update gradients
                disc_train_loss.backward() # removed retain graph for memory consumption
                # Update optimizer
                disc_opt.step()
                D_losses.append(disc_train_loss.item())
                # For testing purposes, to keep track of the generator weights
                if test_generator:
                    old_generator_weights = gen.gen[0][0].weight.detach().clone()
                ### Update generator ###
                gen_opt.zero_grad()
                gen_train_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device, y_real_samples, arch_type)
                gen_train_loss.backward()
                gen_opt.step()
                G_losses.append(gen_train_loss.item())
                # For testing purposes, to check that your code changes the generator weights
                if test_generator:
                    try:
                        assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                        assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
                    except:
                        error = True
                        print("Runtime tests have failed")

                ### Visualization code ###
                if cur_step % display_step == 0 and cur_step > 0:
                    # print(f"Step {cur_step}: Generator loss: {gen_train_loss.item()}, discriminator loss: {disc_train_loss.item()}")
                    if visualize == "YES":
                        p = 'results/MNIST_DCGAN_' + str(nn_type) + '_' +  str(epoch + 1) + '.png'
                        show_result((epoch+1), gen, path=p, isFix=False)
                    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
                    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))  
            
                cur_step += 1
        show_train_hist(train_hist, path='plots/MNIST_DCGAN_{}_train_hist.png'.format(nn_type))
        plt.plot(train_hist['D_losses'])
        plt.plot(train_hist['G_losses'])
        plt.savefig("plots/MNIST_DCGAN_{}_LOSSES.png".format('nn_type'))

    else:
        cur_step = 0
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        test_generator = True # Whether the generator should be tested
        gen_loss = False
        gen_losses = []
        disc_losses = []
        for epoch in range(n_epochs):
  
            # Dataloader returns the batches
            for real, _ in tqdm(dataloader):
                cur_batch_size = len(real)

                # Flatten the batch of real images from the dataset
                real = real.view(cur_batch_size, -1).to(device)

                ### Update discriminator ###
                # Zero out the gradients before backpropagation
                disc_opt.zero_grad()

                # Calculate discriminator loss
                disc_loss = get_disc_loss_1d(gen, disc, criterion, real, cur_batch_size, z_dim, arch_type, device)
                

                # Update gradients
                disc_loss.backward(retain_graph=True)

                # Update optimizer
                disc_opt.step()

                # For testing purposes, to keep track of the generator weights
                if test_generator:
                    old_generator_weights = gen.gen[0][0].weight.detach().clone()
                gen_opt.zero_grad()
                gen_loss = get_gen_loss_1d(gen, disc, criterion, cur_batch_size, z_dim, arch_type, device)
                gen_loss.backward()
                gen_opt.step()

                # For testing purposes, to check that your code changes the generator weights
                if test_generator:
                    try:
                        assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
                        assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
                    except:
                        error = True
                        print("Runtime tests have failed")

                # Keep track of the average discriminator loss
                mean_discriminator_loss += disc_loss.item() / display_step

                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / display_step

                ### Visualization code ###
                if cur_step % display_step == 0 and cur_step > 0:
                    print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                    if visualize == "YES":
                        fake_noise = get_noise(cur_batch_size, z_dim, arch_type, device=device)
                        fake = gen(fake_noise)
                        show_tensor_images(fake, epoch, nn_type)
                        show_tensor_images(real, epoch, nn_type)
                    gen_losses.append(mean_generator_loss)
                    disc_losses.append(mean_discriminator_loss)
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
                cur_step += 1
        plt.plot(gen_losses)
        plt.plot(disc_losses)
        plt.savefig('plots/MNIST_LINEAR_{}_LOSSES.png'.format(nn_type))
    # Save Model Weights
    torch.save(gen.state_dict(), 'models/gen_model_{}.pth'.format(nn_type))
    torch.save(disc.state_dict(), 'models/disc_model_{}.pth'.format(nn_type))


if __name__ == "__main__":
    config_path = 'train_config.json'
    with open(config_path) as json_file:
        train_config = json.load(json_file)
        print("Training Session Configuration as found at '{}' are: {}".format(config_path, train_config.values()))
        train(train_config["arch_type"], train_config["nn_type"], train_config["batch_size"], train_config["n_epochs"], train_config["display_step"] ,train_config["device"], train_config["visualize"])
