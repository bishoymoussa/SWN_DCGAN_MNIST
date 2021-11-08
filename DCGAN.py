import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from torchvision.datasets import MNIST # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from layers import *
from utils import show_result, show_train_hist
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

def show_tensor_images(image_tensor, num_images=25, size=(1, 64, 64)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.plot(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig('gen_disc_loss.png')

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
              get_swn_conv2d_bn2d_generator_block(z_dim, hidden_dim * 8, 4, 1, 0, True),
              get_swn_conv2d_bn2d_generator_block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, True),
              get_swn_conv2d_bn2d_generator_block(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, True),
              get_swn_conv2d_bn2d_generator_block(hidden_dim * 2, hidden_dim, 4, 2 ,1, True),
              get_swn_conv2d_bn2d_generator_block(hidden_dim, 1, 4, 2, 1, False),
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
    def __init__(self, im_dim=784, hidden_dim=128, nn_type='linear'):
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
          self.gen = nn.Sequential(
              get_swn_conv2d_bn2d_discriminator_block(z_dim, hidden_dim * 8, 4, 1, 0, True),
              get_swn_conv2d_bn2d_discriminator_block(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, True),
              get_swn_conv2d_bn2d_discriminator_block(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, True),
              get_swn_conv2d_bn2d_discriminator_block(hidden_dim * 2, hidden_dim, 4, 2 ,1, True),
              get_swn_conv2d_bn2d_discriminator_block(hidden_dim, 1, 4, 2, 1, False),
              nn.Tanh()
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

def get_noise(n_samples, z_dim, arch_type, device='cuda'):
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
    
def train(arch_type, nn_type, batch_size, n_epochs):
    n_epochs = n_epochs
    arch_type = arch_type
    batch_size = batch_size
    nn_type = nn_type
    z_dim = 100
    display_step = 500
    device = 'cuda'
    if arch_type == '2D':
        criterion = nn.BCELoss()    
        lr = 0.0002
        # Load the MNIST data in Grayscale Format
        img_size = 64
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
        dataloader = DataLoader(
        MNIST('.', download=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True)




    # Initialize The Generator and Discriminator Networks
    gen = Generator(nn_type='convbn2d').to(device)
    disc = Discriminator(nn_type='convbn2d').to(device)
    gen.weight_init(mean=0.0, std=0.02)
    disc.weight_init(mean=0.0, std=0.02)

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
                disc_fake_loss, disc_fake_score, disc_real_loss, y_real_samples = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device, arch_type)
                # Update gradients
                disc_fake_loss.backward() # removed retain graph for memory consumption
                # Update optimizer
                disc_opt.step()
                # For testing purposes, to keep track of the generator weights
                if test_generator:
                    old_generator_weights = gen.gen[0][0].weight.detach().clone()
                ### Update generator ###
                gen_opt.zero_grad()
                gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device, y_real_samples, arch_type)
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
                mean_discriminator_loss += disc_fake_loss.item() / display_step
                D_losses.append(disc_fake_loss)
                gen_loss_values.append(gen_loss)
                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / display_step

                ### Visualization code ###
                if cur_step % display_step == 0 and cur_step > 0:
                    print(f"Epoch: {epoch} Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                    p = 'MNIST_DCGAN_' + str(epoch + 1) + '.png'
                    show_result((epoch+1), gen, path=p, isFix=False)
                    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
                    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))  
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0

            
                cur_step += 1
    else:
        for epoch in range(n_epochs):
            # Dataloader returns the batches
            for real, _ in tqdm(train_loader):
                cur_batch_size = len(real)
                # Flatten the batch of real images from the dataset
                real = real.view(cur_batch_size, -1).to(device)
                # Zero out the gradients before backpropagation
                disc_opt.zero_grad()
                # Calculate discriminator loss
                disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device, arch_type)
                # Update gradients
                disc_loss.backward() # removed retain graph for memory consumption
                # Update optimizer
                disc_opt.step()
                # For testing purposes, to keep track of the generator weights
                if test_generator:
                    old_generator_weights = gen.gen[0][0].weight.detach().clone()
                gen_opt.zero_grad()
                gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device, y_real_samples, arch_type)
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
                    fake_noise = get_noise(cur_batch_size, z_dim, arch_type, device=device)
                    fake = gen(fake_noise)
                    show_tensor_images(fake)
                    show_tensor_images(real)
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
                cur_step += 1
    # Save Model Weights
    show_train_hist(train_hist, save=True, path='MNIST_DCGAN_train_hist.png')
    torch.save(gen.state_dict(), 'gen_model.pth')
    torch.save(disc.state_dict(), 'disc_model.pth')


if __name__ == "__main__":
    with open('train_config.json') as json_file:
        train_config = json.load(json_file)
        print(train_config)
        train(train_config["arch_type"], train_config["nn_type"], train_config["batch_size"], train_config["nn_epochs"])