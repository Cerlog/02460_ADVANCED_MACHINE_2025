# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm

from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import glob
import hydra
from omegaconf import DictConfig, OmegaConf

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Initializes the GaussianPrior module.

        This module creates a fixed Gaussian (normal) distribution to serve as the prior
        over the latent space in a Variational Autoencoder (VAE). Each latent variable
        is modeled as an independent Gaussian with zero mean and unit variance.

        Parameters:
        M (int): Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        # Save the latent space dimension.
        self.M = M
        
        # Create a tensor for the means of the latent variables.
        # Since the prior is fixed to be zero-mean, this tensor is initialized to zeros.
        # It is wrapped in nn.Parameter for proper module registration, but gradients are disabled.
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        
        # Create a tensor for the standard deviations of the latent variables.
        # The standard deviation is set to one for a unit variance Gaussian.
        # Like the mean, it is registered as a parameter with gradients disabled.
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Constructs and returns the prior distribution.

        It returns an independent multivariate normal distribution where each dimension is
        assumed to be independent and follows a Normal distribution with the stored mean and std.

        Returns:
        torch.distributions.Distribution: 
            An independent Gaussian distribution representing the prior over the latent space.
        """
        # Create a Normal distribution for each latent dimension.
        # The 'td.Independent' wrapper indicates that the dimensions are independent,
        # combining them into a multivariate distribution.
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class MoGPrior(nn.Module):
    def __init__(self, M, K):
        """
        M: dimension of the latent space
        K: number of mixture components
        """
        super().__init__()
        
        # Mixture weights (initialized to be uniform)
        self.logits = nn.Parameter(torch.zeros(K))  
        
        # Means and standard deviations for each component
        self.means = nn.Parameter(torch.randn(K, M))
        self.log_stds = nn.Parameter(torch.zeros(K, M))

    def forward(self):
        # Convert logits to probabilities
        cat = td.Categorical(logits=self.logits)
        
        # Create a Normal distribution for each mixture component
        comps = td.Independent(
            td.Normal(self.means, torch.exp(self.log_stds)),
            reinterpreted_batch_ndims=1
        )
        
        # Combine them into a mixture
        return td.MixtureSameFamily(
            mixture_distribution=cat,
            component_distribution=comps
        )


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        if isinstance(self.prior, MoGPrior):
            kl_divergence_ = q.log_prob(z) - self.prior().log_prob(z)
        else:
            kl_divergence_ = td.kl_divergence(q, self.prior())
        #print("KL: ", kl_divergence_.shape)
        #########################################
        elbo = torch.mean(self.decoder(z).log_prob(x) - kl_divergence_, dim=0)
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)

import io
from PIL import Image
import torchvision.transforms as transforms


def visualize_samples_from_prior(model, data_loader, device, M, file_name, n_samples=100):
    model.eval()
    latent_samples = []
    labels = []

    with torch.no_grad():
        # Sample from the prior and generate outputs
        for _ in tqdm(range(n_samples), desc="Sampling from prior"):
            # Sample z from the prior
            z = model.prior().sample((1,))  # Single sample from the prior
            
            # Decode the sampled latent variable z
            x_reconstructed = model.decoder(z).mean  # Get the mean of p(x|z)
            
            latent_samples.append(x_reconstructed.cpu().numpy())
        
        # Convert list of samples to an array
        latent_samples = np.concatenate(latent_samples, axis=0)
        
        # Flatten the data for PCA (ensure the shape is 2D)
        latent_samples = latent_samples.reshape(latent_samples.shape[0], -1)  # Flatten to 2D

    # Now apply PCA
    print("Applying PCA to reduce dimensions to 2D...")
    pca = PCA(n_components=2)
    latent_samples_2d = pca.fit_transform(latent_samples)

    plt.figure(figsize=(8, 6))
    plt.scatter(latent_samples_2d[:, 0], latent_samples_2d[:, 1], alpha=0.7)
    plt.title("Latent Space Visualization with PCA")
    plt.savefig(file_name)
    plt.show()



def visualize_latent_space(model, data_loader, device, M, file_name, test_elbo):
    model.eval()
    latent_samples = []
    labels = []

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Sampling latent space"):
            x = x.to(device)
            q = model.encoder(x)  # Get posterior q(z|x)
            z = q.rsample()  # Sample from q(z|x)
            latent_samples.append(z.cpu().numpy())
            labels.append(y.cpu().numpy())

    latent_samples = np.concatenate(latent_samples, axis=0)
    labels = np.concatenate(labels, axis=0)

    if M > 2:
        print("Applying PCA to reduce dimensions to 2D...")
        pca = PCA(n_components=2)
        latent_samples = pca.fit_transform(latent_samples)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_samples[:, 0], latent_samples[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label="Digit Class")
    plt.xlabel("Latent Dimension 1" if M == 2 else "PCA Component 1")
    plt.ylabel("Latent Dimension 2" if M == 2 else "PCA Component 2")
    plt.title("Latent Space Visualization with ELBO: {:.4f}".format(test_elbo))
    plt.savefig(file_name)
    plt.show()




def evaluate_elbo(model, data_loader, device):
    model.eval()
    total_elbo = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            # Assuming your dataloader returns a tuple where the first element is the image tensor
            x = batch[0].to(device)
            # Compute the ELBO for the batch
            elbo = model.elbo(x)
            # Multiply by batch size to weight the average correctly
            total_elbo += elbo.item() * x.size(0)
            total_samples += x.size(0)
    return total_elbo / total_samples

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


#if __name__ == "__main__":
#    from torchvision import datasets, transforms
#    from torchvision.utils import save_image, make_grid
#    import glob
#
#    # Parse arguments
#    import argparse
#    parser = argparse.ArgumentParser()
#    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
#    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
#    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
#    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
#    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
#    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
#    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
#
#    args = parser.parse_args()
#    print('# Options')
#    for key, value in sorted(vars(args).items()):
#        print(key, '=', value)
#
#    device = args.device
#
#    # Load MNIST as binarized at 'thresshold' and create data loaders
#    thresshold = 0.5
#    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
#                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
#                                                    batch_size=args.batch_size, shuffle=True)
#    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
#                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
#                                                    batch_size=args.batch_size, shuffle=True)
#
#    # Define prior distribution
#    M = args.latent_dim
#    prior = GaussianPrior(M)
#
#    # Define encoder and decoder networks
#    encoder_net = nn.Sequential(
#        nn.Flatten(),
#        nn.Linear(784, 512),
#        nn.ReLU(),
#        nn.Linear(512, 512),
#        nn.ReLU(),
#        nn.Linear(512, M*2),
#    )
#
#    decoder_net = nn.Sequential(
#        nn.Linear(M, 512),
#        nn.ReLU(),
#        nn.Linear(512, 512),
#        nn.ReLU(),
#        nn.Linear(512, 784),
#        nn.Unflatten(-1, (28, 28))
#    )
#
#    # Define VAE model
#    decoder = BernoulliDecoder(decoder_net)
#    encoder = GaussianEncoder(encoder_net)
#    model = VAE(prior, decoder, encoder).to(device)
#
#    # Choose mode to run
#    if args.mode == 'train':
#        # Define optimizer
#        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#
#        # Train model
#        train(model, optimizer, mnist_train_loader, args.epochs, args.device)
#
#        # Save model
#        torch.save(model.state_dict(), args.model)
#
#
#        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
#        model.to(device)  # Ensure the model is on the correct device
#        model.eval()      # Set the model to evaluation mode
#
#        train_elbo = evaluate_elbo(model, mnist_train_loader, args.device)
#        test_elbo = evaluate_elbo(model, mnist_test_loader, args.device)
#
#        print(f"ELBO on training set: {train_elbo:.4f}")
#        print(f"ELBO on test set: {test_elbo:.4f}")
#
#    # Optionally, you can also add latent space visualization if you wish:
#        visualize_latent_space(model, mnist_test_loader, args.device, args.latent_dim)
#
#    elif args.mode == 'sample':
#        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
#
#        # Generate samples
#        model.eval()
#        with torch.no_grad():
#            samples = (model.sample(64)).cpu() 
#            save_image(samples.view(64, 1, 28, 28), args.samples)
#

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    #print(OmegaConf.to_yaml(cfg))
    
    device = torch.device(cfg.device)
    data_dir = hydra.utils.to_absolute_path(cfg.data_dir)
    
    # Data transformations and loaders 
    threshold = 0.5
    # Binirize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (threshold < x).float().squeeze())
    ])


    # pixel values in mnist 
    transform_2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.squeeze())])
    
    mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=False, transform=transform),
        batch_size=cfg.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, download=False, transform=transform),
        batch_size=cfg.batch_size, shuffle=False
    )
    
    # Define the prior
    M = cfg.M
    prior = MoGPrior(cfg.M, cfg.K)
    
    # Define encoder network and Gaussian encoder
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2)
    )
    encoder = GaussianEncoder(encoder_net)
    
    # Define decoder network and Bernoulli decoder
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )
    decoder = BernoulliDecoder(decoder_net)
    
    # Create the VAE model and move to device
    model = VAE(prior, decoder, encoder).to(device)
    
    # Mode handling
    if cfg.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        train(model, optimizer, mnist_train_loader, cfg.epochs, device)
        torch.save(model.state_dict(), cfg.model)
        model.load_state_dict(torch.load(cfg.model, map_location=device))
        model.to(device)
        model.eval()

        train_elbo = evaluate_elbo(model, mnist_train_loader, device)
        test_elbo = evaluate_elbo(model, mnist_test_loader, device)
        print(f"ELBO on training set: {train_elbo:.4f}")
        print(f"ELBO on test set: {test_elbo:.4f}")

        # Save the latent space visualization along with metrics
        visualize_latent_space(model, mnist_test_loader, device, cfg.latent_dim, cfg.latent_space_file, test_elbo)
    
    elif cfg.mode == 'sample':
        model.load_state_dict(torch.load(cfg.model, map_location=device))
        model.to(device)
        model.eval()
        #with torch.no_grad():
        #    samples = model.sample(64).cpu()
        #    save_image(samples.view(64, 1, 28, 28), cfg.samples)
        

        # Sample from the prior and visualize THE MEAN
        with torch.no_grad():
            # Sample from the prior (standard normal distribution or your custom prior)
            z = model.prior().sample((64,))  # Sample 64 latent variables from the prior
    
            # Decode the sampled latent variables (this gives you the mean of p(x|z))
            x_reconstructed = model.decoder(z).mean  # Get the mean of the output distribution p(x|z)
    
            # Save and visualize the samples
            save_image(x_reconstructed.view(64, 1, 28, 28), cfg.samples)

if __name__ == "__main__":
    main()