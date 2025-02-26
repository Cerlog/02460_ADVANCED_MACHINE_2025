# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)
#
# This script implements a Denoising Diffusion Probabilistic Model (DDPM) using PyTorch.
# The code defines a diffusion model, a fully connected network to parametrize the diffusion process,
# training and sampling procedures, and also provides a command-line interface for adjusting parameters.

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm  # For displaying progress bars during training

class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize the DDPM model.
        
        Parameters:
        network: nn.Module
            The neural network used to predict noise (or parameters).
        beta_1: float
            The noise level at the first diffusion step.
        beta_T: float
            The noise level at the final diffusion step.
        T: int
            The total number of diffusion steps.
        """
        super(DDPM, self).__init__()
        self.network = network  # The neural network provided by the user
        self.beta_1 = beta_1  # Noise of the first time step
        self.beta_T = beta_T  # Noise of the last time step
        self.T = T  # Total number of diffusion steps

        # Create a beta schedule linearly spaced between beta_1 and beta_T for T steps.
        # No gradient is computed for these parameters hence requires_grad=False.
        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        # Compute alpha, representing the probability of retaining the signal at each step:
        #   α_t = 1 − β_t
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        
        # Compute the cumulative product of alphas up to time t:
        #   ᾱ_t = ∏₍ᵢ₌₁₎ᵗ α_i
        # This represents the overall retained signal over t steps.
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Compute the negative Evidence Lower Bound (ELBO) for a given batch of data.
        
        Parameters:
        x: torch.Tensor
            A batch of input data of shape (batch_size, *).
        
        Returns:
        torch.Tensor
            The computed negative ELBO for the batch with shape (batch_size,).
        
        Note:
        This function is not yet implemented and should replicate Algorithm 1 from the reference paper.
        """
        # Initialize the negative ELBO. In practice, one would accumulate terms according to the algorithm.

        batch_size = x.size(0)

        t = torch.randint(self.T, (batch_size, 1), device=x.device)  # Sample a random time step for each batch element.
        # Create time tensor for network input (needs to match batch dimensions)
        t_tensor = t.unsqueeze(-1).float() / self.T

        # generate noise epsilon from a standard normal 
        epsilon = torch.randn_like(x)
        # 
        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1)

        # crate the noisty samples x_t using forward pocess

        # Create noisy samples x_t using the forward process
        x_t = torch.sqrt(alpha_cumprod_t) * x + torch.sqrt(1 - alpha_cumprod_t) * epsilon

        # Compute the predicted noise for the given input and time step.
        epsilon_theta = self.network(x_t, t_tensor)

        # calculate the negative elbo between the true and predicted noise 
        neg_elbo = F.mse(epsilon, epsilon_theta, reduction='none').sum(dim=1)

        return neg_elbo

    def sample(self, shape):
        """
        Generate samples from the diffusion model.
        
        Parameters:
        shape: tuple
            The desired shape of the generated samples.
        
        Returns:
        torch.Tensor
            The final sample generated after reversing the diffusion process (x_0).
        
        Note:
        This function is not completely implemented. It should follow the reverse process outlined in Algorithm 2.
        """
        # Start from Gaussian noise corresponding to the final diffusion step.
        x_t = torch.randn(shape).to(self.alpha.device)

        # Iteratively sample the previous step given the current step, working backwards from T-1 to 0.
        for t in range(self.T-1, -1, -1):
            # TODO: Implement the reverse diffusion step here following the algorithm.

            # create timestep tensor for the entire batch 
            t_tensor = torch.ones(shape[0], 1, device= device) * t / self.T

            # predict noise using the network 
            epsilon_theta = self.network(x_t, t_tensor)

            # get the corresponding alpha, apha_cumprod for timestep t 
            alpha_t = self.alpha[t]
            alpha_cumprod_t = self.alpha_cumprod[t]

            # Get the corresponding beta for timestep t 
            beta_t = self.beta[t]

            # No noise for the last step (t=0)
            z = torch.zeros_like(x_t) if t == 0 else torch.randn_like(x_t)
            # Formula for x_{t-1} given x_t and predicted noise
            # Implementing equation (11) from the paper
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)
            
            # For numerical stability, compute in steps:
            # 1. x_0_hat is our prediction of the original data
            x_0_hat = (x_t - coef2 * epsilon_theta) / torch.sqrt(alpha_cumprod_t)
            
            # 2. Use x_0_hat to compute mean for the reverse step
            mean = coef1 * (x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * epsilon_theta)
            
            # 3. Add scaled noise (σₜ²=βₜ as specified in the exercise)
            sigma_t = torch.sqrt(beta_t)
            x_t = mean + sigma_t * z
        
        return x_t

    def loss(self, x):
        """
        Compute the overall loss for the DDPM model on a batch.
        
        Parameters:
        x: torch.Tensor
            Input data of shape (batch_size, *).
        
        Returns:
        torch.Tensor
            The mean negative ELBO loss over the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train the diffusion model using data from a DataLoader.
    
    Parameters:
    model: nn.Module
       The diffusion model (DDPM) to be trained.
    optimizer: torch.optim.Optimizer
         The optimizer used to update the model parameters.
    data_loader: torch.utils.data.DataLoader
            DataLoader for accessing training data.
    epochs: int
        Number of complete passes over the training dataset.
    device: torch.device
        The device (CPU, CUDA, etc.) on which the training occurs.
    """
    model.train()  # Set the model to training mode

    # Calculate total number of training steps (epochs * batches per epoch)
    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")  # Initialize a progress bar

    # Loop over each epoch
    for epoch in range(epochs):
        data_iter = iter(data_loader)
        # Loop over each batch in the DataLoader
        for x in data_iter:
            # If batch is a tuple or list (e.g., data and labels), take only the data.
            if isinstance(x, (list, tuple)):
                x = x[0]
            # Move data to the specified device (CPU, GPU)
            x = x.to(device)
            optimizer.zero_grad()  # Reset gradients for the optimizer
            
            # Compute the loss (negative ELBO)
            loss = model.loss(x)
            # Perform backpropagation
            loss.backward()
            # Update the model parameters
            optimizer.step()

            # Update the progress bar with current loss and epoch information.
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Create a fully connected (dense) network for use within the DDPM.
        The network is designed to accept both the input data and the time step.
        
        Parameters:
        input_dim: int
            The dimension of the input data.
        num_hidden: int
            The number of hidden units to use within the network layers.
        """
        super(FcNetwork, self).__init__()
        # The network concatenates the data features and a time step value.
        # Three linear layers with ReLU activations are used for the forward pass.
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, num_hidden),  # Input layer (+1 for time step input)
            nn.ReLU(),                           # Activation function
            nn.Linear(num_hidden, num_hidden),   # Hidden layer
            nn.ReLU(),                           # Activation function
            nn.Linear(num_hidden, input_dim)      # Output layer returning original input dimensions
        )

    def forward(self, x, t):
        """
        Forward pass of the network.
        
        Parameters:
        x: torch.Tensor
            Input data of shape (batch_size, input_dim).
        t: torch.Tensor
            Time steps as a tensor of shape (batch_size, 1) to be concatenated with x.
        
        Returns:
        torch.Tensor
            The output produced by the network, typically used to predict noise.
        """
        # Concatenate the input data with the time step along the feature dimension.
        x_t_cat = torch.cat([x, t], dim=1)
        # Forward the concatenated tensor through the network.
        return self.network(x_t_cat)


if __name__ == "__main__":
    import torch.utils.data  # For handling data loading
    from torchvision import datasets, transforms  # For standard datasets and image transformations
    from torchvision.utils import save_image  # For saving generated images
    import ToyData  # Custom toy dataset module

    # Parse command-line arguments for various options including mode, dataset, and training parameters.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], 
                        help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], 
                        help='dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', 
                        help='file path to save model parameters or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', 
                        help='file name to save generated samples (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], 
                        help='device to use for computations (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', 
                        help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', 
                        help='number of training epochs (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', 
                        help='learning rate for the optimizer (default: %(default)s)')

    args = parser.parse_args()

    # Display parsed options for user verification.
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Generate the sample data for training and testing.
    # n_data defines the total number of data points.
    n_data = 10000000
    # Select the toy dataset based on the chosen data type ('tg' for two Gaussians, 'cb' for chequerboard).
    toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
    # Define a transformation to shift the data scale from [0,1] to [-1,1].
    transform = lambda x: (x - 0.5) * 2.0
    # Create DataLoaders for training and testing splits.
    train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), 
                                              batch_size=args.batch_size, shuffle=True)

    # Determine the data dimension by inspecting the first batch from the train loader.
    D = next(iter(train_loader)).shape[1]

    # Define the network configuration.
    num_hidden = 64
    network = FcNetwork(D, num_hidden)

    # Set the number of diffusion steps in the model.
    T = 1000

    # Instantiate the DDPM model and move it to the chosen device.
    model = DDPM(network, T=T).to(args.device)

    # Execute different operations based on the chosen mode.
    if args.mode == 'train':
        # Create an optimizer to update model parameters.
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Train the model using the provided data_loader.
        train(model, optimizer, train_loader, args.epochs, args.device)

        # Save the trained model parameters to the specified file.
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        import matplotlib.pyplot as plt
        import numpy as np

        # Load the trained model parameters from file.
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Set model to evaluation mode (disable dropout, batch norm, etc.)
        model.eval()
        with torch.no_grad():
            # Generate samples from the model; these samples are still in the transformed data space.
            samples = model.sample((10000, D)).cpu()

        # Reverse the earlier data transformation to map values back to the original scale.
        samples = samples / 2 + 0.5

        # Generate a grid of coordinates to evaluate the toy dataset's probability density.
        coordinates = [[[x, y] for x in np.linspace(*toy.xlim, 1000)] 
                       for y in np.linspace(*toy.ylim, 1000)]
        # Compute the probability of the toy data over the grid using the toy dataset's log_prob function.
        prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

        # Create a plot displaying the probability density and the generated samples.
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], 
                       origin='lower', cmap='YlOrRd')
        # Overlay the generated samples on the density plot.
        ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
        ax.set_xlim(toy.xlim)
        ax.set_ylim(toy.ylim)
        ax.set_aspect('equal')
        fig.colorbar(im)  # Show a colorbar for the density
        # Save the resulting plot to a file.
        plt.savefig(args.samples)
        plt.close()
