# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.3 (2024-02-11)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/flows/realnvp_example.ipynb
# - https://github.com/VincentStimper/normalizing-flows/tree/master

import torch
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm
import os 
class GaussianBase(nn.Module):
    def __init__(self, D):
        """
        Initialize the Gaussian base distribution.

        This class defines an isotropic (diagonal covariance) Gaussian distribution with 
        zero mean and unit variance in D dimensions. It serves as the base distribution
        in normalizing flow models.

        Parameters:
        D: int
            The dimensionality (number of features) of the base distribution.
        """
        super(GaussianBase, self).__init__()
        self.D = D
        
        # Define the mean vector of the Gaussian distribution as a zero vector.
        # The 'requires_grad=False' flag indicates that these parameters should not be updated during training.
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        
        # Define the standard deviation vector as ones indicating unit variance for each dimension.
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        """
        Construct and return the Gaussian base distribution.

        The base distribution is modeled as an Independent Normal distribution.
        Each dimension is treated independently, which is essential for computing 
        joint probabilities over multivariate data in a normalizing flow.

        Returns:
        prior: torch.distributions.Distribution
            An Independent Normal distribution defined by self.mean and self.std.
            The Independent wrapper ensures that the log probability of a batch of data
            corresponds to the sum over individual dimensions, which is often required
            when working with multivariate distributions.
        """
        # Create a Normal distribution with the specified mean and standard deviation.
        # Then wrap it with Independent to treat the last dimension as independent.
        prior = td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)
        return prior

class MaskedCouplingLayer(nn.Module):
    """
    An affine coupling layer for a normalizing flow.

    This layer uses a binary mask to split the input features into two groups.
    One part of the input remains unchanged while the other part is transformed
    using scale and translation networks. The design ensures that the transformation
    is easily invertible, and the Jacobian determinant can be computed efficiently.
    """

    def __init__(self, scale_net, translation_net, mask):
        """
        Initialize the MaskedCouplingLayer.

        Parameters:
        scale_net: torch.nn.Module
            The network that computes scaling factors for the affine transformation.
            It takes a subset of the input features (as determined by the mask) and 
            outputs scaling coefficients.
            
        translation_net: torch.nn.Module
            The network that computes translation factors for the affine transformation.
            It takes the same subset of input features and outputs translation values.
            
        mask: torch.Tensor
            A binary mask of shape (feature_dim,) indicating which features remain
            unchanged (mask value 1) and which features are transformed (mask value 0).
            The mask is registered as a non-trainable parameter.
        """
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net              # Neural network to calculate scaling factors
        self.translation_net = translation_net  # Neural network to calculate translation factors
        self.mask = nn.Parameter(mask, requires_grad=False)  # Fixed binary mask to split features

    def forward(self, z):
        """
        Apply the forward affine transformation to the input latent variable.

        The forward pass uses the mask to separate the input into unmodified and 
        transformed parts. In a complete implementation, the non-masked part would 
        be used to compute the scaling and translation parameters, which are then 
        applied to the masked part. This results in an invertible transformation with 
        a tractable Jacobian determinant.

        Parameters:
        z: torch.Tensor
            Input tensor of shape (batch_size, feature_dim) representing latent variables.

        Returns:
        x: torch.Tensor
            Output tensor after applying the affine transformation.
        log_det_J: torch.Tensor
            Vector of shape (batch_size,) containing the log determinant of the Jacobian
            for each example. Here, it is set to zero as a placeholder.
        """
        # Note: In a full implementation, the following steps would be taken:
        # 1. Split z into two parts using the mask.
        # 2. Pass the unchanged part through scale_net and translation_net to compute
        #    the affine transformation parameters for the other part.
        # 3. Apply the affine transformation to the appropriate features.
        # 4. Compute the log determinant of the scaling factors for the transformed features.

        # split the input 

        x_a = z * self.mask
        x_b = z * (1 - self.mask)

        # compute scale and translation parameters 
        s = self.scale_net(x_a)
        t = self.translation_net(x_a)

        # apply the affine transformation to the transformed part 
        x_b = x_b * torch.exp(s) + t

        # combine the unchanged and transformed parts 
        x = x_a + x_b * (1 - self.mask)

        # compute the log determinant of the Jacobian (sum over transformed dimensions)
        log_det_J = torch.sum(s * (1 - self.mask), dim=1)

        return x, log_det_J
        #x = z  # Placeholder implementation; no transformation is applied.
        #log_det_J = torch.zeros(z.shape[0])  # Placeholder for the log determinant; no scaling applied.
        #return x, log_det_J

    def inverse(self, x):
        """
        Apply the inverse affine transformation to recover the latent variable.

        The inverse operation uses the same mask to identify which parts of the input
        are to be left unchanged and which parts need the inverse affine transformation.
        In a full implementation, this would involve inverting the scaling and translation
        applied during the forward pass and computing the corresponding log Jacobian determinant.

        Parameters:
        x: torch.Tensor
            Input tensor of shape (batch_size, feature_dim) representing data samples.

        Returns:
        z: torch.Tensor
            Output tensor representing the recovered latent variables from the data sample.
        log_det_J: torch.Tensor
            Vector of shape (batch_size,) containing the log determinant of the Jacobian
            for the inverse transformation. Here, it is set to zero as a placeholder.
        """
        # Note: In a complete implementation, the inverse pass would:
        # 1. Use the mask to separate the features.
        # 2. Compute the inverse transformation parameters using scale_net and translation_net.
        # 3. Reverse the affine transformation for the relevant features.
        # 4. Compute the corresponding log determinant of the inverse transformation.
        
        #z = x  # Placeholder implementation; no inverse transformation is applied.
        #log_det_J = torch.zeros(x.shape[0])  # Placeholder for the log determinant; no scaling applied.
        #return z, log_det_J

        # split the output 
        z_a = x * self.mask
        z_b = x * (1 - self.mask)

        # compute scale and translation parameters from the unchanged part
        s = self.scale_net(z_a)
        t = self.translation_net(z_a)

        # invert the asfine transformations 
        x_b = (z_b - t) * torch.exp(-s)

        # combine to form the inverse transformation 
        x = z_a + x_b * (1 - self.mask)

        # compute the log determinant of the Jacobian (sum over transformed dimensions)
        log_det_J = -torch.sum(s * (1 - self.mask), dim=1)
        return x, log_det_J


class Flow(nn.Module):
    def __init__(self, base, transformations):
        """
        Define a normalizing flow model.
        
        Parameters:
        base: [torch.distributions.Distribution]
            The base distribution.
        transformations: [list of torch.nn.Module]
            A list of transformations to apply to the base distribution.
        """
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        """
        Transform a batch of data through the flow (from the base to data).
        
        Parameters:
        x: [torch.Tensor]
            The input to the transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the forward transformations.            
        """
        sum_log_det_J = 0
        for T in self.transformations:
            x, log_det_J = T(z)
            sum_log_det_J += log_det_J
            z = x
        return x, sum_log_det_J
    
    def inverse(self, x):
        """
        Transform a batch of data through the flow (from data to the base).

        Parameters:
        x: [torch.Tensor]
            The input to the inverse transformation of dimension `(batch_size, feature_dim)`
        Returns:
        z: [torch.Tensor]
            The output of the inverse transformation of dimension `(batch_size, feature_dim)`
        sum_log_det_J: [torch.Tensor]
            The sum of the log determinants of the Jacobian matrices of the inverse transformations.
        """
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            x = z
        return z, sum_log_det_J
    
    def log_prob(self, x):
        """
        Compute the log probability of a batch of data under the flow.

        Parameters:
        x: [torch.Tensor]
            The data of dimension `(batch_size, feature_dim)`
        Returns:
        log_prob: [torch.Tensor]
            The log probability of the data under the flow.
        """
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J
    
    def sample(self, sample_shape=(1,)):
        """
        Sample from the flow.

        Parameters:
        n_samples: [int]
            Number of samples to generate.
        Returns:
        z: [torch.Tensor]
            The samples of dimension `(n_samples, feature_dim)`
        """
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]
    
    def loss(self, x):
        """
        Compute the negative mean log likelihood for the given data bath.

        Parameters:
        x: [torch.Tensor] 
            A tensor of dimension `(batch_size, feature_dim)`
        Returns:
        loss: [torch.Tensor]
            The negative mean log likelihood for the given data batch.
        """
        return -torch.mean(self.log_prob(x))


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The Flow model to train.
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

    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")
    losses = []
    for epoch in range(epochs):
        epoch_losses  = []
        data_iter = iter(data_loader)
        for batch in data_iter:
            x = batch[0].to(device)  # Extract the images from (images, labels)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()


            epoch_losses.append(loss.item())
            # Update progress bar
            progress_bar.set_postfix(loss=f"{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        progress_bar.set_postfix(loss=f"{avg_loss:12.4f}", epoch=f"{epoch+1}/{epochs}")

    return losses


def create_random_mask(D):
    """Create a random binary mask of size D"""
    return torch.randint(0, 2, (D,)).float()

def create_checkerboard_mask(height, width):
    """Create a checkerboard mask for a flattened image of size height x width"""
    mask = torch.zeros(height, width)
    for i in range(height):
        for j in range(width):
            mask[i, j] = (i + j) % 2
    return mask.view(-1)  # Flatten to 1D

def plot_samples(model, device, num_samples=25, nrow=5):
    """Generate and plot samples from the model"""
    model.eval()
    with torch.no_grad():
        samples = model.sample((num_samples,)).to('cpu')
    
    # Reshape samples to images
    samples = samples.view(-1, 28, 28)
    
    # Create plot
    fig, axes = plt.subplots(nrow, nrow, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        if i < num_samples:
            ax.imshow(samples[i], cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData

    

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb'], help='toy dataset to use {tg: two Gaussians, cb: chequerboard} (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--num-layers', type=int, default=5, metavar='N', help='number of coupling layers in the flow (default: %(default)s)')
    parser.add_argument('--hidden-dim', type=int, default=256, metavar='N', help='dimension of hidden layers in the coupling layers (default: %(default)s)')


    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)


    device = torch.device(args.device)
    ############################################################################
    # Data: Dequantized MNIST
    ############################################################################
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),                            # shape (1,28,28), range [0,1]
        transforms.Lambda(lambda x: x + torch.rand_like(x)/256.0),  # dequantize
        transforms.Lambda(lambda x: x.clamp(0.0, 1.0)),   # ensure still in [0,1]
        transforms.Lambda(lambda x: x.view(-1))           # flatten to (784,)
    ])
    train_dataset = datasets.MNIST(root='data', train=True, download=True,
                                   transform=transform_mnist)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    D = 784  # flattened MNIST

    ############################################################################
    # Build the Flow Model
    ############################################################################
    # 1) Base distribution
    base = GaussianBase(D)

    # 2) Checkerboard mask for 28x28
    #    We'll alternate it across layers to transform all pixels eventually.
    mask2d = torch.zeros((28,28))
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask2d[i, j] = 1.0
    mask_checkerboard = mask2d.view(-1)  # shape (784,)

    # 3) Create coupling layers
    transformations = []
    for i in range(args.num_layers):
        # Flip checkerboard every other layer
        if i % 2 == 0:
            current_mask = mask_checkerboard
        else:
            current_mask = 1.0 - mask_checkerboard

        # Scale net with tanh at the end
        scale_net = nn.Sequential(
            nn.Linear(D, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, D),
            nn.Tanh()  # ensures log-scale is in (-1,1)
        )
        # Translation net (no tanh needed)
        translation_net = nn.Sequential(
            nn.Linear(D, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, D)
        )

        transformations.append(MaskedCouplingLayer(scale_net, translation_net, current_mask))

    # 4) Flow model
    model = Flow(base, transformations).to(device)

    ############################################################################
    # Run: Train or Sample
    ############################################################################
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(model, optimizer, train_loader, args.epochs, device)

        print(f"Saving model to {args.model} ...")
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        # Load the trained model
        if not os.path.isfile(args.model):
            raise FileNotFoundError(f"No model file found at {args.model}")

        print(f"Loading model from {args.model} ...")
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

        # Generate samples
        print("Generating samples from the flow ...")
        with torch.no_grad():
            samples = model.sample((64,)).cpu()  # 64 samples

        # Reshape to (64, 1, 28, 28) for saving as image grid
        samples = samples.clamp(0,1).view(-1, 1, 28, 28)
        print(f"Saving samples to {args.samples} ...")
        save_image(samples, args.samples, nrow=8)
        print("Done.")


#    # Generate the data
#    n_data = 10000000
#    toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
#    train_loader = torch.utils.data.DataLoader(toy().sample((n_data,)), batch_size=args.batch_size, shuffle=True)
#    test_loader = torch.utils.data.DataLoader(toy().sample((n_data,)), batch_size=args.batch_size, shuffle=True)
#
#    # Define prior distribution
#    D = next(iter(train_loader)).shape[1]
#    base = GaussianBase(D)
#
#    # Define transformations
#    transformations =[]
#    mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(28) for j in range(28)])
#    
#    num_transformations = 5
#    num_hidden = 8
#
#    # Make a mask that is 1 for the first half of the features and 0 for the second half
#    mask = torch.zeros((D,))
#    mask[D//2:] = 1
#    
#    for i in range(num_transformations):
#        mask = (1-mask) # Flip the mask
#        scale_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
#        translation_net = nn.Sequential(nn.Linear(D, num_hidden), nn.ReLU(), nn.Linear(num_hidden, D))
#        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))
#
#    # Define flow model
#    model = Flow(base, transformations).to(args.device)
#
#    # Choose mode to run
#    if args.mode == 'train':
#        # Define optimizer
#        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#
#        # Train model
#        train(model, optimizer, train_loader, args.epochs, args.device)
#
#        # Save model
#        torch.save(model.state_dict(), args.model)
#
#    elif args.mode == 'sample':
#        import matplotlib.pyplot as plt
#        import numpy as np
#
#        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
#
#        # Generate samples
#        model.eval()
#        with torch.no_grad():
#            samples = (model.sample((10000,))).cpu() 
#
#        # Plot the density of the toy data and the model samples
#        coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
#        prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))
#
#        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
#        im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
#        ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
#        ax.set_xlim(toy.xlim)
#        ax.set_ylim(toy.ylim)
#        ax.set_aspect('equal')
#        fig.colorbar(im)
#        plt.savefig(args.samples)
#        plt.close()