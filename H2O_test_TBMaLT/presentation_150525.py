import torch
from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed
from tbmalt.common.maths.interpolation import CubicSpline
from tbmalt.ml.loss_function import Loss, mse_loss

from ase.build import molecule

Tensor = torch.Tensor

# This must be set until typecasting from HDF5 databases has been implemented.
torch.set_default_dtype(torch.float64)

# Set device explicitly
device = torch.device('cpu')

molecule = 'H2O'
target = {'total_energy': torch.tensor([-4.0779], device=device)}
shell_dict = {1: [0], 8: [0, 1]}

parameter_db_path = 'mio.h5'

# Type of ML model
model = 'spline'

# Whether performing model fitting
fit_model = True

# Number of training cycles
number_of_epochs = 50

# Learning rate
lr = 0.002

# Loss function
loss_func = mse_loss

# Create geometry with device specification
geometry = Geometry(
        torch.tensor([8, 1, 1], device=device),
        torch.tensor([
            [0.00000000, -0.71603315, -0.00000000],
            [0.00000000, -0.14200298, 0.77844804],
            [-0.00000000, -0.14200298, -0.77844804]],
            device=device), units='a')

orbs = OrbitalInfo(geometry.atomic_numbers, shell_dict)

# Identify which species are present
species = torch.unique(geometry.atomic_numbers)
# Strip out padding species and convert to a standard list.
species = species[species != 0].tolist()

# Create all feeds with explicit device specification
h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian', device=device)
s_feed = SkFeed.from_database(parameter_db_path, species, 'overlap', device=device)
o_feed = SkfOccupationFeed.from_database(parameter_db_path, species, device=device)
u_feed = HubbardFeed.from_database(parameter_db_path, species, device=device)
r_feed = RepulsiveSplineFeed.from_database(parameter_db_path, species, device=device)

# Verify all feeds are on the same device
for feed in [h_feed, s_feed, o_feed, u_feed, r_feed]:
    print(f"Feed device: {feed.device}")

# Create the DFTB calculator
dftb_calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, r_feed)

# Create a loss entity
loss_entity = Loss(loss_func, target.keys())

# Training loop with error handling
for epoch in range(number_of_epochs):
    try:
        print(f'epoch {epoch}')
        
        # Run DFTB calculation
        results = dftb_calculator(geometry, orbs, grad_mode="direct")
        
        # Calculate loss
        total_loss, raw_losses = loss_entity(dftb_calculator, target)
        print(f'loss: {total_loss}')
        
        # If we're fitting the model, perform backpropagation and optimization
        if fit_model:
            # Setup optimizer if it's the first epoch
            if epoch == 0:
                params = []
                for feed in [h_feed, s_feed, u_feed, r_feed]:
                    if hasattr(feed, 'parameters'):
                        params.extend(list(feed.parameters()))
                optimizer = torch.optim.Adam(params, lr=lr)
            
            # Zero gradients, perform backward pass, and update parameters
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
    except Exception as e:
        print(f"Error in epoch {epoch}: {str(e)}")
        # Add eigenvalue conditioning for numerical stability
        if "linalg.eigh" in str(e):
            print("Adding regularization to improve matrix conditioning...")
            # You might need to adjust the regularization approach based on your specific case
            break