import torch
from tbmalt import Geometry, OrbitalInfo
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed
from tbmalt.common.maths.interpolation import CubicSpline
from tbmalt.ml.loss_function import Loss, mse_loss
import torch.nn as nn

# This must be set until typecasting from HDF5 databases has been implemented.
torch.set_default_dtype(torch.float64)

# Reference of target properties
targets = {'total_energy': torch.tensor([-4.0779]),
           'q_final_atomic': torch.tensor([6.5926, 0.7037, 0.7037])}

# Provide information about the orbitals on each atom
shell_dict = {1: [0], 6: [0, 1], 7: [0, 1], 8: [0, 1]}

# Location at which the DFTB parameter set database is located
parameter_db_path = 'mio.h5'

# Training hyperparameters
number_of_epochs = 250
lr = 0.002

device = torch.device('cpu')

# Construct the Geometry and OrbitalInfo objects
geometry = Geometry(
        torch.tensor([8,1,1], device=device),
        torch.tensor([
            [0.00000000, -0.71603315, -0.00000000],
            [0.00000000, -0.14200298, 0.77844804 ],
            [-0.00000000, -0.14200298, -0.77844804]],
            device=device), units='a')

orbs = OrbitalInfo(geometry.atomic_numbers, shell_dict)

print('Geometry:', geometry)
print('OrbitalInfo:', orbs)

# Identify which species are present
species = torch.unique(geometry.atomic_numbers)
species = species[species != 0].tolist()

# Load all the necessary feed models
h_feed = SkFeed.from_database(parameter_db_path, species, 'hamiltonian',
                              interpolation=CubicSpline, requires_grad_onsite=True, requires_grad_offsite=True)
s_feed = SkFeed.from_database(parameter_db_path, species, 'overlap',
                              interpolation=CubicSpline)
o_feed = SkfOccupationFeed.from_database(parameter_db_path, species)
u_feed = HubbardFeed.from_database(parameter_db_path, species)
r_feed = RepulsiveSplineFeed.from_database(parameter_db_path, species)

original_h = h_feed._on_sites["1"].clone()
original_o = h_feed._on_sites["8"].clone()

torch.manual_seed(4)  # Set random seed for reproducibility

# Create parameters directly in the on_sites dictionaries
# For hydrogen (Z=1)
h_onsite = h_feed._on_sites["1"].clone()
h_onsite[0] = -torch.rand(1, dtype=torch.double, requires_grad=True)[0]  # x parameter
#h_onsite[1:4] = -torch.rand(1, dtype=torch.double, requires_grad=True)[0].repeat(3)  # y parameter
h_feed._on_sites["1"] = h_onsite

# For oxygen (Z=8)
o_onsite = h_feed._on_sites["8"].clone()
o_onsite[0] = -torch.rand(1, dtype=torch.double, requires_grad=True)[0]  # a parameter
o_onsite[1:4] = -torch.rand(1, dtype=torch.double, requires_grad=True)[0].repeat(3)  # b parameters
h_feed._on_sites["8"] = o_onsite

# Get the parameter references for optimization
params = []
for key in h_feed._on_sites:
    h_feed._on_sites[key].requires_grad_(True)
    params.append(h_feed._on_sites[key])

# Create the DFTB calculator
dftb_calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, r_feed, filling_scheme=None)

# Define optimizer with the parameters
optimizer = torch.optim.Adam(params, lr=lr)

# Training loop
loss_list = []
for epoch in range(number_of_epochs):
    # Zero gradients
    optimizer.zero_grad()
    
    # Run DFTB calculation
    dftb_calculator(geometry, orbs, grad_mode="direct")
    
    # Calculate loss
    energy_loss = mse_loss(dftb_calculator.total_energy, targets['total_energy'])
    mulliken_loss = mse_loss(dftb_calculator.q_final_atomic, targets['q_final_atomic'])
    loss = energy_loss + mulliken_loss
    
    # Backward pass
    loss.backward(retain_graph=True)
    
    # Update parameters
    optimizer.step()
    
    # Record loss
    loss_list.append(loss.item())
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

# Print final parameters

print("Original H on-site parameters:", original_h)
print("Original O on-site parameters:", original_o)

print(f"Final H on-site parameters: {h_feed._on_sites['1']}")
print(f"Final O on-site parameters: {h_feed._on_sites['8']}")

# Plot the loss
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.linewidth"] = 1.5
plt.tick_params(direction='in', labelsize='26', width=1.5, length=5, top='on',
                right='on', zorder=10)
plt.plot(range(number_of_epochs), loss_list)
plt.xlabel("Iteration", fontsize=28)
plt.ylabel("Loss", fontsize=28)
plt.show()