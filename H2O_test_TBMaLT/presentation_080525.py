import torch
from tbmalt import Geometry, OrbitalInfo
from tbmalt.tools.downloaders import download_dftb_parameter_set
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed
from tbmalt.data.units import length_units

torch.set_default_dtype(torch.float64)

def feeds_scc(device, skf_file):
    species = [1, 8]
    h_feed = SkFeed.from_database(skf_file, species, 'hamiltonian', device=device)
    s_feed = SkFeed.from_database(skf_file, species, 'overlap', device=device)
    o_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)
    u_feed = HubbardFeed.from_database(skf_file, species, device=device)
    r_feed = RepulsiveSplineFeed.from_database(skf_file, species, device=device)

    return h_feed, s_feed, o_feed, u_feed, r_feed

def H2O_scc(device):
    # Ensure device is a torch device
    if isinstance(device, str):
        device = torch.device(device)
    
    cutoff = torch.tensor([9.98], device=device)

    geometry = Geometry(
        torch.tensor([8,1,1], device=device),
        torch.tensor([
            [0.00000000, -0.71603315, -0.00000000],
            [0.00000000, -0.14200298, 0.77844804 ],
            [-0.00000000, -0.14200298, -0.77844804]],
            device=device),units='a')

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0], 8: [0, 1]})

    return geometry, orbs 

if __name__ == "__main__":
    parameter_url = "https://github.com/dftbparams/mio/releases/download/v1.1.0/mio-1-1.tar.xz"
    file_path = "mio.h5"  # Save as HDF5 file
    #print(f"Downloading DFTB parameter set to {file_path}...")
    #download_dftb_parameter_set(parameter_url, file_path)
    #print(f"Downloaded DFTB parameter set to {file_path}")
    device = torch.device('cpu')
    print(f"Main device: {device}")

    h_feed, s_feed, o_feed, u_feed, r_feed = feeds_scc(device,file_path)

    calculator = Dftb2(h_feed, s_feed, o_feed, u_feed, r_feed)

    geometry, orbs = H2O_scc(device)

    print(h_feed)
    print(h_feed._on_sites["1"])
    print(h_feed._on_sites["8"])

    energy = calculator(geometry, orbs)
    print('Energy:', energy)

    total_energy = calculator.total_energy
    print('Total energy:', total_energy)

    q_final_atomic = calculator.q_final_atomic
    print('Q Final Atomic:', q_final_atomic)

    hamiltonian = calculator.hamiltonian
    print('Hamiltonian:', hamiltonian)

    forces = calculator.forces
    print('forces:', forces)

    homo_lumo = calculator.homo_lumo
    print('homo_lumo:', homo_lumo)