# SurfCat

A Python library for analyzing surface science and electrocatalysis molecular dynamics simulations.

## Features

- **Species Identification**: Automatic identification of OH⁻, H₂O, H₃O⁺ based on bonding topology
- **Spatial Analysis**: Define interface/surface regions for localized analysis  
- **Density Profiles**: Calculate 1D distributions along any dimension
- **Dynamic Properties**: MSD, displacement correlations, residence times

## Quick Start

```python
import surfcat as sc

# Load trajectory
system = sc.System('trajectory.xyz')

# Define interface region
system.define_region('interface', method='z_slice', z_min=10, z_max=15)

# Identify species
oh_atoms = system.find_species('hydroxide')
water_atoms = system.find_species('water')

# Calculate density profile
density_data = sc.analysis.profiles.calculate_density_profile(
    system, selection='name O', region=system.regions['interface']
)

# Plot results
fig, ax = sc.plotting.profiles.plot_density_profile(density_data)
```

## Installation

```bash
pip install -e .
```

## Requirements

- Python ≥ 3.8
- MDAnalysis ≥ 2.0.0
- NumPy, Matplotlib, Pandas, SciPy

## Testing

```bash
python run_tests.py basic
```

## License

MIT License
