"""
SurfCat: A Post-Processing Library for Surface and Interfacial Simulations

SurfCat provides a high-level API for analyses common in surface science and 
electrocatalysis simulations. Built as a high-level wrapper around MDAnalysis,
it offers an intuitive, science-oriented interface.

Key Features:
1. Interface-Centric Abstraction - Analyses framed around "surface" or "interface" concepts
2. Dynamic Species Identification - On-the-fly identification of molecules and ions based on bonding topology
3. Workflow-Oriented API - Supports chaining of commands in logical sequences
4. Integrated Publication-Quality Plotting - Every analysis module has corresponding plotting functions

Basic Usage:
    >>> import surfcat as sc
    >>> system = sc.System('trajectory.xyz')
    >>> system.define_region('interface', method='z_slice', z_min=10, z_max=15)
    >>> oh_data = system.find_species('hydroxide')
"""

__version__ = "0.1.0"
__author__ = "SurfCat Development Team"

# Import core classes and functions
from .system import System
from .region import Region
from . import species
from . import utils
from . import analysis
from . import plotting

# Define public API
__all__ = [
    'System',
    'Region', 
    'species',
    'utils',
    'analysis', 
    'plotting',
    '__version__'
]

# Setup logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())