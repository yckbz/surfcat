"""
Core System class for SurfCat library.

This module contains the central System class that serves as the main entry point
for all trajectory analysis operations.
"""

import MDAnalysis as mda
import numpy as np
import warnings
from typing import Dict, Optional, Union, Any
from pathlib import Path

from .region import Region
from . import species


class System:
    """
    Central object for trajectory analysis in SurfCat.
    
    The System class handles trajectory loading, PBC management, and serves as
    the entry point for all analyses. It maintains regions and provides access
    to species identification and analysis modules.
    
    Parameters
    ----------
    trajectory_file : str or Path
        Path to the trajectory file
    topology_file : str or Path, optional
        Path to the topology file (if different from trajectory)
    **kwargs
        Additional arguments passed to MDAnalysis.Universe
        
    Attributes
    ----------
    u : MDAnalysis.Universe
        The underlying MDAnalysis Universe object
    regions : dict
        Dictionary of defined spatial regions
    trajectory_file : Path
        Path to the trajectory file
    """
    
    def __init__(self, trajectory_file: Union[str, Path], 
                 topology_file: Optional[Union[str, Path]] = None, 
                 **kwargs):
        
        self.trajectory_file = Path(trajectory_file)
        
        # Validate file existence
        if not self.trajectory_file.exists():
            raise FileNotFoundError(f"Trajectory file {trajectory_file} not found")
            
        # Load trajectory with MDAnalysis
        if topology_file:
            self.u = mda.Universe(str(topology_file), str(trajectory_file), **kwargs)
        else:
            self.u = mda.Universe(str(trajectory_file), **kwargs)
            
        # Handle periodic boundary conditions
        self._setup_pbc()
        
        # Initialize regions dictionary
        self.regions: Dict[str, Region] = {}
        
        print(f"System loaded: {len(self.u.atoms)} atoms, {len(self.u.trajectory)} frames")
        print(f"Box dimensions: {self.u.dimensions}")
    
    def _setup_pbc(self):
        """Setup periodic boundary conditions with sensible defaults."""
        if self.u.dimensions is None:
            # Default box dimensions based on common surface simulation setups
            default_dims = [10.69, 10.69, 49.16, 90, 90, 90]
            self.u.dimensions = default_dims
            warnings.warn(
                f"No PBC info found. Using default dimensions: {default_dims}. "
                "Please verify these match your system."
            )
    
    def select_atoms(self, selection_string: str) -> mda.AtomGroup:
        """
        Select atoms using MDAnalysis selection syntax.
        
        Parameters
        ----------
        selection_string : str
            Selection string in MDAnalysis format
            
        Returns
        -------
        MDAnalysis.AtomGroup
            Selected atoms
        """
        return self.u.select_atoms(selection_string)
    
    def define_region(self, name: str, method: str, **params) -> Region:
        """
        Define a spatial region for analysis.
        
        Parameters
        ----------
        name : str
            Name identifier for the region
        method : str
            Method for defining the region. Options:
            - 'z_slice': Define by absolute Z coordinates
            - 'relative_z_slice': Define relative to reference atoms
            - 'cylinder': Define cylindrical region
            - 'sphere': Define spherical region
        **params
            Parameters specific to the chosen method
            
        Returns
        -------
        Region
            The created region object
            
        Examples
        --------
        >>> system.define_region('interface', method='z_slice', z_min=10, z_max=15)
        >>> system.define_region('surface', method='relative_z_slice', 
        ...                      ref_selection='name Cu', z_min=2.0, z_max=8.0)
        """
        region = Region(name, method, self.u, **params)
        self.regions[name] = region
        return region
    
    def find_species(self, species_name: str, frame_idx: Optional[int] = None, 
                     **kwargs) -> Union[mda.AtomGroup, Dict]:
        """
        Find and identify chemical species in the system.
        
        Parameters
        ----------
        species_name : str
            Name of species to identify. Options:
            - 'hydroxide' or 'OH': Hydroxide ions
            - 'water' or 'H2O': Water molecules
            - 'hydronium' or 'H3O': Hydronium ions
        frame_idx : int, optional
            Frame index to analyze. If None, analyze current frame
        **kwargs
            Additional parameters for species identification
            
        Returns
        -------
        MDAnalysis.AtomGroup or dict
            Identified species atoms or analysis results
        """
        if frame_idx is not None:
            self.u.trajectory[frame_idx]
            
        species_map = {
            'hydroxide': species.find_hydroxide,
            'OH': species.find_hydroxide,
            'water': species.find_water,
            'H2O': species.find_water,
            'hydronium': species.find_hydronium,
            'H3O': species.find_hydronium,
        }
        
        if species_name not in species_map:
            available = ', '.join(species_map.keys())
            raise ValueError(f"Unknown species '{species_name}'. Available: {available}")
            
        finder_func = species_map[species_name]
        return finder_func(self, frame_idx, **kwargs)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the system.
        
        Returns
        -------
        dict
            Dictionary containing system information
        """
        info = {
            'n_atoms': len(self.u.atoms),
            'n_frames': len(self.u.trajectory),
            'box_dimensions': self.u.dimensions,
            'trajectory_file': str(self.trajectory_file),
            'atom_types': list(set(self.u.atoms.types)),
            'regions': list(self.regions.keys()),
        }
        
        # Add element counts
        elements, counts = np.unique(self.u.atoms.names, return_counts=True)
        info['element_counts'] = dict(zip(elements, counts))
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the System."""
        return (f"System({self.trajectory_file.name}: "
                f"{len(self.u.atoms)} atoms, {len(self.u.trajectory)} frames)")