"""
Region module for defining spatial regions in SurfCat.

This module provides the Region class for defining and managing spatial regions
within the simulation system for localized analysis.
"""

import numpy as np
import MDAnalysis as mda
from typing import Dict, Any, Optional, Union


class Region:
    """
    Represents a spatial region within the simulation system.
    
    Regions are used to restrict analyses to specific spatial areas,
    such as interfaces, surfaces, or bulk regions.
    
    Parameters
    ----------
    name : str
        Name identifier for the region
    method : str
        Method for defining the region
    universe : MDAnalysis.Universe
        The MDAnalysis Universe object
    **params
        Parameters specific to the chosen method
        
    Attributes
    ----------
    name : str
        Region name
    method : str
        Definition method
    params : dict
        Region parameters
    universe : MDAnalysis.Universe
        Associated universe
    """
    
    def __init__(self, name: str, method: str, universe: mda.Universe, **params):
        self.name = name
        self.method = method
        self.universe = universe
        self.params = params
        
        # Validate method and parameters
        self._validate_method_params()
    
    def _validate_method_params(self):
        """Validate the region definition method and parameters."""
        valid_methods = {
            'z_slice': ['z_min', 'z_max'],
            'relative_z_slice': ['ref_selection', 'z_min', 'z_max'],
            'cylinder': ['center_x', 'center_y', 'radius', 'z_min', 'z_max'],
            'sphere': ['center_x', 'center_y', 'center_z', 'radius'],
        }
        
        if self.method not in valid_methods:
            available = ', '.join(valid_methods.keys())
            raise ValueError(f"Unknown method '{self.method}'. Available: {available}")
        
        required_params = valid_methods[self.method]
        missing_params = [p for p in required_params if p not in self.params]
        
        if missing_params:
            raise ValueError(f"Missing required parameters for '{self.method}': {missing_params}")
    
    def contains_atoms(self, atoms: mda.AtomGroup, frame_idx: Optional[int] = None) -> np.ndarray:
        """
        Check which atoms are contained within this region.
        
        Parameters
        ----------
        atoms : MDAnalysis.AtomGroup
            Atoms to check
        frame_idx : int, optional
            Frame index. If None, use current frame
            
        Returns
        -------
        numpy.ndarray
            Boolean array indicating which atoms are in the region
        """
        if frame_idx is not None:
            self.universe.trajectory[frame_idx]
        
        positions = atoms.positions
        
        if self.method == 'z_slice':
            return self._z_slice_contains(positions)
        elif self.method == 'relative_z_slice':
            return self._relative_z_slice_contains(positions)
        elif self.method == 'cylinder':
            return self._cylinder_contains(positions)
        elif self.method == 'sphere':
            return self._sphere_contains(positions)
        else:
            raise NotImplementedError(f"Method '{self.method}' not implemented")
    
    def _z_slice_contains(self, positions: np.ndarray) -> np.ndarray:
        """Check containment for z_slice method."""
        z_coords = positions[:, 2]
        return (z_coords >= self.params['z_min']) & (z_coords <= self.params['z_max'])
    
    def _relative_z_slice_contains(self, positions: np.ndarray) -> np.ndarray:
        """Check containment for relative_z_slice method."""
        # Get reference atoms and their average Z position
        ref_atoms = self.universe.select_atoms(self.params['ref_selection'])
        if len(ref_atoms) == 0:
            raise ValueError(f"No atoms found for reference selection: {self.params['ref_selection']}")
        
        ref_z = np.mean(ref_atoms.positions[:, 2])
        
        # Calculate relative Z coordinates
        relative_z = positions[:, 2] - ref_z
        
        return (relative_z >= self.params['z_min']) & (relative_z <= self.params['z_max'])
    
    def _cylinder_contains(self, positions: np.ndarray) -> np.ndarray:
        """Check containment for cylinder method."""
        # Check Z bounds
        z_coords = positions[:, 2]
        z_in_bounds = (z_coords >= self.params['z_min']) & (z_coords <= self.params['z_max'])
        
        # Check radial distance from cylinder axis
        center_x = self.params['center_x']
        center_y = self.params['center_y']
        radius = self.params['radius']
        
        dx = positions[:, 0] - center_x
        dy = positions[:, 1] - center_y
        radial_dist = np.sqrt(dx**2 + dy**2)
        
        radial_in_bounds = radial_dist <= radius
        
        return z_in_bounds & radial_in_bounds
    
    def _sphere_contains(self, positions: np.ndarray) -> np.ndarray:
        """Check containment for sphere method."""
        center_x = self.params['center_x']
        center_y = self.params['center_y']
        center_z = self.params['center_z']
        radius = self.params['radius']
        
        dx = positions[:, 0] - center_x
        dy = positions[:, 1] - center_y
        dz = positions[:, 2] - center_z
        
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        return distance <= radius
    
    def select_atoms(self, selection_string: str = 'all', 
                     frame_idx: Optional[int] = None) -> mda.AtomGroup:
        """
        Select atoms within this region.
        
        Parameters
        ----------
        selection_string : str, default 'all'
            MDAnalysis selection string
        frame_idx : int, optional
            Frame index
            
        Returns
        -------
        MDAnalysis.AtomGroup
            Atoms within the region
        """
        if frame_idx is not None:
            self.universe.trajectory[frame_idx]
        
        # First apply the selection string
        atoms = self.universe.select_atoms(selection_string)
        
        # Then filter by region
        in_region = self.contains_atoms(atoms, frame_idx)
        
        # Return subset of atoms that are in the region
        return atoms[in_region]
    
    def get_bounds(self, frame_idx: Optional[int] = None) -> Dict[str, float]:
        """
        Get the spatial bounds of this region.
        
        Parameters
        ----------
        frame_idx : int, optional
            Frame index (relevant for relative regions)
            
        Returns
        -------
        dict
            Dictionary with bound information
        """
        if frame_idx is not None:
            self.universe.trajectory[frame_idx]
        
        if self.method == 'z_slice':
            return {
                'z_min': self.params['z_min'],
                'z_max': self.params['z_max'],
                'type': 'z_slice'
            }
        elif self.method == 'relative_z_slice':
            ref_atoms = self.universe.select_atoms(self.params['ref_selection'])
            ref_z = np.mean(ref_atoms.positions[:, 2])
            return {
                'z_min': ref_z + self.params['z_min'],
                'z_max': ref_z + self.params['z_max'],
                'reference_z': ref_z,
                'type': 'relative_z_slice'
            }
        elif self.method == 'cylinder':
            return {
                'center_x': self.params['center_x'],
                'center_y': self.params['center_y'],
                'radius': self.params['radius'],
                'z_min': self.params['z_min'],
                'z_max': self.params['z_max'],
                'type': 'cylinder'
            }
        elif self.method == 'sphere':
            return {
                'center_x': self.params['center_x'],
                'center_y': self.params['center_y'],
                'center_z': self.params['center_z'],
                'radius': self.params['radius'],
                'type': 'sphere'
            }
    
    def __repr__(self) -> str:
        """String representation of the Region."""
        return f"Region('{self.name}', method='{self.method}')"