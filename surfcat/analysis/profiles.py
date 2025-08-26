"""
Profile analysis module for calculating 1D distributions.

This module provides functions for calculating density profiles, orientation 
profiles, and other 1D distributions along specified dimensions.
"""

import numpy as np
import pandas as pd
import MDAnalysis as mda
from typing import Dict, Optional, Union, TYPE_CHECKING, List
import warnings

if TYPE_CHECKING:
    from ..system import System
    from ..region import Region


def calculate_density_profile(system: 'System', 
                             selection: Union[str, mda.AtomGroup],
                             dim: str = 'z',
                             bin_width: float = 0.1,
                             region: Optional['Region'] = None,
                             start_frame: int = 0,
                             end_frame: Optional[int] = None,
                             use_relative_coords: bool = False,
                             reference_selection: Optional[str] = None,
                             density_type: str = 'number') -> Dict[str, np.ndarray]:
    """
    Calculate density profile along a specified dimension.
    
    This function generalizes the logic from OH_dist_0716.py and z_dist.py to
    calculate density distributions of any species along any dimension.
    
    Parameters
    ----------
    system : System
        The SurfCat System object
    selection : str or MDAnalysis.AtomGroup
        Atom selection string or AtomGroup
    dim : str, default 'z'
        Dimension along which to calculate profile ('x', 'y', or 'z')
    bin_width : float, default 0.1
        Width of histogram bins in Angstroms
    region : Region, optional
        Spatial region to restrict analysis
    start_frame : int, default 0
        Starting frame for analysis
    end_frame : int, optional
        Ending frame for analysis
    use_relative_coords : bool, default False
        Whether to use coordinates relative to reference atoms
    reference_selection : str, optional
        Selection string for reference atoms (required if use_relative_coords=True)
    density_type : str, default 'number'
        Type of density to calculate ('number' or 'mass')
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'bins': bin center coordinates
        - 'density': density values
        - 'counts': raw counts per bin
        - 'bin_edges': bin edge coordinates
        
    Examples
    --------
    >>> density_data = calculate_density_profile(system, 'name O', bin_width=0.2)
    >>> z_coords = density_data['bins']
    >>> densities = density_data['density']
    """
    if end_frame is None:
        end_frame = len(system.u.trajectory)
    
    # Validate dimension
    if dim not in ['x', 'y', 'z']:
        raise ValueError(f"Invalid dimension '{dim}'. Must be 'x', 'y', or 'z'")
    
    dim_idx = {'x': 0, 'y': 1, 'z': 2}[dim]
    
    # Validate reference selection if using relative coordinates
    if use_relative_coords and not reference_selection:
        raise ValueError("reference_selection is required when use_relative_coords=True")
    
    all_coords = []
    total_frames = 0
    
    print(f"Calculating {density_type} density profile along {dim}-axis...")
    
    for frame_idx in range(start_frame, end_frame):
        system.u.trajectory[frame_idx]
        
        if frame_idx % 1000 == 0:
            print(f"Processing frame {frame_idx}")
        
        # Get atoms for analysis
        if isinstance(selection, str):
            atoms = system.u.select_atoms(selection)
        else:
            atoms = selection
        
        # Apply region filter if specified
        if region:
            in_region = region.contains_atoms(atoms, frame_idx)
            atoms = atoms[in_region]
        
        if len(atoms) == 0:
            continue
        
        # Get coordinates
        coords = atoms.positions[:, dim_idx]
        
        # Apply relative coordinate transformation if requested
        if use_relative_coords:
            ref_atoms = system.u.select_atoms(reference_selection)
            if len(ref_atoms) == 0:
                warnings.warn(f"No reference atoms found in frame {frame_idx}")
                continue
            ref_coord = np.mean(ref_atoms.positions[:, dim_idx])
            coords = coords - ref_coord
        
        all_coords.extend(coords)
        total_frames += 1
    
    if not all_coords:
        warnings.warn("No atoms found for density calculation")
        return {
            'bins': np.array([]),
            'density': np.array([]),
            'counts': np.array([]),
            'bin_edges': np.array([])
        }
    
    all_coords = np.array(all_coords)
    
    # Create histogram bins
    coord_min = np.min(all_coords)
    coord_max = np.max(all_coords)
    n_bins = int(np.ceil((coord_max - coord_min) / bin_width))
    bin_edges = np.linspace(coord_min, coord_min + n_bins * bin_width, n_bins + 1)
    
    # Calculate histogram
    counts, _ = np.histogram(all_coords, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate density
    box_dims = system.u.dimensions
    if dim == 'x':
        volume_per_bin = box_dims[1] * box_dims[2] * bin_width  # y * z * bin_width
    elif dim == 'y':
        volume_per_bin = box_dims[0] * box_dims[2] * bin_width  # x * z * bin_width  
    else:  # z
        volume_per_bin = box_dims[0] * box_dims[1] * bin_width  # x * y * bin_width
    
    if density_type == 'number':
        # Number density (atoms/Å³)
        density = counts / (total_frames * volume_per_bin)
    elif density_type == 'mass':
        # Mass density (g/cm³)
        # This is a simplified calculation assuming average atomic mass
        avg_mass = 18.0  # Default to water-like mass
        N_A = 6.022e23
        density = (counts * avg_mass) / (N_A * total_frames * volume_per_bin * 1e-24)
    else:
        raise ValueError(f"Invalid density_type '{density_type}'. Must be 'number' or 'mass'")
    
    print(f"Density profile calculation complete. Analyzed {total_frames} frames.")
    
    return {
        'bins': bin_centers,
        'density': density,
        'counts': counts,
        'bin_edges': bin_edges,
        'total_frames': total_frames,
        'volume_per_bin': volume_per_bin
    }


def calculate_orientation_profile(system: 'System',
                                 selection: Union[str, mda.AtomGroup],
                                 vector_type: str = 'bisector',
                                 dim: str = 'z',
                                 bin_width: float = 0.1,
                                 region: Optional['Region'] = None,
                                 start_frame: int = 0,
                                 end_frame: Optional[int] = None,
                                 bond_min: float = 0.8,
                                 bond_max: float = 1.2,
                                 angle_min: float = 90.0,
                                 angle_max: float = 125.0,
                                 atom1_name: str = 'H',
                                 atom2_name: str = 'H') -> Dict[str, np.ndarray]:
    """
    Calculate orientation profile of molecules along a dimension.
    
    This function generalizes the logic from rho_phi.py to calculate the
    orientational distribution of molecular vectors.
    
    Parameters
    ----------
    system : System
        The SurfCat System object
    selection : str or MDAnalysis.AtomGroup
        Selection for central atoms (e.g., oxygen atoms for water)
    vector_type : str, default 'bisector'
        Type of vector to calculate:
        - 'bisector': Bisector of two bonds (for water)
        - 'bond': Single bond vector
        - 'dipole': Dipole moment vector
    dim : str, default 'z'
        Dimension along which to calculate profile
    bin_width : float, default 0.1
        Bin width for spatial binning
    region : Region, optional
        Spatial region for analysis
    start_frame : int, default 0
        Starting frame
    end_frame : int, optional
        Ending frame
    bond_min : float, default 0.8
        Minimum bond distance for molecule identification
    bond_max : float, default 1.2
        Maximum bond distance for molecule identification
    angle_min : float, default 90.0
        Minimum bond angle for molecule identification (degrees)
    angle_max : float, default 125.0
        Maximum bond angle for molecule identification (degrees)
    atom1_name : str, default 'H'
        Name of first bonded atom type
    atom2_name : str, default 'H'
        Name of second bonded atom type
        
    Returns
    -------
    dict
        Dictionary containing orientation profile data
    """
    if end_frame is None:
        end_frame = len(system.u.trajectory)
    
    dim_idx = {'x': 0, 'y': 1, 'z': 2}[dim]
    reference_vector = np.zeros(3)
    reference_vector[dim_idx] = 1.0  # Unit vector along the specified dimension
    
    all_coords = []
    all_cos_angles = []
    
    print(f"Calculating orientation profile along {dim}-axis...")
    
    for frame_idx in range(start_frame, end_frame):
        system.u.trajectory[frame_idx]
        
        if frame_idx % 1000 == 0:
            print(f"Processing frame {frame_idx}")
        
        # Get central atoms
        if isinstance(selection, str):
            central_atoms = system.u.select_atoms(selection)
        else:
            central_atoms = selection
        
        # Get bonded atoms
        bonded_atoms1 = system.u.select_atoms(f'name {atom1_name}')
        bonded_atoms2 = system.u.select_atoms(f'name {atom2_name}') if atom2_name != atom1_name else bonded_atoms1
        
        for central_atom in central_atoms:
            # Apply region filter if specified
            if region:
                if not region.contains_atoms(mda.AtomGroup([central_atom], system.u), frame_idx)[0]:
                    continue
            
            # Find bonded atoms
            distances1 = mda.lib.distances.distance_array(
                central_atom.position.reshape(1, 3),
                bonded_atoms1.positions,
                box=system.u.dimensions
            )[0]
            
            bonded_indices1 = np.where((distances1 >= bond_min) & (distances1 <= bond_max))[0]
            
            if vector_type == 'bisector':
                # Need exactly 2 bonded atoms for bisector
                if len(bonded_indices1) == 2:
                    h1, h2 = bonded_atoms1[bonded_indices1]
                    
                    # Check bond angle
                    vec1 = h1.position - central_atom.position
                    vec2 = h2.position - central_atom.position
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                    
                    if angle_min <= angle <= angle_max:
                        # Calculate bisector vector
                        vec1_norm = vec1 / np.linalg.norm(vec1)
                        vec2_norm = vec2 / np.linalg.norm(vec2)
                        bisector = vec1_norm + vec2_norm
                        bisector = bisector / np.linalg.norm(bisector)
                        
                        # Calculate angle with reference vector
                        cos_phi = np.dot(bisector, reference_vector)
                        
                        all_coords.append(central_atom.position[dim_idx])
                        all_cos_angles.append(cos_phi)
            
            elif vector_type == 'bond':
                # Need exactly 1 bonded atom for bond vector
                if len(bonded_indices1) == 1:
                    bonded_atom = bonded_atoms1[bonded_indices1[0]]
                    bond_vector = bonded_atom.position - central_atom.position
                    bond_vector = bond_vector / np.linalg.norm(bond_vector)
                    
                    cos_phi = np.dot(bond_vector, reference_vector)
                    
                    all_coords.append(central_atom.position[dim_idx])
                    all_cos_angles.append(cos_phi)
    
    if not all_coords:
        warnings.warn("No molecules found for orientation calculation")
        return {
            'bins': np.array([]),
            'avg_cos_angle': np.array([]),
            'std_cos_angle': np.array([]),
            'counts': np.array([])
        }
    
    all_coords = np.array(all_coords)
    all_cos_angles = np.array(all_cos_angles)
    
    # Create spatial bins
    coord_min = np.min(all_coords)
    coord_max = np.max(all_coords)
    n_bins = int(np.ceil((coord_max - coord_min) / bin_width))
    bin_edges = np.linspace(coord_min, coord_min + n_bins * bin_width, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate orientation statistics in each bin
    avg_cos_angles = np.zeros(n_bins)
    std_cos_angles = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        in_bin = (all_coords >= bin_edges[i]) & (all_coords < bin_edges[i+1])
        cos_angles_in_bin = all_cos_angles[in_bin]
        
        if len(cos_angles_in_bin) > 0:
            avg_cos_angles[i] = np.mean(cos_angles_in_bin)
            std_cos_angles[i] = np.std(cos_angles_in_bin)
            counts[i] = len(cos_angles_in_bin)
    
    print("Orientation profile calculation complete.")
    
    return {
        'bins': bin_centers,
        'avg_cos_angle': avg_cos_angles,
        'std_cos_angle': std_cos_angles,
        'counts': counts,
        'bin_edges': bin_edges
    }