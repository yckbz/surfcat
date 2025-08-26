"""
Species identification module for SurfCat.

This module provides dynamic identification and tracking of chemical species
based on bonding topology. It encapsulates the logic from various analysis
scripts for identifying OH-, H2O, H3O+, and other species.
"""

import numpy as np
import pandas as pd
import MDAnalysis as mda
from typing import Optional, Dict, List, Tuple, Union, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from .system import System


def _calculate_angle_with_pbc(central_pos: np.ndarray, 
                             pos1: np.ndarray, 
                             pos2: np.ndarray, 
                             box_dimensions: np.ndarray) -> float:
    """
    Calculate angle between three atoms with proper periodic boundary condition handling.
    
    Parameters
    ----------
    central_pos : numpy.ndarray
        Position of central atom
    pos1 : numpy.ndarray
        Position of first bonded atom
    pos2 : numpy.ndarray
        Position of second bonded atom
    box_dimensions : numpy.ndarray
        Box dimensions for PBC
        
    Returns
    -------
    float
        Angle in degrees
    """
    # Calculate vectors with minimum image convention
    vec1 = pos1 - central_pos
    vec2 = pos2 - central_pos
    
    # Apply minimum image convention
    box_lengths = box_dimensions[:3]
    vec1 = vec1 - box_lengths * np.round(vec1 / box_lengths)
    vec2 = vec2 - box_lengths * np.round(vec2 / box_lengths)
    
    # Calculate angle
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    
    return angle


def find_hydroxide(system: 'System', frame_idx: Optional[int] = None,
                   bond_min: float = 0.85, bond_max: float = 1.2,
                   strict_level: int = 0,
                   extended_bond_min: float = 0.82, extended_bond_max: float = 1.23,
                   return_details: bool = False) -> Union[mda.AtomGroup, Dict]:
    """
    Identify hydroxide (OH-) ions based on bonding topology with configurable strictness.
    
    A hydroxide ion is identified as an oxygen atom bonded to exactly one hydrogen
    atom within the specified distance range. Additional strictness levels help
    prevent false positives from water molecules.
    
    Parameters
    ----------
    system : System
        The SurfCat System object
    frame_idx : int, optional
        Frame index to analyze. If None, use current frame
    bond_min : float, default 0.85
        Minimum O-H bond distance in Angstroms
    bond_max : float, default 1.2
        Maximum O-H bond distance in Angstroms
    strict_level : int, default 0
        Strictness level for identification:
        - 0: Basic (oxygen with exactly 1 H in bond range)
        - 1: + Extended range check (prevent water with stretched bonds)
        - 2: + Hydrogen exclusivity check (prevent H sharing between molecules)
    extended_bond_min : float, default 0.82
        Extended minimum range for level 1+ checks
    extended_bond_max : float, default 1.23
        Extended maximum range for level 1+ checks
    return_details : bool, default False
        If True, return detailed information including bond distances
        
    Returns
    -------
    MDAnalysis.AtomGroup or dict
        If return_details=False: AtomGroup of oxygen atoms in OH- ions
        If return_details=True: Dictionary with detailed analysis
        
    Examples
    --------
    >>> # Basic identification
    >>> oh_atoms = find_hydroxide(system)
    
    >>> # Strict identification to prevent false positives
    >>> oh_atoms = find_hydroxide(system, strict_level=2)
    
    >>> # Detailed analysis
    >>> details = find_hydroxide(system, return_details=True)
    
    Notes
    -----
    Strict levels explained:
    
    Level 0 (Basic): 
        - Oxygen with exactly 1 H in [bond_min, bond_max]
        - Fastest, but may have false positives from water molecules
    
    Level 1 (Extended range check):
        - Basic check + ensure no additional H in [extended_min, extended_max]
        - Prevents water molecules with one stretched O-H bond from being misidentified
        - Recommended for most applications
    
    Level 2 (Hydrogen exclusivity):
        - Level 1 + ensure each H atom belongs to only one molecule
        - Most accurate but computationally intensive
        - Recommended for critical applications or complex systems
    """
    if frame_idx is not None:
        system.u.trajectory[frame_idx]
    
    oxygen_atoms = system.u.select_atoms('name O')
    hydrogen_atoms = system.u.select_atoms('name H')
    
    oh_oxygens = []
    oh_details = []
    hydrogen_assignments = {}  # For level 2: track which H belongs to which O
    
    # Level 2 preprocessing: identify all water molecules first
    if strict_level >= 2:
        water_oxygens = _identify_water_molecules_for_exclusivity(
            system, oxygen_atoms, hydrogen_atoms, bond_min, bond_max, 
            extended_bond_min, extended_bond_max
        )
        # Build hydrogen assignment map
        for water_info in water_oxygens:
            for h_idx in water_info['hydrogen_indices']:
                hydrogen_assignments[h_idx] = water_info['oxygen_index']
    
    for o_atom in oxygen_atoms:
        # Calculate distances to all hydrogen atoms with PBC
        distances = mda.lib.distances.distance_array(
            o_atom.position.reshape(1, 3), 
            hydrogen_atoms.positions, 
            box=system.u.dimensions
        )[0]
        
        # Level 0: Basic check - exactly 1 H in primary bond range
        primary_bonded_indices = np.where((distances >= bond_min) & (distances <= bond_max))[0]
        
        if len(primary_bonded_indices) != 1:
            continue  # Must have exactly 1 H in primary range
        
        # Level 1+: Extended range check
        if strict_level >= 1:
            extended_bonded_indices = np.where(
                (distances >= extended_bond_min) & (distances <= extended_bond_max)
            )[0]
            
            # Exclude the primary bonded hydrogen
            extended_only_indices = extended_bonded_indices[
                ~np.isin(extended_bonded_indices, primary_bonded_indices)
            ]
            
            if len(extended_only_indices) > 0:
                # Found additional H atoms in extended range - likely water molecule
                continue
        
        # Level 2: Hydrogen exclusivity check
        if strict_level >= 2:
            primary_h_idx = primary_bonded_indices[0]
            if primary_h_idx in hydrogen_assignments:
                # This hydrogen is already assigned to another molecule
                continue
        
        # Passed all checks - this is a hydroxide ion
        oh_oxygens.append(o_atom.index)
        
        if return_details:
            h_atom = hydrogen_atoms[primary_bonded_indices[0]]
            oh_details.append({
                'oxygen_id': o_atom.index,
                'hydrogen_id': h_atom.index,
                'oh_distance': distances[primary_bonded_indices[0]],
                'oxygen_pos': o_atom.position.copy(),
                'hydrogen_pos': h_atom.position.copy(),
                'strict_level_used': strict_level
            })
    
    if not return_details:
        if oh_oxygens:
            return system.u.atoms[oh_oxygens]
        else:
            return system.u.atoms[[]]  # Empty AtomGroup
    else:
        return {
            'n_hydroxide': len(oh_oxygens),
            'oxygen_atoms': system.u.atoms[oh_oxygens] if oh_oxygens else system.u.atoms[[]],
            'details': oh_details,
            'frame': system.u.trajectory.frame,
            'time': getattr(system.u.trajectory, 'time', 0.0),
            'strict_level': strict_level
        }


def _identify_water_molecules_for_exclusivity(system, oxygen_atoms, hydrogen_atoms,
                                            bond_min, bond_max, extended_min, extended_max):
    """
    Helper function to identify water molecules for hydrogen exclusivity checking.
    
    This function is used in strict_level >= 2 to pre-identify water molecules
    and create a hydrogen assignment map.
    """
    water_molecules = []
    
    for o_atom in oxygen_atoms:
        distances = mda.lib.distances.distance_array(
            o_atom.position.reshape(1, 3),
            hydrogen_atoms.positions,
            box=system.u.dimensions
        )[0]
        
        # Look for hydrogen atoms in the extended range
        bonded_indices = np.where((distances >= extended_min) & (distances <= extended_max))[0]
        
        if len(bonded_indices) == 2:
            h1_idx, h2_idx = bonded_indices
            h1_atom = hydrogen_atoms[h1_idx]
            h2_atom = hydrogen_atoms[h2_idx]
            
            # Calculate H-O-H angle with proper PBC handling
            hoh_angle = _calculate_angle_with_pbc(
                o_atom.position, h1_atom.position, h2_atom.position, system.u.dimensions
            )
            
            # Use water identification criteria
            if 85.0 <= hoh_angle <= 125.0:  # Same as find_water default
                water_molecules.append({
                    'oxygen_index': o_atom.index,
                    'hydrogen_indices': [h1_idx, h2_idx],
                    'angle': hoh_angle,
                    'distances': [distances[h1_idx], distances[h2_idx]]
                })
    
    return water_molecules


def find_water(system: 'System', frame_idx: Optional[int] = None,
               bond_min: float = 0.85, bond_max: float = 1.2,
               angle_min: float = 85.0, angle_max: float = 125.0,
               return_details: bool = False) -> Union[mda.AtomGroup, Dict]:
    """
    Identify water (H2O) molecules based on bonding topology and geometry.
    
    A water molecule is identified as an oxygen atom bonded to exactly two hydrogen
    atoms with H-O-H angle within the specified range. Parameters are based on
    physical analysis of liquid water at interfaces.
    
    Parameters
    ----------
    system : System
        The SurfCat System object
    frame_idx : int, optional
        Frame index to analyze
    bond_min : float, default 0.85
        Minimum O-H bond distance in Angstroms (based on liquid water statistics)
    bond_max : float, default 1.2
        Maximum O-H bond distance in Angstroms (allows for hydrogen bonding)
    angle_min : float, default 85.0
        Minimum H-O-H angle in degrees (allows for interface distortion)
    angle_max : float, default 125.0
        Maximum H-O-H angle in degrees (covers thermal fluctuations)
    return_details : bool, default False
        If True, return detailed molecular information
        
    Returns
    -------
    MDAnalysis.AtomGroup or dict
        Water molecule oxygen atoms or detailed analysis
        
    Notes
    -----
    Parameters are optimized for liquid water systems based on:
    - Gas phase H2O: ~0.96 Å, 104.5°
    - Liquid H2O: ~0.98-1.02 Å, 104.5° ± 6°
    - Interface effects: slight bond lengthening and angle variation
    - Statistical analysis: covers >99% of water molecules in typical simulations
    """
    if frame_idx is not None:
        system.u.trajectory[frame_idx]
    
    oxygen_atoms = system.u.select_atoms('name O')
    hydrogen_atoms = system.u.select_atoms('name H')
    
    water_oxygens = []
    water_details = []
    
    for o_atom in oxygen_atoms:
        distances = mda.lib.distances.distance_array(
            o_atom.position.reshape(1, 3),
            hydrogen_atoms.positions,
            box=system.u.dimensions
        )[0]
        
        bonded_indices = np.where((distances >= bond_min) & (distances <= bond_max))[0]
        
        # Water has exactly two bonded hydrogens
        if len(bonded_indices) == 2:
            h1_idx, h2_idx = bonded_indices
            h1_atom = hydrogen_atoms[h1_idx]
            h2_atom = hydrogen_atoms[h2_idx]
            
            # Calculate H-O-H angle with proper PBC handling
            hoh_angle = _calculate_angle_with_pbc(o_atom.position, h1_atom.position, h2_atom.position, system.u.dimensions)
            
            # Check if angle is within water range
            if angle_min <= hoh_angle <= angle_max:
                water_oxygens.append(o_atom.index)
                
                if return_details:
                    # Calculate vectors with PBC for detail recording
                    vec_oh1 = h1_atom.position - o_atom.position
                    vec_oh2 = h2_atom.position - o_atom.position
                    box_lengths = system.u.dimensions[:3]
                    vec_oh1 = vec_oh1 - box_lengths * np.round(vec_oh1 / box_lengths)
                    vec_oh2 = vec_oh2 - box_lengths * np.round(vec_oh2 / box_lengths)
                    
                    water_details.append({
                        'oxygen_id': o_atom.index,
                        'hydrogen1_id': h1_atom.index,
                        'hydrogen2_id': h2_atom.index,
                        'oh1_distance': distances[h1_idx],
                        'oh2_distance': distances[h2_idx],
                        'hoh_angle': hoh_angle,
                        'oxygen_pos': o_atom.position.copy(),
                        'hydrogen1_pos': h1_atom.position.copy(),
                        'hydrogen2_pos': h2_atom.position.copy()
                    })
    
    if not return_details:
        if water_oxygens:
            return system.u.atoms[water_oxygens]
        else:
            return system.u.atoms[[]]
    else:
        return {
            'n_water': len(water_oxygens),
            'oxygen_atoms': system.u.atoms[water_oxygens] if water_oxygens else system.u.atoms[[]],
            'details': water_details,
            'frame': system.u.trajectory.frame,
            'time': getattr(system.u.trajectory, 'time', 0.0)
        }


def find_hydronium(system: 'System', frame_idx: Optional[int] = None,
                   bond_min: float = 0.8, bond_max: float = 1.2,
                   return_details: bool = False) -> Union[mda.AtomGroup, Dict]:
    """
    Identify hydronium (H3O+) ions based on bonding topology.
    
    A hydronium ion is identified as an oxygen atom bonded to exactly three
    hydrogen atoms within the specified distance range.
    
    Parameters
    ----------
    system : System
        The SurfCat System object
    frame_idx : int, optional
        Frame index to analyze
    bond_min : float, default 0.8
        Minimum O-H bond distance in Angstroms
    bond_max : float, default 1.2
        Maximum O-H bond distance in Angstroms
    return_details : bool, default False
        If True, return detailed information
        
    Returns
    -------
    MDAnalysis.AtomGroup or dict
        Hydronium oxygen atoms or detailed analysis
    """
    if frame_idx is not None:
        system.u.trajectory[frame_idx]
    
    oxygen_atoms = system.u.select_atoms('name O')
    hydrogen_atoms = system.u.select_atoms('name H')
    
    h3o_oxygens = []
    h3o_details = []
    
    for o_atom in oxygen_atoms:
        distances = mda.lib.distances.distance_array(
            o_atom.position.reshape(1, 3),
            hydrogen_atoms.positions,
            box=system.u.dimensions
        )[0]
        
        bonded_indices = np.where((distances >= bond_min) & (distances <= bond_max))[0]
        
        # Hydronium has exactly three bonded hydrogens
        if len(bonded_indices) == 3:
            h3o_oxygens.append(o_atom.index)
            
            if return_details:
                h_atoms = hydrogen_atoms[bonded_indices]
                h3o_details.append({
                    'oxygen_id': o_atom.index,
                    'hydrogen_ids': [h.index for h in h_atoms],
                    'oh_distances': distances[bonded_indices].tolist(),
                    'oxygen_pos': o_atom.position.copy(),
                    'hydrogen_positions': [h.position.copy() for h in h_atoms]
                })
    
    if not return_details:
        if h3o_oxygens:
            return system.u.atoms[h3o_oxygens]
        else:
            return system.u.atoms[[]]
    else:
        return {
            'n_hydronium': len(h3o_oxygens),
            'oxygen_atoms': system.u.atoms[h3o_oxygens] if h3o_oxygens else system.u.atoms[[]],
            'details': h3o_details,
            'frame': system.u.trajectory.frame,
            'time': getattr(system.u.trajectory, 'time', 0.0)
        }


def track_species_identity(system: 'System', species_finder_func,
                          start_frame: int = 0, end_frame: Optional[int] = None,
                          progress_freq: int = 100, **finder_kwargs) -> pd.DataFrame:
    """
    Track species identity throughout the trajectory.
    
    This function implements the logic from oh_id.py to track how species identities
    change over time, handling cases like Grotthuss proton hopping.
    
    Parameters
    ----------
    system : System
        The SurfCat System object
    species_finder_func : callable
        Function to identify species (e.g., find_hydroxide)
    start_frame : int, default 0
        Starting frame for analysis
    end_frame : int, optional
        Ending frame. If None, analyze to end of trajectory
    progress_freq : int, default 100
        Print progress every N frames
    **finder_kwargs
        Additional arguments for the species finder function
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: [frame, time, species_id, atom_id, x, y, z, displacement]
        
    Examples
    --------
    >>> from surfcat.species import find_hydroxide, track_species_identity
    >>> identity_df = track_species_identity(system, find_hydroxide)
    >>> print(identity_df.head())
    """
    if end_frame is None:
        end_frame = len(system.u.trajectory)
    
    trajectory_data = []
    previous_positions = None
    species_counter = 0
    
    print(f"Tracking species identity from frame {start_frame} to {end_frame}...")
    
    for frame_idx in range(start_frame, end_frame):
        system.u.trajectory[frame_idx]
        current_time = getattr(system.u.trajectory, 'time', frame_idx * 1.0)
        
        if frame_idx % progress_freq == 0:
            print(f"Processing frame {frame_idx} (time: {current_time:.2f})")
        
        # Find species in current frame
        species_atoms = species_finder_func(system, frame_idx, **finder_kwargs)
        
        if len(species_atoms) == 0:
            previous_positions = None
            continue
        
        current_positions = species_atoms.positions
        
        # Calculate displacement from previous frame
        if previous_positions is not None and len(previous_positions) > 0:
            # Calculate distance matrix between current and previous positions
            dist_matrix = mda.lib.distances.distance_array(
                current_positions, previous_positions, box=system.u.dimensions
            )
            
            # Find minimum distances (nearest species assignment)
            min_distances = np.min(dist_matrix, axis=1)
        else:
            # First frame or no previous species
            min_distances = np.zeros(len(current_positions))
        
        # Store data for each species
        for i, (atom, displacement) in enumerate(zip(species_atoms, min_distances)):
            trajectory_data.append({
                'frame': frame_idx,
                'time': current_time,
                'species_id': species_counter + i + 1,  # 1-indexed
                'atom_id': atom.index + 1,  # 1-indexed for compatibility
                'x': atom.position[0],
                'y': atom.position[1],
                'z': atom.position[2],
                'displacement': displacement
            })
        
        # Update for next iteration
        previous_positions = current_positions.copy()
        species_counter += len(current_positions)
    
    print("Species tracking complete.")
    
    if trajectory_data:
        return pd.DataFrame(trajectory_data)
    else:
        warnings.warn("No species found in the specified frame range.")
        return pd.DataFrame(columns=['frame', 'time', 'species_id', 'atom_id', 'x', 'y', 'z', 'displacement'])


def count_species_over_time(system: 'System', species_finder_func,
                           start_frame: int = 0, end_frame: Optional[int] = None,
                           progress_freq: int = 100, **finder_kwargs) -> pd.DataFrame:
    """
    Count species over time for validation and diagnostics.
    
    This function provides a diagnostic tool to help users validate their
    species identification parameters by tracking species counts over time.
    
    Parameters
    ----------
    system : System
        The SurfCat System object
    species_finder_func : callable
        Function to identify species
    start_frame : int, default 0
        Starting frame
    end_frame : int, optional
        Ending frame
    progress_freq : int, default 100
        Progress reporting frequency
    **finder_kwargs
        Arguments for species finder
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: [frame, time, count]
    """
    if end_frame is None:
        end_frame = len(system.u.trajectory)
    
    counts_data = []
    
    print(f"Counting species from frame {start_frame} to {end_frame}...")
    
    for frame_idx in range(start_frame, end_frame):
        system.u.trajectory[frame_idx]
        current_time = getattr(system.u.trajectory, 'time', frame_idx * 1.0)
        
        if frame_idx % progress_freq == 0:
            print(f"Processing frame {frame_idx}")
        
        species_atoms = species_finder_func(system, frame_idx, **finder_kwargs)
        
        counts_data.append({
            'frame': frame_idx,
            'time': current_time,
            'count': len(species_atoms)
        })
    
    print("Species counting complete.")
    return pd.DataFrame(counts_data)