"""
Dynamics analysis module for time-dependent properties.

This module provides functions for calculating displacement correlations,
diffusion coefficients, residence times, and other dynamic properties.
"""

import numpy as np
import pandas as pd
import MDAnalysis as mda
from typing import Dict, List, Optional, Union, TYPE_CHECKING, Callable
import warnings

if TYPE_CHECKING:
    from ..system import System
    from ..region import Region


def calculate_displacement_correlation(system: 'System',
                                     sel1: Union[str, mda.AtomGroup],
                                     sel2: Union[str, mda.AtomGroup, None] = None,
                                     time_intervals: List[int] = None,
                                     region: Optional['Region'] = None,
                                     start_frame: int = 0,
                                     end_frame: Optional[int] = None,
                                     nearest_neighbor: bool = True,
                                     progress_freq: int = 200) -> Dict[str, np.ndarray]:
    """
    Calculate displacement correlation between two selections.
    
    This function implements the logic from corr.py to calculate the correlation
    ⟨Δr₁(t)·Δr₂(t)⟩ between displacements of two groups of atoms.
    
    Parameters
    ----------
    system : System
        The SurfCat System object
    sel1 : str or MDAnalysis.AtomGroup
        First selection (e.g., 'name O' for OH- oxygens)
    sel2 : str or MDAnalysis.AtomGroup or None
        Second selection (e.g., 'name Na'). If None, use sel1
    time_intervals : list of int, optional
        List of time intervals (in frames) to analyze
    region : Region, optional
        Spatial region to restrict analysis
    start_frame : int, default 0
        Starting frame for analysis
    end_frame : int, optional
        Ending frame for analysis
    nearest_neighbor : bool, default True
        If True, correlate each atom in sel1 with its nearest neighbor in sel2
    progress_freq : int, default 200
        Print progress every N frames
        
    Returns
    -------
    dict
        Dictionary containing correlation results
        
    Examples
    --------
    >>> # OH-Na displacement correlation
    >>> correlation = calculate_displacement_correlation(
    ...     system, 'name O', 'name Na', 
    ...     time_intervals=[1, 10, 20, 50, 100]
    ... )
    """
    if end_frame is None:
        end_frame = len(system.u.trajectory)
    
    if time_intervals is None:
        time_intervals = [1, 10, 20, 50, 75, 100, 200]
    
    if sel2 is None:
        sel2 = sel1
        nearest_neighbor = False  # Self-correlation
    
    box_lengths = system.u.dimensions[:3]
    correlation_results = []
    
    print(f"Calculating displacement correlation for time intervals: {time_intervals}")
    print(f"Analysis from frame {start_frame} to {end_frame}")
    
    for delta_t in time_intervals:
        print(f"\nProcessing time interval: {delta_t} frames...")
        
        dot_products = []
        n_pairs = 0
        
        for t_initial in range(start_frame, end_frame - delta_t):
            if t_initial % progress_freq == 0:
                print(f"  Processing initial frame {t_initial}/{end_frame - delta_t}")
            
            t_final = t_initial + delta_t
            
            # Get initial positions
            system.u.trajectory[t_initial]
            
            if isinstance(sel1, str):
                atoms1_initial = system.u.select_atoms(sel1)
            else:
                atoms1_initial = sel1
            
            if isinstance(sel2, str):
                atoms2_initial = system.u.select_atoms(sel2)
            else:
                atoms2_initial = sel2
            
            # Apply region filter if specified
            if region:
                in_region1 = region.contains_atoms(atoms1_initial, t_initial)
                in_region2 = region.contains_atoms(atoms2_initial, t_initial)
                atoms1_initial = atoms1_initial[in_region1]
                atoms2_initial = atoms2_initial[in_region2]
            
            if len(atoms1_initial) == 0 or len(atoms2_initial) == 0:
                continue
            
            pos1_initial = atoms1_initial.positions.copy()
            pos2_initial = atoms2_initial.positions.copy()
            
            # Get final positions
            system.u.trajectory[t_final]
            
            if isinstance(sel1, str):
                atoms1_final = system.u.select_atoms(sel1)
            else:
                atoms1_final = sel1
            
            if isinstance(sel2, str):
                atoms2_final = system.u.select_atoms(sel2)
            else:
                atoms2_final = sel2
            
            # Apply region filter if specified
            if region:
                in_region1 = region.contains_atoms(atoms1_final, t_final)
                in_region2 = region.contains_atoms(atoms2_final, t_final)
                atoms1_final = atoms1_final[in_region1]
                atoms2_final = atoms2_final[in_region2]
            
            if len(atoms1_final) == 0 or len(atoms2_final) == 0:
                continue
            
            pos1_final = atoms1_final.positions
            pos2_final = atoms2_final.positions
            
            # Calculate displacements with minimum image convention
            if nearest_neighbor:
                # For each atom in sel1, find nearest atom in sel2
                for i in range(len(pos1_initial)):
                    # Find nearest atom in sel2 at initial time
                    distances = mda.lib.distances.distance_array(
                        pos1_initial[i:i+1], pos2_initial, box=system.u.dimensions
                    )[0]
                    nearest_idx = np.argmin(distances)
                    
                    # Calculate displacement for sel1 atom
                    disp1 = pos1_final[i] - pos1_initial[i]
                    disp1 = disp1 - box_lengths * np.round(disp1 / box_lengths)
                    
                    # Calculate displacement for nearest sel2 atom
                    disp2 = pos2_final[nearest_idx] - pos2_initial[nearest_idx]
                    disp2 = disp2 - box_lengths * np.round(disp2 / box_lengths)
                    
                    # Calculate dot product
                    dot_products.append(np.dot(disp1, disp2))
                    n_pairs += 1
            else:
                # Direct pairing (same indices or self-correlation)
                min_len = min(len(pos1_initial), len(pos2_initial), 
                             len(pos1_final), len(pos2_final))
                
                for i in range(min_len):
                    # Calculate displacements
                    disp1 = pos1_final[i] - pos1_initial[i]
                    disp1 = disp1 - box_lengths * np.round(disp1 / box_lengths)
                    
                    disp2 = pos2_final[i] - pos2_initial[i]
                    disp2 = disp2 - box_lengths * np.round(disp2 / box_lengths)
                    
                    dot_products.append(np.dot(disp1, disp2))
                    n_pairs += 1
        
        if dot_products:
            avg_correlation = np.mean(dot_products)
            std_correlation = np.std(dot_products)
            correlation_results.append({
                'time_interval': delta_t,
                'correlation': avg_correlation,
                'std': std_correlation,
                'n_samples': len(dot_products)
            })
            print(f"  Average correlation: {avg_correlation:.4f} ± {std_correlation:.4f} Ų")
            print(f"  Number of samples: {len(dot_products)}")
        else:
            correlation_results.append({
                'time_interval': delta_t,
                'correlation': 0.0,
                'std': 0.0,
                'n_samples': 0
            })
            print(f"  No valid pairs found for this interval")
    
    print("\nDisplacement correlation calculation complete.")
    
    # Convert to arrays for easy plotting
    results_df = pd.DataFrame(correlation_results)
    
    return {
        'time_intervals': results_df['time_interval'].values,
        'correlations': results_df['correlation'].values,
        'std_errors': results_df['std'].values,
        'n_samples': results_df['n_samples'].values,
        'results_dataframe': results_df
    }


def calculate_mean_square_displacement(system: 'System',
                                      selection: Union[str, mda.AtomGroup],
                                      time_intervals: List[int] = None,
                                      region: Optional['Region'] = None,
                                      start_frame: int = 0,
                                      end_frame: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Calculate mean square displacement (MSD) for selected atoms.
    
    Parameters
    ----------
    system : System
        The SurfCat System object
    selection : str or MDAnalysis.AtomGroup
        Atom selection for MSD calculation
    time_intervals : list of int, optional
        Time intervals (in frames) to analyze
    region : Region, optional
        Spatial region to restrict analysis
    start_frame : int, default 0
        Starting frame
    end_frame : int, optional
        Ending frame
        
    Returns
    -------
    dict
        Dictionary containing MSD results
    """
    if end_frame is None:
        end_frame = len(system.u.trajectory)
    
    if time_intervals is None:
        # Default time intervals (logarithmic spacing)
        max_interval = min(200, (end_frame - start_frame) // 4)
        time_intervals = np.unique(np.logspace(0, np.log10(max_interval), 20).astype(int))
    
    box_lengths = system.u.dimensions[:3]
    msd_results = []
    
    print(f"Calculating MSD for time intervals: {time_intervals}")
    
    for delta_t in time_intervals:
        print(f"Processing time interval: {delta_t} frames...")
        
        squared_displacements = []
        
        for t_initial in range(start_frame, end_frame - delta_t):
            t_final = t_initial + delta_t
            
            # Get initial positions
            system.u.trajectory[t_initial]
            if isinstance(selection, str):
                atoms_initial = system.u.select_atoms(selection)
            else:
                atoms_initial = selection
            
            if region:
                in_region = region.contains_atoms(atoms_initial, t_initial)
                atoms_initial = atoms_initial[in_region]
            
            if len(atoms_initial) == 0:
                continue
            
            pos_initial = atoms_initial.positions.copy()
            atom_indices = atoms_initial.indices
            
            # Get final positions
            system.u.trajectory[t_final]
            atoms_final = system.u.atoms[atom_indices]
            
            if region:
                in_region = region.contains_atoms(atoms_final, t_final)
                atoms_final = atoms_final[in_region]
                pos_initial = pos_initial[in_region]
            
            if len(atoms_final) == 0:
                continue
            
            pos_final = atoms_final.positions
            
            # Calculate squared displacements with minimum image convention
            displacements = pos_final - pos_initial
            displacements = displacements - box_lengths * np.round(displacements / box_lengths)
            squared_disps = np.sum(displacements**2, axis=1)
            
            squared_displacements.extend(squared_disps)
        
        if squared_displacements:
            msd = np.mean(squared_displacements)
            msd_std = np.std(squared_displacements)
            msd_results.append({
                'time_interval': delta_t,
                'msd': msd,
                'std': msd_std,
                'n_samples': len(squared_displacements)
            })
        else:
            msd_results.append({
                'time_interval': delta_t,
                'msd': 0.0,
                'std': 0.0,
                'n_samples': 0
            })
    
    results_df = pd.DataFrame(msd_results)
    
    return {
        'time_intervals': results_df['time_interval'].values,
        'msd': results_df['msd'].values,
        'std_errors': results_df['std'].values,
        'n_samples': results_df['n_samples'].values,
        'results_dataframe': results_df
    }


def calculate_residence_time(system: 'System',
                           selection: Union[str, mda.AtomGroup],
                           region: 'Region',
                           start_frame: int = 0,
                           end_frame: Optional[int] = None,
                           min_residence: int = 1) -> Dict[str, Union[float, List]]:
    """
    Calculate residence times of atoms in a specified region.
    
    Parameters
    ----------
    system : System
        The SurfCat System object
    selection : str or MDAnalysis.AtomGroup
        Atom selection
    region : Region
        Region for residence time analysis
    start_frame : int, default 0
        Starting frame
    end_frame : int, optional
        Ending frame
    min_residence : int, default 1
        Minimum residence time (in frames) to count
        
    Returns
    -------
    dict
        Dictionary containing residence time statistics
    """
    if end_frame is None:
        end_frame = len(system.u.trajectory)
    
    print(f"Calculating residence times in region '{region.name}'...")
    
    # Track which atoms are in the region at each frame
    atom_in_region = {}
    residence_times = []
    
    for frame_idx in range(start_frame, end_frame):
        system.u.trajectory[frame_idx]
        
        if isinstance(selection, str):
            atoms = system.u.select_atoms(selection)
        else:
            atoms = selection
        
        in_region = region.contains_atoms(atoms, frame_idx)
        
        for i, (atom, is_in_region) in enumerate(zip(atoms, in_region)):
            atom_id = atom.index
            
            if atom_id not in atom_in_region:
                atom_in_region[atom_id] = {'in_region': False, 'start_frame': None}
            
            if is_in_region and not atom_in_region[atom_id]['in_region']:
                # Atom enters region
                atom_in_region[atom_id]['in_region'] = True
                atom_in_region[atom_id]['start_frame'] = frame_idx
            elif not is_in_region and atom_in_region[atom_id]['in_region']:
                # Atom leaves region
                atom_in_region[atom_id]['in_region'] = False
                if atom_in_region[atom_id]['start_frame'] is not None:
                    residence_time = frame_idx - atom_in_region[atom_id]['start_frame']
                    if residence_time >= min_residence:
                        residence_times.append(residence_time)
    
    # Handle atoms that are still in the region at the end
    for atom_id, data in atom_in_region.items():
        if data['in_region'] and data['start_frame'] is not None:
            residence_time = end_frame - data['start_frame']
            if residence_time >= min_residence:
                residence_times.append(residence_time)
    
    if residence_times:
        results = {
            'mean_residence_time': np.mean(residence_times),
            'std_residence_time': np.std(residence_times),
            'median_residence_time': np.median(residence_times),
            'max_residence_time': np.max(residence_times),
            'min_residence_time': np.min(residence_times),
            'n_events': len(residence_times),
            'residence_times': residence_times
        }
    else:
        results = {
            'mean_residence_time': 0.0,
            'std_residence_time': 0.0,
            'median_residence_time': 0.0,
            'max_residence_time': 0.0,
            'min_residence_time': 0.0,
            'n_events': 0,
            'residence_times': []
        }
    
    print(f"Found {results['n_events']} residence events")
    print(f"Mean residence time: {results['mean_residence_time']:.2f} frames")
    
    return results