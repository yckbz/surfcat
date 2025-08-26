"""
Hydrogen bond analysis module.

This module provides functions for analyzing hydrogen bond networks,
including bond counting, lifetime analysis, and spatial distribution.
"""

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from ..system import System
    from ..region import Region


def analyze_hydrogen_bonds(system: 'System',
                          donors_sel: str = 'name O N',
                          hydrogens_sel: str = 'name H',
                          acceptors_sel: str = 'name O N',
                          d_a_cutoff: float = 3.5,
                          d_h_a_angle_cutoff: float = 150.0,
                          region: Optional['Region'] = None,
                          start_frame: int = 0,
                          end_frame: Optional[int] = None,
                          ignore_atoms: Optional[List[int]] = None) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """
    Analyze hydrogen bonds in the system.
    
    This function implements the logic from Hbond_section.py to analyze
    hydrogen bond networks with spatial filtering capabilities.
    
    Parameters
    ----------
    system : System
        The SurfCat System object
    donors_sel : str, default 'name O N'
        Selection string for hydrogen bond donors
    hydrogens_sel : str, default 'name H'
        Selection string for hydrogen atoms
    acceptors_sel : str, default 'name O N'
        Selection string for hydrogen bond acceptors
    d_a_cutoff : float, default 3.5
        Maximum donor-acceptor distance (Angstroms)
    d_h_a_angle_cutoff : float, default 150.0
        Minimum donor-hydrogen-acceptor angle (degrees)
    region : Region, optional
        Spatial region to restrict analysis
    start_frame : int, default 0
        Starting frame for analysis
    end_frame : int, optional
        Ending frame for analysis
    ignore_atoms : list of int, optional
        List of atom indices to ignore (0-based)
        
    Returns
    -------
    dict
        Dictionary containing hydrogen bond analysis results
        
    Examples
    --------
    >>> hb_results = analyze_hydrogen_bonds(
    ...     system, 
    ...     region=system.regions['interface'],
    ...     d_a_cutoff=3.2
    ... )
    >>> print(f"Average H-bonds per frame: {hb_results['avg_hbonds_per_frame']:.2f}")
    """
    if end_frame is None:
        end_frame = len(system.u.trajectory)
    
    # Prepare selection strings to exclude ignored atoms
    if ignore_atoms:
        ignore_str = ' '.join(map(str, ignore_atoms))
        donors_sel = f'{donors_sel} and not index {ignore_str}'
        hydrogens_sel = f'{hydrogens_sel} and not index {ignore_str}'
        acceptors_sel = f'{acceptors_sel} and not index {ignore_str}'
    
    print(f"Setting up hydrogen bond analysis...")
    print(f"Donors: {donors_sel}")
    print(f"Hydrogens: {hydrogens_sel}")
    print(f"Acceptors: {acceptors_sel}")
    print(f"D-A cutoff: {d_a_cutoff} Å")
    print(f"D-H-A angle cutoff: {d_h_a_angle_cutoff}°")
    
    # Set up HydrogenBondAnalysis
    hbonds = HydrogenBondAnalysis(
        universe=system.u,
        donors_sel=donors_sel,
        hydrogens_sel=hydrogens_sel,
        acceptors_sel=acceptors_sel,
        d_a_cutoff=d_a_cutoff,
        d_h_a_angle_cutoff=d_h_a_angle_cutoff
    )
    
    # Run analysis on specified frame range
    print(f"Running hydrogen bond analysis from frame {start_frame} to {end_frame}...")
    hbonds.run(start=start_frame, stop=end_frame)
    
    # Get results
    hb_results = hbonds.results.hbonds
    
    # Process results with spatial filtering if region is specified
    hbonds_per_frame = []
    hbonds_in_region = []
    detailed_results = []
    
    for frame_idx in range(start_frame, end_frame):
        system.u.trajectory[frame_idx]
        
        # Get hydrogen bonds for this frame
        frame_hbonds = hb_results[hb_results[:, 0] == frame_idx]
        
        count_total = len(frame_hbonds)
        count_in_region = 0
        
        if region is not None:
            # Filter by region
            for hbond in frame_hbonds:
                donor_idx = int(hbond[1])
                acceptor_idx = int(hbond[3])
                
                # Skip ignored atoms
                if ignore_atoms and (donor_idx in ignore_atoms or acceptor_idx in ignore_atoms):
                    continue
                
                # Check if both donor and acceptor are in the region
                donor_atom = system.u.atoms[[donor_idx]]
                acceptor_atom = system.u.atoms[[acceptor_idx]]
                
                donor_in_region = region.contains_atoms(donor_atom, frame_idx)[0]
                acceptor_in_region = region.contains_atoms(acceptor_atom, frame_idx)[0]
                
                if donor_in_region and acceptor_in_region:
                    count_in_region += 1
                    
                    # Store detailed information
                    detailed_results.append({
                        'frame': frame_idx,
                        'donor_idx': donor_idx,
                        'hydrogen_idx': int(hbond[2]),
                        'acceptor_idx': acceptor_idx,
                        'distance': hbond[4],
                        'angle': hbond[5],
                        'donor_pos': system.u.atoms[donor_idx].position.copy(),
                        'acceptor_pos': system.u.atoms[acceptor_idx].position.copy()
                    })
        else:
            count_in_region = count_total
            
            # Store all hydrogen bonds
            for hbond in frame_hbonds:
                donor_idx = int(hbond[1])
                acceptor_idx = int(hbond[3])
                
                # Skip ignored atoms
                if ignore_atoms and (donor_idx in ignore_atoms or acceptor_idx in ignore_atoms):
                    continue
                
                detailed_results.append({
                    'frame': frame_idx,
                    'donor_idx': donor_idx,
                    'hydrogen_idx': int(hbond[2]),
                    'acceptor_idx': acceptor_idx,
                    'distance': hbond[4],
                    'angle': hbond[5],
                    'donor_pos': system.u.atoms[donor_idx].position.copy(),
                    'acceptor_pos': system.u.atoms[acceptor_idx].position.copy()
                })
        
        hbonds_per_frame.append(count_total)
        hbonds_in_region.append(count_in_region)
    
    # Calculate statistics
    avg_hbonds_total = np.mean(hbonds_per_frame) if hbonds_per_frame else 0.0
    std_hbonds_total = np.std(hbonds_per_frame) if hbonds_per_frame else 0.0
    avg_hbonds_region = np.mean(hbonds_in_region) if hbonds_in_region else 0.0
    std_hbonds_region = np.std(hbonds_in_region) if hbonds_in_region else 0.0
    
    print(f"Analysis complete!")
    print(f"Average H-bonds per frame (total): {avg_hbonds_total:.2f} ± {std_hbonds_total:.2f}")
    if region:
        print(f"Average H-bonds per frame (in region): {avg_hbonds_region:.2f} ± {std_hbonds_region:.2f}")
    
    # Create detailed DataFrame
    detailed_df = pd.DataFrame(detailed_results) if detailed_results else pd.DataFrame()
    
    # Prepare time series data
    frames = np.arange(start_frame, end_frame)
    time_series = pd.DataFrame({
        'frame': frames,
        'total_hbonds': hbonds_per_frame,
        'region_hbonds': hbonds_in_region
    })
    
    return {
        'avg_hbonds_per_frame': avg_hbonds_total,
        'std_hbonds_per_frame': std_hbonds_total,
        'avg_hbonds_in_region': avg_hbonds_region,
        'std_hbonds_in_region': std_hbonds_region,
        'time_series': time_series,
        'detailed_bonds': detailed_df,
        'raw_results': hb_results
    }


def calculate_hbond_lifetimes(system: 'System',
                             hbond_results: Dict,
                             max_gap: int = 1) -> Dict[str, Union[float, List]]:
    """
    Calculate hydrogen bond lifetimes from analysis results.
    
    Parameters
    ----------
    system : System
        The SurfCat System object
    hbond_results : dict
        Results from analyze_hydrogen_bonds function
    max_gap : int, default 1
        Maximum gap (in frames) allowed in bond definition
        
    Returns
    -------
    dict
        Dictionary containing lifetime statistics
    """
    if hbond_results['detailed_bonds'].empty:
        return {
            'mean_lifetime': 0.0,
            'std_lifetime': 0.0,
            'median_lifetime': 0.0,
            'n_bonds': 0,
            'lifetimes': []
        }
    
    detailed_bonds = hbond_results['detailed_bonds']
    
    # Group by donor-acceptor pairs
    bond_pairs = detailed_bonds.groupby(['donor_idx', 'acceptor_idx'])
    
    lifetimes = []
    
    for (donor_idx, acceptor_idx), group in bond_pairs:
        frames = sorted(group['frame'].values)
        
        # Find continuous sequences allowing for gaps
        current_start = frames[0]
        current_length = 1
        
        for i in range(1, len(frames)):
            gap = frames[i] - frames[i-1]
            
            if gap <= max_gap + 1:
                # Continue current bond
                current_length += gap
            else:
                # End current bond and start new one
                lifetimes.append(current_length)
                current_start = frames[i]
                current_length = 1
        
        # Add the final bond
        lifetimes.append(current_length)
    
    if lifetimes:
        return {
            'mean_lifetime': np.mean(lifetimes),
            'std_lifetime': np.std(lifetimes),
            'median_lifetime': np.median(lifetimes),
            'max_lifetime': np.max(lifetimes),
            'min_lifetime': np.min(lifetimes),
            'n_bonds': len(lifetimes),
            'lifetimes': lifetimes
        }
    else:
        return {
            'mean_lifetime': 0.0,
            'std_lifetime': 0.0,
            'median_lifetime': 0.0,
            'max_lifetime': 0.0,
            'min_lifetime': 0.0,
            'n_bonds': 0,
            'lifetimes': []
        }


def analyze_hbond_network_connectivity(system: 'System',
                                      hbond_results: Dict,
                                      frame_idx: Optional[int] = None) -> Dict[str, Union[int, float, List]]:
    """
    Analyze hydrogen bond network connectivity.
    
    Parameters
    ----------
    system : System
        The SurfCat System object
    hbond_results : dict
        Results from analyze_hydrogen_bonds function
    frame_idx : int, optional
        Specific frame to analyze. If None, analyze average over all frames
        
    Returns
    -------
    dict
        Dictionary containing network connectivity metrics
    """
    detailed_bonds = hbond_results['detailed_bonds']
    
    if detailed_bonds.empty:
        return {
            'n_donors': 0,
            'n_acceptors': 0,
            'n_unique_atoms': 0,
            'avg_bonds_per_atom': 0.0,
            'max_bonds_per_atom': 0,
            'network_components': []
        }
    
    if frame_idx is not None:
        # Analyze specific frame
        frame_bonds = detailed_bonds[detailed_bonds['frame'] == frame_idx]
    else:
        # Analyze all frames (use most common connections)
        frame_bonds = detailed_bonds
    
    if frame_bonds.empty:
        return {
            'n_donors': 0,
            'n_acceptors': 0,
            'n_unique_atoms': 0,
            'avg_bonds_per_atom': 0.0,
            'max_bonds_per_atom': 0,
            'network_components': []
        }
    
    # Count bonds per atom
    all_atoms = list(frame_bonds['donor_idx'].values) + list(frame_bonds['acceptor_idx'].values)
    unique_atoms = list(set(all_atoms))
    bonds_per_atom = {atom: all_atoms.count(atom) for atom in unique_atoms}
    
    # Network analysis
    donors = set(frame_bonds['donor_idx'].values)
    acceptors = set(frame_bonds['acceptor_idx'].values)
    
    return {
        'n_donors': len(donors),
        'n_acceptors': len(acceptors),
        'n_unique_atoms': len(unique_atoms),
        'avg_bonds_per_atom': np.mean(list(bonds_per_atom.values())),
        'max_bonds_per_atom': max(bonds_per_atom.values()) if bonds_per_atom else 0,
        'bonds_per_atom': bonds_per_atom,
        'donor_indices': list(donors),
        'acceptor_indices': list(acceptors)
    }