"""
Utility functions for SurfCat library.

This module provides general helper functions including file format conversion,
data processing utilities, and other supporting functionality.
"""

import numpy as np
import pandas as pd
import MDAnalysis as mda
from pathlib import Path
from typing import Optional, Union, Dict, Any
import warnings


def convert_trajectory(input_file: Union[str, Path], 
                      output_file: Union[str, Path],
                      in_format: Optional[str] = None,
                      out_format: Optional[str] = None,
                      start: int = 0,
                      stop: Optional[int] = None,
                      step: int = 1) -> None:
    """
    Convert trajectory between different formats using MDAnalysis.
    
    This function provides a simple interface for converting between different
    trajectory formats, leveraging MDAnalysis's comprehensive format support.
    
    Parameters
    ----------
    input_file : str or Path
        Input trajectory file path
    output_file : str or Path
        Output trajectory file path
    in_format : str, optional
        Input format. If None, MDAnalysis will guess from extension
    out_format : str, optional
        Output format. If None, MDAnalysis will guess from extension
    start : int, default 0
        Starting frame index
    stop : int, optional
        Stopping frame index. If None, convert all frames
    step : int, default 1
        Frame step size
        
    Examples
    --------
    >>> convert_trajectory('traj.lammpstrj', 'traj.xyz')
    >>> convert_trajectory('input.xtc', 'output.dcd', start=100, stop=1000)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_file} not found")
    
    print(f"Converting {input_file} to {output_file}...")
    
    # Load trajectory
    try:
        if in_format:
            u = mda.Universe(str(input_path), format=in_format)
        else:
            u = mda.Universe(str(input_path))
    except Exception as e:
        raise ValueError(f"Failed to load input file: {e}")
    
    # Set up frame selection
    if stop is None:
        stop = len(u.trajectory)
    
    total_frames = len(range(start, stop, step))
    print(f"Converting {total_frames} frames...")
    
    # Write trajectory
    try:
        with mda.Writer(str(output_path), n_atoms=len(u.atoms)) as writer:
            for ts in u.trajectory[start:stop:step]:
                writer.write(u.atoms)
    except Exception as e:
        raise ValueError(f"Failed to write output file: {e}")
    
    print(f"Conversion complete: {output_file}")


def setup_default_pbc(system_type: str = "surface") -> np.ndarray:
    """
    Get default periodic boundary conditions for common system types.
    
    Parameters
    ----------
    system_type : str, default "surface"
        Type of system. Options: "surface", "bulk", "interface"
        
    Returns
    -------
    numpy.ndarray
        Box dimensions [a, b, c, alpha, beta, gamma]
    """
    defaults = {
        "surface": [10.69, 10.69, 49.16, 90, 90, 90],  # Common surface slab
        "bulk": [20.0, 20.0, 20.0, 90, 90, 90],        # Cubic bulk
        "interface": [15.0, 15.0, 40.0, 90, 90, 90],   # Interface system
    }
    
    if system_type not in defaults:
        available = ', '.join(defaults.keys())
        raise ValueError(f"Unknown system type '{system_type}'. Available: {available}")
    
    return np.array(defaults[system_type])


def calculate_molecular_weight(formula: str) -> float:
    """
    Calculate molecular weight from chemical formula.
    
    Parameters
    ----------
    formula : str
        Chemical formula (e.g., 'H2O', 'OH', 'H3O')
        
    Returns
    -------
    float
        Molecular weight in g/mol
    """
    atomic_weights = {
        'H': 1.008,
        'C': 12.011,
        'N': 14.007,
        'O': 15.999,
        'Na': 22.990,
        'Mg': 24.305,
        'Al': 26.982,
        'Si': 28.085,
        'P': 30.974,
        'S': 32.065,
        'Cl': 35.453,
        'K': 39.098,
        'Ca': 40.078,
        'Cu': 63.546,
        'Zn': 65.380,
    }
    
    # Simple parser for basic formulas
    weight = 0.0
    i = 0
    while i < len(formula):
        # Read element symbol
        element = formula[i]
        i += 1
        
        # Check for lowercase letter (second character of element)
        if i < len(formula) and formula[i].islower():
            element += formula[i]
            i += 1
        
        # Read count
        count_str = ""
        while i < len(formula) and formula[i].isdigit():
            count_str += formula[i]
            i += 1
        
        count = int(count_str) if count_str else 1
        
        if element not in atomic_weights:
            raise ValueError(f"Unknown element: {element}")
        
        weight += atomic_weights[element] * count
    
    return weight


def apply_minimum_image_convention(positions1: np.ndarray, 
                                  positions2: np.ndarray,
                                  box_lengths: np.ndarray) -> np.ndarray:
    """
    Apply minimum image convention for periodic boundary conditions.
    
    This function calculates the minimum image displacement vector between
    two sets of positions, accounting for periodic boundary conditions.
    
    Parameters
    ----------
    positions1 : numpy.ndarray
        First set of positions, shape (N, 3)
    positions2 : numpy.ndarray
        Second set of positions, shape (N, 3)
    box_lengths : numpy.ndarray
        Box lengths [lx, ly, lz]
        
    Returns
    -------
    numpy.ndarray
        Displacement vectors with minimum image convention applied
    """
    displacement = positions2 - positions1
    displacement = displacement - box_lengths * np.round(displacement / box_lengths)
    return displacement


def memory_usage_mb() -> float:
    """
    Get current memory usage in MB.
    
    Returns
    -------
    float
        Memory usage in MB
    """
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2
    except ImportError:
        warnings.warn("psutil not available. Cannot monitor memory usage.")
        return 0.0


def print_memory_usage(label: str = "") -> None:
    """
    Print current memory usage with optional label.
    
    Parameters
    ----------
    label : str, optional
        Label to include in the output
    """
    memory_mb = memory_usage_mb()
    if memory_mb > 0:
        print(f"[MEMORY] {label}: {memory_mb:.2f} MB")


def validate_frame_range(trajectory_length: int, 
                        start_frame: int = 0,
                        end_frame: Optional[int] = None) -> tuple:
    """
    Validate and normalize frame range parameters.
    
    Parameters
    ----------
    trajectory_length : int
        Total number of frames in trajectory
    start_frame : int, default 0
        Starting frame
    end_frame : int, optional
        Ending frame
        
    Returns
    -------
    tuple
        (start_frame, end_frame) validated and normalized
    """
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")
    
    if start_frame >= trajectory_length:
        raise ValueError(f"start_frame ({start_frame}) >= trajectory length ({trajectory_length})")
    
    if end_frame is None:
        end_frame = trajectory_length
    
    if end_frame <= start_frame:
        raise ValueError(f"end_frame ({end_frame}) must be > start_frame ({start_frame})")
    
    if end_frame > trajectory_length:
        end_frame = trajectory_length
        warnings.warn(f"end_frame truncated to trajectory length ({trajectory_length})")
    
    return start_frame, end_frame


def create_output_filename(base_name: str, 
                          analysis_type: str,
                          extension: str = ".dat") -> str:
    """
    Create standardized output filename.
    
    Parameters
    ----------
    base_name : str
        Base filename (usually trajectory name without extension)
    analysis_type : str
        Type of analysis (e.g., 'density', 'orientation', 'identity')
    extension : str, default ".dat"
        File extension
        
    Returns
    -------
    str
        Formatted output filename
    """
    return f"{base_name}_{analysis_type}{extension}"


def save_data_with_header(filename: str, 
                         data: np.ndarray,
                         header: str,
                         fmt: str = "%.6f") -> None:
    """
    Save numerical data with header comment.
    
    Parameters
    ----------
    filename : str
        Output filename
    data : numpy.ndarray
        Data to save
    header : str
        Header comment
    fmt : str, default "%.6f"
        Format string for numerical data
    """
    np.savetxt(filename, data, header=header, fmt=fmt)
    print(f"Data saved to {filename}")


def load_trajectory_info(trajectory_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load basic information about a trajectory file.
    
    Parameters
    ----------
    trajectory_file : str or Path
        Path to trajectory file
        
    Returns
    -------
    dict
        Dictionary with trajectory information
    """
    try:
        u = mda.Universe(str(trajectory_file))
        
        info = {
            'n_atoms': len(u.atoms),
            'n_frames': len(u.trajectory),
            'dimensions': u.dimensions,
            'dt': getattr(u.trajectory, 'dt', None),
            'total_time': getattr(u.trajectory, 'totaltime', None),
        }
        
        # Element composition
        elements, counts = np.unique(u.atoms.names, return_counts=True)
        info['elements'] = dict(zip(elements, counts))
        
        return info
        
    except Exception as e:
        raise ValueError(f"Failed to load trajectory info: {e}")