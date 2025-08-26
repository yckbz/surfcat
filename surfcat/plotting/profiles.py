"""
Profile plotting functions for 1D distributions.

This module provides functions for plotting density profiles, orientation
profiles, and other 1D distribution data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, Optional, Union, List, Tuple
import warnings


def plot_density_profile(data: Dict[str, np.ndarray],
                        title: str = "Density Profile",
                        xlabel: str = "Z (Å)",
                        ylabel: str = "Number Density (1/Å³)",
                        figsize: Tuple[float, float] = (8, 6),
                        color: str = 'blue',
                        linestyle: str = '-',
                        linewidth: float = 2.0,
                        alpha: float = 1.0,
                        grid: bool = True,
                        save_path: Optional[str] = None,
                        dpi: int = 300,
                        show: bool = True) -> Tuple[Figure, Axes]:
    """
    Plot density profile data.
    
    Parameters
    ----------
    data : dict
        Dictionary containing 'bins' and 'density' arrays
    title : str, default "Density Profile"
        Plot title
    xlabel : str, default "Z (Å)"
        X-axis label
    ylabel : str, default "Number Density (1/Å³)"
        Y-axis label
    figsize : tuple, default (8, 6)
        Figure size (width, height) in inches
    color : str, default 'blue'
        Line color
    linestyle : str, default '-'
        Line style
    linewidth : float, default 2.0
        Line width
    alpha : float, default 1.0
        Line transparency
    grid : bool, default True
        Whether to show grid
    save_path : str, optional
        Path to save the figure
    dpi : int, default 300
        Resolution for saved figure
    show : bool, default True
        Whether to display the plot
        
    Returns
    -------
    tuple
        (Figure, Axes) objects for further customization
        
    Examples
    --------
    >>> fig, ax = plot_density_profile(density_data, title="OH⁻ Density Profile")
    >>> ax.set_ylim(0, 0.1)  # Customize y-axis range
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Validate data
    if 'bins' not in data or 'density' not in data:
        raise ValueError("Data must contain 'bins' and 'density' keys")
    
    bins = data['bins']
    density = data['density']
    
    if len(bins) != len(density):
        raise ValueError("Bins and density arrays must have the same length")
    
    # Plot the profile
    ax.plot(bins, density, color=color, linestyle=linestyle, 
            linewidth=linewidth, alpha=alpha, label='Density')
    
    # Customize the plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    if grid:
        ax.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    if np.max(density) > 0:
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    
    return fig, ax


def plot_orientation_profile(data: Dict[str, np.ndarray],
                           title: str = "Orientation Profile",
                           xlabel: str = "Z (Å)",
                           ylabel: str = "⟨cos θ⟩",
                           figsize: Tuple[float, float] = (8, 6),
                           color: str = 'red',
                           show_error_bars: bool = True,
                           error_alpha: float = 0.3,
                           grid: bool = True,
                           save_path: Optional[str] = None,
                           dpi: int = 300,
                           show: bool = True) -> Tuple[Figure, Axes]:
    """
    Plot orientation profile with optional error bars.
    
    Parameters
    ----------
    data : dict
        Dictionary containing orientation profile data
    title : str, default "Orientation Profile"
        Plot title
    xlabel : str, default "Z (Å)"
        X-axis label
    ylabel : str, default "⟨cos θ⟩"
        Y-axis label
    figsize : tuple, default (8, 6)
        Figure size
    color : str, default 'red'
        Line and fill color
    show_error_bars : bool, default True
        Whether to show error bars/shaded regions
    error_alpha : float, default 0.3
        Transparency for error regions
    grid : bool, default True
        Whether to show grid
    save_path : str, optional
        Path to save figure
    dpi : int, default 300
        Resolution for saved figure
    show : bool, default True
        Whether to display plot
        
    Returns
    -------
    tuple
        (Figure, Axes) objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Validate data
    required_keys = ['bins', 'avg_cos_angle']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Data must contain '{key}' key")
    
    bins = data['bins']
    avg_cos_angle = data['avg_cos_angle']
    
    # Plot the main line
    ax.plot(bins, avg_cos_angle, color=color, linewidth=2, label='⟨cos θ⟩')
    
    # Add error bars or shaded region if available
    if show_error_bars and 'std_cos_angle' in data:
        std_cos_angle = data['std_cos_angle']
        ax.fill_between(bins, avg_cos_angle - std_cos_angle, 
                       avg_cos_angle + std_cos_angle,
                       alpha=error_alpha, color=color, label='± σ')
    
    # Add horizontal reference lines
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=-1, color='gray', linestyle=':', alpha=0.5)
    
    # Customize plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-1.1, 1.1)
    
    if grid:
        ax.grid(True, alpha=0.3)
    
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_multiple_profiles(data_list: List[Dict[str, np.ndarray]],
                          labels: List[str],
                          title: str = "Multiple Profiles",
                          xlabel: str = "Z (Å)",
                          ylabel: str = "Density",
                          figsize: Tuple[float, float] = (10, 6),
                          colors: Optional[List[str]] = None,
                          linestyles: Optional[List[str]] = None,
                          linewidths: Optional[List[float]] = None,
                          grid: bool = True,
                          legend: bool = True,
                          save_path: Optional[str] = None,
                          dpi: int = 300,
                          show: bool = True) -> Tuple[Figure, Axes]:
    """
    Plot multiple profiles on the same axes.
    
    Parameters
    ----------
    data_list : list of dict
        List of data dictionaries, each containing 'bins' and 'density'
    labels : list of str
        Labels for each profile
    title : str, default "Multiple Profiles"
        Plot title
    xlabel : str, default "Z (Å)"
        X-axis label
    ylabel : str, default "Density"
        Y-axis label
    figsize : tuple, default (10, 6)
        Figure size
    colors : list of str, optional
        Colors for each profile. If None, use default color cycle
    linestyles : list of str, optional
        Line styles for each profile
    linewidths : list of float, optional
        Line widths for each profile
    grid : bool, default True
        Whether to show grid
    legend : bool, default True
        Whether to show legend
    save_path : str, optional
        Path to save figure
    dpi : int, default 300
        Resolution for saved figure
    show : bool, default True
        Whether to display plot
        
    Returns
    -------
    tuple
        (Figure, Axes) objects
    """
    if len(data_list) != len(labels):
        raise ValueError("data_list and labels must have the same length")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set default styles
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    if linestyles is None:
        linestyles = ['-'] * len(data_list)
    if linewidths is None:
        linewidths = [2.0] * len(data_list)
    
    # Plot each profile
    for i, (data, label) in enumerate(zip(data_list, labels)):
        if 'bins' not in data or 'density' not in data:
            warnings.warn(f"Skipping profile {i}: missing required data")
            continue
        
        color = colors[i] if i < len(colors) else colors[i % len(colors)]
        linestyle = linestyles[i] if i < len(linestyles) else linestyles[i % len(linestyles)]
        linewidth = linewidths[i] if i < len(linewidths) else linewidths[i % len(linewidths)]
        
        ax.plot(data['bins'], data['density'], 
                color=color, linestyle=linestyle, linewidth=linewidth, 
                label=label)
    
    # Customize plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    if grid:
        ax.grid(True, alpha=0.3)
    
    if legend:
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax


def plot_correlation_function(data: Dict[str, np.ndarray],
                             title: str = "Displacement Correlation",
                             xlabel: str = "Time Interval (ps)",
                             ylabel: str = "Correlation (Ų)",
                             figsize: Tuple[float, float] = (8, 6),
                             color: str = 'green',
                             marker: str = 'o',
                             markersize: float = 6,
                             show_error_bars: bool = True,
                             grid: bool = True,
                             save_path: Optional[str] = None,
                             dpi: int = 300,
                             show: bool = True) -> Tuple[Figure, Axes]:
    """
    Plot correlation function data.
    
    Parameters
    ----------
    data : dict
        Dictionary containing correlation data
    title : str, default "Displacement Correlation"
        Plot title
    xlabel : str, default "Time Interval (ps)"
        X-axis label
    ylabel : str, default "Correlation (Ų)"
        Y-axis label
    figsize : tuple, default (8, 6)
        Figure size
    color : str, default 'green'
        Color for line and markers
    marker : str, default 'o'
        Marker style
    markersize : float, default 6
        Marker size
    show_error_bars : bool, default True
        Whether to show error bars
    grid : bool, default True
        Whether to show grid
    save_path : str, optional
        Path to save figure
    dpi : int, default 300
        Resolution for saved figure
    show : bool, default True
        Whether to display plot
        
    Returns
    -------
    tuple
        (Figure, Axes) objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Validate data
    required_keys = ['time_intervals', 'correlations']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Data must contain '{key}' key")
    
    time_intervals = data['time_intervals']
    correlations = data['correlations']
    
    # Plot with or without error bars
    if show_error_bars and 'std_errors' in data:
        std_errors = data['std_errors']
        ax.errorbar(time_intervals, correlations, yerr=std_errors,
                   color=color, marker=marker, markersize=markersize,
                   linewidth=2, capsize=3, label='Correlation')
    else:
        ax.plot(time_intervals, correlations, color=color, marker=marker,
                markersize=markersize, linewidth=2, label='Correlation')
    
    # Add reference line at y=0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Customize plot
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    if grid:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    
    return fig, ax