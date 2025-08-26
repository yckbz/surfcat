"""
SurfCat Quick Start Example

This script demonstrates the basic usage of SurfCat library with the provided test.xyz file.
It showcases the main features and typical workflow for surface analysis.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add SurfCat to path
sys.path.insert(0, '/Users/liuce/coding/surfcat')

import surfcat as sc
from surfcat.analysis import profiles, dynamics
from surfcat.plotting import profiles as plot_profiles
from surfcat import species

def main():
    print("SurfCat Quick Start Example")
    print("=" * 40)
    
    # Change to correct directory
    os.chdir('/Users/liuce/coding/surfcat')
    
    # 1. Load the system
    print("\n1. Loading trajectory...")
    system = sc.System('test.xyz')
    print(f"   Loaded: {len(system.u.atoms)} atoms, {len(system.u.trajectory)} frames")
    
    # Show system composition
    info = system.get_info()
    print(f"   Composition: {info['element_counts']}")
    
    # 2. Define analysis regions
    print("\n2. Defining analysis regions...")
    
    # Interface region relative to Cu surface
    system.define_region(
        'interface',
        method='relative_z_slice',
        ref_selection='name Cu',
        z_min=3.0,
        z_max=12.0
    )
    print("   ✓ Interface region defined (relative to Cu surface)")
    
    # Bulk region (absolute coordinates)
    system.define_region(
        'bulk',
        method='z_slice',
        z_min=25.0,
        z_max=40.0
    )
    print("   ✓ Bulk region defined")
    
    # 3. Species identification
    print("\n3. Identifying chemical species...")
    
    # Analyze first frame
    oh_atoms = system.find_species('hydroxide', frame_idx=0)
    water_atoms = system.find_species('water', frame_idx=0)
    
    print(f"   OH⁻ ions: {len(oh_atoms)}")
    print(f"   Water molecules: {len(water_atoms)}")
    
    # Get detailed information
    oh_details = species.find_hydroxide(system, frame_idx=0, return_details=True)
    if oh_details['details']:
        for i, detail in enumerate(oh_details['details']):
            print(f"   OH⁻ {i+1}: O-H distance = {detail['oh_distance']:.3f} Å")
    
    # 4. Density profile analysis
    print("\n4. Calculating density profiles...")
    
    # Oxygen density profile in interface region
    interface_density = profiles.calculate_density_profile(
        system,
        selection='name O',
        region=system.regions['interface'],
        bin_width=0.3,
        start_frame=0,
        end_frame=50,  # Use first 50 frames
        use_relative_coords=True,
        reference_selection='name Cu'
    )
    
    # Hydrogen density profile for comparison
    h_density = profiles.calculate_density_profile(
        system,
        selection='name H',
        region=system.regions['interface'],
        bin_width=0.3,
        start_frame=0,
        end_frame=50,
        use_relative_coords=True,
        reference_selection='name Cu'
    )
    
    print(f"   ✓ Interface density profiles calculated")
    print(f"     O peak density: {np.max(interface_density['density']):.6f} atoms/Ų")
    print(f"     H peak density: {np.max(h_density['density']):.6f} atoms/Ų")
    
    # 5. Species counting over time
    print("\n5. Tracking species over time...")
    
    # Count species in first 20 frames
    oh_counts = species.count_species_over_time(
        system,
        species.find_hydroxide,
        start_frame=0,
        end_frame=20,
        progress_freq=10
    )
    
    water_counts = species.count_species_over_time(
        system,
        species.find_water,
        start_frame=0,
        end_frame=20,
        progress_freq=10
    )
    
    print(f"   ✓ Species counting complete")
    print(f"     Average OH⁻ count: {oh_counts['count'].mean():.2f}")
    print(f"     Average water count: {water_counts['count'].mean():.2f}")
    
    # 6. Create visualizations
    print("\n6. Creating visualizations...")
    
    # Create a comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Density profiles comparison
    ax1.plot(interface_density['bins'], interface_density['density'], 
             'b-', linewidth=2, label='Oxygen')
    ax1.plot(h_density['bins'], h_density['density'], 
             'r-', linewidth=2, label='Hydrogen')
    ax1.set_xlabel('Distance from Cu surface (Å)')
    ax1.set_ylabel('Number density (atoms/Ų)')
    ax1.set_title('Density Profiles at Interface')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Species counts over time
    ax2.plot(oh_counts['frame'], oh_counts['count'], 'go-', label='OH⁻')
    ax2.plot(water_counts['frame'], water_counts['count'], 'bo-', label='H₂O')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Species count')
    ax2.set_title('Species Counts Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative density
    ax3.plot(interface_density['bins'], 
             np.cumsum(interface_density['density']) * np.diff(interface_density['bin_edges'])[0],
             'g-', linewidth=2)
    ax3.set_xlabel('Distance from Cu surface (Å)')
    ax3.set_ylabel('Cumulative density')
    ax3.set_title('Cumulative Oxygen Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: System composition
    elements = list(info['element_counts'].keys())
    counts = list(info['element_counts'].values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(elements)))
    
    ax4.pie(counts, labels=elements, colors=colors, autopct='%1.1f%%')
    ax4.set_title('System Composition')
    
    plt.tight_layout()
    plt.savefig('surfcat_quickstart_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Analysis plots saved as 'surfcat_quickstart_analysis.png'")
    
    # 7. Summary and recommendations
    print("\n7. Analysis Summary")
    print("   " + "-" * 30)
    
    # Calculate some key metrics
    interface_atoms = system.regions['interface'].select_atoms('name O', frame_idx=0)
    bulk_atoms = system.regions['bulk'].select_atoms('name O', frame_idx=0)
    
    print(f"   Interface region: {len(interface_atoms)} oxygen atoms")
    print(f"   Bulk region: {len(bulk_atoms)} oxygen atoms")
    print(f"   Interface/bulk ratio: {len(interface_atoms)/max(1, len(bulk_atoms)):.2f}")
    
    # Find peak position
    peak_idx = np.argmax(interface_density['density'])
    peak_position = interface_density['bins'][peak_idx]
    print(f"   Peak density at: {peak_position:.2f} Å from Cu surface")
    
    print("\n8. Next Steps")
    print("   " + "-" * 15)
    print("   • Try different region definitions")
    print("   • Analyze longer trajectories")
    print("   • Calculate orientation profiles")
    print("   • Perform correlation analysis")
    print("   • Study hydrogen bond networks")
    
    print(f"\n🎉 SurfCat analysis complete! Check out the generated plot.")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()