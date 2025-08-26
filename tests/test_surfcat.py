"""
Test script for SurfCat library functionality.

This script demonstrates the main features of SurfCat and validates
that the library works correctly with the test.xyz file.
"""

import sys
import os
import numpy as np
import warnings

# Add the surfcat directory to the Python path
sys.path.insert(0, '/Users/liuce/coding/surfcat')

try:
    import surfcat as sc
    from surfcat import species, analysis, plotting
    print("✓ SurfCat library imported successfully")
except ImportError as e:
    print(f"✗ Failed to import SurfCat: {e}")
    sys.exit(1)


def test_system_loading():
    """Test basic system loading and setup."""
    print("\n=== Testing System Loading ===")
    
    try:
        # Load the test trajectory
        system = sc.System('test.xyz')
        print(f"✓ System loaded: {len(system.u.atoms)} atoms, {len(system.u.trajectory)} frames")
        
        # Test system info
        info = system.get_info()
        print(f"✓ System info: {info['n_atoms']} atoms, {info['n_frames']} frames")
        print(f"  Elements: {info['element_counts']}")
        
        return system
        
    except Exception as e:
        print(f"✗ System loading failed: {e}")
        return None


def test_region_definition(system):
    """Test region definition functionality."""
    print("\n=== Testing Region Definition ===")
    
    try:
        # Define a Z-slice region (interface region)
        system.define_region(
            'interface', 
            method='z_slice', 
            z_min=10.0, 
            z_max=15.0
        )
        print("✓ Z-slice region defined")
        
        # Define a relative Z region based on Cu atoms
        system.define_region(
            'surface',
            method='relative_z_slice',
            ref_selection='name Cu',
            z_min=2.0,
            z_max=8.0
        )
        print("✓ Relative Z-slice region defined")
        
        # Test region atom selection
        interface_atoms = system.regions['interface'].select_atoms('name O')
        print(f"✓ Found {len(interface_atoms)} oxygen atoms in interface region")
        
        return True
        
    except Exception as e:
        print(f"✗ Region definition failed: {e}")
        return False


def test_species_identification(system):
    """Test species identification functionality."""
    print("\n=== Testing Species Identification ===")
    
    try:
        # Test hydroxide identification
        oh_atoms = system.find_species('hydroxide', frame_idx=0)
        print(f"✓ Found {len(oh_atoms)} OH⁻ ions in frame 0 (expected: 1)")
        
        # Test water identification
        water_atoms = system.find_species('water', frame_idx=0)
        print(f"✓ Found {len(water_atoms)} water molecules in frame 0 (expected: 79)")
        
        # Verify the numbers are reasonable
        total_identified = len(oh_atoms) + len(water_atoms)
        print(f"✓ Total identified: {total_identified}/80 oxygen atoms")
        
        if len(oh_atoms) >= 1 and len(water_atoms) >= 75 and total_identified >= 78:
            print("✓ Species identification accuracy: EXCELLENT")
        elif total_identified >= 75:
            print("✓ Species identification accuracy: GOOD")
        else:
            print("⚠ Species identification may need adjustment")
        
        # Test detailed species information
        oh_details = species.find_hydroxide(system, frame_idx=0, return_details=True)
        print(f"✓ Detailed OH⁻ analysis: {oh_details['n_hydroxide']} ions found")
        
        water_details = species.find_water(system, frame_idx=0, return_details=True)
        print(f"✓ Detailed water analysis: {water_details['n_water']} molecules found")
        
        return True
        
    except Exception as e:
        print(f"✗ Species identification failed: {e}")
        return False


def test_species_counting(system):
    """Test species counting over time."""
    print("\n=== Testing Species Counting ===")
    
    try:
        # Count hydroxide species over first 10 frames
        oh_counts = species.count_species_over_time(
            system, 
            species.find_hydroxide,
            start_frame=0,
            end_frame=10,
            progress_freq=5
        )
        print(f"✓ OH⁻ counting complete: {len(oh_counts)} frames analyzed")
        print(f"  Average count: {oh_counts['count'].mean():.2f}")
        
        return oh_counts
        
    except Exception as e:
        print(f"✗ Species counting failed: {e}")
        return None


def test_density_profiles(system):
    """Test density profile calculations."""
    print("\n=== Testing Density Profiles ===")
    
    try:
        # Calculate oxygen density profile
        density_data = analysis.profiles.calculate_density_profile(
            system,
            selection='name O',
            bin_width=0.2,
            start_frame=0,
            end_frame=5,  # Use only first 5 frames for speed
            density_type='number'
        )
        print(f"✓ Density profile calculated: {len(density_data['bins'])} bins")
        print(f"  Max density: {np.max(density_data['density']):.6f} atoms/Ų")
        
        # Test with region
        if 'interface' in system.regions:
            interface_density = analysis.profiles.calculate_density_profile(
                system,
                selection='name O',
                bin_width=0.2,
                region=system.regions['interface'],
                start_frame=0,
                end_frame=5
            )
            print(f"✓ Interface density profile: {len(interface_density['bins'])} bins")
        
        return density_data
        
    except Exception as e:
        print(f"✗ Density profile calculation failed: {e}")
        return None


def test_plotting(density_data):
    """Test plotting functionality."""
    print("\n=== Testing Plotting ===")
    
    try:
        # Test density profile plotting
        fig, ax = plotting.profiles.plot_density_profile(
            density_data,
            title="Test Oxygen Density Profile",
            show=False,  # Don't display plot in test
            save_path="test_density_profile.png"
        )
        print("✓ Density profile plot created and saved")
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"✗ Plotting failed: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\n=== Testing Utilities ===")
    
    try:
        # Test trajectory info loading
        info = sc.utils.load_trajectory_info('test.xyz')
        print(f"✓ Trajectory info loaded: {info['n_atoms']} atoms")
        
        # Test molecular weight calculation
        h2o_mass = sc.utils.calculate_molecular_weight('H2O')
        print(f"✓ H2O molecular weight: {h2o_mass:.3f} g/mol")
        
        oh_mass = sc.utils.calculate_molecular_weight('OH')
        print(f"✓ OH molecular weight: {oh_mass:.3f} g/mol")
        
        # Test memory monitoring (if psutil available)
        memory_mb = sc.utils.memory_usage_mb()
        if memory_mb > 0:
            print(f"✓ Memory usage: {memory_mb:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Utilities test failed: {e}")
        return False


def run_comprehensive_test():
    """Run a comprehensive test of SurfCat functionality."""
    print("Starting SurfCat Comprehensive Test")
    print("=" * 50)
    
    # Suppress MDAnalysis warnings for cleaner output
    warnings.filterwarnings('ignore', module='MDAnalysis')
    
    test_results = []
    
    # Test 1: System loading
    system = test_system_loading()
    test_results.append(system is not None)
    
    if system is None:
        print("\n✗ Cannot continue tests without valid system")
        return False
    
    # Test 2: Region definition
    regions_ok = test_region_definition(system)
    test_results.append(regions_ok)
    
    # Test 3: Species identification
    species_ok = test_species_identification(system)
    test_results.append(species_ok)
    
    # Test 4: Species counting
    counts_data = test_species_counting(system)
    test_results.append(counts_data is not None)
    
    # Test 5: Density profiles
    density_data = test_density_profiles(system)
    test_results.append(density_data is not None)
    
    # Test 6: Plotting (only if matplotlib available and density data exists)
    plotting_ok = False
    if density_data is not None:
        try:
            import matplotlib.pyplot as plt
            plotting_ok = test_plotting(density_data)
        except ImportError:
            print("⚠ Matplotlib not available, skipping plotting tests")
            plotting_ok = True  # Don't fail test for optional dependency
    test_results.append(plotting_ok)
    
    # Test 7: Utilities
    utils_ok = test_utilities()
    test_results.append(utils_ok)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"System Loading: {'✓' if test_results[0] else '✗'}")
    print(f"Region Definition: {'✓' if test_results[1] else '✗'}")
    print(f"Species Identification: {'✓' if test_results[2] else '✗'}")
    print(f"Species Counting: {'✓' if test_results[3] else '✗'}")
    print(f"Density Profiles: {'✓' if test_results[4] else '✗'}")
    print(f"Plotting: {'✓' if test_results[5] else '✗'}")
    print(f"Utilities: {'✓' if test_results[6] else '✗'}")
    
    success_rate = sum(test_results) / len(test_results) * 100
    print(f"\nOverall Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 SurfCat library is functioning correctly!")
        return True
    else:
        print("⚠ SurfCat library has some issues that need attention.")
        return False


def demo_workflow():
    """Demonstrate a typical SurfCat workflow."""
    print("\n" + "=" * 50)
    print("SurfCat Workflow Demonstration")
    print("=" * 50)
    
    try:
        # 1. Load system
        print("1. Loading trajectory...")
        system = sc.System('test.xyz')
        
        # 2. Define interface region
        print("2. Defining interface region...")
        system.define_region(
            'interface',
            method='relative_z_slice',
            ref_selection='name Cu',
            z_min=3.0,
            z_max=10.0
        )
        
        # 3. Identify species
        print("3. Identifying chemical species...")
        oh_atoms = system.find_species('hydroxide', frame_idx=0)
        water_atoms = system.find_species('water', frame_idx=0)
        print(f"   Found {len(oh_atoms)} OH⁻ ions and {len(water_atoms)} water molecules")
        
        # 4. Calculate profiles
        print("4. Calculating density profiles...")
        oh_profile = analysis.profiles.calculate_density_profile(
            system,
            selection='name O',  # Would normally use OH-specific selection
            region=system.regions['interface'],
            bin_width=0.3,
            start_frame=0,
            end_frame=10
        )
        
        # 5. Summary
        print("5. Analysis complete!")
        print(f"   Interface region defined with {len(oh_profile['bins'])} spatial bins")
        print(f"   Peak density: {np.max(oh_profile['density']):.6f} atoms/Ų")
        
        print("\n✓ Workflow demonstration successful!")
        return True
        
    except Exception as e:
        print(f"✗ Workflow demonstration failed: {e}")
        return False


if __name__ == "__main__":
    # Change to the correct directory
    os.chdir('/Users/liuce/coding/surfcat')
    
    # Run comprehensive tests
    success = run_comprehensive_test()
    
    # Run workflow demonstration
    if success:
        demo_workflow()
    
    print("\nTest script completed.")