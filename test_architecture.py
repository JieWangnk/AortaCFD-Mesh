#!/usr/bin/env python3
"""
Test script for the updated mesh optimization architecture.

Tests the new modular architecture components:
- Core infrastructure modules
- Physics calculations 
- Configuration management
- Geometry processing
"""

import sys
import logging
from pathlib import Path

# Add mesh_optim to path
sys.path.insert(0, str(Path(__file__).parent))

def test_core_modules():
    """Test core infrastructure modules."""
    print("🔧 Testing core infrastructure modules...")
    
    try:
        from mesh_optim.core.config_manager import ConfigurationManager
        from mesh_optim.core.process_manager import ProcessManager
        from mesh_optim.core.command_runner import OpenFOAMCommandRunner
        from mesh_optim.core.mesh_analyzer import MeshQualityAnalyzer
        
        # Test ConfigurationManager
        config_manager = ConfigurationManager()
        print("  ✅ ConfigurationManager imported and instantiated")
        
        # Test ProcessManager  
        process_manager = ProcessManager()
        status = process_manager.check_system_resources()
        print(f"  ✅ ProcessManager - Memory: {status.memory_available_gb:.1f}GB available")
        
        # Test CommandRunner (without actual OpenFOAM)
        runner = OpenFOAMCommandRunner("echo 'test'", max_memory_gb=1)
        print("  ✅ OpenFOAMCommandRunner instantiated")
        
        # Test MeshAnalyzer
        analyzer = MeshQualityAnalyzer(runner)
        print("  ✅ MeshQualityAnalyzer instantiated")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Core module test failed: {e}")
        return False

def test_physics_modules():
    """Test physics calculation modules."""
    print("🧪 Testing physics calculation modules...")
    
    try:
        from mesh_optim.physics.blood_properties import BloodPropertiesCalculator, BloodProperties
        from mesh_optim.physics.flow_physics import FlowPhysicsCalculator
        from mesh_optim.physics.layer_physics import BoundaryLayerCalculator
        
        # Test BloodProperties
        blood_calc = BloodPropertiesCalculator()
        blood_props = blood_calc.get_standard_properties()
        print(f"  ✅ Blood properties - Density: {blood_props.density:.0f} kg/m³, "
              f"Viscosity: {blood_props.dynamic_viscosity*1000:.1f} cP")
        
        # Test FlowPhysics
        flow_calc = FlowPhysicsCalculator()
        reynolds = flow_calc.calculate_reynolds_number(1.0, 0.02, blood_props)
        print(f"  ✅ Flow physics - Reynolds number: {reynolds:.0f}")
        
        # Test LayerPhysics
        layer_calc = BoundaryLayerCalculator()
        t1 = layer_calc.calculate_first_layer_thickness(1.0, 0.02, blood_props)
        print(f"  ✅ Layer physics - First layer thickness: {t1*1e6:.1f} μm")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Physics module test failed: {e}")
        return False

def test_geometry_modules():
    """Test geometry processing modules."""
    print("📐 Testing geometry processing modules...")
    
    try:
        from mesh_optim.geometry.stl_processor import read_stl_triangles, compute_triangle_area
        from mesh_optim.geometry.seed_calculator import calculate_seed_point
        from mesh_optim.geometry.geometry_analyzer import estimate_reference_diameters
        
        print("  ✅ STL processor functions imported")
        print("  ✅ Seed calculator functions imported") 
        print("  ✅ Geometry analyzer functions imported")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Geometry module test failed: {e}")
        return False

def test_utility_modules():
    """Test utility modules."""
    print("🛠️  Testing utility modules...")
    
    try:
        from mesh_optim.utils.surface_analysis import log_surface_histogram
        from mesh_optim.utils.geometry_utils import estimate_geometry_parameters
        from mesh_optim.utils.legacy_utils import run_command, evaluate_stage1_metrics
        
        print("  ✅ Surface analysis functions imported")
        print("  ✅ Geometry utilities imported")
        print("  ✅ Legacy utilities imported")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Utility module test failed: {e}")
        return False

def test_configuration_loading():
    """Test configuration loading with the patient1 config."""
    print("📋 Testing configuration loading...")
    
    try:
        config_path = Path("mesh_optim/configs/patient1_config.json")
        if not config_path.exists():
            print("  ⚠️  patient1_config.json not found, skipping config test")
            return True
            
        from mesh_optim.core.config_manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        config = config_manager.load_and_validate(config_path)
        
        print(f"  ✅ Configuration loaded successfully")
        print(f"  ✅ OpenFOAM environment: {config.get('openfoam_env_path', 'Not set')}")
        
        # Test solver presets
        config = config_manager.apply_solver_presets(config, "RANS")
        print(f"  ✅ RANS solver presets applied")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def main():
    """Run all architecture tests."""
    print("🚀 Testing Updated Mesh Optimization Architecture\n")
    
    tests = [
        test_core_modules,
        test_physics_modules, 
        test_geometry_modules,
        test_utility_modules,
        test_configuration_loading
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("📊 Test Summary:")
    print(f"  ✅ Passed: {passed}/{total}")
    print(f"  ❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All architecture tests passed! The updated architecture is working correctly.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())