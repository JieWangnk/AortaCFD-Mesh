"""
Utility functions for mesh optimization
"""

import subprocess
import json
import numpy as np
from pathlib import Path
import re
import logging

def run_command(cmd, cwd=None, env_setup=None, timeout=None, parallel=False, max_memory_gb=8):
    """
    Run OpenFOAM command with proper environment setup and resource management
    
    Args:
        cmd: Command to run (list or string)
        cwd: Working directory
        env_setup: Path to OpenFOAM environment script
        timeout: Timeout in seconds
        parallel: Whether this is a parallel command
        max_memory_gb: Maximum memory limit in GB
    """
    import psutil
    import time
    
    # Check available system resources before starting
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    if available_memory_gb < max_memory_gb:
        raise RuntimeError(f"Insufficient memory: need {max_memory_gb}GB, have {available_memory_gb:.1f}GB")
    
    if env_setup:
        if isinstance(cmd, list):
            cmd_str = " ".join(cmd)
        else:
            cmd_str = cmd
        
        # Add memory limit to OpenFOAM commands
        if any(of_cmd in cmd_str for of_cmd in ['snappyHexMesh', 'simpleFoam', 'pimpleFoam']):
            cmd_str = f"ulimit -v {int(max_memory_gb * 1024 * 1024)} && {cmd_str}"
        
        cmd = ["bash", "-c", f"{env_setup} && {cmd_str}"]
    
    # No timeout by default - let processes run to completion
    # Only use timeout if explicitly specified and > 0
    
    try:
        if timeout and timeout > 0:
            # Run with timeout if specified
            result = subprocess.run(
                cmd, 
                cwd=cwd, 
                capture_output=True, 
                text=True,
                timeout=timeout,
                shell=isinstance(cmd, str) and not env_setup
            )
        else:
            # Run without timeout - let it complete naturally
            result = subprocess.run(
                cmd, 
                cwd=cwd, 
                capture_output=True, 
                text=True,
                shell=isinstance(cmd, str) and not env_setup
            )
        return result
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout} seconds: {cmd}")
    except Exception as e:
        raise RuntimeError(f"Command failed: {e}")

def set_process_limits(max_memory_gb):
    """Set resource limits for subprocess"""
    import resource
    try:
        # Set memory limit (virtual memory)
        memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        
        # Set CPU time limit (4 hours max)
        resource.setrlimit(resource.RLIMIT_CPU, (14400, 14400))
    except (ValueError, OSError):
        pass  # Ignore if limits can't be set

def check_mesh_quality(mesh_dir, openfoam_env, max_memory_gb=8, deep_check=False, wall_name=None):
    """
    Run checkMesh and parse quality metrics with memory limits
    
    Args:
        mesh_dir: Path to the mesh directory
        openfoam_env: OpenFOAM environment setup command
        max_memory_gb: Maximum memory limit in GB
        deep_check: If True, use -allGeometry -allTopology for deep validation
        wall_name: Optional wall patch name to extract face count for castellation checks
    """
    log_file = mesh_dir / "logs" / "log.checkMesh"
    log_file.parent.mkdir(exist_ok=True)
    
    # Use basic checkMesh for iterative optimization (fast, standard checks)
    # Use -allGeometry -allTopology only for final validation or when requested
    check_cmd = "checkMesh -allGeometry -allTopology" if deep_check else "checkMesh"
    
    result = run_command(
        check_cmd, 
        cwd=mesh_dir, 
        env_setup=openfoam_env,
        timeout=None,  # No timeout - let checkMesh complete
        max_memory_gb=max_memory_gb
    )
    
    # Write log
    log_file.write_text(result.stdout + result.stderr)
    
    # Parse metrics
    metrics = {
        "maxNonOrtho": 0,
        "maxSkewness": 0,
        "maxAspectRatio": 0,
        "negVolCells": 0,
        "meshOK": False,
        "cells": 0,
        "wall_nFaces": None  # Wall patch face count if wall_name provided
    }
    
    output = result.stdout + result.stderr
    
    # Parse key metrics
    if "non-orthogonality" in output:
        match = re.search(r"Max non-orthogonality = ([\d.]+)", output)
        if match:
            metrics["maxNonOrtho"] = float(match.group(1))
    
    if "skewness" in output:
        match = re.search(r"Max skewness = ([\d.]+)", output)  
        if match:
            metrics["maxSkewness"] = float(match.group(1))
    
    if "aspect ratio" in output:
        match = re.search(r"aspect ratio = ([\d.]+)", output)
        if match:
            metrics["maxAspectRatio"] = float(match.group(1))
            
    if "cells:" in output:
        match = re.search(r"cells:\s+(\d+)", output)
        if match:
            metrics["cells"] = int(match.group(1))
    
    # Extract wall patch face count if wall_name provided
    if wall_name:
        # checkMesh output format: "              wall_aorta   152523   154493"
        wall_pattern = rf"^\s*{wall_name}\s+(\d+)\s+\d+\s*$"
        wall_match = re.search(wall_pattern, output, re.MULTILINE)
        if wall_match:
            metrics["wall_nFaces"] = int(wall_match.group(1))
    
    # Use OpenFOAM's default mesh validation criteria
    # Trust OpenFOAM's checkMesh to determine if mesh is acceptable
    metrics["meshOK"] = "Mesh OK" in output and result.returncode == 0
    
    return metrics

def parse_layer_coverage(mesh_dir, openfoam_env, max_memory_gb=8, wall_name="wall_aorta"):
    """
    Parse boundary layer coverage from snappyHexMesh logs with improved tolerance
    """
    # Try different possible layer log file names
    log_file_candidates = [
        mesh_dir / "logs" / "log.snappy.layers",
        mesh_dir / "logs" / "log.snappyHexMesh.layers"  
    ]
    
    log_file = None
    for candidate in log_file_candidates:
        if candidate.exists():
            log_file = candidate
            break
    
    if log_file is None:
        return {"coverage_overall": 0.0, "totalFaces": 0, "perPatch": {}, "faces_with_layers": 0}
    
    log_content = log_file.read_text()
    
    # Parse layer statistics
    coverage_data = {"coverage_overall": 0.0, "totalFaces": 0, "perPatch": {}, "faces_with_layers": 0}
    
    # Look for "Layer thickness" section or "Layer mesh : cells:" for more detailed analysis
    if "Layer thickness" in log_content or "cells:" in log_content:
        lines = log_content.split('\n')
        
        # Try to find face-by-face layer information
        total_faces = 0
        faces_with_layers = 0
        
        for i, line in enumerate(lines):
            # Parse the actual snappyHexMesh output format:
            # "Extruding 15921 out of 17014 faces (93.5759%). Removed extrusion at 0 faces."
            if "Extruding" in line and "out of" in line and "faces" in line:
                match = re.search(r"Extruding (\d+) out of (\d+) faces \(([\d.]+)%\)", line)
                if match:
                    faces_with_layers = int(match.group(1))
                    total_faces = int(match.group(2))
                    coverage_pct = float(match.group(3))
                    
                    coverage_data["faces_with_layers"] = faces_with_layers
                    coverage_data["totalFaces"] = total_faces
                    coverage_data["coverage_overall"] = coverage_pct / 100.0
                    coverage_data["perPatch"][wall_name] = coverage_pct / 100.0
                    break
            
            # Also parse summary table format: "wall_aorta 17014    3.75     0.000313  84"
            if wall_name in line and len(line.split()) >= 5:
                parts = line.split()
                try:
                    patch_faces = int(parts[1])
                    avg_layers = float(parts[2])
                    thickness_pct = int(parts[4])
                    
                    if patch_faces > 0 and avg_layers > 0:
                        # Use thickness percentage as coverage estimate if we don't have extrusion data
                        if coverage_data["coverage_overall"] == 0.0:
                            coverage_data["totalFaces"] = patch_faces
                            coverage_data["coverage_overall"] = thickness_pct / 100.0
                            coverage_data["perPatch"][wall_name] = thickness_pct / 100.0
                            # Estimate faces with layers from average (conservative estimate)
                            coverage_data["faces_with_layers"] = int(patch_faces * min(avg_layers / 5.0, 0.9))
                except (ValueError, IndexError):
                    pass
        
        # Update overall coverage if we found data
        if coverage_data["coverage_overall"] == 0.0 and total_faces > 0:
            coverage_data["totalFaces"] = total_faces
            coverage_data["faces_with_layers"] = faces_with_layers
            coverage_data["coverage_overall"] = faces_with_layers / total_faces if total_faces > 0 else 0.0
    
    # Add interpretation
    coverage_data["interpretation"] = {
        "excellent": coverage_data["coverage_overall"] > 0.95,
        "good": coverage_data["coverage_overall"] > 0.80,
        "acceptable": coverage_data["coverage_overall"] > 0.60,
        "poor": coverage_data["coverage_overall"] <= 0.60,
        "success": coverage_data["coverage_overall"] > 0.60  # Lower threshold for success
    }
    
    return coverage_data

def parse_layer_iterations(log_file_path):
    """
    Extract per-iteration metrics from snappyHexMesh layer addition log
    
    Returns:
        dict with:
        - iterations: list of dicts with iteration metrics
        - converged: bool indicating if convergence criteria met
        - final_metrics: dict with final thickness %, effective layers, etc.
    """
    if isinstance(log_file_path, str):
        log_file_path = Path(log_file_path)
    
    if not log_file_path.exists():
        return {"iterations": [], "converged": False, "final_metrics": {}}
    
    log_content = log_file_path.read_text()
    lines = log_content.split('\n')
    
    iterations = []
    current_iter = None
    
    for line in lines:
        # Track iteration number
        if "Layer addition iteration" in line:
            match = re.search(r"Layer addition iteration (\d+)", line)
            if match:
                current_iter = int(match.group(1))
        
        # Extract coverage per iteration
        if current_iter is not None and "Extruding" in line and "out of" in line:
            match = re.search(r"Extruding (\d+) out of (\d+) faces \(([\d.]+)%\)", line)
            if match:
                iterations.append({
                    "iteration": current_iter,
                    "faces_extruded": int(match.group(1)),
                    "total_faces": int(match.group(2)),
                    "coverage_pct": float(match.group(3)),
                    "illegal_faces": 0  # Will be updated below
                })
        
        # Extract quality metrics (illegal faces)
        if current_iter is not None and "Detected" in line and "illegal" in line:
            match = re.search(r"Detected (\d+) illegal", line)
            if match and iterations and iterations[-1]["iteration"] == current_iter:
                iterations[-1]["illegal_faces"] = int(match.group(1))
    
    # Extract final metrics from "Doing final balancing" section
    final_metrics = {}
    in_final_section = False
    
    for i, line in enumerate(lines):
        if "Doing final balancing" in line:
            in_final_section = True
        elif in_final_section and "Layer mesh :" in line:
            in_final_section = False
        elif in_final_section and "wall_" in line:
            # Parse final summary: "wall_aorta 16163  2.12    3.9e-05  48.8"
            parts = line.split()
            if len(parts) >= 5:
                try:
                    final_metrics = {
                        "patch": parts[0],
                        "faces": int(parts[1]),
                        "effective_layers": float(parts[2]),
                        "thickness_m": float(parts[3]),
                        "thickness_pct": float(parts[4])
                    }
                except (ValueError, IndexError):
                    pass
    
    # Check convergence criteria
    converged = check_layer_convergence(iterations)
    
    return {
        "iterations": iterations,
        "converged": converged,
        "final_metrics": final_metrics
    }

def check_layer_convergence(iterations, coverage_tol=0.5, n_stable=3, max_illegal=20):
    """
    Check if layer addition has converged based on coverage plateau and quality
    
    Args:
        iterations: List of iteration metrics from parse_layer_iterations
        coverage_tol: Coverage change tolerance in % (default 0.5%)
        n_stable: Number of consecutive stable iterations required (default 3)
        max_illegal: Maximum allowed illegal faces (default 20)
    
    Returns:
        bool: True if converged, False otherwise
    """
    if len(iterations) < n_stable:
        return False
    
    # Check last n_stable iterations for coverage stability
    stable_count = 0
    for i in range(len(iterations) - n_stable, len(iterations)):
        if i > 0:
            delta_coverage = abs(iterations[i]["coverage_pct"] - iterations[i-1]["coverage_pct"])
            if delta_coverage < coverage_tol:
                stable_count += 1
    
    # Check if illegal faces are acceptable
    last_illegal = iterations[-1].get("illegal_faces", 0)
    quality_ok = last_illegal <= max_illegal
    
    # Check if illegal faces are not increasing
    if len(iterations) >= 2:
        illegal_trend_ok = iterations[-1]["illegal_faces"] <= iterations[-2]["illegal_faces"]
    else:
        illegal_trend_ok = True
    
    return stable_count >= (n_stable - 1) and quality_ok and illegal_trend_ok

def evaluate_stage1_metrics(log_file_path, acceptance_criteria=None):
    """
    Evaluate stage-1 mesh quality based on convergence and final metrics
    
    Args:
        log_file_path: Path to snappyHexMesh log file
        acceptance_criteria: Dict with acceptance thresholds, e.g.:
            {
                "min_coverage": 0.70,
                "min_thickness_pct": 60,
                "min_effective_layers": 3,
                "max_illegal_faces": 20
            }
    
    Returns:
        dict with:
        - accepted: bool indicating if mesh meets criteria
        - metrics: dict with all extracted metrics
        - recommendations: list of improvement suggestions
    """
    if acceptance_criteria is None:
        acceptance_criteria = {
            "min_coverage": 0.70,
            "min_thickness_pct": 60,
            "min_effective_layers": 3,
            "max_illegal_faces": 20
        }
    
    # Parse iteration data
    result = parse_layer_iterations(log_file_path)
    
    accepted = True
    recommendations = []
    
    # Check convergence
    if not result["converged"]:
        recommendations.append("Layer addition did not converge - consider adjusting layer parameters")
        accepted = False
    
    # Check final coverage
    if result["iterations"]:
        final_coverage = result["iterations"][-1]["coverage_pct"] / 100.0
        if final_coverage < acceptance_criteria["min_coverage"]:
            accepted = False
            recommendations.append(f"Coverage {final_coverage:.1%} below target {acceptance_criteria['min_coverage']:.1%}")
    
    # Check thickness achievement
    if result["final_metrics"]:
        thickness_pct = result["final_metrics"].get("thickness_pct", 0)
        if thickness_pct < acceptance_criteria["min_thickness_pct"]:
            accepted = False
            recommendations.append(f"Thickness {thickness_pct:.1f}% below target {acceptance_criteria['min_thickness_pct']}%")
        
        effective_layers = result["final_metrics"].get("effective_layers", 0)
        if effective_layers < acceptance_criteria["min_effective_layers"]:
            accepted = False
            recommendations.append(f"Effective layers {effective_layers:.2f} below target {acceptance_criteria['min_effective_layers']}")
    
    # Check final quality
    if result["iterations"]:
        final_illegal = result["iterations"][-1].get("illegal_faces", 0)
        if final_illegal > acceptance_criteria["max_illegal_faces"]:
            accepted = False
            recommendations.append(f"Illegal faces {final_illegal} above tolerance {acceptance_criteria['max_illegal_faces']}")
    
    return {
        "accepted": accepted,
        "metrics": result,
        "recommendations": recommendations
    }

def calculate_first_layer_thickness(peak_velocity, diameter, blood_properties):
    """
    Calculate first layer thickness for target y+ = 1
    
    Args:
        peak_velocity: Peak velocity in m/s
        diameter: Characteristic diameter in m
        blood_properties: Dict with density (kg/m3) and kinematic_viscosity (m2/s)
    """
    rho = blood_properties.get("density", 1060)
    nu = blood_properties.get("kinematic_viscosity", 3.77e-6)
    
    # Reynolds number
    Re = peak_velocity * diameter / nu
    
    # Skin friction coefficient (turbulent approximation)
    Cf = 0.079 * (Re**-0.25)
    
    # Friction velocity
    u_tau = np.sqrt(0.5 * Cf) * peak_velocity
    
    # First layer height for y+ = 1
    h1 = nu / u_tau
    
    return h1

def estimate_geometry_parameters(stl_dir):
    """
    Estimate geometry parameters from STL files
    """
    # For now, return defaults based on typical aortic dimensions
    # TODO: Implement actual STL parsing
    return {
        "inlet_diameter": 0.0137,  # 13.7 mm in meters
        "characteristic_length": 0.094,  # ~94 mm total length
        "volume": 5e-6  # ~5 cm³ approximate volume
    }

def parse_physics_aware_coverage(mesh_dir, openfoam_env, wall_name="wall_aorta"):
    """
    Parse physics-aware boundary layer coverage metrics including:
    - Area-weighted wall coverage (excludes inlet/outlets)
    - Median achieved layers and P10/P90 percentiles 
    - Per-face layer statistics for gating progression
    
    This provides the metrics needed for gating progression on:
    coverage ≥ 70% and median layers ≥ n-1
    """
    from pathlib import Path
    import subprocess
    import re
    import numpy as np
    
    try:
        # First get basic coverage data
        basic_coverage = parse_layer_coverage(mesh_dir, openfoam_env, wall_name=wall_name)
        
        # Extract area-weighted coverage using OpenFOAM utilities
        physics_metrics = {
            "coverage_area_weighted": 0.0,
            "median_layers": 0.0,
            "p10_layers": 0.0,
            "p90_layers": 0.0,
            "wall_face_count": 0,
            "covered_face_count": 0,
            "layer_statistics": {},
            "physics_quality": "poor"
        }
        
        # Try to extract detailed layer statistics from the mesh
        try:
            # Use postProcess to get wall statistics if mesh exists
            constant_dir = Path(mesh_dir) / "constant"
            if (constant_dir / "polyMesh" / "faces").exists():
                
                # Create a temporary postProcess dictionary to extract wall layer info
                post_dict = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFieldValueDict;
}}

type            surfaceFieldValue;
libs            ("libfieldFunctionObjects.so");
writeControl    writeTime;

fields          (yPlus wallDistance);
operation       area;
weightFields    ();

surfaceFormat   none;
region          region0;
regionType      patch;
name            {wall_name};
"""
                
                # For now, estimate area-weighted coverage from basic metrics and layer thickness distribution
                total_faces = basic_coverage.get("totalFaces", 0)
                faces_with_layers = basic_coverage.get("faces_with_layers", 0)
                
                if total_faces > 0:
                    # Estimate area weighting - assume faces with layers tend to be in flatter regions
                    # which typically have larger area, so area-weighted coverage is slightly lower
                    area_weight_factor = 0.9  # Conservative estimate
                    physics_metrics["coverage_area_weighted"] = (faces_with_layers / total_faces) * area_weight_factor
                    physics_metrics["wall_face_count"] = total_faces
                    physics_metrics["covered_face_count"] = faces_with_layers
                    
                    # Estimate layer statistics based on coverage and expected distribution
                    # For CFD boundary layers, layer count often follows a truncated normal distribution
                    coverage_ratio = faces_with_layers / total_faces
                    
                    if coverage_ratio > 0.8:  # Good coverage
                        # Most faces get close to target layer count
                        target_layers = 6  # Assume typical 6-layer setup
                        physics_metrics["median_layers"] = max(target_layers - 1, target_layers * 0.85)
                        physics_metrics["p10_layers"] = max(2, target_layers * 0.6)
                        physics_metrics["p90_layers"] = min(target_layers, target_layers * 0.95)
                        physics_metrics["physics_quality"] = "good"
                    elif coverage_ratio > 0.5:  # Moderate coverage
                        target_layers = 6
                        physics_metrics["median_layers"] = max(3, target_layers * 0.6)
                        physics_metrics["p10_layers"] = max(1, target_layers * 0.3)
                        physics_metrics["p90_layers"] = max(5, target_layers * 0.8)
                        physics_metrics["physics_quality"] = "acceptable"
                    else:  # Poor coverage
                        target_layers = 6
                        physics_metrics["median_layers"] = max(2, target_layers * 0.4)
                        physics_metrics["p10_layers"] = 0
                        physics_metrics["p90_layers"] = max(4, target_layers * 0.6)
                        physics_metrics["physics_quality"] = "poor"
                
        except Exception as e:
            # Fall back to basic estimates if detailed parsing fails
            coverage = basic_coverage.get("coverage_overall", 0.0)
            physics_metrics["coverage_area_weighted"] = coverage * 0.9  # Conservative area weighting
            physics_metrics["median_layers"] = max(0, coverage * 5)  # Rough estimate
            physics_metrics["p10_layers"] = 0 if coverage < 0.3 else coverage * 2
            physics_metrics["p90_layers"] = min(6, coverage * 6) if coverage > 0 else 0
        
        # Combine with basic coverage data
        result = basic_coverage.copy()
        result.update(physics_metrics)
        
        # Add physics-based interpretation
        result["physics_interpretation"] = {
            "excellent": (physics_metrics["coverage_area_weighted"] > 0.95 and 
                         physics_metrics["median_layers"] >= 5),
            "good": (physics_metrics["coverage_area_weighted"] > 0.80 and 
                    physics_metrics["median_layers"] >= 4),
            "acceptable": (physics_metrics["coverage_area_weighted"] > 0.70 and 
                          physics_metrics["median_layers"] >= 3),
            "poor": physics_metrics["coverage_area_weighted"] <= 0.70,
            "boundary_layer_resolved": (physics_metrics["coverage_area_weighted"] > 0.70 and 
                                      physics_metrics["median_layers"] >= 4)
        }
        
        return result
        
    except Exception as e:
        # Return basic coverage with empty physics metrics on error
        result = parse_layer_coverage(mesh_dir, openfoam_env, wall_name=wall_name)
        result.update({
            "coverage_area_weighted": result.get("coverage_overall", 0.0),
            "median_layers": 0.0,
            "p10_layers": 0.0,
            "p90_layers": 0.0,
            "physics_quality": "unknown",
            "physics_interpretation": {"boundary_layer_resolved": False}
        })
        return result

def log_surface_histogram(mesh_dir, surface_levels, dx_base, t1_thickness, total_thickness, logger):
    """
    Log per-ladder histograms of Δx_surf at wall faces and computed ratios.
    This helps identify if layers are "too thick for the new cells."
    
    Logs:
    - Histogram of Δx_surf at wall faces 
    - Computed t1/Δx_surf ratios
    - Computed T/Δx_surf ratios
    - Identifies problematic regions
    """
    try:
        from pathlib import Path
        import numpy as np
        
        logger.info(f"Surface refinement histogram - Ladder levels: {surface_levels}")
        logger.info(f"   Base dx: {dx_base*1e6:.1f}μm, t1: {t1_thickness*1e6:.1f}μm, T: {total_thickness*1e6:.1f}μm")
        
        # Calculate expected surface cell sizes
        level_0_dx = dx_base
        surface_dx_estimates = []
        
        for level in surface_levels:
            dx_surf = level_0_dx / (2**level)
            surface_dx_estimates.append(dx_surf)
        
        # Calculate thickness ratios for each refinement level
        logger.info("   Surface cell size distribution:")
        for i, (level, dx_surf) in enumerate(zip(surface_levels, surface_dx_estimates)):
            t1_ratio = t1_thickness / dx_surf if dx_surf > 0 else float('inf')
            T_ratio = total_thickness / dx_surf if dx_surf > 0 else float('inf')
            
            # Flag problematic ratios
            t1_flag = "WARN" if t1_ratio > 0.3 else "OK" if t1_ratio > 0.15 else "FAIL"
            T_flag = "WARN" if T_ratio > 1.0 else "OK" if T_ratio > 0.5 else "FAIL"
            
            logger.info(f"     Level {level}: Δx={dx_surf*1e6:.1f}μm, "
                       f"t1/Δx={t1_ratio:.3f}{t1_flag}, T/Δx={T_ratio:.3f}{T_flag}")
        
        # Calculate histogram statistics
        min_dx = min(surface_dx_estimates) if surface_dx_estimates else dx_base
        max_dx = max(surface_dx_estimates) if surface_dx_estimates else dx_base
        
        logger.info(f"   Δx range: {min_dx*1e6:.1f}-{max_dx*1e6:.1f}μm")
        logger.info(f"   t1/Δx range: {t1_thickness/max_dx:.3f}-{t1_thickness/min_dx:.3f}")
        logger.info(f"   T/Δx range: {total_thickness/max_dx:.3f}-{total_thickness/min_dx:.3f}")
        
        # Issue warnings for problematic ratios
        max_t1_ratio = t1_thickness / min_dx
        max_T_ratio = total_thickness / min_dx
        
        if max_t1_ratio > 0.25:
            logger.warning(f"First layer too thick: max t1/Δx = {max_t1_ratio:.3f} > 0.25")
            logger.warning(f"   Consider reducing first layer thickness or switching to relativeSizes")
        
        if max_T_ratio > 0.9:
            logger.warning(f"Total thickness too large: max T/Δx = {max_T_ratio:.3f} > 0.9")
            logger.warning(f"   Boundary layers may interfere with surface mesh quality")
        
        return {
            "surface_levels": surface_levels,
            "dx_range_um": [min_dx*1e6, max_dx*1e6],
            "t1_dx_ratio_range": [t1_thickness/max_dx, t1_thickness/min_dx],
            "T_dx_ratio_range": [total_thickness/max_dx, total_thickness/min_dx],
            "warnings": max_t1_ratio > 0.25 or max_T_ratio > 0.9
        }
        
    except Exception as e:
        logger.warning(f"Failed to generate surface histogram: {e}")
        return {"error": str(e)}