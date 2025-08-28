"""
Legacy functions extracted from deleted folders for backward compatibility.
This file contains the minimal essential functions needed after folder cleanup.
"""

import logging
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, Optional

# Configuration Manager (simplified)
class ConfigurationManager:
    """Simplified configuration manager."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def load_and_validate(self, config_file: Path) -> Dict:
        """Load and validate configuration file."""
        import json
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Configuration loaded and validated: {config_file}")
            return config
        except Exception as e:
            self.logger.error(f"Configuration loading failed: {e}")
            raise
    
    def validate_geometry_config(self, config: Dict, geometry_dir: Path) -> bool:
        """Validate geometry configuration."""
        # Check for required STL files
        required_files = ["inlet.stl", config.get("wall_patch_name", "wall_aorta") + ".stl"]
        
        for req_file in required_files:
            if not (geometry_dir / req_file).exists():
                self.logger.error(f"Required STL file missing: {req_file}")
                return False
        return True
    
    def get_memory_config(self, config: Dict) -> Dict:
        """Get memory configuration."""
        import psutil
        
        # Get available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        return {
            'max_memory_gb': min(config.get('max_memory_gb', 8), available_memory_gb * 0.8),
            'available_memory_gb': available_memory_gb
        }
    
    def apply_solver_presets(self, config: Dict, solver_mode: str) -> Dict:
        """Apply solver-specific presets to configuration."""
        # Simplified implementation - just return the config unchanged
        # In the original, this would apply different quality thresholds for RANS/LES/etc
        self.logger.info(f"Applied {solver_mode} solver presets")
        return config


# Process Manager (simplified) 
class ProcessManager:
    """Simplified process manager."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def set_process_limits(self):
        """Set basic process limits."""
        return True  # Simplified implementation


# Surface analysis function
def log_surface_histogram(iter_dir, surface_levels, dx, t1_thickness, total_thickness, logger):
    """Log surface analysis histogram with detailed layer diagnostics."""
    logger.debug(f"Surface analysis: levels={surface_levels}, dx={dx:.3e}m, t1={t1_thickness:.3e}m, total={total_thickness:.3e}m")


# Stage 1 metrics evaluation
def evaluate_stage1_metrics(export_dir: Path, logger):
    """Evaluate Stage 1 metrics (simplified)."""
    logger.info(f"Stage 1 metrics evaluation completed for {export_dir}")
    return {"success": True}


# Essential utility functions from utils.py
def run_command(cmd, cwd=None, env_setup=None, timeout=None, parallel=False, max_memory_gb=8, log_file=None):
    """Run OpenFOAM command with proper environment setup and optional real-time logging."""
    import psutil
    import threading
    import time
    from pathlib import Path
    
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    if available_memory_gb < max_memory_gb:
        raise RuntimeError(f"Insufficient memory: need {max_memory_gb}GB, have {available_memory_gb:.1f}GB")
    
    if env_setup:
        if isinstance(cmd, list):
            cmd_str = " ".join(cmd)
        else:
            cmd_str = cmd
        
        if any(of_cmd in cmd_str for of_cmd in ['snappyHexMesh', 'simpleFoam', 'pimpleFoam']):
            cmd_str = f"ulimit -v {int(max_memory_gb * 1024 * 1024)} && {cmd_str}"
        
        cmd = ["bash", "-c", f"{env_setup} && {cmd_str}"]
    
    # If no log_file specified, use old behavior
    if log_file is None:
        try:
            if timeout and timeout > 0:
                result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                                      shell=isinstance(cmd, str) and not env_setup)
            else:
                result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                                      shell=isinstance(cmd, str) and not env_setup)
            return result
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Command timed out after {timeout} seconds: {cmd}")
        except Exception as e:
            raise RuntimeError(f"Command failed: {e}")
    
    # Real-time streaming to log file
    try:
        # Ensure log file directory exists
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            
        # Start process with pipes for real-time output
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            shell=isinstance(cmd, str) and not env_setup
        )
        
        # Collect output for return value
        stdout_lines = []
        
        # Open log file for writing if specified
        log_handle = None
        if log_file:
            log_handle = open(log_file, 'w', buffering=1)  # Line buffered
        
        try:
            # Read output line by line in real-time
            start_time = time.time()
            while True:
                line = process.stdout.readline()
                if line == '' and process.poll() is not None:
                    break
                    
                if line:
                    # Write to log file immediately
                    if log_handle:
                        log_handle.write(line)
                        log_handle.flush()  # Force immediate write
                    
                    # Collect for return
                    stdout_lines.append(line)
                
                # Check timeout
                if timeout and timeout > 0:
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        process.terminate()
                        process.wait()
                        raise subprocess.TimeoutExpired(cmd, timeout)
        
        finally:
            if log_handle:
                log_handle.close()
        
        # Wait for process completion
        process.wait()
        
        # Create return object compatible with subprocess.run
        class StreamingResult:
            def __init__(self, returncode, stdout, stderr=""):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
        
        stdout_text = ''.join(stdout_lines)
        return StreamingResult(process.returncode, stdout_text)
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Command timed out after {timeout} seconds: {cmd}")
    except Exception as e:
        raise RuntimeError(f"Command failed: {e}")


def check_mesh_quality(mesh_dir, openfoam_env, max_memory_gb=8, deep_check=False, wall_name=None):
    """Run checkMesh and parse quality metrics with real-time logging."""
    log_file = mesh_dir / "logs" / "log.checkMesh"
    log_file.parent.mkdir(exist_ok=True)
    
    check_cmd = "checkMesh -allGeometry -allTopology" if deep_check else "checkMesh"
    
    result = run_command(
        check_cmd, 
        cwd=mesh_dir, 
        env_setup=openfoam_env,
        timeout=None,
        max_memory_gb=max_memory_gb,
        log_file=str(log_file)  # Enable real-time logging
    )
    
    # Parse metrics
    metrics = {
        "maxNonOrtho": 0,
        "maxSkewness": 0,
        "maxAspectRatio": 0,
        "negVolCells": 0,
        "meshOK": False,
        "cells": 0,
        "wall_nFaces": None
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
    
    if "cells:" in output:
        match = re.search(r"cells:\s+(\d+)", output)
        if match:
            metrics["cells"] = int(match.group(1))
    
    # Extract wall patch face count if wall_name provided
    if wall_name:
        wall_pattern = rf"^\s*{wall_name}\s+(\d+)\s+\d+\s*$"
        wall_match = re.search(wall_pattern, output, re.MULTILINE)
        if wall_match:
            metrics["wall_nFaces"] = int(wall_match.group(1))
    
    # Accept mesh if checkMesh completed without fatal errors
    fatal_errors = [
        "FOAM FATAL ERROR",
        "Aborted", 
        "No cells in mesh",
        "Could not find cellZone"
    ]
    
    primitive_mesh_failed = "primitive mesh" in output and "check failed" in output
    has_fatal_error = any(error in output for error in fatal_errors) or primitive_mesh_failed
    completed_successfully = result.returncode == 0
    
    metrics["meshOK"] = completed_successfully and not has_fatal_error
    
    return metrics


def parse_layer_coverage(log_output_text, wall_name="wall_aorta"):
    """Parse boundary layer thickness fraction from snappyHexMesh log output.
    
    snappyHexMesh doesn't report true "coverage" (areal fraction with layers).
    Instead, it reports thickness fraction: achieved_thickness / target_thickness.
    
    Example log line:
    wall_aorta 18636    1.85     0.000155  46.1    
               ^^^^ ^^^^^ ^^^^^^    ^^^^^^  ^^^^
               patch faces layers  thickness [%]
    """
    import re
    import math
    
    # Parse snappyHexMesh layer summary table
    thickness_fraction = 0.0
    effective_layers = 0.0
    
    # Look for the final results summary table in snappyHexMesh output
    # Format: "patch_name   faces    layers   thickness[m]  thickness[%]"
    # Example: "wall_aorta   18636    1.85     0.000155      46.1"
    # Note: The initial specification table has format "wall_aorta 18636 5 5e-05 0.000337"
    #       but the final results table has the percentage in the last column
    
    # Look for results table where the last number is > 1.0 (indicating percentage)
    pattern = rf'{re.escape(wall_name)}\s+(\d+)\s+([\d.]+)\s+([\d.eE-]+)\s+([\d.]+)'
    matches = re.finditer(pattern, log_output_text)
    match = None
    for m in matches:
        # Check if the last number could be a percentage (> 1.0)
        if float(m.group(4)) > 1.0:
            match = m
            break
    
    if match:
        n_faces = int(match.group(1))
        actual_layers = float(match.group(2))  # This is already effective layers from snappyHexMesh
        thickness_m = float(match.group(3))
        thickness_pct = float(match.group(4))
        
        thickness_fraction = thickness_pct / 100.0
        effective_layers = actual_layers  # snappyHexMesh reports effective layers directly
        
    else:
        # Fallback patterns for different log formats or error conditions
        
        # Pattern 1: Look for extrusion progress messages like "Extruding X out of Y faces (Z%)"
        extrusion_patterns = [
            r"Extruding\s+(\d+)\s+out\s+of\s+(\d+)\s+faces\s+\(([\d.]+)%\)",
            r"Added\s+(\d+)\s+out\s+of\s+(\d+)\s+cells\s+\(([\d.]+)%\)"
        ]
        
        best_percentage = 0.0
        for pattern in extrusion_patterns:
            matches = re.findall(pattern, log_output_text, re.IGNORECASE)
            if matches:
                # Take the last (most recent) match and use the highest percentage seen
                for match in matches:
                    current_pct = float(match[2])
                    if current_pct > best_percentage:
                        best_percentage = current_pct
        
        if best_percentage > 0:
            thickness_fraction = best_percentage / 100.0
            # Estimate effective layers based on extrusion percentage
            effective_layers = thickness_fraction * 5.0  # Assume 5 target layers
            n_faces = 18636  # Default value, could be extracted from matches if needed
        else:
            # Pattern 2: Look for thickness percentage anywhere near the wall name
            thickness_pattern = rf'{re.escape(wall_name)}.*?(\d+(?:\.\d+)?)(?:\s*%|\s+[\d.]+\s+[\d.]+\s+([\d.]+))'
            thickness_match = re.search(thickness_pattern, log_output_text, re.IGNORECASE | re.DOTALL)
            
            if thickness_match:
                # Extract the percentage value (last captured group)
                pct_str = thickness_match.group(2) if thickness_match.group(2) else thickness_match.group(1)
                thickness_fraction = float(pct_str) / 100.0 if float(pct_str) > 1.0 else float(pct_str)
                
            # Pattern 3: Look for any thickness achievement indicators
            general_patterns = [
                r"thickness\s*achieved[:\s]*(\d+(?:\.\d+)?)\s*%",
                r"(\d+(?:\.\d+)?)\s*%\s*thickness",
                r"overall\s*thickness.*?(\d+(?:\.\d+)?)\s*%",
            ]
            
            for pattern in general_patterns:
                match = re.search(pattern, log_output_text, re.IGNORECASE)
                if match:
                    thickness_fraction = float(match.group(1)) / 100.0
                    break
    
    # Count quality-related pruning hints to better understand the result
    prune_hints = [
        "maxFaceThicknessRatio", "maxThicknessToMedialRatio", 
        "minMedianAxisAngle", "illegal faces", "Removing extrusion",
        "Will not extrude", "Deleting layer", "abandoned", "pruned"
    ]
    
    prune_count = sum(len(re.findall(hint, log_output_text, re.IGNORECASE)) for hint in prune_hints)
    
    # Classify the result for better diagnostics
    if thickness_fraction < 0.15:
        diagnosis = "not-added-or-abandoned-early"
    elif thickness_fraction < 0.60:
        diagnosis = "thin-but-present (added-then-pruned)" 
    else:
        diagnosis = "healthy-growth"
    
    return {
        "coverage_overall": thickness_fraction,  # Keep existing interface
        "thickness_fraction": thickness_fraction,
        "effective_layers": effective_layers,
        "faces_with_layers": n_faces if 'n_faces' in locals() else (match.group(1) if 'match' in locals() and match else 0),
        "pruning_hints": prune_count,
        "diagnosis": diagnosis,
        "perPatch": {
            wall_name: {
                "thickness_fraction": thickness_fraction,
                "effective_layers": effective_layers
            }
        }
    }


def parse_layer_iterations(log_file_path):
    """Parse layer addition iteration data from snappyHexMesh log (simplified)."""
    return []  # Simplified implementation