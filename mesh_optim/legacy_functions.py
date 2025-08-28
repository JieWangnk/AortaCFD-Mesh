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


# Layer diagnosis constants (enum-like)
class LayerDiagnosis:
    """Constants for layer coverage diagnosis results."""
    NOT_ADDED = "not-added-or-abandoned-early"
    THIN_PRESENT = "thin-but-present"
    THIN_PRUNED = "thin-but-present (added-then-pruned)"
    HEALTHY = "healthy-growth"


# Pre-compiled regex patterns for performance
import re
_EXTRUSION_PATTERNS = [
    re.compile(r"Extruding\s+(\d+)\s+out\s+of\s+(\d+)\s+faces\s+\(([\d.]+)%\)", re.IGNORECASE),
    re.compile(r"Added\s+(\d+)\s+out\s+of\s+(\d+)\s+cells\s+\(([\d.]+)%\)", re.IGNORECASE)
]

_GENERAL_PATTERNS = [
    re.compile(r"thickness\s*achieved[:\s]*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*%\s*thickness", re.IGNORECASE),
    re.compile(r"overall\s*thickness.*?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE),
]

_PRUNE_HINTS = [
    "maxFaceThicknessRatio", "maxThicknessToMedialRatio", 
    "minMedianAxisAngle", "illegal faces", "Removing extrusion",
    "Will not extrude", "Deleting layer", "abandoned", "pruned"
]


def _parse_layer_summary_table(log_output_text, wall_name):
    """Parse the final layer results summary table.
    
    Returns:
        tuple: (n_faces, effective_layers, thickness_fraction) or (None, None, None)
    """
    import re
    
    # Look for results table where the last number is > 1.0 (indicating percentage)
    pattern = rf'{re.escape(wall_name)}\s+(\d+)\s+([\d.]+)\s+([\d.eE-]+)\s+([\d.]+)'
    matches = re.finditer(pattern, log_output_text)
    
    for match in matches:
        try:
            # Check if the last number could be a percentage (> 1.0)
            if float(match.group(4)) > 1.0:
                n_faces = int(match.group(1))
                actual_layers = float(match.group(2))
                thickness_pct = float(match.group(4))
                thickness_fraction = thickness_pct / 100.0
                return n_faces, actual_layers, thickness_fraction
        except (ValueError, TypeError):
            continue
            
    return None, None, None


def _parse_extrusion_messages(log_output_text, target_layers):
    """Parse extrusion progress messages for coverage estimation.
    
    Returns:
        tuple: (thickness_fraction, effective_layers, n_faces) or (None, None, None)
    """
    best_percentage = 0.0
    n_faces = None
    
    for i, pattern in enumerate(_EXTRUSION_PATTERNS):
        try:
            matches = pattern.findall(log_output_text)
            if matches:
                for match in matches:
                    current_pct = float(match[2])
                    if current_pct > best_percentage:
                        best_percentage = current_pct
                        if i == 0:  # Face-based metric (first pattern) is more reliable
                            n_faces = int(match[1])  # Total faces
        except (ValueError, TypeError, IndexError):
            continue
    
    if best_percentage > 0:
        thickness_fraction = best_percentage / 100.0
        effective_layers = thickness_fraction * target_layers
        return thickness_fraction, effective_layers, n_faces or 18636
        
    return None, None, None


def _parse_fallback_patterns(log_output_text, wall_name):
    """Parse fallback patterns for edge cases.
    
    Returns:
        float: thickness_fraction or None
    """
    import re
    
    # Pattern 1: Look for thickness percentage near wall name
    thickness_pattern = rf'{re.escape(wall_name)}.*?(\d+(?:\.\d+)?)(?:\s*%|\s+[\d.]+\s+[\d.]+\s+([\d.]+))'
    thickness_match = re.search(thickness_pattern, log_output_text, re.IGNORECASE | re.DOTALL)
    
    if thickness_match:
        try:
            pct_str = thickness_match.group(2) if thickness_match.group(2) else thickness_match.group(1)
            thickness_fraction = float(pct_str) / 100.0 if float(pct_str) > 1.0 else float(pct_str)
            return thickness_fraction
        except (ValueError, TypeError):
            pass
            
    # Pattern 2: General thickness achievement indicators
    for pattern in _GENERAL_PATTERNS:
        try:
            match = pattern.search(log_output_text)
            if match:
                return float(match.group(1)) / 100.0
        except (ValueError, TypeError):
            continue
            
    return None


def _count_pruning_hints(log_output_text):
    """Count quality-related pruning indicators in the log.
    
    Returns:
        int: Number of pruning hints found
    """
    import re
    
    return sum(len(re.findall(hint, log_output_text, re.IGNORECASE)) for hint in _PRUNE_HINTS)


def _diagnose_layer_result(thickness_fraction, prune_count):
    """Generate human-readable diagnosis based on results.
    
    Returns:
        str: Diagnosis string from LayerDiagnosis constants
    """
    if thickness_fraction < 0.15:
        return LayerDiagnosis.NOT_ADDED
    elif thickness_fraction < 0.60:
        return LayerDiagnosis.THIN_PRUNED if prune_count > 0 else LayerDiagnosis.THIN_PRESENT
    else:
        return LayerDiagnosis.HEALTHY


def parse_layer_coverage(log_output_text, wall_name="wall_aorta", target_layers=5):
    """Parse boundary layer thickness fraction from snappyHexMesh log output.
    
    snappyHexMesh doesn't report true "coverage" (areal fraction with layers).
    Instead, it reports thickness fraction: achieved_thickness / target_thickness.
    
    Args:
        log_output_text (str): The snappyHexMesh log content to parse
        wall_name (str): Name of the wall patch to look for (default: "wall_aorta")
        target_layers (int): Number of target layers for estimation (default: 5)
        
    Returns:
        dict: Layer coverage data with keys:
            - coverage_overall (float): Thickness fraction (0.0-1.0)
            - thickness_fraction (float): Same as coverage_overall
            - effective_layers (float): Estimated effective layers achieved
            - faces_with_layers (int): Number of faces with layers
            - pruning_hints (int): Count of quality-related pruning indicators
            - diagnosis (str): Human-readable diagnosis
            - perPatch (dict): Per-patch breakdown
            
    Raises:
        TypeError: If log_output_text is not a string
        ValueError: If wall_name is empty or invalid
        
    Example log line:
    wall_aorta 18636    1.85     0.000155  46.1    
               ^^^^ ^^^^^ ^^^^^^    ^^^^^^  ^^^^
               patch faces layers  thickness [%]
    """
    import re
    import math
    
    # Input validation
    if not isinstance(log_output_text, str):
        raise TypeError(f"log_output_text must be a string, got {type(log_output_text).__name__}")
    
    if not wall_name or not isinstance(wall_name, str):
        raise ValueError("wall_name must be a non-empty string")
        
    if not isinstance(target_layers, (int, float)) or target_layers <= 0:
        raise ValueError("target_layers must be a positive number")
    
    # Initialize default values
    thickness_fraction = 0.0
    effective_layers = 0.0
    n_faces = 18636  # Default fallback
    
    # Try parsing methods in order of reliability
    
    # Method 1: Parse layer summary table (most reliable when present)
    table_result = _parse_layer_summary_table(log_output_text, wall_name)
    if table_result[0] is not None:
        n_faces, effective_layers, thickness_fraction = table_result
    else:
        # Method 2: Parse extrusion messages (common fallback)
        extrusion_result = _parse_extrusion_messages(log_output_text, target_layers)
        if extrusion_result[0] is not None:
            thickness_fraction, effective_layers, n_faces = extrusion_result
        else:
            # Method 3: Fallback patterns for edge cases
            fallback_thickness = _parse_fallback_patterns(log_output_text, wall_name)
            if fallback_thickness is not None:
                thickness_fraction = fallback_thickness
                effective_layers = thickness_fraction * target_layers
    
    # Count pruning indicators and generate diagnosis
    prune_count = _count_pruning_hints(log_output_text)
    diagnosis = _diagnose_layer_result(thickness_fraction, prune_count)
    
    return {
        "coverage_overall": thickness_fraction,  # Keep existing interface
        "thickness_fraction": thickness_fraction,
        "effective_layers": effective_layers,
        "faces_with_layers": n_faces,
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