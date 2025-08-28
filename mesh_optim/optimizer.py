"""Optimization loop module for Stage 1 mesh optimization.

Handles the iterative mesh optimization process, quality assessment,
parameter adaptation, and convergence detection.
"""

import csv
import math
import shutil
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging
import numpy as np

from .constants import (
    MICRO_TRIALS_PER_ITERATION,
    COVERAGE_ACCEPTABLE_THRESHOLD,
    EXPANSION_RATIO_MIN,
    EXPANSION_RATIO_REDUCTION,
    RELATIVE_FIRST_LAYER_MIN,
    RELATIVE_FIRST_LAYER_REDUCTION,
    FIRST_LAYER_MIN,
    FIRST_LAYER_REDUCTION_FACTOR,
    COVERAGE_PLATEAU_THRESHOLD,
    COVERAGE_PLATEAU_ITERATIONS
)
from .legacy_functions import (
    run_command,
    check_mesh_quality,
    parse_layer_coverage,
    parse_layer_iterations,
    log_surface_histogram
)


class Stage1Optimizer:
    """Handles the Stage 1 mesh optimization iteration loop."""
    
    def __init__(self, config: dict, wall_name: str, max_memory_gb: float, logger: logging.Logger):
        self.config = config
        self.wall_name = wall_name
        self.max_memory_gb = max_memory_gb
        self.logger = logger
        
        # Cache config sections
        self.stage1 = config.get("STAGE1", {})
        self.snappy = config.get("SNAPPY", {})
        self.layers = config.get("LAYERS", {})
        
        # Optimization state
        self.current_iteration = 0
        self.surface_levels = [1, 1]  # Will be updated during optimization
        
        # Targets from config - simple dataclass to avoid circular imports
        from dataclasses import dataclass
        
        @dataclass
        class Targets:
            max_nonortho: float
            max_skewness: float  
            min_layer_cov: float
            
        acceptance = config.get("acceptance_criteria", {})
        self.targets = Targets(
            max_nonortho=float(acceptance.get("maxNonOrtho", 65)),
            max_skewness=float(acceptance.get("maxSkewness", 4.0)),
            min_layer_cov=float(acceptance.get("min_layer_coverage", 0.65)),
        )

    def run_mesh_generation(self, iter_dir: Path, force_full_remesh: bool = True) -> Tuple[Dict, Dict, Dict]:
        """Run mesh generation with enhanced error handling.
        
        Returns:
            Tuple of (snap_metrics, layer_metrics, layer_coverage)
        """
        env = self.config["openfoam_env_path"]
        logs = iter_dir / "logs"
        logs.mkdir(exist_ok=True)
        
        if not force_full_remesh:
            self.logger.info("Layers-only optimization possible, but running full remesh for robustness")
        
        # Setup parallel decomposition if configured
        pre, post = self._maybe_parallel(iter_dir)
        n_procs = int(self.stage1.get("n_processors", 1))
        
        try:
            # Initial mesh generation (always serial) with real-time logging
            for cmd, log_name in [(["blockMesh"], "log.blockMesh"), (["surfaceFeatures"], "log.surfaceFeatures")]:
                self.logger.info(f"Running: {' '.join(cmd)}")
                try:
                    log_path = logs / log_name
                    res = run_command(cmd, cwd=iter_dir, env_setup=env, timeout=None, 
                                    max_memory_gb=self.max_memory_gb, log_file=str(log_path))
                    
                    # Check for critical failure indicators
                    output_text = res.stdout + res.stderr
                    if "FOAM FATAL ERROR" in output_text or "Aborted" in output_text:
                        error_msg = f"Critical error in {cmd[0]}: Check {log_name} for details"
                        self.logger.error(error_msg)
                        raise RuntimeError(error_msg)
                        
                except Exception as e:
                    self.logger.error(f"Command failed: {' '.join(cmd)} | {e}")
                    # Return error state immediately for critical preprocessing failures
                    if cmd[0] in ["blockMesh", "surfaceFeatures"]:
                        error_metrics = {"meshOK": False, "cells": 0, "error": str(e)}
                        return error_metrics, error_metrics, error_metrics
            
            # Domain decomposition for parallel run with real-time logging
            if pre:
                self.logger.info(f"Running: {' '.join(pre[0])}")
                try:
                    log_path = logs / pre[1]
                    res = run_command(pre[0], cwd=iter_dir, env_setup=env, timeout=None, 
                                    max_memory_gb=self.max_memory_gb, log_file=str(log_path))
                except Exception as e:
                    self.logger.error(f"Decomposition failed: {e}")
            
            # MESH WITHOUT LAYERS phase
            snap_metrics = self._run_mesh_without_layers(iter_dir, logs, env, n_procs)
            
            if not snap_metrics.get("meshOK", False):
                self.logger.warning("Mesh generation (no layers) failed; skipping layer addition")
                empty_coverage = {"coverage_overall": 0.0, "error": "No base mesh"}
                return snap_metrics, snap_metrics, empty_coverage
            
            # LAYERS phase
            layer_metrics, layer_coverage = self._run_layer_addition(iter_dir, logs, env, n_procs)
            
            return snap_metrics, layer_metrics, layer_coverage
            
        except Exception as e:
            self.logger.error(f"Mesh generation failed with exception: {e}")
            error_metrics = {"meshOK": False, "cells": 0, "error": str(e)}
            error_coverage = {"coverage_overall": 0.0, "error": str(e)}
            return error_metrics, error_metrics, error_coverage

    def _run_mesh_without_layers(self, iter_dir: Path, logs: Path, env: str, n_procs: int) -> Dict:
        """Run castellation and snapping phases."""
        # Copy appropriate dictionary
        shutil.copy2(iter_dir / "system" / "snappyHexMeshDict.noLayer", 
                     iter_dir / "system" / "snappyHexMeshDict")
        
        # Build command
        if n_procs > 1:
            snappy_cmd = ["mpirun", "-np", str(n_procs), "snappyHexMesh", "-overwrite", "-parallel"]
        else:
            snappy_cmd = ["snappyHexMesh", "-overwrite"]
        
        self.logger.info(f"Running: {' '.join(snappy_cmd)} (mesh without layers)")
        try:
            log_path = logs / "log.snappy.no_layers"
            res = run_command(snappy_cmd, cwd=iter_dir, env_setup=env, timeout=None, 
                            max_memory_gb=self.max_memory_gb, log_file=str(log_path))
            output_text = res.stdout + res.stderr
            
            # Check for common snappyHexMesh failure patterns
            if any(pattern in output_text for pattern in [
                "FOAM FATAL ERROR", "Aborted", "Could not find cellZone", 
                "locationInMesh", "No cells in mesh", "Maximum number of cells"
            ]):
                error_msg = "snappyHexMesh failed with critical error - check log.snappy.no_layers"
                self.logger.error(error_msg)
                return {"meshOK": False, "cells": 0, "error": error_msg}
                
        except Exception as e:
            self.logger.error(f"snappyHexMesh failed with critical error - check log.snappy.no_layers")
            self.logger.error(f"Error details: {e}")
            # Write error marker file for clear visibility
            error_marker = iter_dir / "MESH_GENERATION_FAILED"
            error_marker.write_text(f"snappyHexMesh failed: {str(e)}\nCheck logs/log.snappy.no_layers for details")
            return {"meshOK": False, "cells": 0, "error": f"snappyHexMesh error: {str(e)}"}
        
        # Validate mesh generation
        wall_faces_total = self._sum_wall_faces_from_processors(iter_dir)
        if wall_faces_total <= 0:
            self.logger.warning("No wall faces found - mesh generation may have failed")
            return {"meshOK": False, "cells": 0, "error": "No wall faces detected"}
        
        self.logger.debug(f"Mesh generation verified: {wall_faces_total} wall faces")
        
        # Quality check
        snap_metrics = self._check_mesh_quality(iter_dir, env, n_procs)
        return snap_metrics

    def _run_layer_addition(self, iter_dir: Path, logs: Path, env: str, n_procs: int) -> Tuple[Dict, Dict]:
        """Run boundary layer addition phase."""
        # Clean up refinement level files from previous snappyHexMesh runs
        # to prevent "Number of cells in mesh does not equal size of cellLevel" errors
        self._cleanup_refinement_levels(iter_dir)
        
        # Copy layers dictionary
        shutil.copy2(iter_dir / "system" / "snappyHexMeshDict.layers",
                     iter_dir / "system" / "snappyHexMeshDict")
        
        # Build command
        if n_procs > 1:
            layers_cmd = ["mpirun", "-np", str(n_procs), "snappyHexMesh", "-overwrite", "-parallel"]
        else:
            layers_cmd = ["snappyHexMesh", "-overwrite"]
        
        self.logger.info(f"Running: {' '.join(layers_cmd)} (adding layers)")
        
        try:
            log_path = logs / "log.snappy.layers"
            res = run_command(layers_cmd, cwd=iter_dir, env_setup=env, timeout=None, 
                            max_memory_gb=self.max_memory_gb, log_file=str(log_path))
            
            # Parse layer coverage from log file (snappyHexMesh writes summary to log, not stdout)
            if log_path.exists():
                log_content = log_path.read_text()
                layer_coverage = parse_layer_coverage(log_content, self.wall_name)
                # Parse layer iteration details from log file too
                layer_iterations = parse_layer_iterations(log_content)
            else:
                # Fallback to stdout/stderr if log file doesn't exist
                output_text = res.stdout + res.stderr
                layer_coverage = parse_layer_coverage(output_text, self.wall_name)
                layer_iterations = parse_layer_iterations(output_text)
            
            layer_coverage.update(layer_iterations)
            
        except Exception as e:
            self.logger.error(f"Layer addition failed: {e}")
            error_coverage = {"coverage_overall": 0.0, "error": str(e)}
            error_metrics = {"meshOK": False, "cells": 0, "error": str(e)}
            return error_metrics, error_coverage
        
        # Quality check after layer addition
        layer_metrics = self._check_mesh_quality(iter_dir, env, n_procs)
        
        return layer_metrics, layer_coverage

    def _cleanup_refinement_levels(self, iter_dir: Path) -> None:
        """Clean up refinement level files from previous snappyHexMesh runs.
        
        This prevents FOAM FATAL ERROR: "Number of cells in mesh does not equal size of cellLevel"
        which occurs when snappyHexMesh tries to continue from inconsistent refinement data.
        """
        refinement_patterns = ["cellLevel", "pointLevel", "refinementHistory"]
        
        for pattern in refinement_patterns:
            # Clean up from processor directories
            for proc_dir in iter_dir.glob("processor*"):
                for level_file in proc_dir.rglob(pattern):
                    try:
                        level_file.unlink()
                        self.logger.debug(f"Removed refinement file: {level_file}")
                    except Exception as e:
                        self.logger.debug(f"Could not remove {level_file}: {e}")
            
            # Clean up from main mesh directory
            for level_file in iter_dir.rglob(pattern):
                if not any(part.startswith("processor") for part in level_file.parts):
                    try:
                        level_file.unlink()
                        self.logger.debug(f"Removed refinement file: {level_file}")
                    except Exception as e:
                        self.logger.debug(f"Could not remove {level_file}: {e}")

    def _check_mesh_quality(self, iter_dir: Path, env: str, n_procs: int) -> Dict:
        """Check mesh quality with parallel/serial fallback."""
        if n_procs > 1:
            try:
                # Parallel checkMesh
                check_cmd = ["mpirun", "-np", str(n_procs), "checkMesh", "-parallel"]
                res = run_command(check_cmd, cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                output_text = res.stdout + res.stderr
                metrics = self._parse_parallel_checkmesh(output_text)
            except Exception as e:
                self.logger.warning(f"Parallel checkMesh failed: {e}")
                # Fallback to serial after reconstruction
                try:
                    self.logger.info("Reconstructing mesh for serial checkMesh")
                    res = run_command(["reconstructPar", "-latestTime"], cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                    metrics = check_mesh_quality(iter_dir, env, wall_name=self.wall_name)
                except Exception as serial_e:
                    self.logger.error(f"Serial checkMesh fallback also failed: {serial_e}")
                    metrics = {"meshOK": False, "cells": 0, "error": str(serial_e)}
        else:
            metrics = check_mesh_quality(iter_dir, env, wall_name=self.wall_name)
        
        return metrics

    def meets_quality_constraints(self, snap_m: Dict, layer_m: Dict, layer_cov: Dict) -> Tuple[bool, List[str]]:
        """Check if mesh meets quality constraints."""
        failures = []
        
        # Check mesh validity
        if not snap_m.get("meshOK", False) or not layer_m.get("meshOK", False):
            failures.append("Mesh failed OpenFOAM checkMesh")
        
        # Check orthogonality
        max_nonortho = max(snap_m.get("maxNonOrtho", 0), layer_m.get("maxNonOrtho", 0))
        if max_nonortho > self.targets.max_nonortho:
            failures.append(f"Non-orthogonality {max_nonortho:.1f}° > {self.targets.max_nonortho}°")
        
        # Check skewness
        max_skewness = max(snap_m.get("maxSkewness", 0), layer_m.get("maxSkewness", 0))
        if max_skewness > self.targets.max_skewness:
            failures.append(f"Skewness {max_skewness:.2f} > {self.targets.max_skewness}")
        
        # Check layer coverage
        coverage = layer_cov.get("coverage_overall", 0.0)
        if coverage < self.targets.min_layer_cov:
            failures.append(f"Layer coverage {coverage:.1%} < {self.targets.min_layer_cov:.1%}")
        
        return len(failures) == 0, failures

    def run_micro_layer_optimization(self, iter_dir: Path, outlet_names: List[str],
                                   internal_point: np.ndarray, dx_base: float,
                                   base_coverage: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Execute micro-loop layer-only optimization for improved coverage."""
        from pathlib import Path
        
        max_retries = int(self.stage1.get("micro_layer_retries", 0))
        if max_retries <= 0:
            return None, None

        # Ensure iter_dir is a Path object
        iter_dir = Path(iter_dir)
        env = self.config["openfoam_env_path"]
        logs = iter_dir / "logs"
        n_procs = int(self.stage1.get("n_processors", 1))

        best_cov = float(base_coverage.get("coverage_overall", 0.0))
        best_metrics = None
        best_cov_obj = base_coverage

        self.logger.info(f"🔁 MICRO-LOOP: up to {max_retries} layers-only retries")
        
        for t in range(1, max_retries + 1):
            # Progressive parameter tuning strategy
            L = self.layers
            is_rel = bool(L.get("relativeSizes", False))
            
            if not is_rel:
                # Switch to relative to tie thickness to local Δx_surf
                L["relativeSizes"] = True
                L["firstLayerThickness"] = max(0.15, float(L.get("firstLayerThickness", 0.20)))
                L["finalLayerThickness"] = min(0.85, float(L.get("finalLayerThickness", 0.75)))
                self.logger.info("  → switched to relativeSizes=true (t1_rel≈0.20, T_rel≈0.75)")
            else:
                # Reduce t1_rel progressively; keep floor at 0.15
                L["firstLayerThickness"] = max(0.15, float(L.get("firstLayerThickness", 0.20)) * 0.90)
                # Cap total thickness to ≤0.85 Δx_surf
                L["finalLayerThickness"] = min(0.85, float(L.get("finalLayerThickness", 0.75)))

            # Lower ER slightly for stability
            L["expansionRatio"] = max(1.10, min(1.18, float(L.get("expansionRatio", 1.15)) - 0.01))

            # Gentle nGrow (0→1→2 max)
            L["nGrow"] = min(int(L.get("nGrow", 0)) + 1, 2)
            # More smoothing/relaxation
            L["nSmoothSurfaceNormals"] = min(12, int(L.get("nSmoothSurfaceNormals", 5)) + 2)
            L["nRelaxIter"] = min(12, int(L.get("nRelaxIter", 5)) + 2)
            L["nLayerIter"] = max(60, int(L.get("nLayerIter", 50)))

            self.logger.info(f"  MICRO try {t}: t1_rel={L.get('firstLayerThickness', 'abs')}, "
                           f"T_rel={L.get('finalLayerThickness', 'abs')}, ER={L['expansionRatio']}, nGrow={L['nGrow']}")

            # Generate updated layer dictionary
            from .openfoam_dicts import OpenFOAMDictGenerator
            
            # Debug: Check parameter types
            self.logger.debug(f"  MICRO debug - iter_dir type: {type(iter_dir)}, value: {iter_dir}")
            self.logger.debug(f"  MICRO debug - outlet_names type: {type(outlet_names)}")
            self.logger.debug(f"  MICRO debug - internal_point type: {type(internal_point)}")
            self.logger.debug(f"  MICRO debug - dx_base type: {type(dx_base)}, value: {dx_base}")
            
            # Ensure all parameters are correct types
            iter_dir = Path(iter_dir) if not isinstance(iter_dir, Path) else iter_dir
            dx_base = float(dx_base) if isinstance(dx_base, str) else dx_base
            
            dict_generator = OpenFOAMDictGenerator(self.config, self.wall_name, self.logger)
            dict_generator._generate_snappy_layers_dict(iter_dir, outlet_names, internal_point, dx_base)

            # Run layers-only pass
            metrics, cov = self._run_layers_only_pass(iter_dir, logs, env, n_procs)
            
            cov_now = float(cov.get("coverage_overall", 0.0))
            self.logger.info(f"  MICRO try {t} result: coverage={cov_now*100:.1f}%, "
                           f"skew={metrics.get('maxSkewness', 0):.2f}, nonOrtho={metrics.get('maxNonOrtho', 0):.1f}")

            if cov_now > best_cov:
                best_cov = cov_now
                best_metrics = metrics
                best_cov_obj = cov

            # Stop early if we crossed an OK band (e.g. 60%+)
            if cov_now >= max(0.60, self.targets.min_layer_cov - 0.05):
                self.logger.info("  MICRO loop early-exit: coverage reached target band")
                break

        if best_metrics and best_cov_obj and best_cov > float(base_coverage.get("coverage_overall", 0.0)):
            self.logger.info(f"Micro-loop improved coverage: {float(base_coverage.get('coverage_overall', 0.0))*100:.1f}% → {best_cov*100:.1f}%")
            return best_metrics, best_cov_obj
        
        return None, None

    def _run_layers_only_pass(self, iter_dir: Path, logs: Path, env: str, n_procs: int) -> Tuple[Dict, Dict]:
        """Run a single layers-only snappyHexMesh pass."""
        # Ensure paths are Path objects
        from pathlib import Path
        iter_dir = Path(iter_dir)
        logs = Path(logs)
        
        # Copy layers dict to main dict
        shutil.copy2(iter_dir / "system" / "snappyHexMeshDict.layers", 
                    iter_dir / "system" / "snappyHexMeshDict")
        
        # Run snappyHexMesh (layers only)
        cmd = ["snappyHexMesh", "-overwrite"]
        if n_procs > 1:
            cmd = ["mpirun", "-np", str(n_procs)] + cmd + ["-parallel"]
        
        log_path = logs / "log.snappy.layers.micro"
        res = run_command(cmd, cwd=iter_dir, env_setup=env, timeout=None, 
                        max_memory_gb=self.max_memory_gb, log_file=str(log_path))

        # Parse layer coverage from log file (snappyHexMesh writes summary to log, not stdout)
        if log_path.exists():
            log_content = log_path.read_text()
            layer_coverage = parse_layer_coverage(log_content, self.wall_name)
        else:
            # Fallback to stdout/stderr if log file doesn't exist
            output_text = res.stdout + res.stderr
            layer_coverage = parse_layer_coverage(output_text, self.wall_name)
        # Parse layer iteration details from the same source as coverage
        if log_path.exists():
            layer_iterations = parse_layer_iterations(log_content)
        else:
            output_text = res.stdout + res.stderr
            layer_iterations = parse_layer_iterations(output_text)
        layer_coverage.update(layer_iterations)

        # Check mesh quality
        layer_metrics = self._check_mesh_quality(iter_dir, env, n_procs)

        return layer_metrics, layer_coverage

    def get_cell_count(self, layer_m: Dict, snap_m: Dict) -> int:
        """Get total cell count from mesh metrics."""
        return layer_m.get("cells", snap_m.get("cells", 0))

    def apply_parameter_adaptations(self, coverage: float, trial: int, max_trials: int):
        """Apply quick parameter back-offs for improved coverage."""
        if trial >= max_trials:
            return  # No more trials
        
        self.logger.info(f"Applying parameter adaptations for trial {trial+1}")
        
        # Apply relative sizing if not already enabled and coverage is poor
        if coverage < 0.40 and not self.layers.get("relativeSizes", False):
            self.logger.info("Enabling relativeSizes for complex geometry adaptation")
            self.layers["relativeSizes"] = True
            self.layers["firstLayerThickness"] = 0.2  # Start with 20% of local cell size
        
        # Reduce expansion ratio slightly
        current_er = self.layers.get("expansionRatio", 1.2)
        if current_er > EXPANSION_RATIO_MIN:
            new_er = max(EXPANSION_RATIO_MIN, current_er - EXPANSION_RATIO_REDUCTION)
            self.layers["expansionRatio"] = new_er
            self.logger.info(f"Reduced expansionRatio: {current_er:.3f} → {new_er:.3f}")
        
        # Reduce first layer thickness
        if self.layers.get("relativeSizes", False):
            old_t1 = self.layers.get("firstLayerThickness", 0.2)
            new_t1 = max(RELATIVE_FIRST_LAYER_MIN, old_t1 * RELATIVE_FIRST_LAYER_REDUCTION)
            self.layers["firstLayerThickness"] = new_t1
            self.logger.info(f"Reduced relative t1: {old_t1:.3f} → {new_t1:.3f}")
        else:
            old_t1 = self.layers.get("firstLayerThickness_abs", 50e-6)
            new_t1 = max(FIRST_LAYER_MIN, old_t1 * FIRST_LAYER_REDUCTION_FACTOR)
            self.layers["firstLayerThickness_abs"] = new_t1
            self.logger.info(f"Reduced absolute t1: {old_t1*1e6:.1f}μm → {new_t1*1e6:.1f}μm")
        
        # Apply consistent minThickness policy
        self._apply_consistent_minThickness_policy()

    def _apply_consistent_minThickness_policy(self):
        """Apply minThickness policy based on current layer settings."""
        if self.layers.get("relativeSizes", False):
            # For relative sizing, minThickness is fraction of firstLayerThickness
            first_layer = self.layers.get("firstLayerThickness", 0.2)
            min_thickness = 0.15 * first_layer  # 15% of first layer
            self.layers["minThickness"] = min_thickness
        else:
            # For absolute sizing, ensure minimum is reasonable
            first_layer = self.layers.get("firstLayerThickness_abs", 50e-6)
            min_thickness = max(1.0e-6, 0.15 * first_layer)  # 1μm minimum or 15% of first layer
            self.layers["minThickness_abs"] = min_thickness

    def check_coverage_gated_progression(self, proposed_levels: List[int], 
                                       previous_coverage: Optional[Dict] = None) -> Tuple[List[int], bool]:
        """Check if surface refinement progression should be allowed based on coverage."""
        if not self.stage1.get("use_coverage_gated_progression", True):
            return proposed_levels, True
        
        if previous_coverage is None:
            return proposed_levels, True  # First iteration, allow progression
        
        prev_coverage = previous_coverage.get("coverage_overall", 0.0)
        coverage_threshold = 0.75  # Only progress if previous coverage >= 75%
        
        # Check if we're increasing refinement
        current_levels = getattr(self, 'surface_levels', [1, 1])
        is_increasing = any(p > c for p, c in zip(proposed_levels, current_levels))
        
        if is_increasing and prev_coverage < coverage_threshold:
            self.logger.info(f"Blocking surface refinement progression: "
                           f"coverage {prev_coverage:.1%} < {coverage_threshold:.0%}")
            return current_levels, False  # Block progression
        
        return proposed_levels, True  # Allow progression

    def log_iteration_summary(self, summary_path: Path, iteration: int, cell_count: int,
                            snap_m: Dict, layer_m: Dict, layer_cov: Dict,
                            thickness_frac: float = None, n_eff: float = None, diag: str = ""):
        """Log iteration results to CSV summary including layer diagnostics."""
        if not summary_path.exists():
            with open(summary_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "iter", "cells", "maxNonOrtho", "maxSkewness", "coverage", 
                    "objective_dummy", "levels_min", "levels_max", "resolveFeatureAngle",
                    "nLayers", "firstLayer", "minThickness",
                    "thicknessPct", "N_eff", "diag"  # Layer diagnostic columns
                ])
        
        # Get current parameters
        coverage = layer_cov.get("coverage_overall", 0.0)
        max_nonortho = max(snap_m.get("maxNonOrtho", 0), layer_m.get("maxNonOrtho", 0))
        max_skewness = max(snap_m.get("maxSkewness", 0), layer_m.get("maxSkewness", 0))
        
        resolve_angle = self.snappy.get("resolveFeatureAngle", 45)
        n_layers = self.layers.get("nSurfaceLayers", 5)
        
        if self.layers.get("relativeSizes", False):
            first_layer = self.layers.get("firstLayerThickness", 0.2)
            min_thickness = self.layers.get("minThickness", 0.03)
        else:
            first_layer = self.layers.get("firstLayerThickness_abs", 50e-6)
            min_thickness = self.layers.get("minThickness_abs", 7.5e-6)
        
        # Write summary row with diagnostic data
        with open(summary_path, "a", newline="") as f:
            csv.writer(f).writerow([
                iteration, cell_count, f"{max_nonortho:.1f}", f"{max_skewness:.2f}",
                f"{coverage:.3f}", "0.000", self.surface_levels[0], self.surface_levels[1],
                resolve_angle, n_layers, f"{first_layer:.3e}", f"{min_thickness:.3e}",
                f"{thickness_frac:.3f}" if thickness_frac is not None else "0.000",
                f"{n_eff:.2f}" if n_eff is not None else "0.00",
                diag if diag else ""
            ])

    def _maybe_parallel(self, iter_dir: Path) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """Determine parallel decomposition commands."""
        n_procs = int(self.stage1.get("n_processors", 1))
        
        if n_procs <= 1:
            return None, None
        
        # Create decomposeParDict with OpenFOAM 12 format
        decomp_dict = iter_dir / "system" / "decomposeParDict"
        
        # Determine optimal decomposition for given number of processors
        if n_procs == 4:
            n_xyz = "(2 2 1)"  # 2x2x1 decomposition
        elif n_procs == 8:
            n_xyz = "(2 2 2)"  # 2x2x2 decomposition  
        elif n_procs == 6:
            n_xyz = "(2 3 1)"  # 2x3x1 decomposition
        else:
            # Simple linear decomposition for other cases
            n_xyz = f"({n_procs} 1 1)"
        
        decomp_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Version:  12
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "system";
    object      decomposeParDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

numberOfSubdomains {n_procs};

decomposer      hierarchical;
distributor     ptscotch;

hierarchicalCoeffs
{{
    n               {n_xyz};
    order           xyz;
}}

// ************************************************************************* //
'''
        decomp_dict.write_text(decomp_content)
        
        pre = (["decomposePar"], "log.decomposePar")
        post = (["reconstructPar"], "log.reconstructPar") 
        
        return pre, post

    def _sum_wall_faces_from_processors(self, iter_dir: Path) -> int:
        """Count wall faces from processor boundary files or main boundary file."""
        total_faces = 0
        
        # First check main boundary file (serial run or reconstructed mesh)
        main_boundary_file = iter_dir / "constant" / "polyMesh" / "boundary"
        if main_boundary_file.exists():
            try:
                content = main_boundary_file.read_text()
                if self.wall_name in content:
                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        if self.wall_name in line:
                            for j in range(i+1, min(i+10, len(lines))):
                                if "nFaces" in lines[j]:
                                    total_faces = int(lines[j].split()[1].rstrip(';'))
                                    break
                            break
                    # If we found faces in main boundary file, return immediately
                    if total_faces > 0:
                        return total_faces
            except Exception as e:
                self.logger.debug(f"Could not parse main boundary file {main_boundary_file}: {e}")
        
        # Check for processor directories (parallel run)
        processor_dirs = list(iter_dir.glob("processor*"))
        
        if processor_dirs:
            # Parallel run - check processor directories
            for proc_dir in processor_dirs:
                boundary_file = proc_dir / "constant" / "polyMesh" / "boundary"
                if boundary_file.exists():
                    try:
                        content = boundary_file.read_text()
                        if self.wall_name in content:
                            # Parse nFaces for wall patch
                            lines = content.splitlines()
                            for i, line in enumerate(lines):
                                if self.wall_name in line:
                                    # Look for nFaces in subsequent lines
                                    for j in range(i+1, min(i+10, len(lines))):
                                        if "nFaces" in lines[j]:
                                            faces = int(lines[j].split()[1].rstrip(';'))
                                            total_faces += faces
                                            break
                                    break
                    except Exception as e:
                        self.logger.debug(f"Could not parse boundary file {boundary_file}: {e}")
        
        return total_faces

    def _parse_parallel_checkmesh(self, output: str) -> Dict:
        """Parse checkMesh output from parallel run."""
        metrics = {"meshOK": False, "cells": 0}
        
        lines = output.lower().splitlines()
        for line in lines:
            if "cells:" in line:
                try:
                    metrics["cells"] = int(line.split()[1])
                except (IndexError, ValueError):
                    pass
            elif "max non-orthogonality" in line:
                try:
                    metrics["maxNonOrtho"] = float(line.split()[-1])
                except (IndexError, ValueError):
                    pass
            elif "max skewness" in line:
                try:
                    metrics["maxSkewness"] = float(line.split()[-1])
                except (IndexError, ValueError):
                    pass
        
        # Check for fatal errors vs warnings
        output_lower = output.lower()
        fatal_errors = [
            "foam fatal error",
            "aborted",
            "no cells in mesh",
            "could not find cellzone"
        ]
        
        has_fatal_error = any(error in output_lower for error in fatal_errors)
        has_end_marker = "end" in output_lower and "finalising" in output_lower
        
        # Accept mesh with warnings (failed checks) but not fatal errors
        metrics["meshOK"] = has_end_marker and not has_fatal_error
        
        return metrics

    def update_surface_levels(self, surface_levels: List[int]):
        """Update current surface refinement levels."""
        self.surface_levels = surface_levels

    def update_current_iteration(self, iteration: int):
        """Update current iteration number."""
        self.current_iteration = iteration