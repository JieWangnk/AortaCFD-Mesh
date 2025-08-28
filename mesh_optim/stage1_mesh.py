"""Stage 1 mesh optimizer - Main orchestrator module.

This module coordinates geometry processing, dictionary generation, and optimization
to provide automated mesh generation for cardiovascular CFD applications.

Refactored to use modular architecture with focused sub-modules.
"""

import csv
import math
import re
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from .constants import (
    MAX_ITERATIONS_DEFAULT,
    MICRO_TRIALS_PER_ITERATION,
    COVERAGE_ACCEPTABLE_THRESHOLD,
    SOLVER_MODES,
    THICKNESS_FRACTION_SEVERE_THRESHOLD,
    THICKNESS_FRACTION_THIN_THRESHOLD,
    FIRST_LAYER_MIN_THICKNESS,
    FIRST_LAYER_REDUCTION_FACTOR,
    EXPANSION_RATIO_MIN,
    EXPANSION_RATIO_SMALL_REDUCTION,
    EXPANSION_RATIO_MEDIUM_REDUCTION,
    LAYER_ITER_CONSERVATIVE_INCREASE,
    RELAXED_ITER_CONSERVATIVE_INCREASE,
    RELATIVE_SIZING_FIRST_LAYER,
    RELATIVE_SIZING_FINAL_LAYER,
    MAX_THICKNESS_TO_CELL_RATIO,
    THICKNESS_RATIO_SAFETY,
    COVERAGE_THRESHOLD_MICRO_OPTIMIZATION
)
from .geometry_handler import GeometryHandler
from .openfoam_dicts import OpenFOAMDictGenerator  
from .optimizer import Stage1Optimizer
from .physics_mesh import PhysicsAwareMeshGenerator

# Import legacy functions (after folder cleanup)
from .legacy_functions import (
    ConfigurationManager,
    ProcessManager, 
    log_surface_histogram,
    evaluate_stage1_metrics
)
from .utils import safe_float, safe_int, safe_bool, get_openfoam_geometry_dir


@dataclass
class Stage1Targets:
    """Quality targets for Stage 1 mesh optimization."""
    max_nonortho: float
    max_skewness: float
    min_layer_cov: float


def _safe_get(d: dict, path: List[str], default=None):
    """Safely navigate nested dictionary paths."""
    x = d
    for k in path:
        if not isinstance(x, dict) or k not in x:
            return default
        x = x[k]
    return x


class Stage1MeshOptimizer:
    """Geometry-aware Stage-1 mesh optimizer (geometry only, meters everywhere)."""

    def __init__(self, geometry_dir, config_file, output_dir=None):
        self.geometry_dir = Path(geometry_dir)
        self.config_file = Path(config_file)
        self.output_dir = Path(output_dir) if output_dir else (self.geometry_dir.parent / "output" / "stage1_mesh")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging first
        self.logger = logging.getLogger(f"Stage1Mesh_{self.geometry_dir.name}")
        
        # Load and validate configuration using new architecture
        self.config_manager = ConfigurationManager(self.logger)
        self.config = self.config_manager.load_and_validate(self.config_file)
        
        # Validate geometry configuration
        if not self.config_manager.validate_geometry_config(self.config, self.geometry_dir):
            raise ValueError("Geometry configuration validation failed")
        
        # Apply legacy structure mapping for backward compatibility
        self._map_config_structure()

        # Initialize modules
        self.geometry_handler = GeometryHandler(self.geometry_dir, self.config, self.logger)
        self.dict_generator = OpenFOAMDictGenerator(self.config, self.geometry_handler.wall_name, self.logger)
        
        # Setup process manager for resource management
        self.process_manager = ProcessManager(logger=self.logger)
        memory_config = self.config_manager.get_memory_config(self.config)
        max_memory_gb = memory_config['max_memory_gb']
        
        # Initialize optimizer with managed memory constraints
        self.optimizer = Stage1Optimizer(self.config, self.geometry_handler.wall_name, max_memory_gb, self.logger)

        # Stage-1 policy and iteration control
        self.stage1 = _safe_get(self.config, ["STAGE1"], {}) or {}
        self.current_iteration = 0
        self.max_iterations = int(self.stage1.get("max_iterations", self.config.get("max_iterations", MAX_ITERATIONS_DEFAULT)))
        
        
        # Surface refinement levels
        self.surface_levels = list(_safe_get(self.config, ["SNAPPY", "surface_level"], [1, 1]))
        
        # Memory budgeting
        self._apply_improved_memory_budgeting()
        
        # Apply solver-specific acceptance criteria and robust defaults
        self._apply_solver_presets()
        self._apply_robust_defaults()

        self.logger.info(f"Stage 1 mesh optimizer initialized: max_iterations={self.max_iterations}, "
                        f"wall_patch={self.geometry_handler.wall_name}")

    def _map_config_structure(self):
        """Map from new two-tier configuration structure to internal format."""
        # Check if using new two-tier structure (has top-level mesh/layers/refinement)
        if "mesh" in self.config and "layers" in self.config:
            # Create internal structure from new simple parameters
            
            # Map paths
            paths = self.config.get("paths", {})
            if "openfoam_env" in paths:
                self.config["openfoam_env_path"] = paths["openfoam_env"]
            
            # Map mesh parameters
            mesh = self.config.get("mesh", {})
            if "STAGE1" not in self.config:
                self.config["STAGE1"] = {}
            
            self.config["STAGE1"]["base_size_mode"] = mesh.get("base_size_mode", "diameter")
            self.config["STAGE1"]["N_D"] = mesh.get("cells_per_diameter", 22)
            self.config["STAGE1"]["N_D_min"] = mesh.get("min_cells_per_throat", 28)
            
            # Map refinement parameters
            refinement = self.config.get("refinement", {})
            if "surface_levels" in refinement:
                if "SNAPPY" not in self.config:
                    self.config["SNAPPY"] = {}
                self.config["SNAPPY"]["surface_level"] = refinement["surface_levels"]
                
            self.config["STAGE1"]["near_band_cells"] = refinement.get("near_band_dx", 4)
            self.config["STAGE1"]["far_band_cells"] = refinement.get("far_band_dx", 10)
            
            # Map feature angle
            feature_angle = refinement.get("feature_angle", {})
            if feature_angle.get("mode") == "adaptive":
                self.config["STAGE1"]["featureAngle_init"] = feature_angle.get("init", 45)
                self.config["STAGE1"]["featureAngle_step"] = feature_angle.get("step", 10)
            
            # Map layers parameters
            layers = self.config.get("layers", {})
            if "LAYERS" not in self.config:
                self.config["LAYERS"] = {}
                
            self.config["LAYERS"]["nSurfaceLayers"] = layers.get("n", 10)
            self.config["LAYERS"]["expansionRatio"] = layers.get("expansion", 1.2)
            
            # Map first layer settings
            first_layer = layers.get("first_layer", {})
            if first_layer.get("mode") == "auto_dx":
                self.config["STAGE1"]["alpha_total_layers"] = first_layer.get("t_over_dx", 0.8)
                self.config["STAGE1"]["t1_min_fraction_of_dx"] = first_layer.get("t1_min_frac", 0.02)
                self.config["STAGE1"]["t1_max_fraction_of_dx"] = first_layer.get("t1_max_frac", 0.08)
                self.config["STAGE1"]["autoFirstLayerFromDx"] = True
            
            # Map acceptance criteria
            accept = self.config.get("accept", {})
            if "acceptance_criteria" not in self.config:
                self.config["acceptance_criteria"] = {}
                
            self.config["acceptance_criteria"]["maxNonOrtho"] = accept.get("maxNonOrtho", 65)
            self.config["acceptance_criteria"]["maxSkewness"] = accept.get("maxSkewness", 4.0)
            self.config["acceptance_criteria"]["min_layer_coverage"] = accept.get("min_layer_coverage", 0.70)
            
            # Map compute settings
            compute = self.config.get("compute", {})
            self.config["STAGE1"]["n_processors"] = compute.get("procs", 4)
            self.config["STAGE1"]["cell_budget_kb_per_cell"] = compute.get("cell_budget_kb_per_cell", 1.0)
            
            # Map iterations
            iterations = self.config.get("iterations", {})
            self.config["max_iterations"] = iterations.get("max", 3)
            if "ladder" in iterations:
                self.config["STAGE1"]["ladder"] = iterations["ladder"]
            
            # Map micro_layer_retries to STAGE1 section
            self.config["STAGE1"]["micro_layer_retries"] = iterations.get("micro_layer_retries", 3)
            
            # Map physics parameters  
            physics = self.config.get("physics", {})
            if "PHYSICS" not in self.config:
                self.config["PHYSICS"] = {}
                
            self.config["PHYSICS"]["solver_mode"] = physics.get("solver_mode", "RANS")
            self.config["PHYSICS"]["flow_model"] = physics.get("flow_model", "turbulent")
            self.config["PHYSICS"]["y_plus"] = physics.get("y_plus", 30)
            self.config["PHYSICS"]["U_peak"] = physics.get("U_peak", 1.0)
            self.config["PHYSICS"]["rho"] = physics.get("rho", 1060.0)
            self.config["PHYSICS"]["mu"] = physics.get("mu", 0.0035)
            self.config["PHYSICS"]["use_womersley_bands"] = physics.get("use_womersley_bands", False)
            self.config["PHYSICS"]["heart_rate_hz"] = physics.get("heart_rate_hz", 1.2)
            
            # Preserve advanced overrides
            if "advanced" in self.config:
                advanced = self.config["advanced"]
                for section in ["BLOCKMESH", "SNAPPY", "LAYERS", "MESH_QUALITY", "SURFACE_FEATURES", 
                               "GEOMETRY_POLICY", "STAGE1", "SCALING", "PHYSICS"]:
                    if section in advanced:
                        if section not in self.config:
                            self.config[section] = {}
                        self.config[section].update(advanced[section])

    def _apply_solver_presets(self):
        """Apply solver-specific acceptance criteria using new configuration manager."""
        physics = _safe_get(self.config, ["PHYSICS"], {}) or {}
        solver_mode = physics.get("solver_mode", "RANS").upper()
        
        # Apply solver presets using new configuration manager
        self.config = self.config_manager.apply_solver_presets(self.config, solver_mode)
        
        # Update optimizer targets from applied presets
        if solver_mode in SOLVER_MODES:
            preset = SOLVER_MODES[solver_mode]
            self.optimizer.targets.max_nonortho = min(self.optimizer.targets.max_nonortho, preset["max_nonortho"])
            self.optimizer.targets.max_skewness = min(self.optimizer.targets.max_skewness, preset["max_skewness"])
            self.optimizer.targets.min_layer_cov = max(self.optimizer.targets.min_layer_cov, preset["min_layer_coverage"])

    def _apply_robust_defaults(self):
        """Apply proven mesh generation best practices from successful manual runs."""
        from .constants import (
            FIRST_LAYER_MIN, FIRST_LAYER_MAX, FIRST_LAYER_DEFAULT,
            FEATURE_ANGLE_MIN, FEATURE_ANGLE_MAX, FEATURE_ANGLE_RECOMMENDED,
            FEATURE_SNAP_ITER_HIGH_ANGLE
        )
        
        # Ensure conservative surface refinement progression
        ladder = self.stage1.get("ladder", [[1,1],[1,2],[2,2]])
        if len(ladder) > 0 and len(ladder[0]) == 2:
            # Ensure no iteration jumps too aggressively
            for i, (min_lvl, max_lvl) in enumerate(ladder):
                if i > 0:
                    prev_max = ladder[i-1][1] 
                    jump = max_lvl - prev_max
                    if jump > 1:
                        self.logger.info(f"Limiting surface level jump in ladder iteration {i}: "
                                       f"({min_lvl},{max_lvl}) → ({min_lvl},{prev_max+1})")
                        ladder[i] = [min_lvl, prev_max + 1]
            self.stage1["ladder"] = ladder
        
        # Ensure reasonable layer settings for vascular meshing
        layers = self.config.get("LAYERS", {})
        
        # Ensure first layer thickness is reasonable for vascular scale
        first_layer = layers.get("firstLayerThickness_abs", FIRST_LAYER_DEFAULT)
        if first_layer is not None and (first_layer < FIRST_LAYER_MIN or first_layer > FIRST_LAYER_MAX):
            recommended = FIRST_LAYER_DEFAULT
            self.logger.info(f"Adjusting first layer thickness: "
                           f"{first_layer*1e6:.1f}μm → {recommended*1e6:.1f}μm")
            layers["firstLayerThickness_abs"] = recommended
        
        # Ensure feature angle detection is in proven range
        snap = self.config.get("SNAPPY", {})
        resolve_angle = snap.get("resolveFeatureAngle", FEATURE_ANGLE_RECOMMENDED)
        if resolve_angle < FEATURE_ANGLE_MIN or resolve_angle > FEATURE_ANGLE_MAX:
            recommended = FEATURE_ANGLE_RECOMMENDED
            self.logger.info(f"Adjusting resolveFeatureAngle: {resolve_angle}° → {recommended}°")
            snap["resolveFeatureAngle"] = recommended
        
        # Ensure sufficient snap iterations for complex geometry
        n_feature_snap = snap.get("nFeatureSnapIter", FEATURE_SNAP_ITER_HIGH_ANGLE)
        if n_feature_snap < 10:
            recommended = FEATURE_SNAP_ITER_HIGH_ANGLE
            self.logger.info(f"Increasing nFeatureSnapIter: {n_feature_snap} → {recommended}")
            snap["nFeatureSnapIter"] = recommended
            
        self.logger.debug("Applied robust meshing defaults based on proven configurations")

    def _apply_improved_memory_budgeting(self) -> None:
        """Apply improved memory-aware cell budgeting."""
        import psutil
        
        # Get current system resources
        available_gb = psutil.virtual_memory().available / (1024**3)
        n_procs = self.stage1.get("n_processors", 1)
        kb_per_cell = self.stage1.get("cell_budget_kb_per_cell", 2.0)
        
        # Calculate memory-aware limits
        usable_gb = available_gb * 0.7  # Use 70% of available memory for safety
        usable_kb = usable_gb * 1024 * 1024
        
        # Calculate total cells budget
        total_cells = int(usable_kb / max(kb_per_cell, 0.5))
        
        # Distribute across processors
        import numpy as np
        max_local = int(np.clip(total_cells // max(n_procs, 1), 100_000, 5_000_000))
        max_global = int(np.clip(total_cells, 500_000, 20_000_000))
        
        # Apply limits but allow config overrides
        original_local = self.config["SNAPPY"].get("maxLocalCells", max_local)
        original_global = self.config["SNAPPY"].get("maxGlobalCells", max_global)
        
        # Use the more conservative (smaller) of the two
        self.config["SNAPPY"]["maxLocalCells"] = min(original_local, max_local)
        self.config["SNAPPY"]["maxGlobalCells"] = min(original_global, max_global)
        
        self.logger.info(f"Memory-aware limits: {available_gb:.1f}GB avail → "
                        f"Local: {self.config['SNAPPY']['maxLocalCells']:,}, "
                        f"Global: {self.config['SNAPPY']['maxGlobalCells']:,}")

    # ---------------------- Layer diagnostics (NEW) -----------------------
    def _parse_snappy_layers_dict_values(self, dict_path: Path) -> Dict:
        """Extract nSurfaceLayers, expansionRatio, relativeSizes from snappyHexMeshDict.layers."""
        try:
            txt = dict_path.read_text(errors="ignore")
        except Exception:
            return {"N": None, "ER": None, "relative": None}
        mN  = re.search(r"\bnSurfaceLayers\s+(\d+)", txt)
        mER = re.search(r"\bexpansionRatio\s+([0-9.]+)", txt)
        mRel= re.search(r"\brelativeSizes\s+(true|false)", txt, re.IGNORECASE)
        N   = int(mN.group(1)) if mN else None
        ER  = float(mER.group(1)) if mER else None
        rel = (mRel and mRel.group(1).lower() == "true")
        return {"N": N, "ER": ER, "relative": rel}

    def _scrape_thickness_fraction_from_log(self, log_path: Path) -> float:
        """Return thickness fraction (0..1) if found, else None."""
        patterns = [
            r"Final layer metrics:\s*[0-9.]+\s*layers,\s*([0-9.]+)\s*%\s*thickness achieved",
            r"thickness\s*([0-9.]+)\s*%\s*achieved",
            r"achieved\s*thickness\s*[:=]\s*([0-9.]+)\s*%",
        ]
        try:
            txt = log_path.read_text(errors="ignore")
        except Exception:
            return None
        for pat in patterns:
            m = re.search(pat, txt, flags=re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1)) / 100.0
                except Exception:
                    pass
        return None

    def _count_prune_hints(self, log_path: Path) -> int:
        """Count pruning/quality-gate hints in the layers log."""
        hints = [
            "maxFaceThicknessRatio",
            "maxThicknessToMedialRatio",
            "minMedianAxisAngle",
            "featureAngle",
            "illegal faces",
            "Removing extrusion",
            "Will not extrude",
            "Deleting layer",
        ]
        try:
            txt = log_path.read_text(errors="ignore")
        except Exception:
            return 0
        return sum(len(re.findall(h, txt, flags=re.IGNORECASE)) for h in hints)

    def _effective_layers_from_fraction(self, N: int, ER: float, frac: float) -> float:
        """
        Geometric stack identity (dimensionless):
          T_target/t1 = (ER^N - 1)/(ER - 1)
          N_eff = log( 1 + frac*(ER^N - 1) ) / log(ER)
        Works for absolute and relative sizing (t1 cancels).
        """
        if N is None or ER is None or frac is None:
            return None
        if ER <= 1.0:
            return frac * float(N)
        try:
            return math.log(1.0 + frac * (ER**N - 1.0), ER)
        except ValueError:
            return None

    def _diagnose_layer_growth(self, frac: float, N_eff: float, prune_hits: int) -> str:
        """Return a compact diagnosis string."""
        if frac is None:
            return "unknown: thickness% not found in layers log"
        label = ""
        if frac < 0.15:
            label = "not-added/abandoned-early"
        elif frac < 0.60:
            label = "thin-but-present (added-then-pruned)"
        elif frac < 0.90:
            label = "moderate-growth"
        else:
            label = "healthy-growth"
        if N_eff is not None:
            if N_eff < 1.0 and "abandoned" not in label:
                label += ", N_eff<1"
            elif N_eff < 3.0 and "healthy" in label:
                label = "moderate-growth, N_eff<3"
        if prune_hits and "healthy" not in label:
            label += f", prune-hints={prune_hits}"
        return label

    def _write_layer_diag(self, iter_dir: Path, iteration_data: Dict = None) -> Tuple[float, float, str]:
        """
        Create iter_xxx/layer_diag.txt and return (thickness_frac, N_eff, diagnosis).
        Tries iteration_data first, then scrapes log; reads N/ER from snappy dict.
        """
        sys_snappy = iter_dir / "system" / "snappyHexMeshDict.layers"
        log_path   = iter_dir / "logs" / "log.snappy.layers"
        vals = self._parse_snappy_layers_dict_values(sys_snappy)
        N, ER = vals["N"], vals["ER"]

        # thickness fraction
        frac = None
        if iteration_data and "final_metrics" in iteration_data:
            fm = iteration_data["final_metrics"] or {}
            # accept either 0..100 or 0..1
            if "thickness_pct" in fm and fm["thickness_pct"] is not None:
                v = float(fm["thickness_pct"])
                frac = v/100.0 if v > 1.0 else v
        if frac is None:
            frac = self._scrape_thickness_fraction_from_log(log_path)

        # effective layers & pruning
        N_eff = self._effective_layers_from_fraction(N, ER, frac)
        prune_hits = self._count_prune_hints(log_path)
        diag = self._diagnose_layer_growth(frac, N_eff, prune_hits)

        # write report
        rpt = iter_dir / "layer_diag.txt"
        try:
            with open(rpt, "w") as f:
                f.write("=== Layer Diagnostics ===\n")
                f.write(f"snappy: {sys_snappy.name}\n")
                f.write(f"N={N}  ER={ER}  relativeSizes={vals['relative']}\n")
                f.write(f"thickness%: {('%.1f%%'%(frac*100)) if frac is not None else '<not-found>'}\n")
                f.write(f"N_eff: {('%.2f'%N_eff) if N_eff is not None else '<n/a>'}\n")
                f.write(f"pruning/illegal-face hints in log: {prune_hits}\n")
                f.write(f"diagnosis: {diag}\n")
        except Exception as e:
            self.logger.debug(f"Could not write {rpt}: {e}")

        return frac, N_eff, diag
    
    def _calculate_effective_layers(self, thickness_fraction, N, ER):
        """
        Calculate effective layers from thickness fraction using geometric series.
        
        Geometric stack: t_k = t1 * ER^(k-1), k=1..N
        T_target/t1 = (ER^N - 1)/(ER - 1)
        N_eff = log(1 + thickness_fraction * (ER^N - 1)) / log(ER)
        """
        import math
        if N is None or ER is None or thickness_fraction is None:
            return None
        if ER <= 1.0:
            return thickness_fraction * N  # degenerate case; treat as linear
        try:
            return math.log(1.0 + thickness_fraction * (ER**N - 1.0)) / math.log(ER)
        except (ValueError, ZeroDivisionError):
            return thickness_fraction * N  # fallback to linear
    
    def _diagnose_layer_result(self, layer_coverage_data):
        """Enhanced layer result diagnosis using thickness fraction and effective layers."""
        thickness_frac = layer_coverage_data.get("thickness_fraction", layer_coverage_data.get("coverage_overall", 0.0))
        effective_layers = layer_coverage_data.get("effective_layers", 0.0)
        prune_count = layer_coverage_data.get("pruning_hints", 0)
        
        # If we don't have effective layers, calculate them
        if effective_layers == 0.0 and thickness_frac > 0.0:
            L = self.config.get("LAYERS", {})
            N = L.get("nSurfaceLayers", 5)
            ER = L.get("expansionRatio", 1.2)
            effective_layers = self._calculate_effective_layers(thickness_frac, N, ER)
        
        diagnosis_notes = []
        
        # Primary thickness-based diagnosis
        if thickness_frac < 0.15:
            primary_diag = "not-added-or-abandoned-early"
            diagnosis_notes.append("Very low thickness suggests blocked at t1 or immediate abandonment")
        elif thickness_frac < 0.60:
            primary_diag = "thin-but-present (added-then-pruned)"
            diagnosis_notes.append("Some growth but significant pruning by quality gates")
        else:
            primary_diag = "healthy-growth"
            diagnosis_notes.append("Good thickness achievement")
            
        # Refine based on effective layers
        if effective_layers is not None:
            if effective_layers < 1.0 and primary_diag != "not-added-or-abandoned-early":
                primary_diag = "barely-one-layer"
                diagnosis_notes.append(f"Effective layers {effective_layers:.2f} < 1")
            elif effective_layers < 3.0 and primary_diag == "healthy-growth":
                primary_diag = "moderate-growth"
                diagnosis_notes.append(f"Effective layers {effective_layers:.2f} still under 3")
        
        # Add pruning information
        if prune_count > 0:
            diagnosis_notes.append(f"{prune_count} pruning/quality hints in log")
            
        return {
            "diagnosis": primary_diag,
            "effective_layers": effective_layers,
            "thickness_fraction": thickness_frac,
            "notes": diagnosis_notes,
            "recommendation": self._get_layer_recommendation(primary_diag, thickness_frac, effective_layers)
        }
    
    def _get_layer_recommendation(self, diagnosis, thickness_frac, n_eff):
        """Get specific recommendations based on layer diagnosis."""
        recommendations = []
        
        if diagnosis in ["not-added-or-abandoned-early", "barely-one-layer"]:
            recommendations.extend([
                "Switch to relativeSizes if on refined surface",
                "Reduce firstLayerThickness by 20-30%", 
                "Cap expansionRatio at 1.15",
                "Increase nSmoothSurfaceNormals and nRelaxIter"
            ])
        elif diagnosis == "thin-but-present (added-then-pruned)":
            recommendations.extend([
                "Reduce expansionRatio by 0.02-0.03",
                "Increase nLayerIter to 70-80", 
                "Widen refinement bands (7-8x for near, 14-16x for far)",
                "Consider relaxing quality gates modestly"
            ])
        elif diagnosis == "moderate-growth":
            recommendations.extend([
                "Good progress, consider minor tuning",
                "Could increase target layers if needed",
                "Monitor mesh quality carefully"
            ])
        else:  # healthy-growth
            recommendations.append("Layer addition is working well")
            
        return recommendations

    # ------------------- Micro reactive tuning (NEW) ---------------------
    def _apply_micro_reactive_tuning(self, dx: float, thickness_frac: float, n_eff: float) -> None:
        """
        Small, conservative tweaks when layers were added then pruned or too thick.
        - If thickness% < 15%  -> aggressive rescue: shrink t1, lower ER, enable relativeSizes when refined.
        - If 15–60%            -> gentle rescue: lower ER, bump layer iters.
        - Always cap T/Δx ≤ 0.9 on the current surface level.
        Applies changes to self.config["LAYERS"] for the next iteration.
        """
        L = self.config["LAYERS"]
        max_level = max(getattr(self, "surface_levels", [1, 1]))
        # surface Δx from base Δx (dx) and refinement level: Δx_surf = dx / 2^level
        # Ensure dx is float to prevent string division error
        dx = float(dx)
        dx_surf = dx / (2 ** max_level)

        # Pull current knobs with safe type conversion from utils
        ER = safe_float(L.get("expansionRatio", 1.2))
        N = safe_int(L.get("nSurfaceLayers", 5))
        rel_mode = safe_bool(L.get("relativeSizes", False))

        # --- Rescue branch based on thickness ratio (0..1)
        if thickness_frac is not None:
            if thickness_frac < THICKNESS_FRACTION_SEVERE_THRESHOLD:  # Only for extremely poor coverage
                # Conservative first layer reduction + gentler ER
                if not rel_mode:
                    t1 = safe_float(L.get("firstLayerThickness_abs", 50e-6))
                    new_t1 = max(FIRST_LAYER_MIN_THICKNESS, FIRST_LAYER_REDUCTION_FACTOR * t1)
                    L["firstLayerThickness_abs"] = new_t1
                    self.logger.info(f"[micro] Severe prune: t1 {t1*1e6:.1f}→{new_t1*1e6:.1f} μm")
                # Smaller ER reduction
                new_ER = max(EXPANSION_RATIO_MIN, ER - EXPANSION_RATIO_MEDIUM_REDUCTION)
                if new_ER < ER:
                    L["expansionRatio"] = new_ER
                    self.logger.info(f"[micro] Severe prune: ER {ER:.3f}→{new_ER:.3f}")
                # Only switch to relative sizes on highly refined surfaces with very poor coverage
                if max_level > 2 and not rel_mode and thickness_frac < 0.05:
                    L["relativeSizes"] = True
                    L["firstLayerThickness"] = RELATIVE_SIZING_FIRST_LAYER
                    L["finalLayerThickness"] = RELATIVE_SIZING_FINAL_LAYER
                    self.logger.info(f"[micro] Severe prune on refined surface: switch to relativeSizes (t1_rel={RELATIVE_SIZING_FIRST_LAYER}, T_rel={RELATIVE_SIZING_FINAL_LAYER})")
            elif thickness_frac < THICKNESS_FRACTION_THIN_THRESHOLD:
                new_ER = max(EXPANSION_RATIO_MIN, ER - EXPANSION_RATIO_SMALL_REDUCTION)
                if new_ER < ER:
                    L["expansionRatio"] = new_ER
                    self.logger.info(f"[micro] Thin layers: ER {ER:.3f}→{new_ER:.3f}")
                # More conservative iteration increases
                current_layer_iter = safe_int(L.get("nLayerIter", 50))
                current_relaxed_iter = safe_int(L.get("nRelaxedIter", 20))
                L["nLayerIter"] = max(current_layer_iter, current_layer_iter + LAYER_ITER_CONSERVATIVE_INCREASE)
                L["nRelaxedIter"] = max(current_relaxed_iter, current_relaxed_iter + RELAXED_ITER_CONSERVATIVE_INCREASE)
                self.logger.info(f"[micro] Thin layers: nLayerIter={L['nLayerIter']}, nRelaxedIter={L['nRelaxedIter']}")

        # --- Cap total thickness relative to surface Δx: T/Δx ≤ MAX_THICKNESS_TO_CELL_RATIO
        # Relative mode: T_rel is already T/Δx_surf
        if rel_mode:
            T_rel = safe_float(L.get("finalLayerThickness", 0.75))
            if T_rel > MAX_THICKNESS_TO_CELL_RATIO:
                new_T_rel = THICKNESS_RATIO_SAFETY
                L["finalLayerThickness"] = new_T_rel
                self.logger.info(f"[micro] Cap T/Δx (relative): {T_rel:.3f}→{new_T_rel:.3f}")
        else:
            # Absolute mode: T_abs = t1 * (ER^N - 1)/(ER-1)
            t1 = safe_float(L.get("firstLayerThickness_abs", 50e-6))
            series = (ER**N - 1.0) / max(ER - 1.0, 1e-12) if ER > 1.0 else N
            T_abs = t1 * series
            T_over_dx = T_abs / max(dx_surf, 1e-12)
            if T_over_dx > MAX_THICKNESS_TO_CELL_RATIO:
                # Prefer reducing N by one (keeps t1 intent), recompute series
                new_N = max(3, N - 1)
                new_series = (ER**new_N - 1.0) / max(ER - 1.0, 1e-12) if ER > 1.0 else new_N
                new_T_over_dx = (t1 * new_series) / max(dx_surf, 1e-12)
                if new_N < N and new_T_over_dx < T_over_dx:
                    L["nSurfaceLayers"] = new_N
                    self.logger.info(f"[micro] Cap T/Δx (abs): N {N}→{new_N} (T/Δx {T_over_dx:.3f}→{new_T_over_dx:.3f})")
                else:
                    # fallback: shave t1 slightly
                    new_t1 = max(5e-6, 0.90 * t1)
                    L["firstLayerThickness_abs"] = new_t1
                    self.logger.info(f"[micro] Cap T/Δx (abs): t1 {t1*1e6:.1f}→{new_t1*1e6:.1f} μm (T/Δx={T_over_dx:.3f})")

    def iterate_until_quality(self):
        """Main optimization loop - coordinates all modules to achieve quality mesh."""
        self.logger.info("Starting Stage‑1 geometry‑aware optimization")
        
        optimization_state = self._initialize_optimization()
        
        # Main iteration loop
        for k in range(1, self.max_iterations + 1):
            iteration_result = self._run_single_iteration(k, optimization_state)
            optimization_state = self._update_optimization_state(optimization_state, k, iteration_result)
            
            if iteration_result.get('early_termination'):
                break
        
        return self._finalize_optimization(optimization_state)
    
    def _initialize_optimization(self) -> Dict:
        """Initialize optimization state and create summary CSV."""
        summary_path = self.output_dir / "stage1_summary.csv"
        if not summary_path.exists():
            with open(summary_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "iter","cells","maxNonOrtho","maxSkewness","coverage","objective_dummy",
                    "levels_min","levels_max","resolveFeatureAngle","nLayers","firstLayer","minThickness",
                    "thicknessPct","N_eff","diag"   # NEW columns for layer diagnostics
                ])
        
        return {
            'best_iter': None,
            'best_cell_count': math.inf,
            'previous_coverage_data': None,
            'summary_path': summary_path
        }
    
    def _run_single_iteration(self, k: int, optimization_state: Dict) -> Dict:
        """Run a single optimization iteration."""
        self.current_iteration = k
        self.optimizer.update_current_iteration(k)
        self.logger.info(f"=== ITERATION {k} ===")
        
        # Setup iteration environment
        iter_dir = self._setup_iteration_directory(k)
        
        # Prepare geometry and mesh parameters
        geometry_params = self._prepare_geometry_parameters(iter_dir)
        
        # Generate mesh dictionaries
        self._generate_mesh_dictionaries(iter_dir, geometry_params)
        
        # Run mesh generation with micro-trials
        mesh_result = self._run_mesh_generation_trials(iter_dir, geometry_params)
        
        # Evaluate and log results
        return self._evaluate_iteration_results(k, iter_dir, mesh_result, optimization_state)
    
    def _setup_iteration_directory(self, k: int) -> Path:
        """Setup directory structure for iteration."""
        # Log configuration for transparency  
        snap = self.config["SNAPPY"]
        self.logger.debug(f"Config: maxGlobal={snap['maxGlobalCells']:,}, "
                         f"maxLocal={snap['maxLocalCells']:,}, "
                         f"featureAngle={snap.get('resolveFeatureAngle', 45)}°")
        
        # Setup iteration directory
        iter_dir = self.output_dir / f"iter_{k:03d}"
        iter_dir.mkdir(exist_ok=True)
        (iter_dir / "system").mkdir(exist_ok=True)
        (iter_dir / "logs").mkdir(exist_ok=True)
        
        return iter_dir
    
    def _prepare_geometry_parameters(self, iter_dir: Path) -> Dict:
        """Prepare all geometry-related parameters for mesh generation."""
        # 1) Copy/scale STL files and get outlet names
        outlet_names = self.geometry_handler.copy_and_scale_stl_files(iter_dir)
        
        # 2) Compute bounding box and reference diameters from scaled STL files
        geometry_dir = get_openfoam_geometry_dir()
        stl_map_norm = {
            "inlet": iter_dir/f"constant/{geometry_dir}/inlet.stl",
            self.geometry_handler.wall_name: iter_dir/f"constant/{geometry_dir}/{self.geometry_handler.wall_name}.stl",
            **{p.stem: iter_dir/f"constant/{geometry_dir}/{p.stem}.stl" 
               for p in self.geometry_handler.stl_files["outlets"]}
        }
        
        # Generate bounding box using physics mesh generator
        gen = PhysicsAwareMeshGenerator()
        bbox_data = gen.compute_stl_bounding_box(stl_map_norm, skip_scaling=True)
        
        # Estimate diameters and derive base cell size
        geometry_path = iter_dir / "constant" / geometry_dir
        D_ref, D_min = self.geometry_handler.estimate_reference_diameters(stl_root=geometry_path)
        dx = self.geometry_handler.derive_base_cell_size(D_ref, D_min)
        
        # 3) Apply adaptive feature angle
        adaptive_angle = self.geometry_handler.adaptive_feature_angle(
            D_ref, D_min, len(outlet_names), iter_dir=iter_dir)
        self.config["SNAPPY"]["resolveFeatureAngle"] = adaptive_angle
        
        # 5) Apply ladder progression and coverage gating  
        surface_levels = self._apply_surface_level_progression()
        
        # 6) Calculate internal seed point from scaled STL files
        internal_point = self.geometry_handler.calculate_seed_point(
            bbox_data, {"inlet": geometry_path / "inlet.stl"}, dx_base=dx)
        
        # 7) Apply dynamic feature snap iterations
        self._configure_dynamic_feature_snapping()
        
        return {
            'outlet_names': outlet_names,
            'bbox_data': bbox_data,
            'dx': dx,
            'surface_levels': surface_levels,
            'internal_point': internal_point,
            'geometry_dir': geometry_path
        }
    
    def _apply_surface_level_progression(self) -> List[int]:
        """Apply ladder progression and coverage gating."""
        ladder = self.stage1.get("ladder", [[1,1],[2,2],[2,3]])
        idx = min(self.current_iteration-1, len(ladder)-1)
        proposed_surface_levels = list(ladder[idx])
        
        if self.stage1.get("use_coverage_gated_progression", True):
            actual_surface_levels, progression_allowed = self.optimizer.check_coverage_gated_progression(
                proposed_surface_levels, getattr(self, '_previous_coverage_data', None))
        else:
            actual_surface_levels, progression_allowed = proposed_surface_levels, True
        
        # Update surface levels and inform modules
        surface_levels_changed = (not hasattr(self, 'surface_levels') or 
                                self.surface_levels != actual_surface_levels)
        self.surface_levels = actual_surface_levels
        self.optimizer.update_surface_levels(actual_surface_levels)
        self.dict_generator.update_surface_levels(actual_surface_levels)
        
        self.logger.info(f"Surface levels from ladder: {proposed_surface_levels} → {self.surface_levels}")
        if not progression_allowed:
            self.logger.info("   (progression blocked by coverage gating)")
        
        return actual_surface_levels
    
    def _configure_dynamic_feature_snapping(self):
        """Configure dynamic feature snap iterations based on feature angle."""
        from .constants import (
            FEATURE_ANGLE_LOW_THRESHOLD, FEATURE_ANGLE_HIGH_THRESHOLD,
            FEATURE_SNAP_ITER_LOW_ANGLE, FEATURE_SNAP_ITER_HIGH_ANGLE
        )
        
        current_feature_angle = int(self.config["SNAPPY"].get("resolveFeatureAngle", 45))
        if current_feature_angle <= FEATURE_ANGLE_LOW_THRESHOLD:
            current_nFeatureSnapIter = FEATURE_SNAP_ITER_LOW_ANGLE
        elif current_feature_angle >= FEATURE_ANGLE_HIGH_THRESHOLD:
            current_nFeatureSnapIter = FEATURE_SNAP_ITER_HIGH_ANGLE
        else:
            # Linear interpolation
            current_nFeatureSnapIter = int(FEATURE_SNAP_ITER_LOW_ANGLE + 
                (FEATURE_SNAP_ITER_HIGH_ANGLE - FEATURE_SNAP_ITER_LOW_ANGLE) * 
                (current_feature_angle - FEATURE_ANGLE_LOW_THRESHOLD) / 
                (FEATURE_ANGLE_HIGH_THRESHOLD - FEATURE_ANGLE_LOW_THRESHOLD))
        
        self.config["SNAPPY"]["nFeatureSnapIter"] = current_nFeatureSnapIter
        self.logger.info(f"Dynamic nFeatureSnapIter: {current_nFeatureSnapIter} "
                       f"(for resolveFeatureAngle={current_feature_angle}°)")
    
    def _generate_mesh_dictionaries(self, iter_dir: Path, geometry_params: Dict):
        """Generate all required OpenFOAM dictionaries."""
        # Generate block mesh dictionary
        self.dict_generator.generate_blockmesh_dict(
            iter_dir, geometry_params['bbox_data'], geometry_params['dx'])
        
        # Generate surface features dictionary
        surface_list = [f"{self.geometry_handler.wall_name}.stl", "inlet.stl"] + \
                      [f"{n}.stl" for n in geometry_params['outlet_names']]
        self.dict_generator.generate_surface_features_dict(iter_dir, surface_list)
        
        # Generate snappy dictionaries
        self.dict_generator.generate_snappy_dict(
            iter_dir, geometry_params['outlet_names'], 
            geometry_params['internal_point'], geometry_params['dx'], "no_layers")
        self.dict_generator.generate_snappy_dict(
            iter_dir, geometry_params['outlet_names'], 
            geometry_params['internal_point'], geometry_params['dx'], "layers")
        
        # Create .foam file for visualization
        self.dict_generator.create_foam_file(iter_dir)
    
    def _run_mesh_generation_trials(self, iter_dir: Path, geometry_params: Dict) -> Dict:
        """Run mesh generation with micro-trials and return best result."""
        K = int(self.stage1.get("micro_layer_retries", MICRO_TRIALS_PER_ITERATION))
        best_local = None
        
        for trial in range(1, K+1):
            self.logger.info(f"Micro-trial {trial}/{K} for iteration {self.current_iteration}")
            
            # Run mesh generation with clean baseline for each micro-trial
            snap_m, layer_m, layer_cov = self.optimizer.run_mesh_generation(
                iter_dir, force_full_remesh=True, trial_num=trial)
            
            # Handle mesh generation failure with graceful fallback
            if not snap_m.get('meshOK', False) or 'error' in snap_m:
                self.logger.warning(f"Mesh generation failed at iteration {self.current_iteration}, trial {trial}")
                self.logger.warning(f"Error: {snap_m.get('error', 'Unknown error')}")
                
                if best_local is not None:
                    best_coverage = best_local[2].get('coverage_overall', 0.0)
                    self.logger.info(f"Falling back to best result so far: {best_coverage*100:.1f}% coverage")
                    self.logger.info("Treating micro-trials as exploratory - accepting best achieved result")
                    break  # Exit trial loop but continue with best result
                else:
                    # No previous result - this is first trial failure
                    self.logger.error(f"CRITICAL: First trial failed at iteration {self.current_iteration}")
                    with open(self.output_dir / "OPTIMIZATION_FAILED.txt", "w") as f:
                        f.write(f"First trial failed at iteration {self.current_iteration}\n")
                        f.write(f"Error: {snap_m.get('error', 'Unknown error')}\n")
                        f.write("Check iter_XXX/logs/ for detailed error information\n")
                    return {'error': 'First trial failed', 'early_termination': True}
            
            # Check if micro-retry is needed for poor layer coverage  
            coverage = layer_cov.get('coverage_overall', 0.0)
            min_threshold = max(0.55, self.optimizer.targets.min_layer_cov - 0.02)
            
            if coverage < min_threshold:
                self.logger.info("Low coverage detected, attempting micro-layer optimization")
                micro_metrics, micro_cov = self.optimizer.run_micro_layer_optimization(
                    iter_dir, geometry_params['outlet_names'], 
                    geometry_params['internal_point'], geometry_params['dx'], layer_cov)
                
                if micro_metrics and micro_cov:
                    layer_cov = micro_cov
                    layer_m.update(micro_metrics)
                    self.logger.info("Micro-loop applied: using improved layers-only outcome")
            
            # Evaluate trial quality and track best
            constraints_ok, failure_reasons = self.optimizer.meets_quality_constraints(snap_m, layer_m, layer_cov)
            coverage = layer_cov.get('coverage_overall', 0.0)
            
            if best_local is None or coverage > best_local[2].get('coverage_overall', 0):
                best_local = (snap_m, layer_m, layer_cov)
                
                diag_result = self._diagnose_layer_result(layer_cov)
                effective_layers = diag_result.get("effective_layers", 0.0)
                diagnosis = diag_result.get("diagnosis", "unknown")
                
                self.logger.info(f"Trial {trial}: thickness={coverage*100:.1f}% (N_eff={effective_layers:.2f}, {diagnosis}) - NEW BEST")
            else:
                self.logger.info(f"Trial {trial}: coverage={coverage*100:.1f}% (not better)")
            
            # Stop if we have good quality or reach target coverage
            if constraints_ok or coverage >= self.optimizer.targets.min_layer_cov:
                self.logger.info(f"Stopping micro-loop: "
                               f"{'constraints OK' if constraints_ok else f'coverage ≥{self.optimizer.targets.min_layer_cov*100:.0f}%'}")
                break
            
            # Apply parameter adaptations for next trial
            if trial < K:
                self.logger.info(f"Applying adaptations for trial {trial+1}")
                self.optimizer.apply_parameter_adaptations(coverage, trial, K)
        
        return {
            'mesh_data': best_local,
            'error': None,
            'early_termination': False
        }
    
    def _evaluate_iteration_results(self, k: int, iter_dir: Path, mesh_result: Dict, optimization_state: Dict) -> Dict:
        """Evaluate iteration results and prepare for next iteration."""
        if mesh_result.get('error'):
            return {'early_termination': True}
        
        snap_m, layer_m, layer_cov = mesh_result['mesh_data']
        
        # Enhanced final reporting
        diag_result = self._diagnose_layer_result(layer_cov)
        thickness_frac = diag_result.get("thickness_fraction", 0.0)
        effective_layers = diag_result.get("effective_layers", 0.0)
        diagnosis = diag_result.get("diagnosis", "unknown")
        
        self.logger.info(f"Selected best trial: thickness={thickness_frac*100:.1f}% (N_eff={effective_layers:.2f}, {diagnosis})")
        
        # Log recommendations for problematic layers
        if diagnosis in ["not-added-or-abandoned-early", "thin-but-present (added-then-pruned)", "barely-one-layer"]:
            recommendations = diag_result.get("recommendation", [])
            self.logger.info(f"Layer improvement suggestions: {'; '.join(recommendations[:3])}")
        
        # Log surface histogram and diagnostics
        self._log_iteration_diagnostics(iter_dir, layer_cov)
        
        # Evaluate constraints and cell count
        constraints_ok, failure_reasons = self.optimizer.meets_quality_constraints(snap_m, layer_m, layer_cov)
        cell_count = self.optimizer.get_cell_count(layer_m, snap_m)
        
        # Log iteration summary with diagnostic data
        thickness_frac, n_eff, diag = self._write_layer_diag(iter_dir, layer_cov.get('iteration_data'))
        self.optimizer.log_iteration_summary(
            optimization_state['summary_path'], k, cell_count, snap_m, layer_m, layer_cov,
            thickness_frac=thickness_frac, n_eff=n_eff, diag=diag)
        
        # Apply micro-reactive tuning for next iteration
        if k < self.max_iterations:
            dx = optimization_state.get('dx', 0.001)  # fallback value
            self._apply_micro_reactive_tuning(dx, thickness_frac, n_eff)
        
        return {
            'constraints_ok': constraints_ok,
            'cell_count': cell_count,
            'layer_coverage': layer_cov,
            'failure_reasons': failure_reasons,
            'early_termination': False
        }
    
    def _log_iteration_diagnostics(self, iter_dir: Path, layer_cov: Dict):
        """Log surface histogram and other diagnostic information."""
        if hasattr(self, 'surface_levels'):
            t1_thickness = self.config["LAYERS"].get("firstLayerThickness_abs", 50e-6)
            n_layers = self.config["LAYERS"].get("nSurfaceLayers", 6)
            expansion = self.config["LAYERS"].get("expansionRatio", 1.2)
            total_thickness = (t1_thickness * (expansion**n_layers - 1) / (expansion - 1) 
                             if expansion != 1.0 else t1_thickness * n_layers)
            
            log_surface_histogram(iter_dir, self.surface_levels, 0.001,  # default dx fallback
                                t1_thickness, total_thickness, self.logger)
    
    def _update_optimization_state(self, state: Dict, k: int, iteration_result: Dict) -> Dict:
        """Update optimization state after each iteration."""
        if iteration_result.get('early_termination'):
            return state
        
        # Track best iteration
        if (iteration_result['constraints_ok'] and 
            iteration_result['cell_count'] < state['best_cell_count']):
            state['best_iter'] = k
            state['best_cell_count'] = iteration_result['cell_count']
            self.logger.info(f"New best iteration: {k} with {iteration_result['cell_count']:,} cells")
        
        # Update coverage data for next iteration's gating
        state['previous_coverage_data'] = iteration_result['layer_coverage']
        self._previous_coverage_data = iteration_result['layer_coverage']  # Store for _apply_surface_level_progression
        
        # Log iteration results
        diag_result = self._diagnose_layer_result(iteration_result['layer_coverage'])
        thickness_frac = diag_result.get("thickness_fraction", 0.0)
        effective_layers = diag_result.get("effective_layers", 0.0)
        diagnosis = diag_result.get("diagnosis", "unknown")
        
        self.logger.info(f"Iteration {k} complete: {iteration_result['cell_count']:,} cells, "
                       f"thickness={thickness_frac*100:.1f}% (N_eff={effective_layers:.2f}, {diagnosis}), "
                       f"{'PASS' if iteration_result['constraints_ok'] else 'FAIL'}")
        
        if iteration_result['failure_reasons']:
            for reason in iteration_result['failure_reasons']:
                self.logger.info(f"  - {reason}")
        
        return state
    
    def _finalize_optimization(self, optimization_state: Dict):
        """Export best mesh and finalize optimization."""
        best_iter = optimization_state['best_iter']
        
        if best_iter:
            best_dir = self.output_dir / f"iter_{best_iter:03d}"
            export_dir = self.output_dir / "best"
            
            if export_dir.exists():
                shutil.rmtree(export_dir)
            shutil.copytree(best_dir, export_dir)
            
            self.logger.info(f"Exported best mesh from iteration {best_iter} to {export_dir}")
            
            # Evaluate Stage 1 metrics
            try:
                evaluate_stage1_metrics(export_dir, self.logger)
            except Exception as e:
                self.logger.warning(f"Stage 1 metrics evaluation failed: {e}")
        else:
            self.logger.warning("No iteration met quality constraints")
        
        self.logger.info("Stage‑1 optimization complete")
        return best_iter

