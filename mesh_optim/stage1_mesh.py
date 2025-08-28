"""Stage 1 mesh optimizer - Main orchestrator module.

This module coordinates geometry processing, dictionary generation, and optimization
to provide automated mesh generation for cardiovascular CFD applications.

Refactored to use modular architecture with focused sub-modules.
"""

import csv
import math
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from .constants import (
    MAX_ITERATIONS_DEFAULT,
    MICRO_TRIALS_PER_ITERATION,
    COVERAGE_ACCEPTABLE_THRESHOLD,
    SOLVER_MODES
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

        # Pull current knobs with safe type conversion
        def safe_float(val, default=0.0):
            try:
                return float(val) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        def safe_int(val, default=0):
            try:
                return int(val) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        def safe_bool(val, default=False):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ('true', '1', 'yes', 'on')
            return bool(val) if val is not None else default
        
        ER = safe_float(L.get("expansionRatio", 1.2))
        N = safe_int(L.get("nSurfaceLayers", 5))
        rel_mode = safe_bool(L.get("relativeSizes", False))

        # --- Rescue branch based on thickness ratio (0..1)
        if thickness_frac is not None:
            if thickness_frac < 0.15:  # abandoned/early prune
                # Prefer thinner first layer + gentler ER
                if not rel_mode:
                    t1 = safe_float(L.get("firstLayerThickness_abs", 50e-6))
                    new_t1 = max(5e-6, 0.80 * t1)
                    L["firstLayerThickness_abs"] = new_t1
                    self.logger.info(f"[micro] Early prune: t1 {t1*1e6:.1f}→{new_t1*1e6:.1f} μm")
                # lower ER a touch
                new_ER = max(1.12, ER - 0.03)
                if new_ER < ER:
                    L["expansionRatio"] = new_ER
                    self.logger.info(f"[micro] Early prune: ER {ER:.3f}→{new_ER:.3f}")
                # switch to relative sizes automatically on refined surfaces
                if max_level > 1 and not rel_mode:
                    L["relativeSizes"] = True
                    L["firstLayerThickness"] = 0.20
                    L["finalLayerThickness"] = 0.70
                    self.logger.info("[micro] Early prune: switch to relativeSizes (t1_rel=0.20, T_rel=0.70)")
            elif thickness_frac < 0.60:  # thin but present
                new_ER = max(1.12, ER - 0.02)
                if new_ER < ER:
                    L["expansionRatio"] = new_ER
                    self.logger.info(f"[micro] Thin layers: ER {ER:.3f}→{new_ER:.3f}")
                L["nLayerIter"]   = max(safe_int(L.get("nLayerIter", 50)), 70)
                L["nRelaxedIter"] = max(safe_int(L.get("nRelaxedIter", 20)), 25)
                self.logger.info(f"[micro] Thin layers: nLayerIter={L['nLayerIter']}, nRelaxedIter={L['nRelaxedIter']}")

        # --- Cap total thickness relative to surface Δx: T/Δx ≤ 0.9
        # Relative mode: T_rel is already T/Δx_surf
        if rel_mode:
            T_rel = safe_float(L.get("finalLayerThickness", 0.75))
            if T_rel > 0.90:
                new_T_rel = 0.85
                L["finalLayerThickness"] = new_T_rel
                self.logger.info(f"[micro] Cap T/Δx (relative): {T_rel:.3f}→{new_T_rel:.3f}")
        else:
            # Absolute mode: T_abs = t1 * (ER^N - 1)/(ER-1)
            t1 = safe_float(L.get("firstLayerThickness_abs", 50e-6))
            series = (ER**N - 1.0) / max(ER - 1.0, 1e-12) if ER > 1.0 else N
            T_abs = t1 * series
            T_over_dx = T_abs / max(dx_surf, 1e-12)
            if T_over_dx > 0.90:
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

        best_iter = None
        best_cell_count = math.inf
        previous_coverage_data = None
        
        # Initialize summary CSV
        summary_path = self.output_dir / "stage1_summary.csv"
        if not summary_path.exists():
            with open(summary_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "iter","cells","maxNonOrtho","maxSkewness","coverage","objective_dummy",
                    "levels_min","levels_max","resolveFeatureAngle","nLayers","firstLayer","minThickness",
                    "thicknessPct","N_eff","diag"   # NEW columns for layer diagnostics
                ])

        # Main iteration loop
        for k in range(1, self.max_iterations + 1):
            self.current_iteration = k
            self.optimizer.update_current_iteration(k)
            self.logger.info(f"=== ITERATION {k} ===")
            
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

            # 1) Copy/scale STL files and get outlet names
            outlet_names = self.geometry_handler.copy_and_scale_stl_files(iter_dir)
            
            # 2) Compute bounding box and reference diameters from scaled STL files
            stl_map_norm = {
                "inlet": iter_dir/"constant/triSurface/inlet.stl",
                self.geometry_handler.wall_name: iter_dir/f"constant/triSurface/{self.geometry_handler.wall_name}.stl",
                **{p.stem: iter_dir/f"constant/triSurface/{p.stem}.stl" 
                   for p in self.geometry_handler.stl_files["outlets"]}
            }
            
            # Generate bounding box using physics mesh generator
            gen = PhysicsAwareMeshGenerator()
            bbox_data = gen.compute_stl_bounding_box(stl_map_norm, skip_scaling=True)
            
            # Estimate diameters and derive base cell size
            tri_dir = iter_dir / "constant" / "triSurface"
            D_ref, D_min = self.geometry_handler.estimate_reference_diameters(stl_root=tri_dir)
            dx = self.geometry_handler.derive_base_cell_size(D_ref, D_min)
            
            # 3) Apply adaptive feature angle
            adaptive_angle = self.geometry_handler.adaptive_feature_angle(
                D_ref, D_min, len(outlet_names), iter_dir=iter_dir)
            self.config["SNAPPY"]["resolveFeatureAngle"] = adaptive_angle
            
            # 4) Generate OpenFOAM dictionaries
            self.dict_generator.generate_blockmesh_dict(iter_dir, bbox_data, dx)
            
            # Generate surface features dictionary
            surface_list = [f"{self.geometry_handler.wall_name}.stl", "inlet.stl"] + \
                          [f"{n}.stl" for n in outlet_names]
            self.dict_generator.generate_surface_features_dict(iter_dir, surface_list)
            
            # 5) Apply ladder progression and coverage gating
            ladder = self.stage1.get("ladder", [[1,1],[2,2],[2,3]])
            idx = min(self.current_iteration-1, len(ladder)-1)
            proposed_surface_levels = list(ladder[idx])
            
            if self.stage1.get("use_coverage_gated_progression", True):
                actual_surface_levels, progression_allowed = self.optimizer.check_coverage_gated_progression(
                    proposed_surface_levels, previous_coverage_data)
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

            # 6) Calculate internal seed point from scaled STL files
            internal_point = self.geometry_handler.calculate_seed_point(
                bbox_data, {"inlet": tri_dir / "inlet.stl"}, dx_base=dx)
            
            # 7) Apply dynamic feature snap iterations
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

            # 8) Generate snappy dictionaries
            self.dict_generator.generate_snappy_dict(iter_dir, outlet_names, internal_point, dx, "no_layers")
            self.dict_generator.generate_snappy_dict(iter_dir, outlet_names, internal_point, dx, "layers")
            
            # Create .foam file for visualization
            self.dict_generator.create_foam_file(iter_dir)

            # 9) Run mesh generation with micro-loop optimization
            K = MICRO_TRIALS_PER_ITERATION
            best_local = None
            
            for trial in range(1, K+1):
                self.logger.info(f"Micro-trial {trial}/{K} for iteration {k}")
                
                # Run mesh generation
                snap_m, layer_m, layer_cov = self.optimizer.run_mesh_generation(
                    iter_dir, force_full_remesh=(trial == 1 and surface_levels_changed))
                
                # Check for critical mesh generation failure
                if not snap_m.get('meshOK', False) or 'error' in snap_m:
                    self.logger.error(f"CRITICAL: Mesh generation failed at iteration {k}, trial {trial}")
                    self.logger.error(f"Error: {snap_m.get('error', 'Unknown error')}")
                    self.logger.error("Stopping optimization due to mesh generation failure")
                    # Write summary indicating failure
                    with open(self.output_dir / "OPTIMIZATION_FAILED.txt", "w") as f:
                        f.write(f"Optimization failed at iteration {k}, trial {trial}\n")
                        f.write(f"Error: {snap_m.get('error', 'Unknown error')}\n")
                        f.write("Check iter_XXX/logs/ for detailed error information\n")
                    return None  # Exit optimization
                
                # Check if micro-retry is needed for poor layer coverage
                coverage = layer_cov.get('coverage_overall', 0.0)
                min_threshold = max(0.55, self.optimizer.targets.min_layer_cov - 0.10)
                
                if coverage < min_threshold:
                    self.logger.info("Low coverage detected, attempting micro-layer optimization")
                    micro_metrics, micro_cov = self.optimizer.run_micro_layer_optimization(
                        iter_dir, outlet_names, internal_point, dx, layer_cov)
                    
                    if micro_metrics and micro_cov:
                        layer_cov = micro_cov
                        layer_m.update(micro_metrics)
                        self.logger.info("Micro-loop applied: using improved layers-only outcome")
                
                # Evaluate trial quality
                constraints_ok, failure_reasons = self.optimizer.meets_quality_constraints(snap_m, layer_m, layer_cov)
                coverage = layer_cov.get('coverage_overall', 0.0)
                
                # Track best trial by coverage and provide enhanced diagnostics
                if best_local is None or coverage > best_local[2].get('coverage_overall', 0):
                    best_local = (snap_m, layer_m, layer_cov)
                    
                    # Enhanced diagnostics
                    diag_result = self._diagnose_layer_result(layer_cov)
                    effective_layers = diag_result.get("effective_layers", 0.0)
                    diagnosis = diag_result.get("diagnosis", "unknown")
                    
                    self.logger.info(f"Trial {trial}: thickness={coverage*100:.1f}% (N_eff={effective_layers:.2f}, {diagnosis}) - NEW BEST")
                else:
                    self.logger.info(f"Trial {trial}: coverage={coverage*100:.1f}% (not better)")
                
                # Stop if we have good quality or acceptable coverage
                if constraints_ok or coverage >= COVERAGE_ACCEPTABLE_THRESHOLD:
                    self.logger.info(f"Stopping micro-loop: "
                                   f"{'constraints OK' if constraints_ok else f'coverage ≥{COVERAGE_ACCEPTABLE_THRESHOLD*100:.0f}%'}")
                    break
                
                # Apply parameter adaptations for next trial
                if trial < K:
                    self.logger.info(f"Applying adaptations for trial {trial+1}")
                    self.optimizer.apply_parameter_adaptations(coverage, trial, K)

            # Use the best trial results
            snap_m, layer_m, layer_cov = best_local
            
            # Enhanced final reporting
            diag_result = self._diagnose_layer_result(layer_cov)
            thickness_frac = diag_result.get("thickness_fraction", 0.0)
            effective_layers = diag_result.get("effective_layers", 0.0)
            diagnosis = diag_result.get("diagnosis", "unknown")
            
            self.logger.info(f"Selected best trial: thickness={thickness_frac*100:.1f}% (N_eff={effective_layers:.2f}, {diagnosis})")
            
            # Log recommendations if layers are problematic
            if diagnosis in ["not-added-or-abandoned-early", "thin-but-present (added-then-pruned)", "barely-one-layer"]:
                recommendations = diag_result.get("recommendation", [])
                self.logger.info(f"Layer improvement suggestions: {'; '.join(recommendations[:3])}")  # Show top 3
            
            # 10) Log surface histogram for analysis
            if hasattr(self, 'surface_levels'):
                t1_thickness = self.config["LAYERS"].get("firstLayerThickness_abs", 50e-6)
                n_layers = self.config["LAYERS"].get("nSurfaceLayers", 6)
                expansion = self.config["LAYERS"].get("expansionRatio", 1.2)
                total_thickness = (t1_thickness * (expansion**n_layers - 1) / (expansion - 1) 
                                 if expansion != 1.0 else t1_thickness * n_layers)
                
                log_surface_histogram(iter_dir, self.surface_levels, dx, 
                                    t1_thickness, total_thickness, self.logger)
            
            # NEW: Write layer diagnostics and get thickness fraction for reactive tuning
            thickness_frac, n_eff, diag = self._write_layer_diag(iter_dir, layer_cov.get('iteration_data'))
            self.logger.info(f"Layer diagnosis: {diag}")
            
            # Update coverage data for next iteration's gating
            previous_coverage_data = layer_cov
            
            # 11) Evaluate iteration and log results
            constraints_ok, failure_reasons = self.optimizer.meets_quality_constraints(snap_m, layer_m, layer_cov)
            cell_count = self.optimizer.get_cell_count(layer_m, snap_m)
            
            # Log iteration summary with diagnostic data
            self.optimizer.log_iteration_summary(summary_path, k, cell_count, snap_m, layer_m, layer_cov,
                                                thickness_frac=thickness_frac, n_eff=n_eff, diag=diag)
            
            # Track best iteration
            if constraints_ok and cell_count < best_cell_count:
                best_iter = k
                best_cell_count = cell_count
                self.logger.info(f"New best iteration: {k} with {cell_count:,} cells")
            
            # Log iteration results with enhanced diagnostics
            diag_result = self._diagnose_layer_result(layer_cov)
            thickness_frac = diag_result.get("thickness_fraction", 0.0)
            effective_layers = diag_result.get("effective_layers", 0.0)
            diagnosis = diag_result.get("diagnosis", "unknown")
            
            self.logger.info(f"Iteration {k} complete: {cell_count:,} cells, "
                           f"thickness={thickness_frac*100:.1f}% (N_eff={effective_layers:.2f}, {diagnosis}), "
                           f"{'PASS' if constraints_ok else 'FAIL'}")
            
            if failure_reasons:
                for reason in failure_reasons:
                    self.logger.info(f"  - {reason}")
            
            # NEW: Apply micro-reactive tuning for the next iteration
            if k < self.max_iterations:  # Only apply if there's a next iteration
                self._apply_micro_reactive_tuning(dx, thickness_frac, n_eff)

        # Final evaluation and export best mesh
        if best_iter:
            best_dir = self.output_dir / f"iter_{best_iter:03d}"
            export_dir = self.output_dir / "best"
            
            if export_dir.exists():
                import shutil
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