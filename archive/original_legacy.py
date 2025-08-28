import json
import csv
import math
import re
import numpy as np
import struct
from pathlib import Path
import shutil
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from .utils import (
    run_command, 
    check_mesh_quality, 
    parse_layer_coverage,
    parse_physics_aware_coverage,
    log_surface_histogram,
    parse_layer_iterations,
    evaluate_stage1_metrics
)
from .physics_mesh import PhysicsAwareMeshGenerator
from .constants import (
    FEATURE_ANGLE_DEFAULT,
    FEATURE_ANGLE_MIN,
    FEATURE_ANGLE_MAX,
    FEATURE_ANGLE_RECOMMENDED,
    FIRST_LAYER_MIN,
    FIRST_LAYER_MAX,
    FIRST_LAYER_DEFAULT,
    MAX_ITERATIONS_DEFAULT,
    MICRO_TRIALS_PER_ITERATION,
    COVERAGE_ACCEPTABLE_THRESHOLD,
    FEATURE_SNAP_ITER_LOW_ANGLE,
    FEATURE_SNAP_ITER_HIGH_ANGLE,
    FEATURE_ANGLE_LOW_THRESHOLD,
    FEATURE_ANGLE_HIGH_THRESHOLD,
    WALL_PATCH_NAMES,
    WALL_PATCH_DEFAULT,
    SOLVER_MODES,
    STL_UNIT_DETECTION_FACTOR,
    STL_MM_TO_M_CONVERSION
)
from .geometry_utils import (
    read_stl_triangles,
    find_internal_seed_point,
    detect_stl_units
)

@dataclass
class Stage1Targets:
    max_nonortho: float
    max_skewness: float
    min_layer_cov: float


def _safe_get(d: dict, path: List[str], default=None):
    x = d
    for k in path:
        if not isinstance(x, dict) or k not in x:
            return default
        x = x[k]
    return x


class Stage1MeshOptimizer:
    """Geometry‑aware Stage‑1 mesh optimizer (geometry only, meters everywhere)."""

    def __init__(self, geometry_dir, config_file, output_dir=None):
        self.geometry_dir = Path(geometry_dir)
        self.config_file = Path(config_file)
        self.output_dir = Path(output_dir) if output_dir else (self.geometry_dir.parent / "output" / "stage1_mesh")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config_file) as f:
            self.config = json.load(f)
        
        # Ensure SNAPPY section exists to prevent KeyError
        self.config.setdefault("SNAPPY", {})
        
        # Map from new two-tier structure to internal format if needed
        self._map_config_structure()

        self.logger = logging.getLogger(f"Stage1Mesh_{self.geometry_dir.name}")

        # Resource management
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        self.max_memory_gb = min(available_memory_gb * 0.7, 12)  # 70% of available, cap 12GB
        self.logger.info(f"Stage 1 memory limit: {self.max_memory_gb:.1f}GB")

        # Stage‑1 policy (needed for max_iterations)
        self.stage1 = _safe_get(self.config, ["STAGE1"], {}) or {}
        
        # Iteration state
        self.current_iteration = 0
        # Check STAGE1 section first, then root level, then default
        self.max_iterations = int(self.stage1.get("max_iterations", self.config.get("max_iterations", MAX_ITERATIONS_DEFAULT)))
        self.logger.debug(f"Max iterations: {self.max_iterations} (from {'STAGE1' if 'max_iterations' in self.stage1 else 'root' if 'max_iterations' in self.config else 'default'})")
        self.surface_levels = list(_safe_get(self.config, ["SNAPPY", "surface_level"], [1, 1]))

        # Discover wall patch name generically
        self.wall_name = self._discover_wall_name()
        
        # STL discovery (now that we know wall name)
        self.stl_files = self._discover_stl_files()
        if "featureAngle_init" in self.stage1:
            self.config["SNAPPY"]["resolveFeatureAngle"] = self.stage1["featureAngle_init"]
        
        # Apply improved memory-aware cell budgeting
        self._apply_improved_memory_budgeting()
        
        # Geometry policy for vessel-agnostic sizing
        self.geometry_policy = _safe_get(self.config, ["GEOMETRY_POLICY"], {}) or {}
        
        # Optional physics block for y+ based layer sizing
        self.physics = _safe_get(self.config, ["PHYSICS"], {}) or {}

        # Acceptance/targets - can be overridden by solver mode
        self.targets = Stage1Targets(
            max_nonortho=float(_safe_get(self.config, ["acceptance_criteria", "maxNonOrtho"], 65)),
            max_skewness=float(_safe_get(self.config, ["acceptance_criteria", "maxSkewness"], 4.0)),
            min_layer_cov=float(_safe_get(self.config, ["acceptance_criteria", "min_layer_coverage"], 0.65)),
        )
        
        # Apply solver-specific acceptance criteria if configured
        self._apply_solver_presets()
        
        # Apply proven mesh generation best practices
        self._apply_robust_defaults()

    def _map_config_structure(self):
        """Map from new two-tier configuration structure to internal format"""
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
            
            # Preserve advanced overrides from the advanced section
            if "advanced" in self.config:
                advanced = self.config["advanced"]
                
                # Override with advanced settings if they exist
                for section in ["BLOCKMESH", "SNAPPY", "LAYERS", "MESH_QUALITY", "SURFACE_FEATURES", 
                               "GEOMETRY_POLICY", "STAGE1", "SCALING", "PHYSICS"]:
                    if section in advanced:
                        if section not in self.config:
                            self.config[section] = {}
                        self.config[section].update(advanced[section])

    def _apply_solver_presets(self):
        """Apply solver-specific acceptance criteria based on intended physics"""
        solver_mode = self.physics.get("solver_mode", "").upper()
        
        if solver_mode == "LES":
            # LES with near-wall resolution: tighter quality requirements, higher coverage for WSS
            self.targets.max_nonortho = min(self.targets.max_nonortho, 60.0)
            self.targets.max_skewness = min(self.targets.max_skewness, 3.5)
            self.targets.min_layer_cov = max(self.targets.min_layer_cov, 0.80)  # Higher for WSS accuracy
            self.logger.info(f"Applied LES solver presets: maxNonOrtho≤{self.targets.max_nonortho}, "
                           f"maxSkewness≤{self.targets.max_skewness}, coverage≥{self.targets.min_layer_cov:.0%}")
            
        elif solver_mode == "RANS":
            # RANS with wall functions: moderate quality requirements
            self.targets.max_nonortho = min(self.targets.max_nonortho, 65.0)
            self.targets.max_skewness = min(self.targets.max_skewness, 4.0)
            self.targets.min_layer_cov = max(self.targets.min_layer_cov, 0.70)
            self.logger.info(f"Applied RANS solver presets: maxNonOrtho≤{self.targets.max_nonortho}, "
                           f"maxSkewness≤{self.targets.max_skewness}, coverage≥{self.targets.min_layer_cov:.0%}")
            
        elif solver_mode == "LAMINAR":
            # Laminar flow: relaxed layer coverage requirement
            self.targets.max_nonortho = min(self.targets.max_nonortho, 65.0)
            self.targets.max_skewness = min(self.targets.max_skewness, 4.0)
            self.targets.min_layer_cov = max(self.targets.min_layer_cov, 0.65)
            self.logger.info(f"Applied Laminar solver presets: maxNonOrtho≤{self.targets.max_nonortho}, "
                           f"maxSkewness≤{self.targets.max_skewness}, coverage≥{self.targets.min_layer_cov:.0%}")
        
        # If solver_mode not specified or unrecognized, keep user-configured values
    
    def _apply_robust_defaults(self):
        """Apply proven mesh generation best practices from successful manual runs"""
        
        # Ensure conservative surface refinement progression
        ladder = self.stage1.get("ladder", [[1,1],[1,2],[2,2]])
        if len(ladder) > 0 and len(ladder[0]) == 2:
            # Ensure no iteration jumps too aggressively
            max_jump = 0
            for i, (min_lvl, max_lvl) in enumerate(ladder):
                if i > 0:
                    prev_max = ladder[i-1][1] 
                    jump = max_lvl - prev_max
                    if jump > 1:
                        self.logger.info(f"Limiting surface level jump in ladder iteration {i}: ({min_lvl},{max_lvl}) → ({min_lvl},{prev_max+1})")
                        ladder[i] = [min_lvl, prev_max + 1]
            self.stage1["ladder"] = ladder
        
        # Ensure reasonable layer settings for vascular meshing
        layers = self.config.get("LAYERS", {})
                
        # Ensure first layer thickness is reasonable for vascular scale
        first_layer = layers.get("firstLayerThickness_abs", FIRST_LAYER_DEFAULT)
        if first_layer is not None and (first_layer < FIRST_LAYER_MIN or first_layer > FIRST_LAYER_MAX):
            recommended = FIRST_LAYER_DEFAULT
            self.logger.info(f"Adjusting first layer thickness: {first_layer*1e6:.1f}μm → {recommended*1e6:.1f}μm")
            layers["firstLayerThickness_abs"] = recommended
            
        # Ensure feature angle detection is in proven range
        snap = self.config.get("SNAPPY", {})
        resolve_angle = snap.get("resolveFeatureAngle", FEATURE_ANGLE_DEFAULT)
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

    # ------------------------ Geometry & base size -------------------------
    def _discover_stl_files(self) -> Dict:
        required = ["inlet.stl"]
        found = {"required": {}, "outlets": []}
        
        # Find inlet
        for name in required:
            p = self.geometry_dir / name
            if not p.exists():
                raise FileNotFoundError(f"Required STL file not found: {name}")
            found["required"][name.split('.')[0]] = p
        
        # Find wall (using discovered name)
        wall_path = self.geometry_dir / f"{self.wall_name}.stl"
        if not wall_path.exists():
            raise FileNotFoundError(f"Wall STL file not found: {self.wall_name}.stl")
        found["required"][self.wall_name] = wall_path
        
        for p in self.geometry_dir.glob("outlet*.stl"):
            found["outlets"].append(p)
        if not found["outlets"]:
            raise FileNotFoundError("No outlet STL files found")
        self.logger.info(f"Found {len(found['outlets'])} outlet files")
        return found
    
    def _apply_improved_memory_budgeting(self) -> None:
        """Apply improved memory-aware cell budgeting."""
        import psutil
        
        # Get current system resources
        available_gb = psutil.virtual_memory().available / (1024**3)
        n_procs = self.stage1.get("n_processors", 1)
        kb_per_cell = self.stage1.get("cell_budget_kb_per_cell", 2.0)
        
        # Calculate memory-aware limits
        # Use 70% of available memory for safety
        usable_gb = available_gb * 0.7
        usable_kb = usable_gb * 1024 * 1024
        
        # Calculate total cells budget
        total_cells = int(usable_kb / max(kb_per_cell, 0.5))
        
        # Distribute across processors
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
        
        if original_local > max_local or original_global > max_global:
            self.logger.info(f"   Reduced from config: Local {original_local:,}→{self.config['SNAPPY']['maxLocalCells']:,}, "
                           f"Global {original_global:,}→{self.config['SNAPPY']['maxGlobalCells']:,}")

    def _get_bbox_dimensions(self, bbox_dict):
        """
        Return (Lx, Ly, Lz) in the *same units as the STL coordinates*.
        Supports either:
          - {"dimensions": {"length":..., "width":..., "height":...}} or {"dx","dy","dz"}
          - {"mesh_domain": {"x_min","x_max","y_min","y_max","z_min","z_max"}}
        """
        # 1) direct dimensions block
        dims = bbox_dict.get("dimensions", {})
        if all(k in dims for k in ("length", "width", "height")):
            return dims["length"], dims["width"], dims["height"]
        if all(k in dims for k in ("dx", "dy", "dz")):
            return dims["dx"], dims["dy"], dims["dz"]
        
        # 2) compute from mesh_domain extents
        md = bbox_dict.get("mesh_domain", {})
        keys = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        if all(k in md for k in keys):
            Lx = float(md["x_max"]) - float(md["x_min"])
            Ly = float(md["y_max"]) - float(md["y_min"])
            Lz = float(md["z_max"]) - float(md["z_min"])
            return Lx, Ly, Lz
        
        # 3) last resort defaults (log once)
        self.logger.warning("BBox dict missing expected keys; returning conservative defaults.")
        return 0.02, 0.02, 0.02

    def _compute_block_divisions(self, bbox_size, dx_base):
        """Compute blockMesh divisions with hierarchical override control
        
        Precedence: divisions > cell_size_m > resolution > geometry-aware
        """
        bm = self.config.get("BLOCKMESH", {}) or {}
        # Handle null/None values in config properly
        min_per_axis_config = bm.get("min_per_axis", [10, 10, 10])
        if min_per_axis_config is None:
            min_per_axis_config = [10, 10, 10]
        mins = np.array(min_per_axis_config, dtype=int)

        if "divisions" in bm and bm["divisions"] is not None:
            divs = np.array(bm["divisions"], dtype=int)
            return np.maximum(divs, mins)

        if "cell_size_m" in bm and bm["cell_size_m"] is not None:
            dx = float(bm["cell_size_m"])
            divs = np.ceil(bbox_size / max(dx, 1e-9)).astype(int)
            return np.maximum(divs, mins)

        if "resolution" in bm:
            R = int(bm["resolution"])
            Lmax = float(max(bbox_size))
            scale = R / max(Lmax, 1e-12)  # cells per meter
            divs = np.ceil(bbox_size * scale).astype(int)
            return np.maximum(divs, mins)

        # default: geometry-aware from dx_base
        if dx_base is None:
            self.logger.error(f"dx_base is None in _compute_block_divisions - this should not happen")
            dx_base = 1e-3  # 1mm fallback
        divs = np.ceil(bbox_size / max(dx_base, 1e-9)).astype(int)
        return np.maximum(divs, mins)
    
    def _discover_wall_name(self) -> str:
        """Discover wall patch name generically across vascular beds"""
        for name in WALL_PATCH_NAMES:
            if (self.geometry_dir / f"{name}.stl").exists():
                self.logger.info(f"Wall patch discovered: {name}")
                return name
        # Fallback to default
        self.logger.warning(f"No standard wall patch found, using '{WALL_PATCH_DEFAULT}'")
        return WALL_PATCH_DEFAULT

    def _estimate_reference_diameters(self, stl_root=None) -> Tuple[float, float]:
        """Return (D_ref, D_min) in METERS - vessel-agnostic approach.
        Derives sizes from actual geometry without anatomy-specific assumptions.
        
        Args:
            stl_root: Optional directory containing STL files to read from.
                     If None, uses original STL files. If provided, reads from scaled STLs.
        """
        policy = self.geometry_policy
        mode = str(policy.get("diameter_mode", "auto")).lower()
        clamp = str(policy.get("clamp_mode", "none")).lower()
        s_guard = float(policy.get("throat_guard_scale", 0.85))
        
        Deq = []  # Initialize for all branches to prevent NameError
        
        if mode == "fixed":
            # User-provided fixed values
            D_ref = float(policy["fixed_D_ref_m"])
            D_min = float(policy["fixed_D_min_m"])
        else:
            # Compute equivalent diameters from outlet areas
            Deq = []
            env = self.config["openfoam_env_path"]
            gen = PhysicsAwareMeshGenerator()  # Instantiate early for bbox calculations
            
            # Determine which STL files to use
            if stl_root:
                # Use scaled STL files from triSurface directory
                outlet_paths = [stl_root / p.name for p in self.stl_files["outlets"]]
                inlet_path = stl_root / "inlet.stl"
                wall_path = stl_root / f"{self.wall_name}.stl"
            else:
                # Use original STL files
                outlet_paths = list(self.stl_files["outlets"])
                inlet_path = self.stl_files["required"]["inlet"]
                wall_path = self.stl_files["required"][self.wall_name]
            
            # Try to get areas from outlet patches
            for p in outlet_paths:
                try:
                    # This should not create constant/ at output_dir level
                    # All OpenFOAM commands should run from iter_dir context
                    # Use direct STL path for surfaceCheck
                    res = run_command(["surfaceCheck", str(p)],
                                    cwd=self.geometry_dir.parent, env_setup=env, timeout=600, 
                                    max_memory_gb=self.max_memory_gb)
                    txt = (res.stdout + res.stderr).lower()
                    
                    # Parse surface area from output
                    area = None
                    for line in txt.splitlines():
                        if "area" in line and ("total" in line or "surface" in line):
                            # Extract the last float on the line
                            tokens = line.replace("=", " ").replace(":", " ").split()
                            for t in reversed(tokens):
                                try:
                                    val = float(t)
                                    if val > 0:
                                        area = val
                                        break
                                except ValueError:
                                    continue
                    
                    if area and area > 0:
                        if stl_root:
                            # For scaled STLs, area should already be in correct units (m²)
                            stem_name = p.name.replace('.stl', '')
                            self.logger.debug(f"Outlet {stem_name} (scaled): area={area:.6f} m²")
                        else:
                            # For original STLs, apply improved unit detection
                            ob = gen.compute_stl_bounding_box({p.stem: p})
                            bbox_dims = np.array(self._get_bbox_dimensions(ob))
                            
                            # Use improved unit detection function
                            detected_units = detect_stl_units(area, bbox_dims, self.logger)
                            
                            if detected_units == 'mm':
                                area_mm2 = area
                                area = area * STL_MM_TO_M_CONVERSION
                                self.logger.info(f"Outlet {p.stem}: detected mm² units ({area_mm2:.1f} mm²) → converted to {area:.6f} m²")
                            else:
                                self.logger.debug(f"Outlet {p.stem}: area={area:.6f} m² (units confirmed as meters)")
                        
                        Deq_outlet = 2.0 * math.sqrt(area / math.pi)
                        Deq.append(Deq_outlet)
                        stem_name = p.name.replace('.stl', '') if stl_root else p.stem
                        self.logger.debug(f"Outlet {stem_name}: area={area:.6f} m², D_eq={Deq_outlet:.4f} m")
                except Exception as e:
                    self.logger.debug(f"Could not compute area for {p.name}: {e}")
            
            # Add inlet diameter to the mix
            if mode == "inlet_only" or (mode == "auto" and len(Deq) < len(self.stl_files["outlets"])):
                # Use inlet bbox as robust proxy
                # If using scaled STLs, skip additional scaling
                ib = gen.compute_stl_bounding_box({"inlet": inlet_path}, skip_scaling=(stl_root is not None))
                lx, ly, lz = self._get_bbox_dimensions(ib)
                # Use median of inlet dimensions as proxy
                D_in = float(np.median([lx, ly, lz]))
                if mode == "inlet_only":
                    Deq = [D_in]
                else:
                    Deq.append(D_in)
                scale_info = " (scaled)" if stl_root else ""
                self.logger.debug(f"Inlet bbox proxy{scale_info}: D_in={D_in:.4f} m")
            
            if not Deq:
                # Final fallback: use global bbox
                if stl_root:
                    # Use scaled wall STL - skip additional scaling
                    bbox = gen.compute_stl_bounding_box({self.wall_name: wall_path}, skip_scaling=True)
                else:
                    # Use original STL map - apply scaling
                    stl_map = {**self.stl_files["required"], **{p.stem: p for p in self.stl_files["outlets"]}}
                    bbox = gen.compute_stl_bounding_box(stl_map, skip_scaling=False)
                lx, ly, lz = self._get_bbox_dimensions(bbox)
                D_ref = min(lx, ly)
                D_min = D_ref * s_guard
                scale_info = " (scaled)" if stl_root else ""
                self.logger.info(f"Diameter fallback{scale_info}: using bbox → D_ref={D_ref:.4f}m, D_min={D_min:.4f}m (no outlet area data)")
            else:
                # Use median for reference, minimum for throat
                D_ref = float(np.median(Deq))
                D_min = float(min(Deq)) if len(Deq) else D_ref
                # Apply throat guard scale for internal narrowings
                D_min = min(D_min, s_guard * D_ref)
        
        # Optional loose sanity clamp (only for unit mishaps)
        if clamp == "loose":
            # Very wide band: 1mm to 50mm - covers most vascular beds
            D_ref = float(np.clip(D_ref, 1e-3, 5e-2))
            D_min = float(np.clip(D_min, 5e-4, D_ref))
        
        # Log detailed diameter derivation
        if Deq:
            self.logger.info(f"Diameter derivation: {len(Deq)} measurements → D_ref={D_ref:.4f}m (median), D_min={D_min:.4f}m (min)")
            self.logger.debug(f"All equivalent diameters: {[f'{d:.4f}m' for d in Deq]}")
            
            # Sanity check: warn if any diameter is unrealistically small
            min_realistic_diameter = 0.2e-3  # 0.2 mm in meters
            for i, d in enumerate(Deq):
                if d < min_realistic_diameter:
                    self.logger.warning(f"Equivalent diameter D_eq[{i}]={d*1e3:.3f}mm < 0.2mm - likely a scaling issue or degenerate outlet!")
                    self.logger.warning(f"   Check your STL units or outlet geometry. Common causes:")
                    self.logger.warning(f"   - STL file in wrong units (m vs mm)")
                    self.logger.warning(f"   - Outlet patch is collapsed or has near-zero area")
                    self.logger.warning(f"   - Incorrect scale_m setting in config (currently {self.config.get('SCALING', {}).get('scale_m', 1.0)})")
        
        self.logger.info(f"[GeomStats] D_ref={D_ref:.4f} m, D_min={D_min:.4f} m "
                        f"(mode={mode}, clamp={clamp}, n_outlets={len(self.stl_files['outlets'])})")
        return D_ref, D_min

    def _adaptive_feature_angle(self, D_ref: float, D_min: float, n_outlets: int, iter_dir=None) -> int:
        """Calculate adaptive resolveFeatureAngle for vascular curvature (30-45°)
        
        Per OpenFOAM guidance: start at 30°, increase only if refinement is excessive
        Lower angles capture more curvature/features (good for tortuous vessels)
        Higher angles reduce over-refinement (good for smoother vessels)
        
        Enhanced with curvature analysis for vessel-aware adaptation.
        """
        beta = D_min / max(D_ref, 1e-6)  # Narrowness ratio
        
        # Baseline within documented range (30-45°)
        base = 35  # Start more aggressive for vessel branching
        
        # Narrower sections: capture more curvature (reduce angle)
        if beta < 0.6:
            base -= 5  # Down to ~32°
            
        # More branches: capture more features (reduce angle)
        if n_outlets >= 3:
            base -= 4  # Additional reduction for complex branching
        
        # Curvature-aware adjustment
        curvature_adjustment = 0
        if iter_dir:
            wall_stl = iter_dir / "constant" / "triSurface" / f"{self.wall_name}.stl"
            if wall_stl.exists():
                try:
                    cs = self._estimate_curvature_strength(wall_stl)
                    strength = cs["strength"]
                    # Map curvature strength [0,1] to angle adjustment [-5°, +3°]
                    # Higher curvature → lower angle → more feature detection
                    curvature_adjustment = int(3 - (strength * 8))
                    base += curvature_adjustment
                    self.logger.info(f"Curvature adjustment: strength={strength:.3f} → {curvature_adjustment:+d}° ({cs['nTris']} tris)")
                except Exception as e:
                    self.logger.debug(f"Curvature analysis failed for resolveFeatureAngle: {e}")
            
        # Constrain to documented vascular range (30-45°)
        angle = int(np.clip(base, 30, 45))
        
        components = f"β={beta:.2f}, n_out={n_outlets}"
        if curvature_adjustment != 0:
            components += f", curv={curvature_adjustment:+d}°"
        
        self.logger.info(f"Adaptive resolveFeatureAngle: {components} → {angle}° (vascular curvature)")
        return angle

    def _derive_base_cell_size(self, D_ref=None, D_min=None) -> float:
        mode = self.stage1.get("base_size_mode", "diameter").lower()
        if mode not in ("diameter", "density"):
            mode = "diameter"
        if mode == "diameter":
            N_D = int(self.stage1.get("N_D", 22))
            N_Dmin = int(self.stage1.get("N_D_min", 28))
            
            # Use provided diameters or compute them
            if D_ref is None or D_min is None:
                D_ref, D_min = self._estimate_reference_diameters()
            
            dx = min(D_ref / max(N_D, 1), D_min / max(N_Dmin, 1))
        else:
            k = float(self.stage1.get("cells_per_cm", 12))
            dx = 0.01 / max(k, 1e-9)
        # bound dx to avoid extremes
        dx = float(np.clip(dx, 1e-4, 5e-3))  # 0.1 mm .. 5 mm
        self.logger.info(f"Stage‑1 base cell size Δx = {dx*1e3:.2f} mm")
        return dx

    def _point_inside_stl(self, stl_path: Path, p, n_axes=3) -> bool:
        """Test if point p is inside closed STL using odd/even ray casting along multiple axes."""
        import numpy as _np
        axes = _np.eye(3, dtype=float)
        hits = 0
        for sign in (-1.0, 1.0):
            for a in range(n_axes):
                ray_o = _np.array(p, float)
                ray_d = axes[a] * sign
                count = 0
                for n, v1, v2, v3 in self._iter_stl_triangles(stl_path):
                    # Möller–Trumbore ray-triangle intersection
                    e1 = _np.array(v2) - _np.array(v1)
                    e2 = _np.array(v3) - _np.array(v1)
                    h = _np.cross(ray_d, e2)
                    det = e1.dot(h)
                    if abs(det) < 1e-14: 
                        continue
                    inv = 1.0/det
                    s = ray_o - _np.array(v1)
                    u = inv * s.dot(h)
                    if u < 0.0 or u > 1.0: 
                        continue
                    q = _np.cross(s, e1)
                    v = inv * ray_d.dot(q)
                    if v < 0.0 or u + v > 1.0: 
                        continue
                    t = inv * e2.dot(q)
                    if t > 1e-12:  # forward hit
                        count += 1
                if count % 2 == 1:
                    hits += 1
        # inside if majority of rays say "inside"
        return hits >= 3

    def _calculate_robust_seed_point(self, bbox_data, stl_files, dx_base=None):
        """
        Robustly compute a locationInMesh that is guaranteed to be inside the lumen.
        Uses inlet centroid + inward step + inside verification.

        Args:
            bbox_data: dict from PhysicsAwareMeshGenerator.compute_stl_bounding_box(...)
            stl_files: dict with scaled STL paths
            dx_base:   base cell size (meters) for step lengths
        Returns:
            np.ndarray shape (3,) point inside lumen (meters)
        """
        # Locate triSurface paths
        if "required" in stl_files and "inlet" in stl_files["required"]:
            inlet_path = Path(stl_files["required"]["inlet"])
        else:
            # Fallback for different dict structure
            inlet_path = stl_files.get("inlet", self.stl_files["required"]["inlet"])
            if not isinstance(inlet_path, Path):
                inlet_path = Path(inlet_path)
        
        # Ensure inlet exists
        if not inlet_path.exists():
            raise FileNotFoundError(f"Inlet STL not found at {inlet_path}")
        
        # Read inlet triangles using new utility function
        inlet_triangles = read_stl_triangles(inlet_path)
        
        # Extract bbox dimensions
        bbox_size = np.array([
            bbox_data["mesh_domain"]["x_max"] - bbox_data["mesh_domain"]["x_min"],
            bbox_data["mesh_domain"]["y_max"] - bbox_data["mesh_domain"]["y_min"],
            bbox_data["mesh_domain"]["z_max"] - bbox_data["mesh_domain"]["z_min"]
        ])
        
        bbox_center = np.array([
            (bbox_data["mesh_domain"]["x_min"] + bbox_data["mesh_domain"]["x_max"]) * 0.5,
            (bbox_data["mesh_domain"]["y_min"] + bbox_data["mesh_domain"]["y_max"]) * 0.5,
            (bbox_data["mesh_domain"]["z_min"] + bbox_data["mesh_domain"]["z_max"]) * 0.5
        ])
        
        # Use the new find_internal_seed_point function
        seed_point = find_internal_seed_point(
            bbox_center, 
            bbox_size, 
            inlet_triangles,
            dx_base,
            self.logger
        )
        
        return seed_point

    # ------------------------ Curvature analysis helpers -------------------------
    def _iter_stl_triangles(self, stl_path: Path):
        """Yield (normal, v1, v2, v3) for each triangle in an STL.
        Works for ASCII and binary STL. If stored normal is zero, recompute from vertices.
        """
        try:
            # Try ASCII first
            with open(stl_path, "r", encoding="utf-8") as f:
                line = f.readline()
                if not line.lower().startswith("solid"):
                    raise UnicodeDecodeError("","",0,0,"not ascii")
                f.seek(0)
                normal = None
                verts = []
                for line in f:
                    s = line.strip()
                    if s.startswith("facet normal"):
                        parts = s.split()
                        try:
                            normal = [float(parts[-3]), float(parts[-2]), float(parts[-1])]
                        except Exception:
                            normal = None
                    elif s.startswith("vertex"):
                        parts = s.split()
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        if len(verts) == 3:
                            # recompute normal if missing/zero
                            import numpy as _np
                            v1, v2, v3 = _np.array(verts[0]), _np.array(verts[1]), _np.array(verts[2])
                            if normal is None or abs(normal[0])+abs(normal[1])+abs(normal[2]) < 1e-20:
                                n = _np.cross(v2 - v1, v3 - v1)
                                n_norm = _np.linalg.norm(n) or 1.0
                                normal = (n / n_norm).tolist()
                            yield normal, v1.tolist(), v2.tolist(), v3.tolist()
                            normal = None; verts = []
                return
        except UnicodeDecodeError:
            pass

        # Binary STL
        with open(stl_path, "rb") as f:
            f.seek(80)
            n_triangles = struct.unpack("<I", f.read(4))[0]
            for _ in range(n_triangles):
                nx, ny, nz = struct.unpack("<3f", f.read(12))
                v1 = list(struct.unpack("<3f", f.read(12)))
                v2 = list(struct.unpack("<3f", f.read(12)))
                v3 = list(struct.unpack("<3f", f.read(12)))
                f.read(2)
                # recompute normal if zero
                import numpy as _np
                n = _np.array([nx, ny, nz], dtype=float)
                if _np.linalg.norm(n) < 1e-20:
                    a, b, c = _np.array(v1), _np.array(v2), _np.array(v3)
                    n = _np.cross(b - a, c - a)
                    nn = _np.linalg.norm(n) or 1.0
                    n = n / nn
                else:
                    n = n / (_np.linalg.norm(n) or 1.0)
                yield n.tolist(), v1, v2, v3

    def _estimate_curvature_strength(self, stl_path: Path, sample_max: int = 200000):
        """Return {'strength':[0..1], 'nTris':N} based on dispersion of wall normals.
        strength≈0 (very straight/smooth); strength≈1 (highly tortuous/branched).
        We sample up to 'sample_max' triangles for speed.
        """
        import numpy as _np, random as _random
        normals = []
        count = 0
        # Reservoir sampling to avoid loading all triangles
        reservoir = []
        k = 0
        for tri in self._iter_stl_triangles(stl_path):
            if tri is None:
                continue
            if k < sample_max:
                reservoir.append(tri)
            else:
                j = _random.randint(0, k)
                if j < sample_max:
                    reservoir[j] = tri
            k += 1
        
        # Extract normals from reservoir
        for n, v1, v2, v3 in reservoir:
            normals.append(n)
        
        if not normals:
            return {"strength": 0.0, "nTris": 0}
        
        N = _np.array(normals, dtype=float)
        # normalize just in case
        nrm = _np.linalg.norm(N, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        N = N / nrm
        
        # mean direction
        mu = _np.mean(N, axis=0)
        mu = mu / (_np.linalg.norm(mu) or 1.0)
        
        # dispersion metric: 1 - |n·mu| averaged
        disp = float(_np.mean(1.0 - _np.abs(N @ mu)))
        
        # map dispersion (~0..~0.5) to [0,1]
        strength = max(0.0, min(1.0, disp / 0.4))
        
        return {"strength": strength, "nTris": len(N)}

    def _estimate_first_layer_from_yplus(self, D_ref, U_peak, rho, mu, y_plus, model="turbulent"):
        """Estimate first layer thickness from y+ target for wall-resolved meshing"""
        # Rough Cf approximation (Dean formula for turbulent, Hagen-Poiseuille for laminar)
        Re = rho * U_peak * D_ref / max(mu, 1e-9)
        
        if model == "laminar" or Re < 2300:
            # Laminar: Cf = 16/Re (Hagen-Poiseuille)
            Cf = 16.0 / max(Re, 1.0)
        else:
            # Turbulent: Cf ≈ 0.073 Re^-0.25 (Dean correlation)
            Cf = 0.073 * max(Re, 1.0) ** (-0.25)
        
        # Wall shear stress and friction velocity
        tauw = 0.5 * rho * (U_peak ** 2) * Cf
        u_tau = math.sqrt(max(tauw, 1e-12) / rho)
        
        # First cell center distance from y+ definition: y+ = y*u_tau*rho/mu
        y = y_plus * mu / max(rho * u_tau, 1e-12)
        
        # Convert to first-layer thickness (cell height) ~ y_center * 2 for safety
        first_layer = 2.0 * y
        
        self.logger.info(f"y+ based first layer: Re={Re:.0f}, Cf={Cf:.4f}, u_tau={u_tau:.3f}, y+={y_plus} → {first_layer*1e6:.1f} μm")
        return first_layer

    def _womersley_boundary_layer(self, heart_rate_hz, nu):
        """Calculate Womersley boundary layer thickness for pulsatile flow
        δ_ω ≈ sqrt(2ν/ω) where ω = 2πf
        """
        omega = 2.0 * math.pi * max(heart_rate_hz, 1e-6)
        delta = math.sqrt(2.0 * nu / omega)
        self.logger.info(f"Womersley boundary layer: f={heart_rate_hz:.2f} Hz, ν={nu:.2e} m²/s → δ_ω={delta*1e3:.2f} mm")
        return delta

    def _first_layer_from_dx(self, dx, N, ER, alpha=0.8, near_dist=None):
        """
        Auto-size first layer thickness to maintain T ≈ α·Δx relationship.
        
        Args:
            dx: Base cell size (meters)
            N: Number of surface layers
            ER: Expansion ratio
            alpha: Target total thickness as fraction of dx (default 0.8)
            
        Returns:
            dict with 'firstLayerThickness_abs', 'minThickness_abs', 'nGrow', 'nSurfaceLayers'
        """
        import numpy as np
        
        # Get config parameters
        stage1 = self.stage1 or {}
        t1_min_frac = float(stage1.get("t1_min_fraction_of_dx", 0.02))  # 2% of Δx
        t1_max_frac = float(stage1.get("t1_max_fraction_of_dx", 0.08))  # 8% of Δx
        
        # Target total thickness
        T_target = alpha * dx
        
        # Raw first layer thickness from geometric series
        if N <= 1:
            t1_raw = T_target
        else:
            t1_raw = T_target * (ER - 1.0) / (ER**N - 1.0)
        
        # Clamp to sensible band
        t1_min = max(t1_min_frac * dx, 1e-6)   # At least 2% of Δx and 1 µm
        t1_max = min(t1_max_frac * dx, 200e-6)  # At most 8% of Δx and 200 µm
        t1 = float(np.clip(t1_raw, t1_min, t1_max))
        
        # Keep total thickness near alpha*dx if clamped
        if abs(t1 - t1_raw) > 0.05 * max(t1_raw, 1e-12):
            # Significant clamping occurred, adjust N to maintain target thickness
            N_star = np.log(1.0 + (ER - 1.0) * (alpha * dx) / t1) / np.log(ER)
            N_adj = int(np.clip(round(N_star), 3, 20))
            self.logger.debug(f"  Clamping adjusted N: {N} → {N_adj} (N*={N_star:.2f})")
        else:
            N_adj = N
        
        # Calculate actual total thickness with clamped t1 and adjusted N
        T_actual = t1 * (ER**N_adj - 1.0) / (ER - 1.0) if N_adj > 1 else t1
        
        # Ensure total thickness fits within near refinement band to avoid growing into coarser zone
        if near_dist and T_actual > 0.6 * near_dist:
            # Trim N until T fits - keep a cushion to avoid quality issues
            while N_adj > 3 and T_actual > 0.6 * near_dist:
                N_adj -= 1
                T_actual = t1 * (ER**N_adj - 1.0) / (ER - 1.0) if N_adj > 1 else t1
            self.logger.info(f"Trimmed layers to fit near band: N→{N_adj}, T={T_actual*1e3:.3f}mm (≤{0.6*near_dist*1e3:.2f}mm)")
        
        # CRITICAL FIX: minThickness MUST be ≤ firstLayerThickness for layer growth
        # Rule: minThickness = 0.15-0.2 × firstLayerThickness (NOT total thickness)
        # Use gentle floor, not ceiling - keep 10-20% of t1, but don't go microscopic
        min_thickness = max(0.15 * t1, 1e-6)  # 15% of first layer, floor at 1µm
        
        # Conservative nGrow: Start with 0 for best coverage, fallback to 1
        n_grow = int(stage1.get("nGrow", 0))  # 0 gives best coverage
        
        self.logger.debug(f"Layer sizing: dx={dx*1e3:.2f}mm, N={N_adj} (orig={N}), ER={ER:.2f}")
        self.logger.debug(f"  Target T={T_target*1e3:.3f}mm ({alpha:.1f}×dx), actual T={T_actual*1e3:.3f}mm")
        self.logger.debug(f"  t1={t1*1e6:.1f}µm (raw={t1_raw*1e6:.1f}µm), minThick={min_thickness*1e6:.1f}µm")
        
        return {
            'firstLayerThickness_abs': t1,
            'minThickness_abs': min_thickness,
            'nGrow': n_grow,
            'nSurfaceLayers': N_adj  # Return adjusted N
        }

    # ------------------------ Dict generation -----------------------------
    def _generate_blockmesh_dict(self, iter_dir, bbox_info, dx_base):
        bbox_min = np.array([
            bbox_info["mesh_domain"]["x_min"],
            bbox_info["mesh_domain"]["y_min"],
            bbox_info["mesh_domain"]["z_min"],
        ])
        bbox_max = np.array([
            bbox_info["mesh_domain"]["x_max"],
            bbox_info["mesh_domain"]["y_max"],
            bbox_info["mesh_domain"]["z_max"],
        ])
        bbox_size = bbox_max - bbox_min
        # Compute block divisions with override hierarchy
        divisions = self._compute_block_divisions(bbox_size, dx_base)
        
        # Get grading from config
        bm = self.config.get("BLOCKMESH", {}) or {}
        grading = bm.get("grading", [1, 1, 1])

        vertices = []
        for z in [bbox_min[2], bbox_max[2]]:
            vertices.append([bbox_min[0], bbox_min[1], z])
            vertices.append([bbox_max[0], bbox_min[1], z])
            vertices.append([bbox_max[0], bbox_max[1], z])
            vertices.append([bbox_min[0], bbox_max[1], z])

        blockmesh = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM
   \\    /   O peration     |
    \\  /    A nd           | Version: 12
     \\/     M anipulation  |
*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1;

vertices
(
{chr(10).join(f"    ({v[0]} {v[1]} {v[2]})  // {i}" for i, v in enumerate(vertices))}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({divisions[0]} {divisions[1]} {divisions[2]}) simpleGrading ({grading[0]} {grading[1]} {grading[2]})
);

edges ()
;

boundary
(
    background
    {{
        type patch;  // avoid accidental walls if snap fails
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
            (0 1 5 4)
            (3 7 6 2)
            (0 4 7 3)
            (1 2 6 5)
        );
    }}
);
"""
        sys = iter_dir / "system"
        sys.mkdir(exist_ok=True)
        (sys / "blockMeshDict").write_text(blockmesh)
        control = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}

application     blockMesh;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
        (sys / "controlDict").write_text(control)
        self.logger.info(f"blockMesh: divisions={divisions.tolist()}, Δx={dx_base*1e3:.2f} mm")

    def _copy_trisurfaces(self, iter_dir) -> List[str]:
        """Process STL files with smart scaling and robust error handling"""
        tri = iter_dir / "constant" / "triSurface"
        tri.mkdir(parents=True, exist_ok=True)
        outlet_names = []
        env = self.config["openfoam_env_path"]
        
        all_stl_files = list(self.stl_files["outlets"]) + list(self.stl_files["required"].values())
        scale_m = self.config.get("SCALING", {}).get("scale_m", 1.0)
        
        self.logger.info(f"Processing {len(all_stl_files)} STL files with scale_m={scale_m}")
        
        for p in all_stl_files:
            dest_path = tri / p.name
            
            # Integrated copy/scale logic with smart detection
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # If explicit scale factor provided
                if scale_m != 1.0 and abs(scale_m - 1.0) > 1e-12:
                    cmd = f'surfaceTransformPoints "scale=({scale_m} {scale_m} {scale_m})" "{p.absolute()}" "{dest_path.absolute()}"'
                    self.logger.info(f"SCALING {p.name} (scale={scale_m})")
                    result = run_command(cmd, cwd=iter_dir, env_setup=env, timeout=600, max_memory_gb=self.max_memory_gb)
                    if result.returncode != 0:
                        raise RuntimeError(f"Failed scaling {p.name}: {result.stderr}")
                        
                # Auto-detection mode: check if needs scaling
                elif self._check_stl_units(p):
                    # Use configured scale factor for auto-detected cases too
                    auto_scale = self.config.get("SCALING", {}).get("scale_m", 0.001)
                    cmd = f'surfaceTransformPoints "scale=({auto_scale} {auto_scale} {auto_scale})" "{p.absolute()}" "{dest_path.absolute()}"'
                    self.logger.info(f"AUTO-SCALING {p.name} (scale={auto_scale})")
                    result = run_command(cmd, cwd=iter_dir, env_setup=env, timeout=600, max_memory_gb=self.max_memory_gb)
                    if result.returncode != 0:
                        self.logger.warning(f"Auto-scaling failed for {p.name}, copying instead")
                        shutil.copy2(p, dest_path)
                else:
                    # Just copy the file
                    self.logger.debug(f"COPYING {p.name}")
                    shutil.copy2(p, dest_path)
                    
            except Exception as e:
                self.logger.error(f"STL processing failed for {p.name}: {e}")
                # Fallback to copy
                try:
                    shutil.copy2(p, dest_path)
                    self.logger.info(f"Fallback copy successful for {p.name}")
                except Exception as copy_e:
                    raise RuntimeError(f"Both scaling and copy failed for {p.name}: {copy_e}")
            
            # Track outlet names for mesh generation
            if p in self.stl_files["outlets"]:
                outlet_names.append(p.stem)

        return outlet_names
    

    def _bbox_maxdim_py(self, stl_path: Path) -> float:
        """Max bbox dimension in RAW STL coordinates (no unit conversion) for unit detection."""
        xmin = ymin = zmin = float("+inf")
        xmax = ymax = zmax = float("-inf")

        try:
            # Try ASCII first
            with open(stl_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.lstrip()
                    if s.startswith("vertex"):
                        _, x, y, z, *rest = s.split()
                        x = float(x); y = float(y); z = float(z)
                        if x<xmin: xmin=x
                        if y<ymin: ymin=y
                        if z<zmin: zmin=z
                        if x>xmax: xmax=x
                        if y>ymax: ymax=y
                        if z>zmax: zmax=z
            if xmin != float("+inf"):
                return float(max(xmax-xmin, ymax-ymin, zmax-zmin))
        except UnicodeDecodeError:
            pass  # Fall through to binary

        try:
            # Binary STL
            with open(stl_path, "rb") as f:
                f.seek(80)
                ntri = struct.unpack("<I", f.read(4))[0]
                for _ in range(ntri):
                    f.read(12)  # Skip normal vector
                    for _ in range(3):  # 3 vertices per triangle
                        x, y, z = struct.unpack("<3f", f.read(12))
                        if x<xmin: xmin=x
                        if y<ymin: ymin=y
                        if z<zmin: zmin=z
                        if x>xmax: xmax=x
                        if y>ymax: ymax=y
                        if z>zmax: zmax=z
                    f.read(2)  # Skip attribute byte count
            if xmin != float("+inf"):
                return float(max(xmax-xmin, ymax-ymin, zmax-zmin))
        except Exception as e:
            self.logger.warning(f"Failed to read STL dimensions from {stl_path}: {e}")
        
        # Conservative fallback if no vertices found
        return 0.02
    
    def _read_stl_vertices_raw(self, stl_path: Path):
        """Read STL vertices in original coordinates without unit conversion."""
        vertices = []
        try:
            # Try ASCII first
            with open(stl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('vertex'):
                        coords = [float(x) for x in line.split()[1:4]]
                        vertices.append(coords)
            return vertices
            
        except UnicodeDecodeError:
            # Binary STL
            try:
                with open(stl_path, 'rb') as f:
                    f.seek(80)  # Skip header
                    n_triangles = struct.unpack('<I', f.read(4))[0]
                    
                    for _ in range(n_triangles):
                        f.read(12)  # Skip normal vector
                        for _ in range(3):  # Read 3 vertices per triangle
                            x, y, z = struct.unpack('<3f', f.read(12))
                            vertices.append([x, y, z])
                        f.read(2)  # Skip attribute byte count
                        
                return vertices
            except Exception as e:
                self.logger.warning(f"Binary STL read failed: {e}")
                return []

    def _check_stl_units(self, stl_path: Path, env: str = None) -> bool:
        """
        Return True if coordinates look like millimetres (need ×0.001 scaling).
        For human vasculature: 0.02-0.1m (20-100mm) is correct physical scale - no scaling needed.
        Only scale if dimensions are unreasonably large (>1m) suggesting mm-as-meter encoding.
        """
        try:
            mx = self._bbox_maxdim_py(stl_path)
            # Vessel geometry >1 m is implausible; interpret as mm numbers needing 0.001 scaling
            needs_scaling = mx > 1.0
            
            if needs_scaling:
                self.logger.info(f"[units] {stl_path.name} maxDim={mx:.6f} → looks like millimetres (scale by 0.001).")
            else:
                self.logger.info(f"[units] {stl_path.name} maxDim={mx:.6f} → already plausible metres.")
            
            return needs_scaling
        except Exception as e:
            self.logger.warning(f"Unit check failed for {stl_path.name}: {e}")
            return False  # fail-safe: don't scale blindly

    def _scale_stl_to_meters(self, src_path: Path, dest_path: Path, iter_dir: Path, env: str):
        """Scale STL from medical imaging scale to proper physical scale with surfaceTransformPoints."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Use configured scale factor, with fallback to common mm->m scaling
        scale_m = self.config.get("SCALING", {}).get("scale_m", 0.001)
        
        # Use string command to handle shell escaping properly
        cmd_str = f'surfaceTransformPoints "scale=({scale_m} {scale_m} {scale_m})" "{src_path.absolute()}" "{dest_path.absolute()}"'
        self.logger.info(f"Running scaling command with configured scale={scale_m}: {cmd_str}")
        self.logger.info(f"Working directory: {iter_dir}")
        # Sanitize env path to avoid logging secrets
        env_display = env if env.startswith("/") else "***" 
        self.logger.info(f"Environment: {env_display}")
        
        res = run_command(
            cmd_str, cwd=iter_dir, env_setup=env, timeout=300, max_memory_gb=self.max_memory_gb
        )
        
        self.logger.info(f"Scaling command return code: {res.returncode}")
        if res.stdout:
            self.logger.debug(f"STDOUT: {res.stdout}")
        if res.stderr:
            self.logger.warning(f"STDERR: {res.stderr}")

        # Basic verification: file exists and is non-zero size
        try:
            if not dest_path.exists():
                raise RuntimeError(f"Scaled file was not created: {dest_path}")
                
            if dest_path.stat().st_size == 0:
                raise RuntimeError(f"Scaled file is empty: {dest_path}")
            
            # Log success without detailed verification (to avoid bbox reading issues)
            mx_src = self._bbox_maxdim_py(src_path)
            self.logger.info(f"{src_path.name}: maxDim src={mx_src:.6f} → scaled successfully (0.001x)")
                
        except Exception as e:
            # hard-stop instead of copying unscaled STL
            raise RuntimeError(f"Scaling verification failed for {src_path.name}: {e}")
    
    def _parse_parallel_checkmesh(self, output: str) -> Dict:
        """Parse parallel checkMesh output to extract metrics (robust for both serial/parallel)"""
        metrics = {
            "maxNonOrtho": 0.0,
            "maxSkewness": 0.0,
            "maxAspectRatio": 0.0,
            "negVolCells": 0,
            "meshOK": False,
            "cells": 0,
            "wall_nFaces": None
        }
        
        # non-ortho / skewness (several OF variants)
        m = re.search(r"Max non-orthogonality\s*=\s*([\d.]+)", output)
        if m: metrics["maxNonOrtho"] = float(m.group(1))
        
        m = re.search(r"Max skewness\s*=\s*([\d.]+)", output)
        if m: metrics["maxSkewness"] = float(m.group(1))
        
        m = re.search(r"aspect ratio\s*=\s*([\d.]+)", output)
        if m: metrics["maxAspectRatio"] = float(m.group(1))
        
        # cells: accept both serial and parallel formats
        m = re.search(r"\bcells:\s+(\d+)", output, re.IGNORECASE)
        if not m:
            m = re.search(r"\bNumber of cells:\s+(\d+)", output, re.IGNORECASE)
        if m:
            metrics["cells"] = int(m.group(1))
        
        # success flag: tolerate punctuation/duplication
        metrics["meshOK"] = ("Mesh OK" in output) or ("Mesh OK." in output)
        
        # don't try to scrape wall faces from the text here; we'll compute it from boundary files
        return metrics
    
    def _sum_wall_faces_from_processors(self, iter_dir: Path) -> int:
        """Count wall patch faces from boundary files (parallel-safe)"""
        total = 0
        
        # Check if processor directories exist (parallel case)
        processor_dirs = list(iter_dir.glob("processor*"))
        
        if processor_dirs:
            # Prefer processor totals when they exist
            for b in iter_dir.glob("processor*/constant/polyMesh/boundary"):
                try:
                    txt = b.read_text()
                    m = re.search(rf"{self.wall_name}\s*\{{[^{{}}]*?nFaces\s+(\d+);", txt, re.DOTALL)
                    if m:
                        total += int(m.group(1))
                        self.logger.debug(f"Found {m.group(1)} {self.wall_name} faces in {b.parent.parent.parent.name}")
                except Exception as e:
                    self.logger.debug(f"Could not parse boundary file {b}: {e}")
        else:
            # Only use root mesh if no processor directories exist (serial case)
            b = iter_dir / "constant" / "polyMesh" / "boundary"
            if b.exists():
                try:
                    txt = b.read_text()
                    m = re.search(rf"{self.wall_name}\s*\{{[^{{}}]*?nFaces\s+(\d+);", txt, re.DOTALL)
                    if m: 
                        total += int(m.group(1))
                        self.logger.debug(f"Found {m.group(1)} {self.wall_name} faces in root mesh")
                except Exception as e:
                    self.logger.debug(f"Could not parse root boundary file: {e}")
        
        return total

    def _write_surfaceFeatures(self, iter_dir, all_surfaces: List[str]):
        # Curvature-aware includedAngle selection
        wall_stl = iter_dir / "constant" / "triSurface" / f"{self.wall_name}.stl"
        
        if wall_stl.exists():
            try:
                cs = self._estimate_curvature_strength(wall_stl)
                strength = cs["strength"]
                # Map curvature strength [0,1] to includedAngle [170°, 145°]
                # Higher curvature → lower angle → more features detected
                included_angle = 170 - (strength * 25)
                self.logger.info(f"Curvature-aware includedAngle: {included_angle:.1f}° (strength={strength:.3f}, {cs['nTris']} tris)")
            except Exception as e:
                # Fallback to configuration or default
                included_angle = self.config.get("SURFACE_FEATURES", {}).get("includedAngle", 160)
                self.logger.warning(f"Curvature analysis failed, using fallback angle {included_angle}°: {e}")
        else:
            # Fallback if wall STL not found
            included_angle = self.config.get("SURFACE_FEATURES", {}).get("includedAngle", 160)
            self.logger.info(f"Wall STL not found, using configured includedAngle: {included_angle}°")
        
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeaturesDict;
}}

surfaces ({' '.join('"'+s+'"' for s in all_surfaces)});

includedAngle   {included_angle:.1f};  // curvature-adaptive feature detection
"""
        (iter_dir / "system" / "surfaceFeaturesDict").write_text(content)

    def _calculate_refinement_bands(self, dx_base):
        """
        Calculate near and far refinement band distances with adaptive scaling.
        
        Key insight: When surface refinement increases, surface cells get smaller by 2^level,
        so we need to widen the bands proportionally to maintain proper cell coverage.
        """
        if self.physics.get("use_womersley_bands", False):
            # Physics-aware bands based on Womersley boundary layer thickness
            D_ref, _ = self._estimate_reference_diameters()
            U_peak = float(self.physics.get("U_peak", 1.0))        # m/s
            rho = float(self.physics.get("rho", 1060.0))           # kg/m³
            mu = float(self.physics.get("mu", 3.5e-3))             # Pa·s
            hr_hz = float(self.physics.get("heart_rate_hz", 1.2))  # Hz
            
            # Calculate Womersley boundary layer thickness δ_W ~ sqrt(ν / ω)
            nu = mu / rho                                          # m²/s
            omega = 2.0 * np.pi * hr_hz                           # rad/s
            delta_w = np.sqrt(nu / omega)                         # Womersley BL thickness
            
            # Near band: 2-4 δ_W, far band: 10-20 δ_W
            near_dist = 3.0 * delta_w
            far_dist = 15.0 * delta_w
            
            self.logger.info(f"Womersley boundary layers: δ_W={delta_w*1e6:.1f}μm, near={near_dist*1e6:.1f}μm, far={far_dist*1e6:.1f}μm")
        else:
            # Geometry-based bands with adaptive scaling for surface refinement
            near_dist, far_dist = self._calculate_adaptive_refinement_bands(dx_base)
            
        return near_dist, far_dist

    def _calculate_adaptive_refinement_bands(self, dx_base):
        """
        Calculate refinement bands that adapt to surface refinement level.
        
        Maintains proper prism-friendly zone coverage as surface cells get smaller.
        Original: near=4Δx, far=10Δx  
        Improved: near=6-8Δx, far=12-16Δx (scaled by refinement level)
        """
        # Get current maximum surface refinement level
        current_max_level = self._get_current_surface_refinement_level()
        
        # Base cell counts (improved from original 4/10 to 6-8/12-16 range)  
        base_near_cells = self.stage1.get("near_band_cells", 4)
        base_far_cells = self.stage1.get("far_band_cells", 10)
        
        # Apply improved baseline (6-8 for near, 12-16 for far)
        if base_near_cells == 4:  # Original default
            improved_near_cells = 7  # Middle of 6-8 range
        else:
            improved_near_cells = max(6, base_near_cells)  # Ensure at least 6
            
        if base_far_cells == 10:  # Original default  
            improved_far_cells = 14  # Middle of 12-16 range
        else:
            improved_far_cells = max(12, base_far_cells)  # Ensure at least 12
        
        # Scale bands based on surface refinement level
        # Surface cells get smaller by factor 2^(level-1), so widen bands accordingly
        if current_max_level > 1:
            refinement_factor = 2 ** (current_max_level - 1)
            
            # Progressive scaling: modest increase for level 2, more for higher levels
            if current_max_level == 2:
                # Level 2: Surface cells are 2x smaller, increase bands by 1.5x
                band_scale_factor = 1.5
            elif current_max_level == 3:
                # Level 3: Surface cells are 4x smaller, increase bands by 2.0x  
                band_scale_factor = 2.0
            else:
                # Level ≥4: Surface cells are ≥8x smaller, increase bands by 2.5x
                band_scale_factor = 2.5
                
            scaled_near_cells = improved_near_cells * band_scale_factor
            scaled_far_cells = improved_far_cells * band_scale_factor
            
            self.logger.info(f"ADAPTIVE BAND SCALING for surface level {current_max_level}")
            self.logger.info(f"   Refinement factor: {refinement_factor}x smaller surface cells")
            self.logger.info(f"   Band scale factor: {band_scale_factor}x")
            self.logger.info(f"   Near band: {improved_near_cells} → {scaled_near_cells:.1f} cells")
            self.logger.info(f"   Far band: {improved_far_cells} → {scaled_far_cells:.1f} cells")
            
        else:
            # Level 1: Use improved baseline without additional scaling
            scaled_near_cells = improved_near_cells
            scaled_far_cells = improved_far_cells
            
            if base_near_cells == 4 or base_far_cells == 10:
                self.logger.info(f"📏 IMPROVED BAND SIZING (level 1)")
                self.logger.info(f"   Near band: {base_near_cells} → {scaled_near_cells:.1f} cells (6-8Δx range)")  
                self.logger.info(f"   Far band: {base_far_cells} → {scaled_far_cells:.1f} cells (12-16Δx range)")
        
        # Calculate distances
        near_dist = scaled_near_cells * dx_base  
        far_dist = scaled_far_cells * dx_base
        
        return near_dist, far_dist

    def _snappy_no_layers_dict(self, iter_dir, outlet_names, internal_pt, dx_base):
        snap = self.config["SNAPPY"]
        
        # Compute Δx-aware mergeTolerance to avoid over-merging at throats
        merge_tol = float(min(1e-5, 0.05 * dx_base))
        merge_tol = max(1e-6, merge_tol)  # cap at reasonable bounds
        
        # Calculate refinement band distances
        near_dist, far_dist = self._calculate_refinement_bands(dx_base)
            
        self.logger.info(f"Cell-based refinement bands: near={near_dist*1e3:.2f}mm ({near_dist/dx_base:.0f} cells), far={far_dist*1e3:.2f}mm ({far_dist/dx_base:.0f} cells)")

        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    inlet.stl       {{ type triSurfaceMesh; name inlet;       file "inlet.stl"; }}
{chr(10).join(f'    {n}.stl        {{ type triSurfaceMesh; name {n};           file "{n}.stl"; }}' for n in outlet_names)}
    {self.wall_name}.stl  {{ type triSurfaceMesh; name {self.wall_name};  file "{self.wall_name}.stl"; }}
}};

castellatedMeshControls
{{
    maxLocalCells {snap.get("maxLocalCells", 2000000)};
    maxGlobalCells {snap.get("maxGlobalCells", 8000000)};
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels {snap.get("nCellsBetweenLevels", 1)};
    
    features
    (
        {{
            file "{self.wall_name}.eMesh";
            level {self.surface_levels[0]};
        }}
        {{
            file "inlet.eMesh";
            level {self.surface_levels[0]};
        }}
{chr(10).join(f'        {{ file "{n}.eMesh"; level {self.surface_levels[0]}; }}' for n in outlet_names)}
    );
    
    refinementSurfaces
    {{
        {self.wall_name}
        {{
            level ({self.surface_levels[0]} {self.surface_levels[1]});
        }}
        inlet
        {{
            level ({self.surface_levels[0]} {self.surface_levels[1]});
        }}
{chr(10).join(f'''        {n}
        {{
            level ({self.surface_levels[0]} {self.surface_levels[1]});
        }}''' for n in outlet_names)}
    }}
    
    refinementRegions
    {{
        {self.wall_name}
        {{
            mode distance;
            levels (({near_dist:.6f} 2) ({far_dist:.6f} 1));
        }}
    }}

    locationInMesh ({internal_pt[0]:.6f} {internal_pt[1]:.6f} {internal_pt[2]:.6f});
    allowFreeStandingZoneFaces true;
    resolveFeatureAngle {snap.get("resolveFeatureAngle", 30)};
}}

snapControls
{{
    nSmoothPatch {snap.get("nSmoothPatch", 3)};
    tolerance 1.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter {snap.get("nFeatureSnapIter", 10)};
    implicitFeatureSnap {str(snap.get("implicitFeatureSnap", False)).lower()};
    explicitFeatureSnap {str(snap.get("explicitFeatureSnap", True)).lower()};
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes false;
    layers
    {{
    }}
    
    firstLayerThickness 1e-6;
    expansionRatio 1.0;
    minThickness 1e-6;
    nGrow 0;
    featureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 3;
    nSmoothNormals 5;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
}}

meshQualityControls
{{
    maxNonOrtho {_safe_get(self.config,["MESH_QUALITY","snap","maxNonOrtho"],65)};
    maxBoundarySkewness {_safe_get(self.config,["MESH_QUALITY","snap","maxBoundarySkewness"],4.0)};
    maxInternalSkewness {_safe_get(self.config,["MESH_QUALITY","snap","maxInternalSkewness"],4.0)};
    maxConcave 80;
    minFlatness 0.5;
    minVol {_safe_get(self.config,["MESH_QUALITY","snap","minVol"],1e-13)};
    minTetQuality {_safe_get(self.config,["MESH_QUALITY","snap","minTetQuality"],-1e15)};
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight {_safe_get(self.config,["MESH_QUALITY","snap","minFaceWeight"],0.02)};
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
    
    // Relaxed quality criteria for castellation (OpenFOAM 12 requirement)
    relaxed
    {{
        maxNonOrtho 75;
        maxBoundarySkewness 12.0;
        maxInternalSkewness 12.0;
        maxConcave 90;
        minFlatness 0.3;
        minVol 1e-15;
        minTetQuality 1e-9;
        minFaceWeight 0.005;
        minVolRatio 0.005;
        minDeterminant 0.0005;
    }}
}}

mergeTolerance {merge_tol:.1e};
"""
        (iter_dir / "system" / "snappyHexMeshDict.noLayer").write_text(content)

    def _snappy_layers_dict(self, iter_dir, outlet_names, internal_pt, dx_base):
        # Apply consistent minThickness policy before generating layer dict
        self._apply_consistent_minThickness_policy()
        
        # Compute Δx-aware mergeTolerance to avoid over-merging at throats
        merge_tol = float(min(1e-5, 0.05 * dx_base))
        merge_tol = max(1e-6, merge_tol)  # cap at reasonable bounds
        
        L = self.config["LAYERS"].copy()  # Make a copy to avoid modifying config
        
        # Calculate refinement bands for layer thickness validation
        near_dist, _ = self._calculate_refinement_bands(dx_base)
        
        # Auto-size layers if enabled (default behavior)
        auto_sizing = self.stage1.get("autoFirstLayerFromDx", True)
        if auto_sizing:
            N = int(L["nSurfaceLayers"])
            ER = float(L.get("expansionRatio", 1.2))
            alpha = float(self.stage1.get("alpha_total_layers", 0.8))
            
            # Apply auto-sizing (will adjust N if clamping occurs)
            sizing = self._first_layer_from_dx(dx_base, N, ER, alpha, near_dist)
            L.update(sizing)  # This now includes adjusted nSurfaceLayers
            
            # Use the potentially adjusted N for logging
            N_actual = sizing.get('nSurfaceLayers', N)
            self.logger.info(f"Auto layer sizing: N={N_actual} (orig={N}), t1={sizing['firstLayerThickness_abs']*1e6:.1f}μm, "
                           f"minThick={sizing['minThickness_abs']*1e6:.1f}μm, nGrow={sizing['nGrow']}")
        
        # Apply physics-aware first layer sizing if configured (overrides auto-sizing above)
        if self.physics.get("autoFirstLayer", False):
            D_ref, _ = self._estimate_reference_diameters()
            U_peak = float(self.physics.get("U_peak", 1.0))        # m/s (peak velocity)
            rho = float(self.physics.get("rho", 1060.0))           # kg/m³ (blood density)
            mu = float(self.physics.get("mu", 3.5e-3))             # Pa·s (blood viscosity)
            y_plus = float(self.physics.get("y_plus", 1.0))        # 1 for LES near-wall, 30 for wall-fn
            model = str(self.physics.get("flow_model", "turbulent"))  # "turbulent" or "laminar"
            
            # Calculate first layer thickness from y+ target
            first_layer = self._estimate_first_layer_from_yplus(D_ref, U_peak, rho, mu, y_plus, model)
            L["firstLayerThickness_abs"] = max(5e-6, first_layer)  # Minimum 5 μm for numerical stability
            
            # CONSERVATIVE layer parameters - start small and build coverage first
            if y_plus <= 5:  # Near-wall LES
                # Start conservative: fewer layers, gentler expansion
                L["nSurfaceLayers"] = min(L.get("nSurfaceLayers", 8), 10)  # Start 8-10, not 16+
                L["expansionRatio"] = min(L.get("expansionRatio", 1.15), 1.2)
            elif y_plus >= 20:  # Wall functions
                # Even fewer layers for wall functions
                L["nSurfaceLayers"] = min(L.get("nSurfaceLayers", 6), 8)   # Start 6-8
                L["expansionRatio"] = max(L.get("expansionRatio", 1.2), 1.2)
            
            self.logger.info(f"Auto first layer: {L['firstLayerThickness_abs']*1e6:.1f} μm for y+={y_plus}, {model} flow")
        
        # CRITICAL: Final validation to ensure minThickness ≤ firstLayerThickness
        first_layer = L.get("firstLayerThickness_abs", 50e-6)
        min_thickness = L.get("minThickness_abs", 20e-6)
        
        if min_thickness > first_layer * 0.2:  # minThickness should be ≤ 20% of firstLayer
            old_min = min_thickness
            # Use consistent minThickness policy instead of hardcoded 15% rule
            self._apply_consistent_minThickness_policy()
            L["minThickness_abs"] = self.config["LAYERS"]["minThickness_abs"]
            self.logger.warning(f"FIXED minThickness constraint using consistent policy: {old_min*1e6:.1f}μm → {L['minThickness_abs']*1e6:.1f}μm")
        
        # Determine if using relative or absolute sizing
        is_rel = bool(L.get("relativeSizes", False))
        
        # Generate the correct thickness lines based on sizing mode
        if is_rel:
            first_layer_line = f"firstLayerThickness {L.get('firstLayerThickness', 0.20):.3f};"
            final_layer_line = f"finalLayerThickness {L.get('finalLayerThickness', 0.75):.3f};" if L.get('finalLayerThickness') else ""
            min_thick_line = f"minThickness {L.get('minThickness', max(0.18, min(0.25, L.get('firstLayerThickness', 0.2) * 0.75))):.3f};"
        else:
            first_layer_line = f"firstLayerThickness {L.get('firstLayerThickness_abs', 50e-6):.2e};"
            final_layer_line = ""  # not used in absolute mode
            min_thick_line = f"minThickness {L.get('minThickness_abs', 2e-6):.2e};"
        
        # Get consistent memory limits from SNAPPY config (fix inconsistency with snap phase)
        mlc = self.config["SNAPPY"].get("maxLocalCells", 2000000)
        mgc = self.config["SNAPPY"].get("maxGlobalCells", 8000000)
        
        # Sync effective layer parameters back to config for metrics consistency
        self.config["LAYERS"].update({
            "nSurfaceLayers": L["nSurfaceLayers"],
            "expansionRatio": L["expansionRatio"],
            "firstLayerThickness_abs": L.get("firstLayerThickness_abs", 50e-6),
            "minThickness_abs": L.get("minThickness_abs", 20e-6),
            "nGrow": L.get("nGrow", 0),
            "featureAngle": L.get("featureAngle", 60),
            "maxThicknessToMedialRatio": L.get("maxThicknessToMedialRatio", 0.3),
            "minMedianAxisAngle": L.get("minMedianAxisAngle", 90),
        })
        
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh false;
snap            false;
addLayers       true;

geometry
{{
    inlet.stl       {{ type triSurfaceMesh; name inlet;       file "inlet.stl"; }}
{chr(10).join(f'    {n}.stl        {{ type triSurfaceMesh; name {n};           file "{n}.stl"; }}' for n in outlet_names)}
    {self.wall_name}.stl  {{ type triSurfaceMesh; name {self.wall_name};  file "{self.wall_name}.stl"; }}
}};

castellatedMeshControls
{{
    maxLocalCells {mlc};
    maxGlobalCells {mgc};
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 1;
    locationInMesh ({internal_pt[0]:.6f} {internal_pt[1]:.6f} {internal_pt[2]:.6f});
    allowFreeStandingZoneFaces true;
    resolveFeatureAngle {self.config['SNAPPY'].get('resolveFeatureAngle', 45)};
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes {str(L.get("relativeSizes", False)).lower()};
    {final_layer_line}
    layers
    {{
        "{self.wall_name}"
        {{
            nSurfaceLayers {L["nSurfaceLayers"]};
        }}
    }}

    {first_layer_line}
    expansionRatio {L["expansionRatio"]};
    {min_thick_line}
    nGrow {L.get("nGrow", 0)};
    featureAngle {min(L.get("featureAngle",60), 90)};
    nRelaxIter {L.get("nRelaxIter", 5)};
    nSmoothSurfaceNormals {L.get("nSmoothSurfaceNormals", 3)};
    nSmoothNormals 5;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio {L.get("maxThicknessToMedialRatio",0.3)};
    minMedianAxisAngle {L.get("minMedianAxisAngle",90)};
    nBufferCellsNoExtrude 0;
    nLayerIter {L.get("nLayerIter", 50)};
    nRelaxedIter {L.get("nRelaxedIter", 20)};
    {f'slipFeatureAngle {L.get("slipFeatureAngle", 30)};' if "slipFeatureAngle" in L else ''}
}}

meshQualityControls
{{
    maxNonOrtho {_safe_get(self.config,["MESH_QUALITY","layer","maxNonOrtho"],65)};
    maxBoundarySkewness {_safe_get(self.config,["MESH_QUALITY","layer","maxBoundarySkewness"],4.0)};
    maxInternalSkewness {_safe_get(self.config,["MESH_QUALITY","layer","maxInternalSkewness"],4.0)};
    maxConcave 80;
    minFlatness 0.5;
    minVol {_safe_get(self.config,["MESH_QUALITY","layer","minVol"],1e-13)};
    minTetQuality {_safe_get(self.config,["MESH_QUALITY","layer","minTetQuality"],1e-6)};
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight {_safe_get(self.config,["MESH_QUALITY","layer","minFaceWeight"],0.02)};
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
    
    // Relaxed quality criteria for layer addition (OpenFOAM 12 requirement)
    relaxed
    {{
        maxNonOrtho 75;
        maxBoundarySkewness 12.0;
        maxInternalSkewness 12.0;
        maxConcave 90;
        minFlatness 0.3;
        minVol 1e-15;
        minTetQuality 1e-9;
        minFaceWeight 0.005;
        minVolRatio 0.005;
        minDeterminant 0.0005;
    }}
}}

mergeTolerance {merge_tol:.1e};
"""
        (iter_dir / "system" / "snappyHexMeshDict.layers").write_text(content)

    def _snappy_dict(self, iter_dir, outlet_names, internal_pt, dx_base, phase):
        """Unified method to generate snappyHexMeshDict for any phase"""
        if phase not in ["no_layers", "layers"]:
            raise ValueError(f"Invalid phase: {phase}. Must be 'no_layers' or 'layers'")
        
        if phase == "no_layers":
            self._snappy_no_layers_dict(iter_dir, outlet_names, internal_pt, dx_base)
        elif phase == "layers":
            self._snappy_layers_dict(iter_dir, outlet_names, internal_pt, dx_base)

    # ----------------------------- Execution ------------------------------
    def _maybe_parallel(self, iter_dir):
        """Setup parallel decomposition if configured"""
        n = int(self.stage1.get("n_processors", 1))
        if n <= 1:
            return [], []
        
        # Write decomposeParDict for domain decomposition
        decompose_dict = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}

numberOfSubdomains {n};
method          scotch;
"""
        (iter_dir / "system" / "decomposeParDict").write_text(decompose_dict)
        
        self.logger.info(f"Parallel processing enabled with {n} processors")
        return (["decomposePar"], "log.decompose"), (["reconstructPar", "-latestTime"], "log.reconstruct")
    
    def _surface_check(self, iter_dir):
        """Run surfaceCheck and gate on critical STL health issues"""
        env = self.config["openfoam_env_path"]
        tri = iter_dir / "constant" / "triSurface"
        
        for p in [tri / f"{self.wall_name}.stl", tri / "inlet.stl", *[tri / f.name for f in self.stl_files["outlets"]]]:
            try:
                rel_path = Path("constant/triSurface") / p.name
                result = run_command(["surfaceCheck", str(rel_path)], cwd=iter_dir, env_setup=env, timeout=600, max_memory_gb=self.max_memory_gb)
                
                # Parse critical issues from surfaceCheck output
                output = result.stdout + result.stderr
                
                # Check for self-intersections
                if "self-intersecting" in output.lower() or "intersecting faces" in output.lower():
                    raise RuntimeError(f"Critical STL error in {p.name}: Self-intersecting faces detected")
                
                # Check for high non-manifold count (>5% of faces)
                if "non-manifold" in output.lower():
                    # Log entire non-manifold block for triage
                    nm_lines = []
                    lines = output.split('\n')
                    in_nm_block = False
                    
                    for line in lines:
                        line_l = line.lower()
                        if "non-manifold" in line_l:
                            in_nm_block = True
                            nm_lines.append(line.strip())
                        elif in_nm_block:
                            if line.strip() and not line.startswith(' '):
                                break  # End of block
                            nm_lines.append(line.strip())
                        
                        if "faces" in line_l and "non-manifold" in line_l:
                            # Try to extract numbers: "X non-manifold faces out of Y"
                            words = line.split()
                            try:
                                nm_idx = [w.lower() for w in words].index("non-manifold")
                                if nm_idx > 0:
                                    non_manifold = int(words[nm_idx-1])
                                    # Find total faces
                                    for i, word in enumerate(words):
                                        if word.lower() == "of" and i+1 < len(words):
                                            total_faces = int(words[i+1])
                                            nm_ratio = non_manifold / max(total_faces, 1)
                                            if nm_ratio > 0.05:  # >5% non-manifold
                                                self.logger.error(f"Non-manifold analysis for {p.name}:")
                                                for nm_line in nm_lines:
                                                    self.logger.error(f"  {nm_line}")
                                                raise RuntimeError(f"Critical STL error in {p.name}: High non-manifold ratio {nm_ratio:.1%} ({non_manifold}/{total_faces})")
                                            else:
                                                self.logger.debug(f"Acceptable non-manifold ratio in {p.name}: {nm_ratio:.1%} ({non_manifold}/{total_faces})")
                                            break
                            except (ValueError, IndexError):
                                # Log the problematic lines for debugging
                                if nm_lines:
                                    self.logger.warning(f"Could not parse non-manifold data in {p.name}:")
                                    for nm_line in nm_lines:
                                        self.logger.warning(f"  {nm_line}")
                                continue
                                
                self.logger.info(f"STL health check passed: {p.name}")
                
            except RuntimeError:
                # Re-raise critical errors to abort iteration
                raise
            except Exception as e:
                self.logger.warning(f"surfaceCheck failed on {p.name}: {e}")

    def _create_foam_file(self, iter_dir: Path):
        """Create a .foam dummy file for easy ParaView visualization"""
        try:
            foam_file = iter_dir / f"{iter_dir.name}.foam"
            foam_file.write_text("// OpenFOAM dummy file for ParaView\n")
            self.logger.debug(f"Created .foam file: {foam_file}")
        except Exception as e:
            self.logger.debug(f"Failed to create .foam file: {e}")

    def _run_snap_then_layers(self, iter_dir, force_full_remesh: bool = True) -> Tuple[Dict, Dict, Dict]:
        env = self.config["openfoam_env_path"]
        logs = iter_dir / "logs"
        logs.mkdir(exist_ok=True)
        
        if not force_full_remesh:
            self.logger.info("Layers-only optimization possible, but running full remesh for robustness")
            
        # Setup parallel decomposition if configured
        pre, post = self._maybe_parallel(iter_dir)
        n_procs = int(self.stage1.get("n_processors", 1))
        
        # Initial mesh generation (always serial)
        for cmd, log_name in [(["blockMesh"], "log.blockMesh"), (["surfaceFeatures"], "log.surfaceFeatures")]:
            self.logger.info(f"Running: {' '.join(cmd)}")
            try:
                res = run_command(cmd, cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                (logs / log_name).write_text(res.stdout + res.stderr)
                
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
        
        # Domain decomposition for parallel run
        if pre:
            self.logger.info(f"Running: {' '.join(pre[0])}")
            try:
                res = run_command(pre[0], cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                (logs / pre[1]).write_text(res.stdout + res.stderr)
                
                # OpenFOAM automatically handles triSurface distribution in parallel runs
                
            except Exception as e:
                self.logger.error(f"Decomposition failed: {e}")
        
        # MESH WITHOUT LAYERS phase (combined castellation + snap)
        shutil.copy2(iter_dir / "system" / "snappyHexMeshDict.noLayer", iter_dir / "system" / "snappyHexMeshDict")
        
        # Build snappyHexMesh command (parallel or serial)
        if n_procs > 1:
            snappy_cmd = ["mpirun", "-np", str(n_procs), "snappyHexMesh", "-overwrite", "-parallel"]
        else:
            snappy_cmd = ["snappyHexMesh", "-overwrite"]
        
        self.logger.info(f"Running: {' '.join(snappy_cmd)} (mesh without layers)")
        try:
            res = run_command(snappy_cmd, cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
            output_text = res.stdout + res.stderr
            (logs / "log.snappy.no_layers").write_text(output_text)
            
            # Check for common snappyHexMesh failure patterns
            if any(pattern in output_text for pattern in [
                "FOAM FATAL ERROR", "Aborted", "Could not find cellZone", 
                "locationInMesh", "No cells in mesh", "Maximum number of cells"
            ]):
                error_msg = "snappyHexMesh failed with critical error - check log.snappy.no_layers"
                self.logger.error(error_msg)
                # Continue with error state but don't fail immediately - allow recovery
                
        except Exception as e:
            self.logger.error(f"Mesh generation (no layers) failed: {e}")
            # Return failed state but continue to try layer addition
            error_metrics = {"meshOK": False, "cells": 0, "error": f"snappyHexMesh error: {str(e)}"}
            return error_metrics, error_metrics, {"coverage_overall": 0.0, "error": str(e)}
        
        # Basic mesh generation validation using boundary file fallback
        mesh_generation_ok = False
        
        # Parallel-safe fallback: trust boundary files if they show faces
        wall_faces_total = self._sum_wall_faces_from_processors(iter_dir)
        if wall_faces_total > 0:
            mesh_generation_ok = True
            self.logger.debug(f"Mesh generation verified via boundary files: {wall_faces_total} wall faces")
        
        # Always reconstruct after castellation for parallel runs (removes catch-22)
        if n_procs > 1:
            try:
                self.logger.info("Reconstructing mesh (post-castellation) to enable serial checkMesh")
                res = run_command(["reconstructPar", "-latestTime"], cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                (logs / "log.reconstruct.no_layers").write_text(res.stdout + res.stderr)
                
                # Serial checkMesh now for more reliable parsing
                castellation_metrics = check_mesh_quality(iter_dir, env, wall_name=self.wall_name)
                castellated_ok = castellation_metrics.get("meshOK", False)
                self.logger.debug(f"Serial checkMesh after reconstruct: meshOK={castellated_ok}")
                
            except Exception as e:
                self.logger.warning(f"Reconstruct/checkMesh (serial) after castellation failed: {e}")
        
        if not mesh_generation_ok:
            self.logger.warning("Mesh generation not verified; skipping layers")
            # Return early with empty metrics
            empty_metrics = {"meshOK": False, "cells": 0}
            return empty_metrics, empty_metrics, empty_metrics
        
        # Check combined mesh quality (castellation + snap)
        if n_procs > 1:
            self.logger.info("Running checkMesh -parallel on generated mesh")
            check_cmd = ["mpirun", "-np", str(n_procs), "checkMesh", "-parallel"]
            try:
                res = run_command(check_cmd, cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                output_text = res.stdout + res.stderr
                (logs / "log.checkMesh.no_layers").write_text(output_text)
                snap_metrics = self._parse_parallel_checkmesh(output_text)
            except Exception as e:
                self.logger.warning(f"Parallel checkMesh failed: {e}")
                # Fall back to serial checkMesh on reconstructed mesh
                try:
                    self.logger.info("Falling back to serial checkMesh on reconstructed mesh")
                    snap_metrics = check_mesh_quality(iter_dir, env, wall_name=self.wall_name)
                except Exception as serial_e:
                    self.logger.error(f"Serial checkMesh fallback also failed: {serial_e}")
                    snap_metrics = {"meshOK": False, "cells": 0}
        else:
            snap_metrics = check_mesh_quality(iter_dir, env, wall_name=self.wall_name)
        
        # Reconstruct mesh for analysis (if parallel and generation succeeded)
        if n_procs > 1 and snap_metrics.get("meshOK", False):
            self.logger.info("Reconstructing mesh for analysis")
            try:
                res = run_command(["reconstructPar", "-latestTime"], cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                (logs / "log.reconstruct.no_layers").write_text(res.stdout + res.stderr)
            except Exception as e:
                self.logger.warning(f"Mesh reconstruction failed: {e}")
        
        # Check for catastrophic mesh failure (zero cells) - only skip layers for complete failure
        cell_count = snap_metrics.get("cells", 0)
        if cell_count == 0:
            self.logger.warning("Snap mesh has zero cells; skipping layers for this iteration")
            layer_metrics = snap_metrics
            layer_cov = {"coverage_overall": 0.0, "perPatch": {}}
            
            # If parallel, clean up processor directories since we're not continuing
            if n_procs > 1:
                for proc_dir in iter_dir.glob("processor*"):
                    if proc_dir.is_dir():
                        shutil.rmtree(proc_dir)
            
            # Create .foam file for easy visualization
            self._create_foam_file(iter_dir)
            
            return snap_metrics, layer_metrics, layer_cov
        
        # Log snap mesh status but continue to layers regardless of quality issues
        mesh_ok = snap_metrics.get("meshOK", False)
        if not mesh_ok:
            self.logger.info("Snap mesh has quality issues but proceeding with layers (may improve quality)")
        
        # LAYERS phase - continue with decomposed mesh if parallel
        shutil.copy2(iter_dir / "system" / "snappyHexMeshDict.layers", iter_dir / "system" / "snappyHexMeshDict")
        
        self.logger.info(f"Running: {' '.join(snappy_cmd)} (layers)")
        try:
            res = run_command(snappy_cmd, cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
            (logs / "log.snappy.layers").write_text(res.stdout + res.stderr)
        except Exception as e:
            self.logger.error(f"Layer meshing failed: {e}")
        
        # Parse layer coverage from snappyHexMesh log
        layer_cov = parse_physics_aware_coverage(iter_dir, env, wall_name=self.wall_name)
        
        # Enhanced convergence tracking
        layer_log = logs / "log.snappy.layers"
        if layer_log.exists():
            # Parse detailed iteration metrics
            iteration_data = parse_layer_iterations(layer_log)
            
            # Evaluate against acceptance criteria
            acceptance_criteria = {
                "min_coverage": self.targets.min_layer_cov,
                "min_thickness_pct": 60,  # From config or default
                "min_effective_layers": 3,
                "max_illegal_faces": 20
            }
            evaluation = evaluate_stage1_metrics(layer_log, acceptance_criteria)
            
            # Log convergence status
            if iteration_data["converged"]:
                self.logger.info(f"Layer addition converged after {len(iteration_data['iterations'])} iterations")
            else:
                self.logger.warning("Layer addition did not fully converge")
            
            # Log final metrics
            if iteration_data["final_metrics"]:
                fm = iteration_data["final_metrics"]
                self.logger.info(f"Final layer metrics: {fm.get('effective_layers', 0):.2f} layers, "
                               f"{fm.get('thickness_pct', 0):.1f}% thickness achieved")
            
            # Add enhanced metrics to layer_cov
            layer_cov["iteration_data"] = iteration_data
            layer_cov["evaluation"] = evaluation
        
        # Check final mesh quality - in parallel if still decomposed
        if n_procs > 1:
            # Run checkMesh in parallel mode
            self.logger.info("Running final checkMesh -parallel on decomposed mesh")
            check_cmd = ["mpirun", "-np", str(n_procs), "checkMesh", "-parallel"]
            try:
                res = run_command(check_cmd, cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                output_text = res.stdout + res.stderr
                (logs / "log.checkMesh.layers").write_text(output_text)
                
                # Parse the checkMesh output
                layer_metrics = self._parse_parallel_checkmesh(output_text)
                self.logger.debug(f"Parsed layer metrics: meshOK={layer_metrics.get('meshOK')}, cells={layer_metrics.get('cells')}")
                
            except Exception as e:
                self.logger.error(f"Final parallel checkMesh failed: {e}")
                # Try to read from log file if it exists
                log_file = logs / "log.checkMesh.layers"
                if log_file.exists():
                    self.logger.info("Attempting to parse final checkMesh from written log file")
                    layer_metrics = self._parse_parallel_checkmesh(log_file.read_text())
                else:
                    layer_metrics = {"meshOK": False, "cells": 0}
        else:
            # Serial checkMesh for non-parallel runs
            layer_metrics = check_mesh_quality(iter_dir, env, wall_name=self.wall_name)
        
        # Now reconstruct after all checks are done
        if post and n_procs > 1:
            self.logger.info(f"Final reconstruction after successful layers")
            try:
                res = run_command(post[0], cwd=iter_dir, env_setup=env, timeout=None, max_memory_gb=self.max_memory_gb)
                (logs / "log.reconstruct.final").write_text(res.stdout + res.stderr)
                
                # Clean up processor directories after successful reconstruction
                for proc_dir in iter_dir.glob("processor*"):
                    if proc_dir.is_dir():
                        shutil.rmtree(proc_dir)
                self.logger.debug(f"Cleaned up {n_procs} processor directories after final reconstruction")
                        
            except Exception as e:
                self.logger.error(f"Final reconstruction failed: {e}")
        
        # Create .foam file for easy visualization
        self._create_foam_file(iter_dir)
        
        return snap_metrics, layer_metrics, layer_cov

    # --------------------------- Objective & updates -----------------------
    def _meets_quality_constraints(self, snap_m: Dict, layer_m: Dict, layer_cov: Dict) -> Tuple[bool, List[str]]:
        """Check if mesh meets hard quality constraints (OpenFOAM defaults)
        
        Returns:
            Tuple[bool, List[str]]: (constraints_met, list_of_failure_reasons)
        """
        reasons = []
        
        # Wall coverage gate for WSS reliability
        wall_cov = layer_cov.get("perPatch", {}).get(self.wall_name, layer_cov.get("coverage_overall", 0.0))
        
        # Check individual constraints
        if not layer_m.get("meshOK", False):
            reasons.append("checkMesh failed")
        
        max_nonortho = float(layer_m.get("maxNonOrtho", 1e9))
        if max_nonortho > self.targets.max_nonortho:
            reasons.append(f"NonOrtho {max_nonortho:.1f} > target {self.targets.max_nonortho}")
        
        max_skewness = float(layer_m.get("maxSkewness", 1e9))
        if max_skewness > self.targets.max_skewness:
            reasons.append(f"Skewness {max_skewness:.2f} > target {self.targets.max_skewness}")
        
        if wall_cov < self.targets.min_layer_cov:
            reasons.append(f"Coverage {wall_cov:.1%} < target {self.targets.min_layer_cov:.1%}")
        
        # Enhanced evaluation check (if available)
        if "evaluation" in layer_cov:
            evaluation = layer_cov["evaluation"]
            if not evaluation.get("accepted", True):
                reasons.extend(evaluation.get("recommendations", []))
        
        constraints_met = len(reasons) == 0
        return constraints_met, reasons
    
    def _get_cell_count(self, layer_m: Dict, no_layers_m: Dict) -> int:
        """Get cell count for mesh minimization objective"""
        return int(layer_m.get("cells", no_layers_m.get("cells", 0)))

    def _apply_updates(self, snap_m: Dict, layer_m: Dict, layer_cov: Dict) -> None:
        # Decide minimal change for next iteration
        NO = float(layer_m.get("maxNonOrtho", 0))
        SK = float(layer_m.get("maxSkewness", 0))
        coverage = float(layer_cov.get("coverage_overall", 0.0))
        # Deltas
        dNO = NO - float(snap_m.get("maxNonOrtho", 0))
        dSK = SK - float(snap_m.get("maxSkewness", 0))

        # IMPROVED LAYER COVERAGE STRATEGY: Check poor coverage first, before other conditions
        current_layers = self.config["LAYERS"]["nSurfaceLayers"]
        if current_layers >= 3 and coverage < self.targets.min_layer_cov:
            # Poor coverage with adequate layers - fix the root causes, NOT layer count
            
            # 1. Reduce first layer thickness more aggressively (main fix)
            current_first = self.config["LAYERS"].get("firstLayerThickness_abs", 50e-6)
            if current_first > 10e-6:  # Don't go below 10 microns
                self.config["LAYERS"]["firstLayerThickness_abs"] = current_first * 0.7
                # Apply consistent minThickness policy after reducing first layer
                self._apply_consistent_minThickness_policy()
                self.logger.info(f"Reduced first layer: {current_first*1e6:.1f}→{self.config['LAYERS']['firstLayerThickness_abs']*1e6:.1f}μm for better coverage")
            
            # 2. Increase smoothing iterations for better layer normals
            self.config["LAYERS"]["nSmoothSurfaceNormals"] = min(10, self.config["LAYERS"].get("nSmoothSurfaceNormals", 5) + 2)
            self.config["LAYERS"]["nRelaxIter"] = min(10, self.config["LAYERS"].get("nRelaxIter", 5) + 2)
            
            # 3. Enable slipFeatureAngle for sharp corners (critical for aorta branches)
            if "slipFeatureAngle" not in self.config["LAYERS"] or self.config["LAYERS"]["slipFeatureAngle"] < 30:
                self.config["LAYERS"]["slipFeatureAngle"] = 45
                self.logger.info("Enabled slipFeatureAngle=45° for sharp corner handling")
            
            # 4. Switch to relativeSizes when surfaces are refined (CRITICAL for stability)
            self._apply_relative_sizing_for_refinement(coverage)
            
            # 5. Consider relativeSizes for complex geometries (if very poor coverage)
            if coverage < 0.4 and not self.config["LAYERS"].get("relativeSizes", False):
                self.config["LAYERS"]["relativeSizes"] = True
                self.config["LAYERS"]["finalLayerThickness"] = 0.7  # 70% of base cell size
                self.logger.info("Switched to relativeSizes=true for complex geometry adaptation")
            
            # 5. Relax quality controls that prevent layer addition
            # Apply consistent minThickness policy
            self._apply_consistent_minThickness_policy()
            self.config["LAYERS"]["nRelaxedIter"] = max(self.config["LAYERS"].get("nRelaxedIter", 10), 20)
            
            self.logger.info(f"Improved layer strategy: first={self.config['LAYERS']['firstLayerThickness_abs']*1e6:.1f}μm, "
                           f"smooth={self.config['LAYERS']['nSmoothSurfaceNormals']}, "
                           f"relax={self.config['LAYERS']['nRelaxIter']}, keeping {current_layers} layers")
            return

        # PROGRESSIVE LAYER STRATEGY: Fix gates first, don't just scale everything (legacy path)
        if (dNO > 5 or dSK > 0.5) and coverage < self.targets.min_layer_cov:
            # 1. First try relaxing geometric constraints (your approach)
            self.config["LAYERS"]["featureAngle"] = min(80, self.config["LAYERS"].get("featureAngle", 70) + 5)
            self.config["LAYERS"]["maxThicknessToMedialRatio"] = min(0.6, self.config["LAYERS"].get("maxThicknessToMedialRatio", 0.45) + 0.05)
            
            # If coverage stalls < 40% after one relaxation, also drop minMedianAxisAngle
            if coverage < 0.4:
                self.config["LAYERS"]["minMedianAxisAngle"] = 70  # Relax cornering constraints
            
            # 2. Only if still failing, reduce first layer but keep minThickness constraint
            if coverage < 0.3:  # Very poor coverage, reduce thickness
                self.config["LAYERS"]["firstLayerThickness_abs"] *= 0.8
                # Apply consistent minThickness policy after reducing firstLayer
                self._apply_consistent_minThickness_policy()
            
            # 3. Make expansion ratio adaptive - smaller ER helps extrusion when layers hurt quality
            if self.config["LAYERS"].get("expansionRatio", 1.2) > 1.15:
                self.config["LAYERS"]["expansionRatio"] = 1.15
                self.logger.info(f"Reduced expansion ratio to {self.config['LAYERS']['expansionRatio']} for better extrusion")
            
            # 4. Boost layer iterations when coverage < 40% (temporary bump for first failing iteration)
            if coverage < 0.4:
                self.config["LAYERS"]["nLayerIter"] = 80
                self.config["LAYERS"]["nRelaxedIter"] = 30
                self.logger.info("Boosted layer iterations for poor coverage")
            
            # 5. Conservative nGrow increase (0→1, 1→2, max 2)
            self.config["LAYERS"]["nGrow"] = min(self.config["LAYERS"].get("nGrow",0) + 1, 2)
            
            self.logger.info(f"Layer gates relaxed: featureAngle={self.config['LAYERS']['featureAngle']}°, "
                           f"medialRatio={self.config['LAYERS']['maxThicknessToMedialRatio']:.2f}, "
                           f"first={self.config['LAYERS']['firstLayerThickness_abs']*1e6:.1f}μm, "
                           f"min={self.config['LAYERS']['minThickness_abs']*1e6:.1f}μm, nGrow={self.config['LAYERS']['nGrow']}")
        
        # Apply complex curvature controls based on coverage
        area_weighted_coverage = layer_cov.get("coverage_area_weighted", coverage)
        self._apply_complex_curvature_controls(area_weighted_coverage)
        
        return

    def _apply_relative_sizing_for_refinement(self, coverage: float) -> None:
        """
        CRITICAL: Switch to relative thickness when surface refinement increases.
        
        Key insight: When ladder goes up (finer surface mesh), absolute layer thickness
        becomes too large relative to surface cell size, causing extrusion failures.
        
        Solution: Use relativeSizes=true with proper T_rel and t1_rel ratios.
        """
        # Check if we have surface levels to evaluate refinement
        if not hasattr(self, 'surface_levels') or self.surface_levels is None:
            return
            
        current_max_level = max(self.surface_levels)
        
        # Switch to relative sizing when surface refinement increases beyond level 1
        if current_max_level > 1 and not self.config["LAYERS"].get("relativeSizes", False):
            self.config["LAYERS"]["relativeSizes"] = True
            
            # Apply optimal relative thickness ratios based on your specifications:
            # t1/Δx_surf ≈ 0.18–0.25 (first layer thickness relative to surface cell size)  
            # T/Δx_surf ≈ 0.6–0.9 (total thickness relative to surface cell size)
            # ER = 1.12–1.18 (expansion ratio)
            
            # Conservative values for stability
            t1_rel = 0.20  # t1/Δx_surf = 0.20 (middle of 0.18-0.25 range)
            T_rel = 0.75   # T/Δx_surf = 0.75 (middle of 0.6-0.9 range)
            ER = 1.15      # Expansion ratio = 1.15 (middle of 1.12-1.18 range)
            
            self.config["LAYERS"]["firstLayerThickness"] = t1_rel
            self.config["LAYERS"]["finalLayerThickness"] = T_rel  
            self.config["LAYERS"]["expansionRatio"] = ER
            
            # Remove absolute sizing parameters when using relative
            if "firstLayerThickness_abs" in self.config["LAYERS"]:
                del self.config["LAYERS"]["firstLayerThickness_abs"]
            if "minThickness_abs" in self.config["LAYERS"]:
                del self.config["LAYERS"]["minThickness_abs"]
                
            self.logger.info(f"SWITCHED TO RELATIVE SIZING for surface level {current_max_level}")
            self.logger.info(f"   t1_rel={t1_rel} (t1/Δx_surf), T_rel={T_rel} (T/Δx_surf), ER={ER}")
            
        elif current_max_level > 1 and self.config["LAYERS"].get("relativeSizes", False):
            # Already using relative sizing - adjust ratios based on coverage performance
            current_t1_rel = self.config["LAYERS"].get("firstLayerThickness", 0.2)
            
            if coverage < 0.3:
                # Very poor coverage - reduce first layer relative thickness
                new_t1_rel = max(0.15, current_t1_rel * 0.85)  # Reduce but stay above 0.15 minimum
                self.config["LAYERS"]["firstLayerThickness"] = new_t1_rel
                self.logger.info(f"   Reduced t1_rel: {current_t1_rel:.3f} → {new_t1_rel:.3f} for better coverage")
                
            elif coverage < 0.5 and current_t1_rel > 0.18:
                # Poor coverage - slight reduction
                new_t1_rel = max(0.18, current_t1_rel * 0.9)
                self.config["LAYERS"]["firstLayerThickness"] = new_t1_rel
                self.logger.info(f"   Adjusted t1_rel: {current_t1_rel:.3f} → {new_t1_rel:.3f}")
                
        # For uniform meshes (level 1 only), keep absolute sizing
        elif current_max_level == 1 and self.config["LAYERS"].get("relativeSizes", False):
            # Switch back to absolute for uniform meshes if coverage is good
            if coverage > 0.6:
                self.config["LAYERS"]["relativeSizes"] = False
                
                # Restore reasonable absolute values
                self.config["LAYERS"]["firstLayerThickness_abs"] = 50e-6  # 50 microns
                # Apply consistent minThickness policy instead of hardcoded value
                self._apply_consistent_minThickness_policy()
                
                # Remove relative parameters
                if "firstLayerThickness" in self.config["LAYERS"]:
                    del self.config["LAYERS"]["firstLayerThickness"] 
                if "finalLayerThickness" in self.config["LAYERS"]:
                    del self.config["LAYERS"]["finalLayerThickness"]
                    
                self.logger.info("🔄 Switched back to ABSOLUTE sizing for uniform mesh (level 1)")
                
    def _get_current_surface_refinement_level(self) -> int:
        """Get the maximum surface refinement level from current ladder step"""
        if hasattr(self, 'surface_levels') and self.surface_levels:
            return max(self.surface_levels)
        return 1  # Default to level 1

    def _apply_relative_sizing_on_ladder_progression(self) -> None:
        """
        Apply relative sizing proactively when ladder progression increases surface refinement.
        
        This is called whenever surface_levels change due to ladder progression.
        Key principle: Switch to relativeSizes=true BEFORE layers fail.
        """
        if not hasattr(self, 'surface_levels') or self.surface_levels is None:
            return
            
        current_max_level = max(self.surface_levels)
        
        # Proactively switch to relative sizing when moving beyond level 1
        if current_max_level > 1:
            if not self.config["LAYERS"].get("relativeSizes", False):
                self.config["LAYERS"]["relativeSizes"] = True
                
                # Apply conservative relative thickness ratios for stability
                # Based on your specifications: t1/Δx_surf ≈ 0.18–0.25, T/Δx_surf ≈ 0.6–0.9, ER = 1.12–1.18
                
                # Start conservatively for higher surface levels
                if current_max_level >= 3:
                    t1_rel = 0.18  # Lower end for high refinement
                    T_rel = 0.65   
                    ER = 1.12      
                elif current_max_level == 2:
                    t1_rel = 0.22  # Middle range
                    T_rel = 0.75
                    ER = 1.15
                else:  # level > 1 but < 2 (e.g., mixed levels like [1,2])
                    t1_rel = 0.25  # Higher end for moderate refinement  
                    T_rel = 0.85
                    ER = 1.18
                
                self.config["LAYERS"]["firstLayerThickness"] = t1_rel
                self.config["LAYERS"]["finalLayerThickness"] = T_rel  
                self.config["LAYERS"]["expansionRatio"] = ER
                
                # Remove absolute sizing parameters
                if "firstLayerThickness_abs" in self.config["LAYERS"]:
                    del self.config["LAYERS"]["firstLayerThickness_abs"]
                if "minThickness_abs" in self.config["LAYERS"]:
                    del self.config["LAYERS"]["minThickness_abs"]
                    
                self.logger.info(f"PROACTIVE RELATIVE SIZING for ladder progression to level {current_max_level}")
                self.logger.info(f"   Applied ratios: t1_rel={t1_rel}, T_rel={T_rel}, ER={ER}")
                self.logger.info(f"   This prevents layer extrusion failures on refined surfaces")
                
            else:
                # Already using relative sizing - adjust for new surface level
                current_t1_rel = self.config["LAYERS"].get("firstLayerThickness", 0.2)
                
                # Make ratios more conservative for higher refinement levels
                if current_max_level >= 3 and current_t1_rel > 0.20:
                    new_t1_rel = 0.18  # Conservative for high refinement
                    new_T_rel = 0.65
                    new_ER = 1.12
                    
                    self.config["LAYERS"]["firstLayerThickness"] = new_t1_rel
                    self.config["LAYERS"]["finalLayerThickness"] = new_T_rel
                    self.config["LAYERS"]["expansionRatio"] = new_ER
                    
                    self.logger.info(f"Adjusted relative sizing for high refinement level {current_max_level}")
                    self.logger.info(f"   t1_rel: {current_t1_rel:.3f} → {new_t1_rel}, T_rel: → {new_T_rel}, ER: → {new_ER}")
                    
        else:
            # Surface level is 1 - can use absolute sizing for uniformity
            if self.config["LAYERS"].get("relativeSizes", False):
                self.logger.info("Surface level 1: keeping relative sizing (if already enabled)")
                # Don't switch back automatically - let coverage-based logic handle it
    
    def _apply_consistent_minThickness_policy(self) -> None:
        """
        Apply consistent minThickness scaling policy across both relative and absolute sizing modes.
        
        Policy options (as requested):
        1. relativeSizes=true: minThickness_rel ≈ 0.18–0.25 (same ballpark as t1_rel)
        2. absoluteSizing: scale-aware absolute values:
           - Fine scale (≤1mm): 1–2 μm for small vessels  
           - Coarse scale (≥10mm): 2–5 μm for adult aorta
           
        This ensures users aren't surprised by inconsistent minThickness behavior.
        """
        is_relative_mode = self.config["LAYERS"].get("relativeSizes", False)
        
        if is_relative_mode:
            # Relative mode: minThickness_rel ≈ 0.18–0.25 (same ballpark as t1_rel)
            t1_rel = self.config["LAYERS"].get("firstLayerThickness", 0.2)
            
            # Use slightly smaller minThickness_rel than t1_rel for stability
            # Target: minThickness_rel = 0.75 * t1_rel, clamped to 0.18–0.25 range
            min_thickness_rel = max(0.18, min(0.25, t1_rel * 0.75))
            
            self.config["LAYERS"]["minThickness"] = min_thickness_rel
            
            # Remove absolute minThickness if present
            if "minThickness_abs" in self.config["LAYERS"]:
                del self.config["LAYERS"]["minThickness_abs"]
                
            self.logger.info(f"Applied relative minThickness policy: {min_thickness_rel:.3f} "
                           f"(based on t1_rel={t1_rel:.3f})")
            
        else:
            # Absolute mode: geometry-aware absolute values (drive from dx or D_ref, not scale_m)
            # Get a representative size from actual mesh/geometry, not conversion factor
            try:
                D_ref, D_min = self._estimate_reference_diameters()
                dx = self._derive_base_cell_size(D_ref, D_min)
            except Exception:
                dx = 3e-4  # safe middle fallback (300 μm)
            
            # Pick minThickness floors based on actual mesh size (tunable parameters)
            if dx <= 2e-4:          # Fine meshes (≤200μm): coronaries/small pediatrics
                min_thickness_abs = 1.0e-6  # 1 μm
            elif dx >= 1e-3:        # Coarse meshes (≥1mm): adult aorta
                min_thickness_abs = 3.0e-6  # 3 μm
            else:
                # Linear interpolate 1–3 μm based on actual mesh size
                t = (dx - 2e-4) / (1e-3 - 2e-4)  # normalize to [0,1]
                min_thickness_abs = 1.0e-6 + t * (3.0e-6 - 1.0e-6)
                
            self.config["LAYERS"]["minThickness_abs"] = min_thickness_abs
            
            # Remove relative minThickness if present  
            if "minThickness" in self.config["LAYERS"]:
                del self.config["LAYERS"]["minThickness"]
                
            self.logger.info(f"Applied geometry-aware minThickness policy: {min_thickness_abs*1e6:.1f}μm "
                           f"(based on dx={dx*1e6:.0f}μm)")
    
    def _check_coverage_gated_progression(self, proposed_surface_levels, previous_coverage_data=None) -> Tuple[List[int], bool]:
        """
        Gate progression on coverage ≥ 70% and median layers ≥ n-1.
        You'll move forward less often, but never by sacrificing boundary-layer quality.
        
        Args:
            proposed_surface_levels: Proposed new surface refinement levels from ladder
            previous_coverage_data: Coverage metrics from previous iteration
            
        Returns:
            (actual_surface_levels, progression_allowed): Final levels to use and whether progression was allowed
        """
        if previous_coverage_data is None:
            # No previous data - allow initial progression
            return proposed_surface_levels, True
            
        # Extract physics-aware metrics 
        area_weighted_coverage = previous_coverage_data.get("coverage_area_weighted", 0.0)
        median_layers = previous_coverage_data.get("median_layers", 0.0)
        target_layers = self.config["LAYERS"].get("nSurfaceLayers", 6)
        
        # Coverage gating criteria: coverage ≥ 70% and median layers ≥ n-1
        coverage_ok = area_weighted_coverage >= 0.70
        layers_ok = median_layers >= (target_layers - 1)
        
        current_levels = getattr(self, 'surface_levels', [1, 1])
        max_current = max(current_levels)
        max_proposed = max(proposed_surface_levels)
        
        progression_requested = max_proposed > max_current
        
        if progression_requested:
            if coverage_ok and layers_ok:
                # Allow progression - quality criteria met
                self.logger.info(f"COVERAGE GATED PROGRESSION: ALLOWED")
                self.logger.info(f"   Coverage: {area_weighted_coverage*100:.1f}% >= 70% OK")
                self.logger.info(f"   Median layers: {median_layers:.1f} >= {target_layers-1} OK") 
                self.logger.info(f"   Surface levels: {current_levels} → {proposed_surface_levels}")
                return proposed_surface_levels, True
            else:
                # Block progression - quality criteria not met
                self.logger.warning(f"COVERAGE GATED PROGRESSION: BLOCKED")
                self.logger.warning(f"   Coverage: {area_weighted_coverage*100:.1f}% {'OK' if coverage_ok else 'FAIL'} (need >=70%)")
                self.logger.warning(f"   Median layers: {median_layers:.1f} {'OK' if layers_ok else 'FAIL'} (need >={target_layers-1})")
                self.logger.warning(f"   Keeping current surface levels: {current_levels} (blocked {proposed_surface_levels})")
                self.logger.warning(f"   Focus on improving boundary layer quality before surface refinement")
                return current_levels, False
        else:
            # No progression requested - allow (maintaining or reducing levels)
            return proposed_surface_levels, True
    
    def _apply_complex_curvature_controls(self, coverage: float) -> None:
        """
        Apply distance-to-medial and angle controls for complex curvature.
        
        For complex curvature, use:
        - maxThicknessToMedialRatio ≈ 0.45–0.6
        - minMedianAxisAngle ≈ 60–70°  
        - featureAngle (layers) ≈ 70–80°
        
        Raise these only when coverage is <40% after a retry with ER/t1 fixes.
        """
        current_medial_ratio = self.config["LAYERS"].get("maxThicknessToMedialRatio", 0.3)
        current_axis_angle = self.config["LAYERS"].get("minMedianAxisAngle", 90)
        current_feature_angle = self.config["LAYERS"].get("featureAngle", 60)
        
        if coverage < 0.40:
            # Poor coverage - apply complex curvature settings
            target_medial_ratio = min(0.60, max(0.45, current_medial_ratio + 0.05))
            target_axis_angle = max(60, min(70, current_axis_angle - 5))  
            target_feature_angle = min(80, max(70, current_feature_angle + 5))
            
            self.config["LAYERS"]["maxThicknessToMedialRatio"] = target_medial_ratio
            self.config["LAYERS"]["minMedianAxisAngle"] = target_axis_angle
            self.config["LAYERS"]["featureAngle"] = target_feature_angle
            
            self.logger.info(f"Applied complex curvature controls for coverage {coverage*100:.1f}%")
            self.logger.info(f"   maxThicknessToMedialRatio: {current_medial_ratio:.2f} → {target_medial_ratio:.2f}")
            self.logger.info(f"   minMedianAxisAngle: {current_axis_angle}° → {target_axis_angle}°")  
            self.logger.info(f"   featureAngle: {current_feature_angle}° → {target_feature_angle}°")
            
        elif coverage >= 0.60:
            # Good coverage - can use more conservative settings
            if current_medial_ratio > 0.35:
                target_medial_ratio = 0.35  # More conservative for stable mesh
                self.config["LAYERS"]["maxThicknessToMedialRatio"] = target_medial_ratio
                self.logger.info(f"Reset to conservative medial ratio: {current_medial_ratio:.2f} → {target_medial_ratio:.2f}")

    # ------------------------------- Loop ---------------------------------
    def iterate_until_quality(self):
        self.logger.info("Starting Stage‑1 geometry‑aware optimization")
        # Note: bbox_data, diameters, and dx will be computed from scaled STL files in each iteration
        # This avoids loading STL files twice (once original, once scaled)

        best_iter = None
        best_cell_count = math.inf
        previous_coverage_data = None  # Track coverage for gated progression
        summary_path = self.output_dir / "stage1_summary.csv"
        if not summary_path.exists():
            with open(summary_path, "w", newline="") as f:
                csv.writer(f).writerow(["iter","cells","maxNonOrtho","maxSkewness","coverage","objective_dummy","levels_min","levels_max","resolveFeatureAngle","nLayers","firstLayer","minThickness"]) 

        for k in range(1, self.max_iterations + 1):
            self.current_iteration = k
            self.logger.info(f"=== ITERATION {k} ===")
            
            # Log original config values for transparency  
            snap = self.config["SNAPPY"]
            self.logger.debug(f"Config: maxGlobal={snap['maxGlobalCells']:,}, maxLocal={snap['maxLocalCells']:,}, "
                             f"featureAngle={snap.get('resolveFeatureAngle', 45)}°, "
                             f"budget={self.stage1.get('cell_budget_kb_per_cell', 1.0)}KB/cell")
            
            iter_dir = self.output_dir / f"iter_{k:03d}"
            iter_dir.mkdir(exist_ok=True)
            
            # Create required directory structure
            (iter_dir / "system").mkdir(exist_ok=True)
            (iter_dir / "logs").mkdir(exist_ok=True)

            # 1) Copy/scale STL files using config scale_m
            outlet_names = self._copy_trisurfaces(iter_dir)
            
            # 2) Compute bounding box from original STL files (in mm)
            # Scaling handled via config "scale_m": 1.0e-3
            gen = PhysicsAwareMeshGenerator()
            stl_map_orig = {"inlet": self.stl_files["required"]["inlet"],
                           self.wall_name: self.stl_files["required"][self.wall_name],
                           **{p.stem: p for p in self.stl_files["outlets"]}}
            
            # Create map to scaled STL files (now in meters)
            stl_map_norm = {"inlet": iter_dir/"constant/triSurface/inlet.stl",
                            self.wall_name: iter_dir/f"constant/triSurface/{self.wall_name}.stl",
                            **{p.stem: iter_dir/f"constant/triSurface/{p.stem}.stl" for p in self.stl_files["outlets"]}}
            
            # Compute bounding box from scaled STL files (already in meters - skip additional scaling)
            bbox_data = gen.compute_stl_bounding_box(stl_map_norm, skip_scaling=True)
            
            # Compute diameters from scaled STL files (already in meters)
            tri_dir = iter_dir / "constant" / "triSurface"
            D_ref, D_min = self._estimate_reference_diameters(stl_root=tri_dir)
            
            # Apply curvature-aware resolveFeatureAngle for vascular geometry
            adaptive_angle = self._adaptive_feature_angle(D_ref, D_min, len(outlet_names), iter_dir=iter_dir)
            self.config["SNAPPY"]["resolveFeatureAngle"] = adaptive_angle
            
            dx = self._derive_base_cell_size(D_ref, D_min)
            
            # 3) Now write blockMesh/snappy dicts using the bbox_data & dx
            self._generate_blockmesh_dict(iter_dir, bbox_data, dx)
            self._write_surfaceFeatures(iter_dir, [f"{self.wall_name}.stl", "inlet.stl", *[f"{n}.stl" for n in outlet_names]])
            
            # Apply ladder progression per iteration (before writing snappy dicts)
            ladder = self.stage1.get("ladder", [[1,1],[2,2],[2,3]])
            idx = min(self.current_iteration-1, len(ladder)-1)
            proposed_surface_levels = list(ladder[idx])
            
            # COVERAGE GATED PROGRESSION: Check if progression should be allowed
            if self.stage1.get("use_coverage_gated_progression", True):
                actual_surface_levels, progression_allowed = self._check_coverage_gated_progression(
                    proposed_surface_levels, previous_coverage_data)
            else:
                actual_surface_levels, progression_allowed = proposed_surface_levels, True
            
            # Detect if surface levels changed - if so, need full remesh
            surface_levels_changed = (not hasattr(self, 'surface_levels') or 
                                    self.surface_levels != actual_surface_levels)
            
            self.surface_levels = actual_surface_levels
            self.logger.info(f"Surface levels from ladder: {proposed_surface_levels} → {self.surface_levels}")
            if not progression_allowed:
                self.logger.info(f"   (progression blocked by coverage gating)")
            
            if surface_levels_changed:
                self.logger.info("Surface refinement levels changed - full remesh required")
                # CRITICAL: Apply relative sizing when surface refinement increases
                self._apply_relative_sizing_on_ladder_progression()
                # Apply consistent minThickness policy after sizing mode changes
                self._apply_consistent_minThickness_policy()
            else:
                self.logger.info("Surface levels unchanged - could optimize for layers-only")
            
            # Calculate seed point using robust geometric analysis
            scaled_inlet_path = iter_dir / "constant" / "triSurface" / "inlet.stl"
            internal_point = self._calculate_robust_seed_point(
                bbox_data,
                {"required": {"inlet": scaled_inlet_path}},
                dx_base=dx
            )
            
            # Calculate and apply dynamic nFeatureSnapIter before generating snappy dicts
            current_feature_angle = int(self.config["SNAPPY"].get("resolveFeatureAngle", FEATURE_ANGLE_DEFAULT))
            if current_feature_angle <= FEATURE_ANGLE_LOW_THRESHOLD:
                current_nFeatureSnapIter = FEATURE_SNAP_ITER_LOW_ANGLE
            elif current_feature_angle >= FEATURE_ANGLE_HIGH_THRESHOLD:
                current_nFeatureSnapIter = FEATURE_SNAP_ITER_HIGH_ANGLE
            else:
                # Linear interpolation between thresholds
                current_nFeatureSnapIter = int(FEATURE_SNAP_ITER_LOW_ANGLE + 
                    (FEATURE_SNAP_ITER_HIGH_ANGLE - FEATURE_SNAP_ITER_LOW_ANGLE) * 
                    (current_feature_angle - FEATURE_ANGLE_LOW_THRESHOLD) / 
                    (FEATURE_ANGLE_HIGH_THRESHOLD - FEATURE_ANGLE_LOW_THRESHOLD))
            
            # Apply the calculated nFeatureSnapIter to config so it's used in snappy dicts
            self.config["SNAPPY"]["nFeatureSnapIter"] = current_nFeatureSnapIter
            self.logger.info(f"Dynamic nFeatureSnapIter: {current_nFeatureSnapIter} (for resolveFeatureAngle={current_feature_angle}°)")

            self._snappy_dict(iter_dir, outlet_names, internal_point, dx, "no_layers")
            self._snappy_dict(iter_dir, outlet_names, internal_point, dx, "layers")
            self._surface_check(iter_dir)

            # Run mesh generation with micro-loop for layer parameter back-offs
            K = MICRO_TRIALS_PER_ITERATION
            best_local = None
            
            for trial in range(1, K+1):
                self.logger.info(f"Micro-trial {trial}/{K} for iteration {k}")
                
                snap_m, layer_m, layer_cov = self._run_snap_then_layers(
                    iter_dir, 
                    force_full_remesh=(trial == 1 and surface_levels_changed)
                )
                
                # Evaluate trial quality
                constraints_ok, failure_reasons = self._meets_quality_constraints(snap_m, layer_m, layer_cov)
                coverage = layer_cov.get('coverage_overall', 0.0)
                
                # Track best trial by coverage
                if best_local is None or coverage > best_local[2].get('coverage_overall', 0):
                    best_local = (snap_m, layer_m, layer_cov)
                    self.logger.info(f"Trial {trial}: coverage={coverage*100:.1f}% (new best)")
                else:
                    self.logger.info(f"Trial {trial}: coverage={coverage*100:.1f}% (not better)")
                
                # Stop if we have good quality or acceptable coverage
                if constraints_ok or coverage >= COVERAGE_ACCEPTABLE_THRESHOLD:
                    self.logger.info(f"Stopping micro-loop: {'constraints OK' if constraints_ok else f'coverage ≥{COVERAGE_ACCEPTABLE_THRESHOLD*100:.0f}%'}")
                    break
                
                # Quick parameter back-offs for next trial (no re-castellation)
                if trial < K:
                    self.logger.info(f"Applying quick back-offs for trial {trial+1}")
                    
                    # Apply relative sizing if not already enabled
                    self._apply_relative_sizing_for_refinement(coverage)
                    
                    # Reduce expansion ratio slightly
                    if self.config["LAYERS"].get("expansionRatio", 1.2) > 1.12:
                        old_er = self.config["LAYERS"]["expansionRatio"]
                        self.config["LAYERS"]["expansionRatio"] = max(1.12, old_er - 0.02)
                        self.logger.info(f"  Reduced expansionRatio: {old_er:.3f} → {self.config['LAYERS']['expansionRatio']:.3f}")
                    
                    # Reduce first layer thickness (relative or absolute)
                    if self.config["LAYERS"].get("relativeSizes", False):
                        old_t1 = self.config["LAYERS"].get("firstLayerThickness", 0.2)
                        new_t1 = max(0.15, 0.9 * old_t1)
                        self.config["LAYERS"]["firstLayerThickness"] = new_t1
                        self.logger.info(f"  Reduced relative t1: {old_t1:.3f} → {new_t1:.3f}")
                    else:
                        old_t1 = self.config["LAYERS"].get("firstLayerThickness_abs", 50e-6)
                        new_t1 = max(10e-6, 0.8 * old_t1)
                        self.config["LAYERS"]["firstLayerThickness_abs"] = new_t1
                        self.logger.info(f"  Reduced absolute t1: {old_t1*1e6:.1f}μm → {new_t1*1e6:.1f}μm")
                    
                    # Update minThickness policy after first layer changes
                    self._apply_consistent_minThickness_policy()
            
            # Use the best trial results for subsequent processing
            snap_m, layer_m, layer_cov = best_local
            self.logger.info(f"Using best trial with coverage={layer_cov.get('coverage_overall', 0)*100:.1f}%")
            
            # Log surface histogram to identify thickness/cell size issues
            if hasattr(self, 'surface_levels'):
                t1_thickness = self.config["LAYERS"].get("firstLayerThickness_abs", 50e-6)
                n_layers = self.config["LAYERS"].get("nSurfaceLayers", 6)
                expansion = self.config["LAYERS"].get("expansionRatio", 1.2)
                total_thickness = t1_thickness * (expansion**n_layers - 1) / (expansion - 1) if expansion != 1.0 else t1_thickness * n_layers
                
                log_surface_histogram(iter_dir, self.surface_levels, dx, t1_thickness, total_thickness, self.logger)
            
            # Update coverage data for next iteration's gating
            previous_coverage_data = layer_cov
            
            # Objective
            # Check quality constraints and get cell count
            constraints_ok, failure_reasons = self._meets_quality_constraints(snap_m, layer_m, layer_cov)
            cell_count = self._get_cell_count(layer_m, snap_m)
            
            # Get current feature snap iter for logging (already applied to config earlier)
            current_nFeatureSnapIter = self.config["SNAPPY"].get("nFeatureSnapIter", 20)

            # Log
            cov = float(layer_cov.get("coverage_overall", 0.0))
            wall_cov = layer_cov.get("perPatch", {}).get(self.wall_name, cov)
            status = "PASS" if constraints_ok else "FAIL"
            self.logger.info(f"RESULTS: cells={cell_count:,}, maxNonOrtho={layer_m.get('maxNonOrtho',0):.1f}, maxSkewness={layer_m.get('maxSkewness',0):.2f}, wall_cov={wall_cov*100:.1f}% [{status}]")
            
            # Log failure reasons if any
            if failure_reasons:
                self.logger.info(f"Constraint failures: {'; '.join(failure_reasons)}")

            # Calculate deltas for triage
            delta_nonortho = float(layer_m.get("maxNonOrtho", 0)) - float(snap_m.get("maxNonOrtho", 0))
            delta_skewness = float(layer_m.get("maxSkewness", 0)) - float(snap_m.get("maxSkewness", 0))
            
            # Save per‑iter metrics JSON
            metrics = {
                "iteration": k,
                "checkMesh_snap": snap_m,
                "checkMesh_layer": layer_m,
                "layerCoverage": layer_cov,
                "delta": {
                    "maxNonOrtho": delta_nonortho,
                    "maxSkewness": delta_skewness,
                    "interpretation": {
                        "layers_degrade_quality": delta_nonortho > 5 or delta_skewness > 0.5,
                        "layers_improve_quality": delta_nonortho < -2 or delta_skewness < -0.2,
                        "layers_neutral": abs(delta_nonortho) <= 2 and abs(delta_skewness) <= 0.2
                    }
                },
                "surface_levels": self.surface_levels,
                "resolveFeatureAngle": int(self.config["SNAPPY"].get("resolveFeatureAngle",45)),
                "nFeatureSnapIter": current_nFeatureSnapIter,
                "layers": {
                    "nSurfaceLayers": self.config["LAYERS"]["nSurfaceLayers"],
                    "firstLayerThickness_abs": self.config["LAYERS"].get("firstLayerThickness_abs", 50e-6),
                    "minThickness_abs": self.config["LAYERS"].get("minThickness_abs", 20e-6),
                },
                "constraints_met": constraints_ok,
                "cell_count": cell_count
            }
            (iter_dir / "stage1_metrics.json").write_text(json.dumps(metrics, indent=2))

            # Append CSV summary (before plateau check to ensure final iteration is always logged)
            # Generate accurate CSV values based on sizing mode
            L = self.config["LAYERS"]
            if L.get("relativeSizes", False):
                t1_out = f"{L.get('firstLayerThickness', 0.2):.3f}rel"
                min_out = f"{L.get('minThickness', 0.15):.3f}rel"
            else:
                t1_out = f"{L.get('firstLayerThickness_abs', 50e-6):.3e}"
                min_out = f"{L.get('minThickness_abs', 2e-6):.3e}"
            
            with open(summary_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    k,
                    layer_m.get("cells",0),
                    f"{layer_m.get('maxNonOrtho',0):.1f}",
                    f"{layer_m.get('maxSkewness',0):.2f}",
                    f"{cov:.3f}",
                    "0.000",  # Dummy - no objective function in constraint-based approach
                    self.surface_levels[0], self.surface_levels[1],
                    int(self.config["SNAPPY"].get("resolveFeatureAngle",45)),
                    self.config["LAYERS"]["nSurfaceLayers"],
                    t1_out,
                    min_out,
                ])

            # Constraint-based acceptance: minimize cells subject to quality constraints
            if constraints_ok:
                if cell_count < best_cell_count:
                    best_cell_count = cell_count
                    best_iter = iter_dir
                    self.logger.info(f"ACCEPTED: iter {k} with {cell_count:,} cells (new best)")
                else:
                    self.logger.info(f"CONSTRAINTS MET: iter {k} with {cell_count:,} cells (not better than {best_cell_count:,})")
                
                # If we have a valid solution, we can stop (constraint-based approach)
                # Continue only if we want to try to find a better (smaller) mesh
                # BUT: allow more iterations if we've made recent improvements to coverage/layer strategy
                if best_iter is not None and k >= 3:  # Give at least 3 iterations for layer improvements
                    self.logger.info(f"Constraint-based optimization complete: best mesh has {best_cell_count:,} cells")
                    break
            else:
                self.logger.info(f"CONSTRAINTS NOT MET: iter {k} - continuing optimization")

            # CRITICAL: Always apply updates to improve coverage/layers, even when constraints pass
            # The enhanced layer strategy should run regardless of constraint status
            self._apply_updates(snap_m, layer_m, layer_cov)

        if best_iter is None:
            self.logger.warning("Stage-1 did not meet all quality constraints within max_iterations")
            best_iter = iter_dir
            final_status = "INCOMPLETE"
        else:
            final_status = "COMPLETE"
        
        # Export best iteration to stable location for Stage 2
        best_out = self.output_dir / "best"
        if best_out.exists():
            shutil.rmtree(best_out)
        shutil.copytree(best_iter, best_out)
        
        # Save config for Stage 2 compatibility
        config_out = best_out / "config.json"
        with open(config_out, "w") as f:
            json.dump(self.config, f, indent=2)
        
        # Log final summary
        if best_iter != iter_dir:
            # Load metrics from best iteration
            best_metrics_path = best_iter / "stage1_metrics.json"
            if best_metrics_path.exists():
                with open(best_metrics_path) as f:
                    best_metrics = json.load(f)
                best_cells = best_metrics.get("cell_count", 0)
                self.logger.info(f"Stage-1 {final_status}: Best geometry-based mesh has {best_cells:,} cells")
        
        self.logger.info(f"Best Stage-1 mesh exported to: {best_out}")
        self.logger.info(f"Ready for Stage-2 physics verification (GCI analysis)")
        
        return best_out

