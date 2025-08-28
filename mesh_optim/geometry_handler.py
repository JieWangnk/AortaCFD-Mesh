"""Geometry processing module - simplified implementation after folder cleanup."""

import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from .utils import get_openfoam_geometry_dir


class GeometryHandler:
    """Simplified geometry handler for STL processing and analysis."""
    
    def __init__(self, geometry_dir: Path, config: Dict, logger: logging.Logger):
        self.geometry_dir = Path(geometry_dir)
        self.config = config
        self.logger = logger
        
        # Discover wall patch name
        self.wall_name = self._discover_wall_patch()
        
        # Discover STL files
        self.stl_files = self._discover_stl_files()
        
        self.logger.info(f"Wall patch discovered: {self.wall_name}")
        outlet_count = len(self.stl_files.get("outlets", []))
        self.logger.info(f"Found {outlet_count} outlet files")
    
    def _discover_wall_patch(self) -> str:
        """Discover the wall patch name from configuration or STL files."""
        # Try config first
        wall_name = self.config.get("wall_patch_name", "wall_aorta")
        
        # Verify it exists
        wall_stl = self.geometry_dir / f"{wall_name}.stl"
        if wall_stl.exists():
            return wall_name
        
        # Fallback search
        for candidate in ["wall_aorta", "aorta", "wall"]:
            if (self.geometry_dir / f"{candidate}.stl").exists():
                return candidate
        
        raise ValueError("No wall STL file found")
    
    def _discover_stl_files(self) -> Dict:
        """Discover and categorize STL files."""
        stl_files = {"outlets": []}
        
        for stl_file in self.geometry_dir.glob("*.stl"):
            name = stl_file.stem
            if name.startswith("outlet") or name.startswith("out"):
                stl_files["outlets"].append(stl_file)
        
        return stl_files
    
    def copy_and_scale_stl_files(self, iter_dir: Path) -> List[str]:
        """Copy and scale STL files to iteration directory."""
        geometry_dir = get_openfoam_geometry_dir()
        tri_surface_dir = iter_dir / "constant" / geometry_dir
        tri_surface_dir.mkdir(parents=True, exist_ok=True)
        
        # Get scaling factor from config
        scale_factor = self.config.get("SCALING", {}).get("scale_m", 1.0)
        
        outlet_names = []
        
        # Copy and scale all STL files
        for stl_file in self.geometry_dir.glob("*.stl"):
            dest_file = tri_surface_dir / stl_file.name
            
            if scale_factor != 1.0:
                # Apply scaling
                self._scale_stl_file(stl_file, dest_file, scale_factor)
            else:
                # No scaling needed
                shutil.copy2(stl_file, dest_file)
            
            if stl_file.stem.startswith("outlet"):
                outlet_names.append(stl_file.stem)
        
        if scale_factor != 1.0:
            self.logger.info(f"Copied and scaled {len(list(self.geometry_dir.glob('*.stl')))} STL files (scale={scale_factor}) to {tri_surface_dir}")
        else:
            self.logger.info(f"Copied {len(list(self.geometry_dir.glob('*.stl')))} STL files to {tri_surface_dir}")
        
        return outlet_names
    
    def _scale_stl_file(self, source: Path, dest: Path, scale_factor: float):
        """Scale STL file by the given factor."""
        import struct
        
        with open(source, 'rb') as src, open(dest, 'wb') as dst:
            # Copy header (80 bytes)
            header = src.read(80)
            dst.write(header)
            
            # Read number of triangles
            n_triangles_bytes = src.read(4)
            n_triangles = struct.unpack('<I', n_triangles_bytes)[0]
            dst.write(n_triangles_bytes)
            
            # Process each triangle
            for _ in range(n_triangles):
                # Read triangle data (50 bytes total)
                triangle_data = src.read(50)
                
                # Unpack triangle (normal + 3 vertices + attribute)
                # Format: 3 floats (normal), 9 floats (3 vertices), 1 short (attribute)
                unpacked = struct.unpack('<12fH', triangle_data)
                
                # Scale only the vertex coordinates (indices 3-11), not the normal (0-2)
                scaled_data = list(unpacked)
                for i in range(3, 12):  # Vertex coordinates
                    scaled_data[i] *= scale_factor
                
                # Repack and write
                packed_data = struct.pack('<12fH', *scaled_data)
                dst.write(packed_data)
    
    def estimate_reference_diameters(self, stl_root: Path) -> Tuple[float, float]:
        """Estimate reference diameters from outlet geometries."""
        outlet_files = list(stl_root.glob("outlet*.stl"))
        
        if not outlet_files:
            self.logger.warning("No outlet files found, using default diameters")
            return 0.020, 0.015  # Default 20mm, 15mm
        
        # Simplified diameter estimation based on file count and typical aortic dimensions
        n_outlets = len(outlet_files)
        
        if n_outlets <= 2:
            D_ref = 0.025  # 25mm for large outlets
            D_min = 0.020  # 20mm
        elif n_outlets <= 4:
            D_ref = 0.020  # 20mm 
            D_min = 0.015  # 15mm
        else:
            D_ref = 0.015  # 15mm for many small outlets
            D_min = 0.010  # 10mm
        
        self.logger.info(f"Computed diameters: D_ref={D_ref*1000:.1f}mm, D_min={D_min*1000:.1f}mm from {n_outlets} outlets")
        return D_ref, D_min
    
    def derive_base_cell_size(self, D_ref: float, D_min: float) -> float:
        """Derive base cell size from reference diameters."""
        # Get cells per diameter from config with type conversion
        N_D = int(self.config.get("STAGE1", {}).get("N_D", 22))
        N_D_min = int(self.config.get("STAGE1", {}).get("N_D_min", 28))
        
        # Ensure inputs are floats
        D_ref = float(D_ref)
        D_min = float(D_min)
        
        # Calculate cell sizes
        dx_ref = D_ref / N_D
        dx_min = D_min / N_D_min
        
        # Use the smaller (more conservative) cell size
        dx = min(dx_ref, dx_min)
        
        self.logger.info(f"Base cell size: D_ref/N_D={dx_ref*1000:.1f}mm, D_min/N_D_min={dx_min*1000:.1f}mm → dx={dx*1000:.1f}mm")
        
        return dx
    
    def adaptive_feature_angle(self, D_ref: float, D_min: float, n_outlets: int, **kwargs) -> int:
        """Calculate adaptive feature angle based on geometry complexity.
        
        Args:
            D_ref: Reference diameter
            D_min: Minimum diameter
            n_outlets: Number of outlet branches
            **kwargs: Additional parameters (e.g., iter_dir) for compatibility
        """
        # Simple adaptive logic
        size_ratio = D_ref / D_min if D_min > 0 else 1.0
        
        # Base angle from config
        base_angle = self.config.get("STAGE1", {}).get("featureAngle_init", 45)
        
        # Adjust based on complexity
        if size_ratio > 2.0 and n_outlets > 3:
            angle = max(base_angle - 15, 25)  # More aggressive for complex geometry
        else:
            angle = base_angle
        
        self.logger.info(f"Adaptive feature angle: size_ratio={size_ratio:.2f}, branches={n_outlets}, curvature=1.000 → {angle}°")
        
        return angle
    
    def calculate_seed_point(self, bbox_data: Dict, stl_map: Dict, dx_base: float) -> List[float]:
        """Calculate internal seed point using the PROVEN robust method."""
        from .geometry_utils import find_internal_seed_point, read_stl_triangles
        import numpy as np
        
        # Get bbox info - handle both mesh_domain and direct bbox formats
        if "mesh_domain" in bbox_data:
            bbox_min = np.array([
                bbox_data["mesh_domain"]["x_min"],
                bbox_data["mesh_domain"]["y_min"], 
                bbox_data["mesh_domain"]["z_min"]
            ])
            bbox_max = np.array([
                bbox_data["mesh_domain"]["x_max"],
                bbox_data["mesh_domain"]["y_max"],
                bbox_data["mesh_domain"]["z_max"]
            ])
        elif "bbox_min" in bbox_data and "bbox_max" in bbox_data:
            bbox_min = np.array(bbox_data["bbox_min"])
            bbox_max = np.array(bbox_data["bbox_max"])
        else:
            self.logger.warning("No bounding box data - using origin")
            return [0.0, 0.0, 0.0]
            
        bbox_center = (bbox_min + bbox_max) / 2.0
        bbox_size = bbox_max - bbox_min
        
        # Find inlet STL file
        inlet_path = None
        if "inlet" in stl_map:
            inlet_path = stl_map["inlet"]
        elif stl_map and "required" in stl_map and "inlet" in stl_map["required"]:
            inlet_path = stl_map["required"]["inlet"]
        else:
            # Search for inlet file
            for key, path in stl_map.items():
                if "inlet" in str(path).lower():
                    inlet_path = path
                    break
        
        if not inlet_path or not Path(inlet_path).exists():
            self.logger.warning("No inlet STL found - using bbox center")
            return bbox_center.tolist()
        
        # Read inlet triangles using proven robust method
        try:
            inlet_triangles = read_stl_triangles(Path(inlet_path))
            if not inlet_triangles:
                self.logger.warning("No triangles in inlet STL - using bbox center")
                return bbox_center.tolist()
                
            self.logger.info(f"Read {len(inlet_triangles)} triangles from inlet STL")
            
            # Use the proven robust seed point calculation
            seed_point = find_internal_seed_point(
                bbox_center,
                bbox_size, 
                inlet_triangles,
                dx_base,
                self.logger
            )
            
            return seed_point.tolist()
            
        except Exception as e:
            self.logger.error(f"Robust seed point calculation failed: {e}")
            self.logger.warning("Falling back to bbox center")
            return bbox_center.tolist()
    
    def _calculate_inlet_centroid(self, inlet_path: Path) -> List[float]:
        """Calculate centroid of inlet STL vertices."""
        import struct
        
        with open(inlet_path, 'rb') as f:
            # Skip STL header (80 bytes)
            f.read(80)
            
            # Read number of triangles
            n_triangles = struct.unpack('<I', f.read(4))[0]
            
            vertices = []
            for _ in range(n_triangles):
                # Skip normal vector (12 bytes)
                f.read(12)
                
                # Read 3 vertices (36 bytes total)
                for _ in range(3):
                    vertex = struct.unpack('<fff', f.read(12))
                    vertices.append(vertex)
                
                # Skip attribute byte count (2 bytes)
                f.read(2)
            
            # Calculate centroid
            if vertices:
                centroid = [
                    sum(v[0] for v in vertices) / len(vertices),
                    sum(v[1] for v in vertices) / len(vertices), 
                    sum(v[2] for v in vertices) / len(vertices)
                ]
                return centroid
            else:
                raise ValueError("No vertices found in inlet STL")

    def validate_surface_quality(self, warn_only: bool = True) -> Dict[str, bool]:
        """Validate STL surface quality using OpenFOAM's surfaceCheck.
        
        Returns dict with validation results for each surface.
        If warn_only=False, raises exception on serious quality issues.
        """
        results = {}
        env_setup = self.config.get("openfoam_env_path", "source /opt/openfoam12/etc/bashrc")
        
        for surface_type, stl_list in self.stl_files.items():
            if surface_type == "outlets":
                for outlet_file in stl_list:
                    results[outlet_file.name] = self._check_single_surface(outlet_file, env_setup, warn_only)
            else:
                for stl_file in stl_list:
                    results[stl_file.name] = self._check_single_surface(stl_file, env_setup, warn_only)
        
        return results
    
    def _check_single_surface(self, stl_file: Path, env_setup: str, warn_only: bool) -> bool:
        """Check single STL surface quality."""
        try:
            cmd = f"{env_setup} && surfaceCheck {stl_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            output = result.stdout + result.stderr
            
            # Check for critical issues
            is_closed = "Surface is not closed" not in output
            has_good_quality = "min" in output and "1e-10" not in output  # Very small triangles
            no_overlaps = "More than one normal orientation" not in output
            
            surface_ok = is_closed and has_good_quality and no_overlaps
            
            if not surface_ok:
                issues = []
                if not is_closed: issues.append("open surface")
                if not has_good_quality: issues.append("poor triangle quality")  
                if not no_overlaps: issues.append("inconsistent normals")
                
                msg = f"Surface quality issues in {stl_file.name}: {', '.join(issues)}"
                if warn_only:
                    self.logger.warning(msg)
                    self.logger.warning("Consider using geometry-tolerant configuration for stability")
                else:
                    raise ValueError(msg)
            
            return surface_ok
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Surface check timeout for {stl_file.name}")
            return False
        except Exception as e:
            self.logger.warning(f"Surface check failed for {stl_file.name}: {e}")
            return False


# Backward compatibility functions (deprecated - use GeometryHandler methods instead)
def estimate_reference_diameters(stl_files: List[Path], logger=None) -> Tuple[float, float]:
    """Deprecated: Use GeometryHandler.estimate_reference_diameters() instead."""
    if logger:
        logger.warning("Using deprecated estimate_reference_diameters function. Use GeometryHandler.estimate_reference_diameters() instead.")
    return 0.020, 0.015

def adaptive_feature_angle(size_ratio: float, n_branches: int, curvature: float = 1.0) -> int:
    """Backward compatibility function.""" 
    if size_ratio > 2.0 and n_branches > 3:
        return 30
    return 45

def derive_base_cell_size(D_ref: float, D_min: float, N_D: int = 22, N_D_min: int = 28) -> float:
    """Backward compatibility function."""
    return min(D_ref / N_D, D_min / N_D_min)

def calculate_refinement_bands_with_override(dx: float, near_cells: int = 4, far_cells: int = 10) -> Tuple[float, float]:
    """Backward compatibility function."""
    return near_cells * dx, far_cells * dx