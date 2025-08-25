"""
Physics-aware mesh generation with proper boundary layer resolution

This module implements the first-layer thickness calculations and physics-aware 
mesh parameters for laminar/RANS/LES flow regimes.
"""

import math
import json
import numpy as np
import struct
from pathlib import Path
from typing import Dict, Tuple, Optional

class PhysicsAwareMeshGenerator:
    """Generate physics-aware mesh parameters for different flow regimes"""
    
    def __init__(self, flow_model: str = "RANS"):
        """
        Initialize physics-aware mesh generator
        
        Args:
            flow_model: Flow regime ('LAMINAR', 'RANS', or 'LES')
        """
        self.flow_model = flow_model.upper()
        
        # Blood properties at 37°C
        self.blood_properties = {
            "density": 1060,  # kg/m³
            "kinematic_viscosity": 3.3e-6,  # m²/s
            "dynamic_viscosity": 3.5e-3  # Pa·s
        }
        
    def calculate_first_layer_thickness(self, peak_velocity: float, diameter: float, 
                                      target_yplus: float = 1.0, frequency: float = None) -> Dict:
        """
        Calculate first layer thickness for target y+ value with correct friction laws
        
        Args:
            peak_velocity: Peak velocity in m/s
            diameter: Characteristic diameter in m (minimum diameter for coarctation)
            target_yplus: Target y+ value
            frequency: Heart rate frequency in Hz (for Womersley number)
            
        Returns:
            Dictionary with calculated parameters
        """
        nu = self.blood_properties["kinematic_viscosity"]
        rho = self.blood_properties["density"]
        
        # Calculate Reynolds number based on minimum diameter
        Re = peak_velocity * diameter / nu
        
        # Choose correct friction coefficient correlation
        if Re < 2100:
            # Laminar flow: Cf = 16/Re (exact for Hagen-Poiseuille)
            Cf = 16.0 / Re
            flow_regime = "laminar"
        elif Re < 4000:
            # Transitional flow: interpolate between laminar and turbulent
            Re_crit = 2100
            Cf_lam = 16.0 / Re_crit
            Cf_turb = 0.079 * (Re ** -0.25)
            # Linear interpolation
            alpha = (Re - 2100) / (4000 - 2100)
            Cf = (1 - alpha) * Cf_lam + alpha * Cf_turb
            flow_regime = "transitional"
        else:
            # Turbulent flow: Blasius correlation for smooth pipes
            Cf = 0.079 * (Re ** -0.25)
            flow_regime = "turbulent"
        
        # Friction velocity
        u_tau = math.sqrt(0.5 * Cf) * peak_velocity
        
        # Calculate Womersley number for pulsatile effects
        womersley = None
        if frequency is not None:
            omega = 2 * math.pi * frequency  # rad/s
            womersley = diameter * math.sqrt(rho * omega / (8 * self.blood_properties["dynamic_viscosity"]))
            
            # For high Womersley number (α > 3), viscous sublayer thickens
            if womersley > 3.0:
                # Empirical correction for pulsatile boundary layer thickness
                pulsatile_factor = 1.0 + 0.1 * min(womersley - 3.0, 7.0)  # Cap at α=10
                target_yplus *= pulsatile_factor
        
        # First layer height for target y+
        h1 = target_yplus * nu / u_tau
        
        # Calculate recommended parameters
        results = {
            "reynolds": Re,
            "flow_regime": flow_regime,
            "friction_coefficient": Cf,
            "friction_velocity": u_tau,
            "first_layer_thickness": h1,
            "first_layer_microns": h1 * 1e6,
            "target_yplus": target_yplus,
            "womersley": womersley,
            "pulsatile_correction": frequency is not None and womersley > 3.0
        }
        
        return results
    
    def compute_stl_bounding_box(self, stl_files: Dict[str, Path], skip_scaling: bool = False) -> Dict:
        """
        Compute bounding box from STL files for automatic mesh domain calculation
        
        Args:
            stl_files: Dictionary with STL file names and paths
            skip_scaling: If True, keep original units (no mm->m conversion)
            
        Returns:
            Dictionary with bounding box coordinates and mesh domain parameters
        """
        try:
            all_vertices = []
            
            for stl_name, stl_path in stl_files.items():
                if not stl_path.exists():
                    print(f"Warning: STL file {stl_path} not found")
                    continue
                    
                # Try ASCII format first
                vertices = []
                try:
                    with open(stl_path, 'r') as f:
                        lines = f.readlines()
                        
                    for line in lines:
                        line = line.strip()
                        if line.startswith('vertex'):
                            coords = [float(x) for x in line.split()[1:4]]
                            vertices.append(coords)
                    
                    if vertices:
                        all_vertices.extend(vertices)
                        print(f"Loaded {len(vertices)} vertices from ASCII {stl_name}")
                        
                except UnicodeDecodeError:
                    # Try binary STL format
                    try:
                        vertices = self._read_binary_stl(stl_path)
                        if vertices:
                            all_vertices.extend(vertices)
                            print(f"Loaded {len(vertices)} vertices from binary {stl_name}")
                    except Exception as e:
                        print(f"Failed to read binary STL {stl_name}: {e}")
                        continue
                        
                except Exception as e:
                    print(f"Error reading {stl_name}: {e}")
                    continue
            
            if not all_vertices:
                print("Warning: No vertices found in STL files, using default bounding box")
                return self._get_default_bounding_box()
                
            all_vertices = np.array(all_vertices)
            
            # Calculate bounding box
            x_min, x_max = all_vertices[:, 0].min(), all_vertices[:, 0].max()
            y_min, y_max = all_vertices[:, 1].min(), all_vertices[:, 1].max() 
            z_min, z_max = all_vertices[:, 2].min(), all_vertices[:, 2].max()
            
            # Apply scaling if requested
            if skip_scaling:
                # Keep original coordinates (e.g., mm)
                bbox_m = {
                    "x_min": x_min, "x_max": x_max,
                    "y_min": y_min, "y_max": y_max,
                    "z_min": z_min, "z_max": z_max
                }
            else:
                # Convert from mm to meters (default behavior)
                bbox_m = {
                    "x_min": x_min * 1e-3, "x_max": x_max * 1e-3,
                    "y_min": y_min * 1e-3, "y_max": y_max * 1e-3,
                    "z_min": z_min * 1e-3, "z_max": z_max * 1e-3
                }
            
            # Calculate dimensions
            length = bbox_m["z_max"] - bbox_m["z_min"]
            width = bbox_m["x_max"] - bbox_m["x_min"]  
            height = bbox_m["y_max"] - bbox_m["y_min"]
            
            # Add safety margins for mesh domain (20% expansion)
            margin_factor = 1.2
            center_x = (bbox_m["x_max"] + bbox_m["x_min"]) / 2
            center_y = (bbox_m["y_max"] + bbox_m["y_min"]) / 2
            center_z = (bbox_m["z_max"] + bbox_m["z_min"]) / 2
            
            mesh_domain = {
                "x_min": center_x - width * margin_factor / 2,
                "x_max": center_x + width * margin_factor / 2,
                "y_min": center_y - height * margin_factor / 2, 
                "y_max": center_y + height * margin_factor / 2,
                "z_min": center_z - length * margin_factor / 2,
                "z_max": center_z + length * margin_factor / 2
            }
            
            return {
                "geometry_bbox": bbox_m,
                "mesh_domain": mesh_domain,
                "dimensions": {
                    "length": length,
                    "width": width, 
                    "height": height
                },
                "center": [center_x, center_y, center_z],
                "total_vertices": len(all_vertices)
            }
            
        except Exception as e:
            print(f"Error computing bounding box: {e}")
            return self._get_default_bounding_box()
    
    def _get_default_bounding_box(self) -> Dict:
        """Default bounding box for typical aortic geometry"""
        return {
            "geometry_bbox": {
                "x_min": -0.05, "x_max": 0.05,
                "y_min": -0.05, "y_max": 0.05, 
                "z_min": -0.05, "z_max": 0.15
            },
            "mesh_domain": {
                "x_min": -0.08, "x_max": 0.08,
                "y_min": -0.08, "y_max": 0.08,
                "z_min": -0.08, "z_max": 0.18  
            },
            "dimensions": {
                "length": 0.20,
                "width": 0.10,
                "height": 0.10
            },
            "center": [0.0, 0.0, 0.05],
            "total_vertices": 0
        }
    
    def _read_binary_stl(self, stl_path: Path) -> list:
        """
        Read vertices from binary STL file
        
        Args:
            stl_path: Path to binary STL file
            
        Returns:
            List of vertices as [x, y, z] coordinates
        """
        vertices = []
        try:
            with open(stl_path, 'rb') as f:
                # Skip 80-byte header
                f.seek(80)
                
                # Read number of triangles (4 bytes, little-endian unsigned int)
                num_triangles_data = f.read(4)
                if len(num_triangles_data) != 4:
                    return vertices
                    
                num_triangles = struct.unpack('<I', num_triangles_data)[0]
                
                # Read each triangle
                for i in range(num_triangles):
                    # Skip normal vector (12 bytes)
                    normal = f.read(12)
                    if len(normal) != 12:
                        break
                        
                    # Read 3 vertices (36 bytes total: 3 vertices × 3 coords × 4 bytes)
                    for v in range(3):
                        vertex_data = f.read(12)  # 3 floats × 4 bytes
                        if len(vertex_data) != 12:
                            break
                        x, y, z = struct.unpack('<fff', vertex_data)
                        vertices.append([x, y, z])
                    
                    # Skip attribute byte count (2 bytes)
                    attr = f.read(2)
                    if len(attr) != 2:
                        break
                        
        except Exception as e:
            print(f"Error reading binary STL: {e}")
            return []
            
        return vertices

    def extract_minimum_diameter_from_stl(self, wall_stl_path: Path) -> float:
        """
        Extract minimum diameter from wall STL file
        
        Args:
            wall_stl_path: Path to wall_aorta.stl file
            
        Returns:
            Minimum diameter in meters
        """
        try:
            # Try ASCII format first
            vertices = []
            try:
                with open(wall_stl_path, 'r') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if line.startswith('vertex'):
                        coords = [float(x) for x in line.split()[1:4]]
                        vertices.append(coords)
                        
            except UnicodeDecodeError:
                # Try binary STL format
                vertices = self._read_binary_stl(wall_stl_path)
                if not vertices:
                    print(f"Failed to read binary STL, using default diameter")
                    return 25e-3
            
            if not vertices:
                raise ValueError("No vertices found in STL file")
            
            vertices = np.array(vertices)
            
            # Find minimum cross-sectional diameter along z-axis (flow direction)
            z_coords = vertices[:, 2]
            z_min, z_max = z_coords.min(), z_coords.max()
            
            min_diameter = float('inf')
            n_slices = 50
            
            for i in range(n_slices):
                z = z_min + (z_max - z_min) * i / n_slices
                # Get vertices in thin slice
                slice_mask = (np.abs(vertices[:, 2] - z) < (z_max - z_min) / (2 * n_slices))
                slice_verts = vertices[slice_mask]
                
                if len(slice_verts) > 3:  # Need enough points for diameter estimation
                    # Estimate diameter as range in x and y directions
                    x_range = slice_verts[:, 0].max() - slice_verts[:, 0].min()
                    y_range = slice_verts[:, 1].max() - slice_verts[:, 1].min()
                    estimated_diameter = min(x_range, y_range)  # Conservative estimate
                    min_diameter = min(min_diameter, estimated_diameter)
            
            # Convert from mm to m (assuming STL in mm)
            return max(min_diameter * 1e-3, 10e-3)  # Minimum 10mm for safety
            
        except Exception as e:
            # Fallback to typical aortic diameter
            print(f"Warning: Could not extract diameter from STL: {e}")
            return 20e-3  # 20mm default
    
    def load_flow_data(self, flow_csv_path: Path) -> Dict:
        """
        Load peak velocity and frequency from BPM75.csv or similar
        
        Args:
            flow_csv_path: Path to CSV with time, velocity data
            
        Returns:
            Dictionary with peak_velocity and heart_rate
        """
        try:
            import csv
            
            times, velocities = [], []
            with open(flow_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    times.append(float(row['time']))
                    velocities.append(float(row['velocity']))
            
            if not velocities:
                raise ValueError("No velocity data found")
            
            peak_velocity = max(velocities)
            
            # Estimate heart rate from cycle time
            times = np.array(times)
            cycle_time = times[-1] - times[0]  # Assuming one cardiac cycle
            heart_rate = 1.0 / cycle_time if cycle_time > 0 else 75/60  # Default 75 BPM
            
            return {
                "peak_velocity": peak_velocity,
                "heart_rate": heart_rate,
                "cycle_time": cycle_time
            }
            
        except Exception as e:
            print(f"Warning: Could not load flow data: {e}")
            # Conservative defaults for aortic flow
            return {
                "peak_velocity": 1.0,  # 1.0 m/s typical
                "heart_rate": 75/60,   # 75 BPM
                "cycle_time": 0.8      # 0.8s cycle
            }
    
    def get_flow_regime_parameters(self, diameter: float = None, peak_velocity: float = None, 
                                 geometry_dir: Path = None, frequency: float = None) -> Dict:
        """
        Get recommended mesh parameters for the specified flow regime with automatic data loading
        
        Args:
            diameter: Characteristic diameter in meters (minimum for coarctation). If None, extracted from STL
            peak_velocity: Peak velocity in m/s. If None, loaded from CSV
            geometry_dir: Directory with STL files and flow data
            frequency: Heart rate frequency in Hz. If None, estimated from flow data
            
        Returns:
            Dictionary with mesh parameters optimized for flow regime including bounding box
        """
        
        bounding_box_data = None
        
        # Auto-load missing parameters from geometry directory
        if geometry_dir is not None:
            geometry_dir = Path(geometry_dir)
            
            # Compute bounding box from all STL files
            stl_files = {}
            for stl_file in geometry_dir.glob("*.stl"):
                stl_files[stl_file.stem] = stl_file
                
            if stl_files:
                bounding_box_data = self.compute_stl_bounding_box(stl_files)
            
            # Load diameter from STL if not provided
            if diameter is None:
                wall_stl = geometry_dir / "wall_aorta.stl"
                if wall_stl.exists():
                    diameter = self.extract_minimum_diameter_from_stl(wall_stl)
                else:
                    diameter = 25e-3  # 25mm default
                    
            # Load flow data if not provided
            if peak_velocity is None or frequency is None:
                flow_files = list(geometry_dir.glob("*.csv"))
                if flow_files:
                    flow_data = self.load_flow_data(flow_files[0])
                    if peak_velocity is None:
                        peak_velocity = flow_data["peak_velocity"]
                    if frequency is None:
                        frequency = flow_data["heart_rate"]
        
        # Use defaults if still missing
        if diameter is None:
            diameter = 25e-3  # 25mm typical aortic diameter
        if peak_velocity is None:
            peak_velocity = 1.0  # 1.0 m/s typical
        if frequency is None:
            frequency = 75/60  # 75 BPM
        
        if self.flow_model == "LAMINAR":
            params = self._get_laminar_parameters(diameter, peak_velocity, frequency)
        elif self.flow_model == "RANS":
            params = self._get_rans_parameters(diameter, peak_velocity, frequency)
        elif self.flow_model == "LES":
            params = self._get_les_parameters(diameter, peak_velocity, frequency)
        else:
            raise ValueError(f"Unknown flow model: {self.flow_model}")
            
        # Add bounding box data if available
        if bounding_box_data is not None:
            params["bounding_box"] = bounding_box_data
            
        return params
    
    def _get_laminar_parameters(self, diameter: float, peak_velocity: float, frequency: float = None) -> Dict:
        """Laminar flow mesh parameters"""
        
        # Calculate first layer for y+ ≈ 1 with pulsatile effects
        layer_calc = self.calculate_first_layer_thickness(peak_velocity, diameter, 1.0, frequency)
        
        # Use slightly thicker first layer for laminar (y+ < 1 is fine)
        first_layer = max(layer_calc["first_layer_thickness"], 70e-6)  # At least 70 µm
        
        return {
            "flow_model": "LAMINAR",
            "target_yplus": 1.0,
            "first_layer_thickness": first_layer,
            "first_layer_microns": first_layer * 1e6,
            "n_surface_layers": 10,
            "expansion_ratio": 1.25,
            "base_cell_target": diameter / 45,  # D/40-D/50
            "diameter_resolution_factor": 45,  # For Stage 2 consistency
            "surface_refinement_level": [2, 3],
            "target_cells_range": [2e6, 5e6],
            "distance_refinement": {
                "near_distance": max(1.5e-3, first_layer * 20),  # At least 20x first layer
                "far_distance": max(3.0e-3, diameter * 0.15),   # 15% of diameter
                "transition_distance": max(6.0e-3, diameter * 0.3)
            },
            "minimum_diameter": diameter,
            "peak_velocity": peak_velocity,
            "characteristic_length": diameter * 3.76,  # ~4D length for aortic arch
            "reynolds": layer_calc["reynolds"],
            "flow_regime": layer_calc["flow_regime"],
            "friction_coefficient": layer_calc["friction_coefficient"],
            "womersley": layer_calc.get("womersley"),
            "pulsatile_correction": layer_calc.get("pulsatile_correction", False),
            "calculated_h1_microns": layer_calc["first_layer_microns"]
        }
    
    def _get_rans_parameters(self, diameter: float, peak_velocity: float, frequency: float = None) -> Dict:
        """RANS flow mesh parameters"""
        
        # Calculate first layer for y+ ≈ 1 with pulsatile effects
        layer_calc = self.calculate_first_layer_thickness(peak_velocity, diameter, 1.0, frequency)
        
        # Use calculated first layer or minimum 50 µm
        first_layer = max(layer_calc["first_layer_thickness"], 50e-6)
        
        return {
            "flow_model": "RANS", 
            "target_yplus": 1.0,
            "first_layer_thickness": first_layer,
            "first_layer_microns": first_layer * 1e6,
            "n_surface_layers": 12,
            "expansion_ratio": 1.20,
            "base_cell_target": diameter / 55,  # D/50-D/60
            "diameter_resolution_factor": 55,  # For Stage 2 consistency
            "surface_refinement_level": [3, 4],
            "target_cells_range": [5e6, 10e6],
            "distance_refinement": {
                "near_distance": max(1.0e-3, first_layer * 15),  # At least 15x first layer
                "far_distance": max(2.5e-3, diameter * 0.10),   # 10% of diameter
                "transition_distance": max(5.0e-3, diameter * 0.25)
            },
            "minimum_diameter": diameter,
            "peak_velocity": peak_velocity,
            "characteristic_length": diameter * 3.76,  # ~4D length for aortic arch
            "reynolds": layer_calc["reynolds"],
            "flow_regime": layer_calc["flow_regime"],
            "friction_coefficient": layer_calc["friction_coefficient"],
            "womersley": layer_calc.get("womersley"),
            "pulsatile_correction": layer_calc.get("pulsatile_correction", False),
            "calculated_h1_microns": layer_calc["first_layer_microns"]
        }
    
    def _get_les_parameters(self, diameter: float, peak_velocity: float, frequency: float = None) -> Dict:
        """Wall-resolved LES mesh parameters"""
        
        # Calculate first layer for y+ ≈ 1-2 with pulsatile effects
        layer_calc = self.calculate_first_layer_thickness(peak_velocity, diameter, 1.5, frequency)
        
        # Use thinner first layer for LES wall resolution
        first_layer = min(layer_calc["first_layer_thickness"], 40e-6)  # Max 40 µm
        
        return {
            "flow_model": "LES",
            "target_yplus": 1.5,
            "first_layer_thickness": first_layer, 
            "n_surface_layers": 20,
            "expansion_ratio": 1.15,
            "base_cell_target": diameter / 100,  # D/80-D/120
            "surface_refinement_level": [4, 5],
            "target_cells_range": [20e6, 60e6],
            "distance_refinement": {
                "near_distance": 0.8e-3,  # 0.8 mm - very tight for LES
                "far_distance": 2.0e-3,   # 2.0 mm
                "transition_distance": 4.0e-3
            },
            "on_wall_spacing_target": 2e-3,  # 2 mm for Δx+ ≈ 40-80
            "reynolds": layer_calc["reynolds"],
            "calculated_h1_microns": layer_calc["first_layer_microns"]
        }
    
    def validate_mesh_for_physics(self, mesh_params: Dict, flow_conditions: Dict) -> Dict:
        """
        Validate mesh parameters against physics requirements
        
        Args:
            mesh_params: Generated mesh parameters 
            flow_conditions: Flow conditions (velocity, diameter, etc.)
            
        Returns:
            Validation results with warnings and recommendations
        """
        
        diameter = flow_conditions["diameter"]
        velocity = flow_conditions["peak_velocity"]
        
        validation = {
            "valid": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Check Reynolds number regime
        Re = velocity * diameter / self.blood_properties["kinematic_viscosity"]
        
        if self.flow_model == "LAMINAR" and Re > 2300:
            validation["warnings"].append(f"Re={Re:.0f} may be transitional/turbulent for laminar model")
        elif self.flow_model == "RANS" and Re < 2000:
            validation["warnings"].append(f"Re={Re:.0f} may be laminar - consider LAMINAR model")
        
        # Check first layer thickness
        h1_microns = mesh_params["first_layer_thickness"] * 1e6
        
        if self.flow_model == "LAMINAR" and h1_microns > 90:
            validation["warnings"].append(f"First layer {h1_microns:.1f}µm may be too thick for laminar WSS")
        elif self.flow_model == "LES" and h1_microns > 50:
            validation["warnings"].append(f"First layer {h1_microns:.1f}µm may be too thick for LES y+≈1")
        
        # Check surface refinement vs on-wall spacing
        if self.flow_model == "LES":
            # Estimate on-wall spacing from surface refinement
            base_size = mesh_params["base_cell_target"] 
            surf_level = mesh_params["surface_refinement_level"][1]
            on_wall_spacing = base_size / (2 ** surf_level)
            
            if on_wall_spacing > 3e-3:  # 3mm
                validation["recommendations"].append(f"On-wall spacing {on_wall_spacing*1000:.1f}mm may be too coarse for LES")
        
        return validation
    
    def generate_physics_aware_config(self, geometry_params: Dict, 
                                    output_file: Path = None) -> Dict:
        """
        Generate complete physics-aware configuration
        
        Args:
            geometry_params: Geometry parameters (diameter, velocity, etc.)
            output_file: Optional output file path
            
        Returns:
            Complete configuration dictionary
        """
        
        diameter = geometry_params["diameter"]
        velocity = geometry_params["peak_velocity"]
        
        # Get flow regime parameters
        mesh_params = self.get_flow_regime_parameters(diameter, velocity)
        
        # Validate parameters
        validation = self.validate_mesh_for_physics(mesh_params, geometry_params)
        
        # Create complete configuration
        config = {
            "description": f"Physics-aware mesh configuration for {self.flow_model}",
            "flow_model": self.flow_model,
            "geometry_params": geometry_params,
            "mesh_parameters": mesh_params,
            "validation": validation,
            "openfoam_env_path": "/opt/openfoam12/etc/bashrc"
        }
        
        # Save configuration if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config

# Convenience function for quick calculations
def calculate_aortic_first_layer(peak_velocity: float, diameter_mm: float, 
                                flow_model: str = "RANS") -> Dict:
    """
    Quick calculation of first layer thickness for aortic flow
    
    Args:
        peak_velocity: Peak velocity in m/s
        diameter_mm: Diameter in millimeters 
        flow_model: Flow model ('LAMINAR', 'RANS', 'LES')
        
    Returns:
        Dictionary with recommended parameters
    """
    
    generator = PhysicsAwareMeshGenerator(flow_model)
    diameter_m = diameter_mm * 1e-3
    
    return generator.get_flow_regime_parameters(diameter_m, peak_velocity)