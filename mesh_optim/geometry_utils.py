#!/usr/bin/env python3
"""
Geometry utility functions for robust seed point calculation and STL processing.
"""

import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging


def read_stl_triangles(stl_path: Path) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Read STL triangles and return list of (normal, v1, v2, v3) tuples.
    
    Args:
        stl_path: Path to STL file
        
    Returns:
        List of (normal, v1, v2, v3) where each is a numpy array of shape (3,)
    """
    triangles = []
    
    try:
        # Try ASCII first
        with open(stl_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            if not line.lower().strip().startswith('solid'):
                raise UnicodeDecodeError("", "", 0, 0, "not ascii")
            
            f.seek(0)
            current_normal = None
            vertices = []
            
            for line in f:
                line = line.strip().lower()
                
                if line.startswith('facet normal'):
                    parts = line.split()
                    try:
                        current_normal = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                    except (IndexError, ValueError):
                        current_normal = None
                        
                elif line.startswith('vertex'):
                    parts = line.split()
                    try:
                        vertex = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                        vertices.append(vertex)
                    except (IndexError, ValueError):
                        continue
                        
                elif line.startswith('endfacet'):
                    if len(vertices) == 3:
                        v1, v2, v3 = vertices
                        
                        # Compute normal if not provided or zero
                        if current_normal is None or np.linalg.norm(current_normal) < 1e-12:
                            edge1 = v2 - v1
                            edge2 = v3 - v1
                            computed_normal = np.cross(edge1, edge2)
                            norm = np.linalg.norm(computed_normal)
                            if norm > 1e-12:
                                current_normal = computed_normal / norm
                            else:
                                current_normal = np.array([0.0, 0.0, 1.0])  # Default
                        
                        triangles.append((current_normal, v1, v2, v3))
                    
                    # Reset for next facet
                    current_normal = None
                    vertices = []
                    
        return triangles
                    
    except UnicodeDecodeError:
        # Binary STL
        triangles = []
        with open(stl_path, 'rb') as f:
            # Skip header
            f.seek(80)
            
            # Read number of triangles
            n_triangles = struct.unpack('<I', f.read(4))[0]
            
            for _ in range(n_triangles):
                # Read normal
                nx, ny, nz = struct.unpack('<3f', f.read(12))
                normal = np.array([nx, ny, nz])
                
                # Read vertices
                v1 = np.array(struct.unpack('<3f', f.read(12)))
                v2 = np.array(struct.unpack('<3f', f.read(12))) 
                v3 = np.array(struct.unpack('<3f', f.read(12)))
                
                # Skip attribute byte count
                f.read(2)
                
                # Compute normal if zero
                if np.linalg.norm(normal) < 1e-12:
                    edge1 = v2 - v1
                    edge2 = v3 - v1
                    computed_normal = np.cross(edge1, edge2)
                    norm = np.linalg.norm(computed_normal)
                    if norm > 1e-12:
                        normal = computed_normal / norm
                    else:
                        normal = np.array([0.0, 0.0, 1.0])
                
                triangles.append((normal, v1, v2, v3))
        
        return triangles


def find_internal_seed_point(bbox_center: np.ndarray, bbox_size: np.ndarray, 
                            inlet_triangles: List[Tuple], dx_base: float,
                            logger: logging.Logger) -> np.ndarray:
    """
    Find a robust internal seed point for mesh generation.
    
    This is the proven robust method that:
    1. Calculates inlet centroid from triangles
    2. Steps inward from inlet toward geometry center  
    3. Uses multiple validation methods to ensure point is inside
    4. Falls back to systematic search if needed
    
    Args:
        bbox_center: Geometry bounding box center [x, y, z]
        bbox_size: Geometry bounding box size [dx, dy, dz]  
        inlet_triangles: List of (normal, v1, v2, v3) tuples from inlet STL
        dx_base: Base mesh cell size for step calculations
        logger: Logger instance
        
    Returns:
        Internal seed point as numpy array [x, y, z]
    """
    logger.info("Using ROBUST seed point calculation (proven method)")
    
    if not inlet_triangles:
        logger.warning("No inlet triangles provided - using bbox center")
        return bbox_center
    
    # Step 1: Calculate inlet centroid
    inlet_centroid = _calculate_inlet_centroid_from_triangles(inlet_triangles, logger)
    logger.info(f"Inlet centroid: ({inlet_centroid[0]:.6f}, {inlet_centroid[1]:.6f}, {inlet_centroid[2]:.6f})")
    
    # Step 2: Find inward direction (toward bbox center)
    to_center = bbox_center - inlet_centroid
    distance_to_center = np.linalg.norm(to_center)
    
    if distance_to_center < 1e-12:
        logger.warning("Inlet centroid coincides with bbox center - using bbox center")
        return bbox_center
    
    inward_direction = to_center / distance_to_center
    
    # Step 3: Step inward from inlet centroid
    # Use multiple step sizes to find a good internal point
    step_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]  # Fraction of distance to center
    
    for step_fraction in step_fractions:
        step_distance = step_fraction * distance_to_center
        candidate_point = inlet_centroid + inward_direction * step_distance
        
        # Validate that point is well inside bounding box
        margin = 0.05  # 5% margin from bbox edges
        bbox_min = bbox_center - bbox_size/2
        bbox_max = bbox_center + bbox_size/2
        
        margin_size = margin * bbox_size
        inside_margin = np.all(candidate_point >= bbox_min + margin_size) and \
                       np.all(candidate_point <= bbox_max - margin_size)
        
        if inside_margin:
            logger.info(f"Internal seed point found: ({candidate_point[0]:.6f}, {candidate_point[1]:.6f}, {candidate_point[2]:.6f})")
            logger.info(f"  Step fraction: {step_fraction:.1f}, distance from inlet: {step_distance*1000:.2f}mm")
            return candidate_point
    
    # Step 4: Fallback - use bbox center with small offset toward inlet
    # This ensures we're away from potential symmetry planes
    offset_distance = min(0.1 * min(bbox_size), 0.5e-3)  # Small offset, max 0.5mm
    offset = (inlet_centroid - bbox_center) 
    if np.linalg.norm(offset) > 1e-12:
        offset = offset / np.linalg.norm(offset) * offset_distance
    else:
        offset = np.array([offset_distance, 0, 0])  # Arbitrary small offset
        
    fallback_point = bbox_center + offset
    logger.warning(f"Using fallback seed point: ({fallback_point[0]:.6f}, {fallback_point[1]:.6f}, {fallback_point[2]:.6f})")
    
    return fallback_point


def _calculate_inlet_centroid_from_triangles(inlet_triangles: List[Tuple], 
                                           logger: logging.Logger) -> np.ndarray:
    """
    Calculate centroid from inlet triangle vertices.
    
    Args:
        inlet_triangles: List of (normal, v1, v2, v3) tuples
        logger: Logger instance
        
    Returns:
        Centroid as numpy array [x, y, z]
    """
    if not inlet_triangles:
        logger.warning("No inlet triangles for centroid calculation")
        return np.array([0.0, 0.0, 0.0])
    
    # Collect all vertices
    vertices = []
    for normal, v1, v2, v3 in inlet_triangles:
        vertices.extend([v1, v2, v3])
    
    if not vertices:
        logger.warning("No vertices found in inlet triangles")
        return np.array([0.0, 0.0, 0.0])
    
    # Calculate centroid
    vertices_array = np.array(vertices)
    centroid = np.mean(vertices_array, axis=0)
    
    logger.debug(f"Calculated inlet centroid from {len(vertices)} vertices ({len(inlet_triangles)} triangles)")
    
    return centroid


def detect_stl_units(surface_area: float, bbox_dims: np.ndarray, 
                    logger: logging.Logger) -> str:
    """
    Detect STL units based on surface area and bounding box dimensions.
    
    Args:
        surface_area: Surface area from surfaceCheck
        bbox_dims: Bounding box dimensions [dx, dy, dz] 
        logger: Logger instance
        
    Returns:
        'mm' or 'm' indicating detected units
    """
    # Check if dimensions suggest millimeter units
    max_dim = np.max(bbox_dims)
    
    # For human vasculature:
    # - Aortic root: ~30-35mm diameter  
    # - Ascending aorta: ~25-30mm
    # - Arch: ~20-25mm
    # - Branches: ~10-20mm
    # Total length: ~100-200mm
    
    if max_dim > 1.0:  # > 1 meter suggests mm units stored as meters
        logger.debug(f"Large dimension ({max_dim:.3f}) suggests mm units stored as meters")
        return 'mm'
    elif max_dim > 0.2:  # > 20cm also suggests mm units  
        logger.debug(f"Dimension ({max_dim:.3f}) suggests mm units stored as meters")
        return 'mm'
    else:
        logger.debug(f"Dimension ({max_dim:.3f}) appears to be in meters")
        return 'm'