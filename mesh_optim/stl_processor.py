"""
Unified STL processing module for mesh_optim.

Clean implementation focused only on the essential functionality needed
for ray-casting interior point detection.
"""

import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging


class STLProcessor:
    """
    Clean STL file processor that handles both ASCII and binary STL formats.
    
    Simplified version focused on ray-casting requirements.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def read_triangles(self, stl_path: Path) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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