"""
AortaCFD Stage 1 Mesh Optimization Package

Automated geometry-driven mesh generation for vascular CFD:
- Physics-aware boundary layer generation
- Adaptive surface refinement
- Quality-constrained optimization
- OpenFOAM snappyHexMesh automation

Tutorial 1: Learn automated mesh generation for complex vascular geometries.
"""

from .stage1_mesh import Stage1MeshOptimizer
from .utils import run_command, check_mesh_quality, parse_layer_coverage

__version__ = "1.0.0-tutorial1"
__all__ = ["Stage1MeshOptimizer", "run_command", "check_mesh_quality", "parse_layer_coverage"]