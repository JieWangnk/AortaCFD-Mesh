"""
AortaCFD Stage 1 Mesh Generation Package V2

Clean, simplified single-pass mesh generation for vascular CFD:
- Ray-casting interior point detection
- Single-pass snappyHexMesh execution
- No optimization loops or complex orchestration
- JSON-based configuration
"""

from .stage1_mesh import Stage1MeshOptimizer

__version__ = "2.0.0-clean"
__all__ = ["Stage1MeshOptimizer"]