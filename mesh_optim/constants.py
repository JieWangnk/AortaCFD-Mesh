"""
Essential constants for mesh_optim package.

Simplified constants focused on the core functionality.
"""

# OpenFOAM version constants
OPENFOAM_VERSION = "v12"
OPENFOAM_BASE_PATH = "/opt/openfoam12"

# STL processing constants
STL_NORMAL_TOLERANCE = 1e-12
STL_DEFAULT_NORMAL = [0.0, 0.0, 1.0]
STL_MM_TO_M_SCALE = 1e-3
STL_MIN_DIAMETER_SAFETY = 10e-3  # 10mm minimum
STL_DEFAULT_DIAMETER = 25e-3     # 25mm default

# Mesh generation defaults
DEFAULT_CELL_SIZE = 0.002  # 2mm
DEFAULT_REFINEMENT_LEVELS = [1, 1]
DEFAULT_FEATURE_ANGLE = 40
DEFAULT_MARGIN_FRACTION = 0.1

# Ray-casting defaults
DEFAULT_MAX_CANDIDATES = 50
DEFAULT_GRID_SIZE = 20
DEFAULT_RAY_DIRECTIONS = 3

# Mesh quality thresholds
MAX_NON_ORTHOGONALITY = 65
MAX_SKEWNESS = 4
MIN_VOLUME = 1e-13