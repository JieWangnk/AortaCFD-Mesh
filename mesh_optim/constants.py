"""Constants and default values for mesh optimization.

This module centralizes all magic numbers and default parameters used throughout
the mesh optimization pipeline to improve maintainability and configurability.
"""

# ============================================================================
# GEOMETRY CONSTANTS
# ============================================================================

# Feature angle detection ranges (degrees)
FEATURE_ANGLE_DEFAULT = 35.0
FEATURE_ANGLE_MIN = 25.0
FEATURE_ANGLE_MAX = 60.0
FEATURE_ANGLE_RECOMMENDED = 35.0  # Conservative but effective for vascular geometry

# Vessel sizing thresholds (meters)
FIRST_LAYER_MIN = 10e-6  # 10 microns minimum
FIRST_LAYER_MAX = 200e-6  # 200 microns maximum  
FIRST_LAYER_DEFAULT = 50e-6  # 50 microns - good for most vascular flows
FIRST_LAYER_REDUCTION_FACTOR = 0.7  # Reduce by 30% when adapting

# Minimum thickness constraints (meters)
MIN_THICKNESS_ABSOLUTE = 1.0e-6  # 1 micron absolute minimum
MIN_THICKNESS_FRACTION = 0.15  # 15% of first layer thickness

# ============================================================================
# MESH QUALITY CONSTRAINTS
# ============================================================================

# Quality metrics thresholds
MAX_NON_ORTHO_LES = 60.0  # Degrees - LES requirement
MAX_NON_ORTHO_RANS = 65.0  # Degrees - RANS requirement
MAX_NON_ORTHO_LAMINAR = 65.0  # Degrees - Laminar requirement

MAX_SKEWNESS_LES = 3.5  # LES requirement
MAX_SKEWNESS_RANS = 4.0  # RANS requirement
MAX_SKEWNESS_LAMINAR = 4.0  # Laminar requirement

MIN_LAYER_COVERAGE_LES = 0.80  # 80% for WSS accuracy in LES
MIN_LAYER_COVERAGE_RANS = 0.70  # 70% for RANS
MIN_LAYER_COVERAGE_LAMINAR = 0.65  # 65% for laminar

# Acceptance criteria defaults
DEFAULT_MAX_NON_ORTHO = 65.0
DEFAULT_MAX_SKEWNESS = 4.0
DEFAULT_MIN_LAYER_COVERAGE = 0.65

# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

# Iteration control
MAX_ITERATIONS_DEFAULT = 4
MICRO_TRIALS_PER_ITERATION = 3  # Number of layer parameter trials

# Coverage convergence criteria
COVERAGE_PLATEAU_THRESHOLD = 0.005  # 0.5% change threshold
COVERAGE_PLATEAU_ITERATIONS = 3  # Must plateau for 3 iterations
COVERAGE_ACCEPTABLE_THRESHOLD = 0.70  # 70% coverage is acceptable

# Feature snap iterations based on angle
FEATURE_SNAP_ITER_LOW_ANGLE = 30  # For angles <= 35°
FEATURE_SNAP_ITER_HIGH_ANGLE = 20  # For angles >= 55°
FEATURE_ANGLE_LOW_THRESHOLD = 35.0
FEATURE_ANGLE_HIGH_THRESHOLD = 55.0

# ============================================================================
# LAYER GENERATION PARAMETERS
# ============================================================================

# Expansion ratio constraints
EXPANSION_RATIO_DEFAULT = 1.2
EXPANSION_RATIO_MIN = 1.12
EXPANSION_RATIO_REDUCTION = 0.02  # Reduce by this amount when adapting

# Layer control
N_SURFACE_LAYERS_DEFAULT = 10
N_GROW_DEFAULT = 0
FEATURE_ANGLE_LAYERS = 75.0  # Degrees - for layer growth control
SLIP_FEATURE_ANGLE = 45.0  # Degrees - for sharp corners

# Relative sizing thresholds
RELATIVE_SIZING_COVERAGE_THRESHOLD = 0.40  # Switch to relative when coverage < 40%
RELATIVE_FIRST_LAYER_DEFAULT = 0.2
RELATIVE_FIRST_LAYER_MIN = 0.15
RELATIVE_FIRST_LAYER_REDUCTION = 0.9  # Multiply by this when adapting

# ============================================================================
# MEMORY AND RESOURCE MANAGEMENT
# ============================================================================

# Memory allocation
MEMORY_SAFETY_FACTOR = 0.7  # Use 70% of available memory
MAX_MEMORY_CAP_GB = 12.0  # Cap at 12GB regardless of available
KB_PER_CELL_DEFAULT = 1.0  # Default memory per cell estimate
KB_PER_CELL_MIN = 0.5  # Minimum for safety

# Cell count limits
MIN_CELLS_PER_PROCESSOR = 100_000
MAX_CELLS_PER_PROCESSOR = 5_000_000
MIN_GLOBAL_CELLS = 500_000
MAX_GLOBAL_CELLS = 20_000_000

# ============================================================================
# STL PROCESSING
# ============================================================================

# Unit detection heuristics
STL_UNIT_DETECTION_FACTOR = 1000.0  # If area > 1000× plausible, assume mm²
STL_MM_TO_M_CONVERSION = 1e-6  # Convert mm² to m²

# Bounding box defaults (meters)
BBOX_DEFAULT_SIZE = 0.02  # 20mm default if can't determine

# ============================================================================
# PHYSICS PARAMETERS
# ============================================================================

# Flow parameters defaults
U_PEAK_DEFAULT = 1.0  # m/s
BLOOD_DENSITY = 1060.0  # kg/m³
BLOOD_VISCOSITY = 0.0035  # Pa·s
HEART_RATE_DEFAULT = 1.2  # Hz

# y+ targets for different models
Y_PLUS_WALL_FUNCTION = 30.0  # For wall functions
Y_PLUS_RESOLVED = 1.0  # For near-wall resolution

# ============================================================================
# GEOMETRY ANALYSIS
# ============================================================================

# Curvature analysis
CURVATURE_SAMPLE_MAX = 200_000  # Maximum triangles to sample
CURVATURE_PERCENTILE = 95  # Use 95th percentile for robustness

# Ray casting parameters
RAY_CAST_EPSILON = 1e-10  # Epsilon for ray-triangle intersection
RAY_DIRECTION_DEFAULT = [0.57735, 0.70711, 0.40825]  # Normalized diagonal

# Seed point search
SEED_POINT_MAX_ATTEMPTS = 100  # Maximum attempts to find internal point
SEED_POINT_GRID_DIVISIONS = 10  # Grid resolution for seed point search

# ============================================================================
# FILE PROCESSING
# ============================================================================

# Supported wall patch names (in priority order)
WALL_PATCH_NAMES = ["wall_aorta", "wall", "vessel_wall", "arterial_wall"]
WALL_PATCH_DEFAULT = "wall_aorta"

# Required STL files
REQUIRED_STL_FILES = ["inlet.stl"]

# ============================================================================
# SOLVER PRESETS
# ============================================================================

SOLVER_MODES = {
    "LES": {
        "max_nonortho": MAX_NON_ORTHO_LES,
        "max_skewness": MAX_SKEWNESS_LES,
        "min_layer_coverage": MIN_LAYER_COVERAGE_LES
    },
    "RANS": {
        "max_nonortho": MAX_NON_ORTHO_RANS,
        "max_skewness": MAX_SKEWNESS_RANS,
        "min_layer_coverage": MIN_LAYER_COVERAGE_RANS
    },
    "LAMINAR": {
        "max_nonortho": MAX_NON_ORTHO_LAMINAR,
        "max_skewness": MAX_SKEWNESS_LAMINAR,
        "min_layer_coverage": MIN_LAYER_COVERAGE_LAMINAR
    }
}