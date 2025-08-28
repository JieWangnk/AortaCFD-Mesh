"""
Utility functions for AortaCFD-Mesh optimization.

This module contains common utility functions used throughout the mesh optimization
process, including type conversion, validation, and helper functions.
"""

import os
import subprocess
from typing import Any, Union


def safe_float(val: Any, default: float = 0.0) -> float:
    """
    Convert value to float with fallback default.
    
    Args:
        val: Value to convert (can be None, string, number, etc.)
        default: Default value to return if conversion fails
        
    Returns:
        float: Converted value or default
        
    Examples:
        >>> safe_float("1.5")
        1.5
        >>> safe_float(None, 0.0)
        0.0
        >>> safe_float("invalid", 1.0)
        1.0
    """
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    """
    Convert value to int with fallback default.
    
    Args:
        val: Value to convert (can be None, string, number, etc.)
        default: Default value to return if conversion fails
        
    Returns:
        int: Converted value or default
        
    Examples:
        >>> safe_int("5")
        5
        >>> safe_int(3.7)
        3
        >>> safe_int(None, 10)
        10
        >>> safe_int("invalid", 1)
        1
    """
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def safe_bool(val: Any, default: bool = False) -> bool:
    """
    Convert value to bool with intelligent parsing.
    
    Args:
        val: Value to convert (can be None, string, number, bool, etc.)
        default: Default value to return if val is None
        
    Returns:
        bool: Converted value or default
        
    Examples:
        >>> safe_bool(True)
        True
        >>> safe_bool("true")
        True
        >>> safe_bool("1")
        True
        >>> safe_bool("false")
        False
        >>> safe_bool(0)
        False
        >>> safe_bool(None, True)
        True
    """
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ('true', '1', 'yes', 'on')
    return bool(val)


def validate_positive_number(val: Union[int, float], param_name: str) -> Union[int, float]:
    """
    Validate that a number is positive.
    
    Args:
        val: Number to validate
        param_name: Parameter name for error messages
        
    Returns:
        The validated number
        
    Raises:
        ValueError: If the number is not positive
    """
    if val <= 0:
        raise ValueError(f"{param_name} must be positive, got {val}")
    return val


def clamp(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> Union[int, float]:
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clamped value between min_val and max_val
        
    Examples:
        >>> clamp(5, 1, 10)
        5
        >>> clamp(-2, 1, 10)
        1
        >>> clamp(15, 1, 10)
        10
    """
    return max(min_val, min(max_val, value))


def format_memory_size(bytes_count: int) -> str:
    """
    Format memory size in human-readable format.
    
    Args:
        bytes_count: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB", "256 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} PB"


def parse_openfoam_dict_value(value_str: str) -> Union[str, float, int, bool]:
    """
    Parse a value from OpenFOAM dictionary format.
    
    Args:
        value_str: String value from OpenFOAM dict
        
    Returns:
        Parsed value with appropriate type
        
    Examples:
        >>> parse_openfoam_dict_value("1.5")
        1.5
        >>> parse_openfoam_dict_value("true")
        True
        >>> parse_openfoam_dict_value("wall_patch")
        'wall_patch'
    """
    value_str = value_str.strip().rstrip(';')
    
    # Try boolean first
    if value_str.lower() in ('true', 'yes', 'on'):
        return True
    if value_str.lower() in ('false', 'no', 'off'):
        return False
    
    # Try integer
    try:
        if '.' not in value_str and 'e' not in value_str.lower():
            return int(value_str)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Return as string
    return value_str


def detect_openfoam_version() -> int:
    """
    Detect OpenFOAM major version number.
    
    Returns:
        int: OpenFOAM major version number, or default if detection fails
        
    Examples:
        >>> detect_openfoam_version()  # With OpenFOAM-12 installed
        12
        >>> detect_openfoam_version()  # With OpenFOAM-11 installed  
        11
    """
    from .constants import (
        DEFAULT_OPENFOAM_MAJOR_VERSION, 
        OPENFOAM_VERSION_COMMAND,
        OPENFOAM_ENV_VARS
    )
    
    # Try environment variables first
    for env_var in OPENFOAM_ENV_VARS:
        version_str = os.environ.get(env_var, "")
        if version_str:
            try:
                # Extract major version number (e.g., "12" from "12.0" or "OpenFOAM-12")
                version_parts = version_str.replace("OpenFOAM-", "").split(".")
                return int(version_parts[0])
            except (ValueError, IndexError):
                continue
    
    # Try foamVersion command
    try:
        result = subprocess.run(
            OPENFOAM_VERSION_COMMAND, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            # Parse output like "OpenFOAM-12" or "12.0"
            if "OpenFOAM-" in output:
                version_str = output.split("OpenFOAM-")[1].split()[0]
            else:
                version_str = output.split()[0] if output.split() else ""
            
            try:
                return int(version_str.split(".")[0])
            except (ValueError, IndexError):
                pass
                
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass
    
    # Fallback to default version
    return DEFAULT_OPENFOAM_MAJOR_VERSION


def get_openfoam_geometry_dir() -> str:
    """
    Get the correct OpenFOAM geometry directory name based on version.
    
    OpenFOAM v12+ uses "geometry", earlier versions use "triSurface".
    
    Returns:
        str: Directory name ("geometry" or "triSurface")
        
    Examples:
        >>> get_openfoam_geometry_dir()  # With OpenFOAM-12
        'geometry'
        >>> get_openfoam_geometry_dir()  # With OpenFOAM-11
        'triSurface'
    """
    from .constants import (
        OPENFOAM_GEOMETRY_DIR_V11,
        OPENFOAM_GEOMETRY_DIR_V12, 
        OPENFOAM_VERSION_THRESHOLD_GEOMETRY
    )
    
    version = detect_openfoam_version()
    
    if version >= OPENFOAM_VERSION_THRESHOLD_GEOMETRY:
        return OPENFOAM_GEOMETRY_DIR_V12  # "geometry"
    else:
        return OPENFOAM_GEOMETRY_DIR_V11  # "triSurface"


def get_openfoam_geometry_path(base_path: str) -> str:
    """
    Get the full path to OpenFOAM geometry directory.
    
    Args:
        base_path: Base directory path (typically "constant")
        
    Returns:
        str: Full path to geometry directory
        
    Examples:
        >>> get_openfoam_geometry_path("/case/constant")
        '/case/constant/geometry'  # For OpenFOAM-12+
        >>> get_openfoam_geometry_path("/case/constant") 
        '/case/constant/triSurface'  # For OpenFOAM-11-
    """
    import os
    geometry_dir = get_openfoam_geometry_dir()
    return os.path.join(base_path, geometry_dir)