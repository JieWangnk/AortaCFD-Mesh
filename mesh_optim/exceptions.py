"""
Exception hierarchy for AortaCFD-Mesh optimization.

This module defines a comprehensive exception hierarchy for mesh optimization
errors, providing clear error handling and debugging capabilities.
"""


class MeshOptimizationError(Exception):
    """
    Base exception for all mesh optimization failures.
    
    This is the root exception that all mesh optimization related errors
    inherit from. It provides a consistent interface for error handling
    throughout the optimization pipeline.
    """
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        """
        Initialize mesh optimization error.
        
        Args:
            message: Human-readable error message
            error_code: Optional error code for programmatic handling
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.error_code = error_code or "MESH_ERROR"
        self.details = details or {}
        self.message = message
    
    def __str__(self):
        if self.details:
            return f"{self.message} (Code: {self.error_code}, Details: {self.details})"
        return f"{self.message} (Code: {self.error_code})"


class GeometryError(MeshOptimizationError):
    """
    Exception for geometry-related errors.
    
    Raised when STL geometry processing fails, including surface quality
    issues, scaling problems, or file format errors.
    """
    def __init__(self, message: str, geometry_file: str = None, **kwargs):
        super().__init__(message, error_code="GEOMETRY_ERROR", **kwargs)
        self.geometry_file = geometry_file
        if geometry_file:
            self.details["geometry_file"] = geometry_file


class SurfaceQualityError(GeometryError):
    """
    Exception for surface quality validation failures.
    
    Raised when STL surfaces have quality issues that prevent successful
    mesh generation, such as open surfaces or poor triangle quality.
    """
    def __init__(self, message: str, quality_issues: list = None, **kwargs):
        super().__init__(message, error_code="SURFACE_QUALITY_ERROR", **kwargs)
        self.quality_issues = quality_issues or []
        if quality_issues:
            self.details["quality_issues"] = quality_issues


class MeshGenerationError(MeshOptimizationError):
    """
    Exception for mesh generation failures.
    
    Raised when OpenFOAM mesh generation fails, including blockMesh,
    snappyHexMesh, or other mesh generation tool failures.
    """
    def __init__(self, message: str, generation_stage: str = None, openfoam_error: str = None, **kwargs):
        super().__init__(message, error_code="MESH_GENERATION_ERROR", **kwargs)
        self.generation_stage = generation_stage
        self.openfoam_error = openfoam_error
        if generation_stage:
            self.details["generation_stage"] = generation_stage
        if openfoam_error:
            self.details["openfoam_error"] = openfoam_error


class LayerGenerationError(MeshGenerationError):
    """
    Exception for boundary layer generation failures.
    
    Raised when snappyHexMesh layer addition fails, including insufficient
    coverage, layer collapse, or quality constraint violations.
    """
    def __init__(self, message: str, coverage_achieved: float = None, target_coverage: float = None, **kwargs):
        super().__init__(message, generation_stage="layer_addition", 
                        error_code="LAYER_GENERATION_ERROR", **kwargs)
        self.coverage_achieved = coverage_achieved
        self.target_coverage = target_coverage
        if coverage_achieved is not None:
            self.details["coverage_achieved"] = coverage_achieved
        if target_coverage is not None:
            self.details["target_coverage"] = target_coverage


class QualityConstraintError(MeshOptimizationError):
    """
    Exception for mesh quality constraint violations.
    
    Raised when generated mesh fails to meet quality requirements such as
    maximum non-orthogonality or skewness limits.
    """
    def __init__(self, message: str, quality_metrics: dict = None, constraints: dict = None, **kwargs):
        super().__init__(message, error_code="QUALITY_CONSTRAINT_ERROR", **kwargs)
        self.quality_metrics = quality_metrics or {}
        self.constraints = constraints or {}
        self.details.update({
            "quality_metrics": self.quality_metrics,
            "constraints": self.constraints
        })


class ConfigurationError(MeshOptimizationError):
    """
    Exception for configuration file and parameter errors.
    
    Raised when configuration files are invalid, missing required parameters,
    or contain incompatible parameter combinations.
    """
    def __init__(self, message: str, config_file: str = None, invalid_parameters: list = None, **kwargs):
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        self.config_file = config_file
        self.invalid_parameters = invalid_parameters or []
        if config_file:
            self.details["config_file"] = config_file
        if invalid_parameters:
            self.details["invalid_parameters"] = invalid_parameters


class ResourceError(MeshOptimizationError):
    """
    Exception for resource-related errors.
    
    Raised when system resources (memory, disk space, CPU) are insufficient
    for mesh generation or when resource limits are exceeded.
    """
    def __init__(self, message: str, resource_type: str = None, required_amount: float = None, 
                 available_amount: float = None, **kwargs):
        super().__init__(message, error_code="RESOURCE_ERROR", **kwargs)
        self.resource_type = resource_type
        self.required_amount = required_amount
        self.available_amount = available_amount
        if resource_type:
            self.details["resource_type"] = resource_type
        if required_amount is not None:
            self.details["required_amount"] = required_amount
        if available_amount is not None:
            self.details["available_amount"] = available_amount


class MemoryError(ResourceError):
    """
    Exception for memory-related errors.
    
    Raised when mesh generation requires more memory than available or
    when memory usage exceeds configured limits.
    """
    def __init__(self, message: str, required_memory_gb: float = None, available_memory_gb: float = None, **kwargs):
        super().__init__(message, resource_type="memory", 
                        required_amount=required_memory_gb, available_amount=available_memory_gb,
                        error_code="MEMORY_ERROR", **kwargs)


class ConvergenceError(MeshOptimizationError):
    """
    Exception for optimization convergence failures.
    
    Raised when the optimization process fails to converge within the
    specified number of iterations or when optimization gets stuck.
    """
    def __init__(self, message: str, iterations_completed: int = None, max_iterations: int = None, 
                 stagnation_threshold: int = None, **kwargs):
        super().__init__(message, error_code="CONVERGENCE_ERROR", **kwargs)
        self.iterations_completed = iterations_completed
        self.max_iterations = max_iterations
        self.stagnation_threshold = stagnation_threshold
        if iterations_completed is not None:
            self.details["iterations_completed"] = iterations_completed
        if max_iterations is not None:
            self.details["max_iterations"] = max_iterations
        if stagnation_threshold is not None:
            self.details["stagnation_threshold"] = stagnation_threshold


class OpenFOAMError(MeshOptimizationError):
    """
    Exception for OpenFOAM execution errors.
    
    Raised when OpenFOAM commands fail to execute, including missing
    installation, environment setup issues, or command-specific failures.
    """
    def __init__(self, message: str, command: str = None, return_code: int = None, 
                 stdout: str = None, stderr: str = None, **kwargs):
        super().__init__(message, error_code="OPENFOAM_ERROR", **kwargs)
        self.command = command
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        if command:
            self.details["command"] = command
        if return_code is not None:
            self.details["return_code"] = return_code
        if stdout:
            self.details["stdout"] = stdout[:1000] + "..." if len(stdout) > 1000 else stdout
        if stderr:
            self.details["stderr"] = stderr[:1000] + "..." if len(stderr) > 1000 else stderr


# Convenience functions for common error scenarios

def raise_geometry_error(message: str, geometry_file: str = None, **kwargs):
    """Raise a GeometryError with standardized formatting."""
    raise GeometryError(message, geometry_file=geometry_file, **kwargs)


def raise_surface_quality_error(message: str, issues: list = None, geometry_file: str = None, **kwargs):
    """Raise a SurfaceQualityError with issue details."""
    raise SurfaceQualityError(message, quality_issues=issues, geometry_file=geometry_file, **kwargs)


def raise_layer_generation_error(message: str, coverage: float = None, target: float = None, **kwargs):
    """Raise a LayerGenerationError with coverage information."""
    raise LayerGenerationError(message, coverage_achieved=coverage, target_coverage=target, **kwargs)


def raise_quality_constraint_error(message: str, metrics: dict = None, constraints: dict = None, **kwargs):
    """Raise a QualityConstraintError with quality metrics."""
    raise QualityConstraintError(message, quality_metrics=metrics, constraints=constraints, **kwargs)


def raise_memory_error(message: str, required_gb: float = None, available_gb: float = None, **kwargs):
    """Raise a MemoryError with memory usage details."""
    raise MemoryError(message, required_memory_gb=required_gb, available_memory_gb=available_gb, **kwargs)


def raise_openfoam_error(message: str, command: str = None, return_code: int = None, 
                        stdout: str = None, stderr: str = None, **kwargs):
    """Raise an OpenFOAMError with command execution details."""
    raise OpenFOAMError(message, command=command, return_code=return_code, 
                       stdout=stdout, stderr=stderr, **kwargs)