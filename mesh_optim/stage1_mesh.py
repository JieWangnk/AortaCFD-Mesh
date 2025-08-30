"""
AortaCFD Stage 1 Mesh Generator V2 - Clean Implementation

Single-pass mesh generation with ray-casting interior point detection.
No optimization loops, no complex orchestration - just reliable snappyHexMesh automation.

This is a clean implementation that combines:
- SimpleMeshGenerator template approach
- Ray-casting interior point detection
- Single-pass execution
- Robust error handling
"""

import json
import logging
import numpy as np
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
import sys

# Import ray-casting functionality
from .point_in_mesh import find_interior_point_raycast


class Stage1MeshOptimizer:
    """
    AortaCFD Stage 1 mesh generator with ray-casting interior point detection.
    
    Key features:
    - Single-pass snappyHexMesh execution
    - Ray-casting for robust interior point calculation
    - JSON-based configuration
    - No optimization loops or quality checking
    - Clean, focused codebase
    """

    def __init__(self, geometry_dir: Union[str, Path], config_file: Union[str, Path], output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the Stage 1 mesh generator.
        
        Args:
            geometry_dir: Path to directory containing STL geometry files
            config_file: Path to JSON configuration file
            output_dir: Optional output directory (default: geometry_dir/../output/stage1_mesh_v2)
        """
        self.geometry_dir = Path(geometry_dir).resolve()
        self.config_file = Path(config_file).resolve()
        self.output_dir = Path(output_dir).resolve() if output_dir else (self.geometry_dir.parent / "output" / "stage1_mesh_v2")
        
        # Setup logging
        self.logger = logging.getLogger(f"Stage1MeshV2_{self.geometry_dir.name}")
        
        # Load configuration
        self._load_configuration()
        
        self.logger.info(f"🚀 Stage 1 mesh generator V2 initialized")
        self.logger.info(f"📁 Geometry: {self.geometry_dir}")
        self.logger.info(f"⚙️  Config: {self.config_file}")
        self.logger.info(f"📁 Output: {self.output_dir}")

    def _load_configuration(self) -> None:
        """Load and validate configuration file."""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
            self.logger.info(f"✅ Configuration loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {self.config_file}: {e}")

    def generate_mesh(self) -> Path:
        """
        Main mesh generation method - single pass with ray-casting.
        
        Returns:
            Path to the generated mesh case directory
        """
        self.logger.info("🔄 Starting single-pass mesh generation with ray-casting")
        
        try:
            # Step 1: Setup case directory
            case_dir = self._setup_case_directory()
            
            # Step 2: Copy and process geometry
            self._copy_stl_files(case_dir)
            
            # Step 3: Generate OpenFOAM dictionaries
            self._generate_openfoam_dictionaries(case_dir)
            
            # Step 4: Execute mesh generation
            self._run_mesh_generation(case_dir)
            
            self.logger.info(f"✅ Mesh generation completed successfully")
            self.logger.info(f"📁 Case directory: {case_dir}")
            
            return case_dir
            
        except Exception as e:
            self.logger.error(f"❌ Mesh generation failed: {e}")
            raise

    def _setup_case_directory(self) -> Path:
        """Setup OpenFOAM case directory structure."""
        case_name = f"{self.geometry_dir.name}_mesh_v2"
        case_dir = self.output_dir / case_name
        
        # Create directory structure
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "0").mkdir(exist_ok=True)
        (case_dir / "constant").mkdir(exist_ok=True)
        (case_dir / "constant" / "triSurface").mkdir(exist_ok=True)
        (case_dir / "system").mkdir(exist_ok=True)
        
        self.logger.info(f"📁 Created case directory: {case_dir}")
        return case_dir

    def _copy_stl_files(self, case_dir: Path) -> None:
        """Copy STL files to triSurface directory."""
        tri_surface_dir = case_dir / "constant" / "triSurface"
        
        stl_files = list(self.geometry_dir.glob("*.stl"))
        if not stl_files:
            raise ValueError(f"No STL files found in {self.geometry_dir}")
        
        for stl_file in stl_files:
            dest_file = tri_surface_dir / stl_file.name
            shutil.copy2(stl_file, dest_file)
            self.logger.debug(f"Copied: {stl_file.name}")
        
        self.logger.info(f"📋 Copied {len(stl_files)} STL files")

    def _generate_openfoam_dictionaries(self, case_dir: Path) -> None:
        """Generate all required OpenFOAM dictionary files."""
        self._generate_controldict(case_dir)
        self._generate_blockdict(case_dir) 
        self._generate_snappy_dict(case_dir)
        self.logger.info("📝 Generated OpenFOAM dictionaries")

    def _generate_controldict(self, case_dir: Path) -> None:
        """Generate controlDict file."""
        content = '''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v12                                   |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         100;
deltaT          1;
writeControl    timeStep;
writeInterval   20;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

// ************************************************************************* //'''
        
        (case_dir / "system" / "controlDict").write_text(content)

    def _generate_blockdict(self, case_dir: Path) -> None:
        """Generate blockMeshDict with automatic sizing."""
        # Calculate bounding box from STL files
        mesh_min, mesh_max, cell_size = self._calculate_bbox_and_cell_size(case_dir)
        
        # Generate block divisions
        domain_size = mesh_max - mesh_min
        divisions = np.maximum(np.round(domain_size / cell_size).astype(int), 1)
        
        content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v12                                   |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

vertices
(
    ({mesh_min[0]:.6f} {mesh_min[1]:.6f} {mesh_min[2]:.6f})
    ({mesh_max[0]:.6f} {mesh_min[1]:.6f} {mesh_min[2]:.6f})
    ({mesh_max[0]:.6f} {mesh_max[1]:.6f} {mesh_min[2]:.6f})
    ({mesh_min[0]:.6f} {mesh_max[1]:.6f} {mesh_min[2]:.6f})
    ({mesh_min[0]:.6f} {mesh_min[1]:.6f} {mesh_max[2]:.6f})
    ({mesh_max[0]:.6f} {mesh_min[1]:.6f} {mesh_max[2]:.6f})
    ({mesh_max[0]:.6f} {mesh_max[1]:.6f} {mesh_max[2]:.6f})
    ({mesh_min[0]:.6f} {mesh_max[1]:.6f} {mesh_max[2]:.6f})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({divisions[0]} {divisions[1]} {divisions[2]}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    walls
    {{
        type patch;
        faces
        (
            (0 4 7 3)
            (2 6 5 1)
            (1 5 4 0)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);

mergePatchPairs
(
);

// ************************************************************************* //'''
        
        (case_dir / "system" / "blockMeshDict").write_text(content)
        self.logger.info(f"📐 Block divisions: {divisions} (total: {np.prod(divisions)} cells)")

    def _calculate_bbox_and_cell_size(self, case_dir: Path) -> Tuple[np.ndarray, np.ndarray, float]:
        """Calculate bounding box and cell size from ORIGINAL STL files (no rescaling)."""
        from .stl_processor import STLProcessor
        
        processor = STLProcessor(self.logger)
        
        # Use ORIGINAL STL files from geometry_dir, not the copied ones
        original_stl_files = list(self.geometry_dir.glob("*.stl"))
        
        all_vertices = []
        for stl_file in original_stl_files:
            triangles = processor.read_triangles(stl_file)
            for normal, v1, v2, v3 in triangles:
                all_vertices.extend([v1, v2, v3])
        
        if not all_vertices:
            raise ValueError("No vertices found in original STL files")
        
        vertices_array = np.array(all_vertices)
        mesh_min = np.min(vertices_array, axis=0)
        mesh_max = np.max(vertices_array, axis=0)
        
        # Calculate domain size in original units
        domain_size = mesh_max - mesh_min
        self.logger.info(f"📐 Original geometry size: {domain_size}")
        
        # Add margin to bounding box
        margin = 0.1 * (mesh_max - mesh_min)
        mesh_min -= margin
        mesh_max += margin
        
        # Calculate cell size using diameter-based approach (same as simple_mesh_generator.py)
        mesh_config = self.config.get('mesh', {})
        cells_per_diameter = mesh_config.get('cells_per_diameter', 20)  # Default 20
        
        # Calculate reference diameter (same as simple_mesh_generator.py)
        bbox_size = mesh_max - mesh_min - 2 * margin  # Remove margin for diameter calc
        ref_diameter = np.mean(bbox_size[:2])  # Average of x,y dimensions
        
        # Cell size calculation (same as simple_mesh_generator.py)
        cell_size = ref_diameter / cells_per_diameter
        
        # Apply scaling (same as simple_mesh_generator.py)
        scaling = self.config.get('SCALING', {})
        scale_m = scaling.get('scale_m', 0.001)  # Default 0.001 (mm to m)
        effective_cell_size = cell_size * scale_m
        
        self.logger.info(f"📐 Original bounding box: {mesh_min} to {mesh_max}")
        self.logger.info(f"📐 Reference diameter: {ref_diameter:.3f}, cells_per_diameter: {cells_per_diameter}")
        self.logger.info(f"📐 Cell size: {cell_size:.3f}, scale_m: {scale_m}")
        self.logger.info(f"📐 Effective cell size: {effective_cell_size:.6f}")
        
        return mesh_min, mesh_max, cell_size  # Return UNSCALED cell size for divisions

    def _generate_snappy_dict(self, case_dir: Path) -> None:
        """Generate snappyHexMeshDict with ray-casting interior point."""
        tri_surface_dir = case_dir / "constant" / "triSurface"
        stl_files = list(tri_surface_dir.glob("*.stl"))
        
        # Categorize STL files
        inlet_files = [f for f in stl_files if 'inlet' in f.name.lower()]
        outlet_files = [f for f in stl_files if 'outlet' in f.name.lower()]
        wall_files = [f for f in stl_files if f not in inlet_files + outlet_files]
        
        # Get configuration
        refinement = self.config.get('refinement', {})
        surface_levels = refinement.get('surface_levels', [1, 1])
        feature_angle = refinement.get('feature_angle', {}).get('init', 40)
        
        layers = self.config.get('layers', {})
        add_layers = layers.get('n', 0) > 0
        
        # Calculate bounding box and use ray-casting for interior point
        mesh_min, mesh_max, _ = self._calculate_bbox_and_cell_size(case_dir)
        
        try:
            # Use ray-casting on ORIGINAL STL files for robust interior point detection
            original_stl_files = list(self.geometry_dir.glob("*.stl"))
            interior_point = find_interior_point_raycast(
                original_stl_files, 
                mesh_min, 
                mesh_max,
                logger=self.logger,
                max_candidates=50,
                margin_fraction=0.15
            )
            self.logger.info(f"🎯 Ray-casting found interior point: ({interior_point[0]:.6f}, {interior_point[1]:.6f}, {interior_point[2]:.6f})")
        except Exception as e:
            self.logger.warning(f"⚠️ Ray-casting failed: {e}, using center point")
            interior_point = (mesh_min + mesh_max) / 2

        # Generate geometry section
        geometry_section = ""
        for stl_file in stl_files:
            name = stl_file.stem
            geometry_section += f'    {name}.stl {{ type triSurfaceMesh; name {name}; file "{stl_file.name}"; }}\n'
        
        # Generate refinement surfaces
        refinement_surfaces = ""
        for stl_file in stl_files:
            name = stl_file.stem
            if 'wall' in name.lower():
                patch_type = "wall"
                level = f"({surface_levels[0]} {surface_levels[1]})"
            else:
                patch_type = "patch"
                level = f"({surface_levels[0]} {surface_levels[0]})"
            
            refinement_surfaces += f'        {name} {{ level {level}; patchInfo {{ type {patch_type}; }} }}\n'
        
        # Generate layers section if needed
        if add_layers:
            wall_names = [f.stem for f in wall_files]
            layers_config = f'''
    relativeSizes {str(layers.get('relativeSizes', True)).lower()};
    layers
    {{
        {wall_names[0] if wall_names else "wall"}
        {{
            nSurfaceLayers {layers.get('n', 4)};
        }}
    }}
    
    firstLayerThickness {layers.get('first_layer', {}).get('t1_rel', 0.5) if layers.get('relativeSizes', True) else layers.get('first_layer', {}).get('t1_abs', 50e-6):.2e};
    expansionRatio {layers.get('expansion', 1.2)};
    minThickness {layers.get('minThickness', 0.1)};
    nGrow 0;
    featureAngle 60;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
    nRelaxedIter 20;
'''
        else:
            layers_config = "    // No boundary layers"
        
        content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v12                                   |
|   \\  /    A nd           | Website:  www.openfoam.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

castellatedMesh true;
snap            true;
addLayers       {str(add_layers).lower()};

geometry
{{
{geometry_section}
}};

castellatedMeshControls
{{
    maxLocalCells 100000;
    maxGlobalCells 2000000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    
    features
    (
    );
    
    refinementSurfaces
    {{
{refinement_surfaces}
    }};
    
    resolveFeatureAngle {feature_angle};
    refinementRegions {{}};
    locationInMesh ({interior_point[0]:.6f} {interior_point[1]:.6f} {interior_point[2]:.6f});
    allowFreeStandingZoneFaces false;
}};

snapControls
{{
    nSmoothPatch 3;
    tolerance 4.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}};

addLayersControls
{{
{layers_config}
}};

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minFlatness 0.5;
    minVol 1e-13;
    minTetQuality 1e-30;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.02;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
    
    relaxed
    {{
        maxNonOrtho 75;
        maxBoundarySkewness 25;
        maxInternalSkewness 6;
    }}
}};

writeFlags
(
    scalarLevels
    layerSets
    layerFields
);

mergeTolerance 1e-6;

// ************************************************************************* //'''
        
        (case_dir / "system" / "snappyHexMeshDict").write_text(content)

    def _run_mesh_generation(self, case_dir: Path) -> None:
        """Execute OpenFOAM mesh generation commands."""
        self.logger.info("⚙️ Running OpenFOAM mesh generation")
        
        # Commands to run
        commands = [
            "source /opt/openfoam12/etc/bashrc && blockMesh",
            "source /opt/openfoam12/etc/bashrc && snappyHexMesh -overwrite",
            "source /opt/openfoam12/etc/bashrc && checkMesh"
        ]
        
        for i, cmd in enumerate(commands, 1):
            self.logger.info(f"🔧 Step {i}/{len(commands)}: {cmd.split('&&')[-1].strip()}")
            try:
                result = subprocess.run(
                    ["bash", "-c", cmd], 
                    cwd=case_dir,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
                if result.returncode != 0:
                    self.logger.error(f"❌ Command failed: {cmd}")
                    self.logger.error(f"stderr: {result.stderr}")
                    raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
                
                self.logger.info(f"✅ Step {i} completed successfully")
                
            except subprocess.TimeoutExpired:
                self.logger.error(f"❌ Command timed out: {cmd}")
                raise
            except Exception as e:
                self.logger.error(f"❌ Command failed: {e}")
                raise

    # Backward compatibility methods
    
    def iterate_until_quality(self) -> Path:
        """Backward compatibility method - calls generate_mesh()."""
        return self.generate_mesh()
        
    def update_current_iteration(self, iteration: int) -> None:
        """Backward compatibility method - does nothing (no iterations in v2)."""
        pass