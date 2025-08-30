#!/usr/bin/env python3
"""
Simple SnappyHexMesh Automation Tool

A streamlined mesh generator that reads JSON configuration and produces 
OpenFOAM meshes without optimization loops, iterations, or quality checking.

Usage:
    python simple_mesh_generator.py --geometry geometry/ --config config.json --output output/
"""

import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class SimpleMeshGenerator:
    """Simple single-pass mesh generator."""
    
    def __init__(self, geometry_dir: Path, config_path: Path, output_dir: Path):
        self.geometry_dir = Path(geometry_dir)
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        
        # Load and validate configuration
        self.config = self._load_config()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized: geometry={geometry_dir}, output={output_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Basic validation - ensure required sections exist
            required_sections = ['mesh', 'refinement', 'layers']
            for section in required_sections:
                if section not in config:
                    logger.warning(f"Missing config section: {section}")
                    config[section] = {}
            
            return config
        except Exception as e:
            raise ValueError(f"Failed to load config from {self.config_path}: {e}")
    
    def generate_mesh(self) -> Path:
        """Generate mesh using single-pass snappyHexMesh."""
        logger.info("Starting mesh generation")
        
        # Create case directory structure
        case_dir = self.output_dir / "mesh_case"
        self._setup_case_structure(case_dir)
        
        # Generate OpenFOAM dictionaries
        self._generate_blockmesh_dict(case_dir)
        self._generate_snappy_dict(case_dir)
        self._generate_control_dict(case_dir)
        
        # Run mesh generation
        self._run_meshing_commands(case_dir)
        
        logger.info(f"✅ Mesh generation completed: {case_dir}")
        return case_dir
    
    def _setup_case_structure(self, case_dir: Path):
        """Create OpenFOAM case directory structure."""
        # Create directories
        (case_dir / "system").mkdir(parents=True, exist_ok=True)
        (case_dir / "constant").mkdir(parents=True, exist_ok=True)
        (case_dir / "constant" / "triSurface").mkdir(parents=True, exist_ok=True)
        
        # Copy STL files
        tri_surface_dir = case_dir / "constant" / "triSurface"
        stl_files = list(self.geometry_dir.glob("*.stl"))
        
        if not stl_files:
            raise ValueError(f"No STL files found in {self.geometry_dir}")
        
        for stl_file in stl_files:
            target = tri_surface_dir / stl_file.name
            shutil.copy2(stl_file, target)
            logger.info(f"Copied: {stl_file.name}")
        
        # Create .foam file for ParaView
        (case_dir / f"{case_dir.name}.foam").touch()
    
    def _calculate_bbox_and_cell_size(self, stl_files: List[Path]) -> tuple:
        """Calculate bounding box and cell size from STL files."""
        all_vertices = []
        
        # Binary STL vertex extraction
        for stl_file in stl_files:
            try:
                import struct
                with open(stl_file, 'rb') as f:
                    # Skip 80-byte header
                    f.read(80)
                    # Read number of triangles
                    num_triangles = struct.unpack('<I', f.read(4))[0]
                    
                    for _ in range(num_triangles):
                        # Skip normal vector (3 floats)
                        f.read(12)
                        # Read 3 vertices (9 floats total)
                        for _ in range(3):
                            vertex = struct.unpack('<fff', f.read(12))
                            all_vertices.append(vertex)
                        # Skip attribute byte count
                        f.read(2)
                        
                logger.info(f"Read {len(all_vertices)} vertices from {stl_file.name}")
            except Exception as e:
                logger.warning(f"Failed to read vertices from {stl_file}: {e}")
        
        if not all_vertices:
            logger.warning("No vertices found, using default bounding box")
            bbox_min = np.array([-0.05, -0.05, -0.1])
            bbox_max = np.array([0.05, 0.05, 0.1])
        else:
            all_vertices = np.array(all_vertices)
            bbox_min = all_vertices.min(axis=0)
            bbox_max = all_vertices.max(axis=0)
        
        # Add 10% margin to bounding box
        margin = 0.1 * (bbox_max - bbox_min)
        mesh_min = bbox_min - margin
        mesh_max = bbox_max + margin
        
        # Calculate cell size from config
        mesh_config = self.config.get('mesh', {})
        cells_per_diameter = mesh_config.get('cells_per_diameter', 20)
        
        # Estimate reference diameter (simple approximation)
        bbox_size = bbox_max - bbox_min
        ref_diameter = np.mean(bbox_size[:2])  # Average of x,y dimensions
        
        cell_size = ref_diameter / cells_per_diameter
        
        # Apply scaling
        scale_m = self.config.get('SCALING', {}).get('scale_m', 0.001)
        cell_size_scaled = cell_size * scale_m
        
        logger.info(f"Bounding box: {mesh_min} to {mesh_max}")
        logger.info(f"Reference diameter: {ref_diameter:.3f}mm, Cell size: {cell_size:.3f}mm")
        logger.info(f"Scaled cell size: {cell_size_scaled*1000:.1f}μm")
        
        return mesh_min, mesh_max, cell_size
    
    def _generate_blockmesh_dict(self, case_dir: Path):
        """Generate blockMeshDict."""
        system_dir = case_dir / "system"
        
        # Get STL files and calculate bounding box
        stl_files = list((case_dir / "constant" / "triSurface").glob("*.stl"))
        mesh_min, mesh_max, cell_size = self._calculate_bbox_and_cell_size(stl_files)
        
        # Calculate divisions
        bbox_size = mesh_max - mesh_min
        divisions = np.ceil(bbox_size / cell_size).astype(int)
        
        # Minimum divisions to avoid degenerate mesh
        divisions = np.maximum(divisions, [10, 10, 10])
        
        total_cells = np.prod(divisions)
        logger.info(f"BlockMesh divisions: {divisions} → {total_cells:,} background cells")
        
        # Generate blockMeshDict content
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

convertToMeters 1;

vertices
(
    ({mesh_min[0]:.6f} {mesh_min[1]:.6f} {mesh_min[2]:.6f})  // 0
    ({mesh_max[0]:.6f} {mesh_min[1]:.6f} {mesh_min[2]:.6f})  // 1
    ({mesh_max[0]:.6f} {mesh_max[1]:.6f} {mesh_min[2]:.6f})  // 2
    ({mesh_min[0]:.6f} {mesh_max[1]:.6f} {mesh_min[2]:.6f})  // 3
    ({mesh_min[0]:.6f} {mesh_min[1]:.6f} {mesh_max[2]:.6f})  // 4
    ({mesh_max[0]:.6f} {mesh_min[1]:.6f} {mesh_max[2]:.6f})  // 5
    ({mesh_max[0]:.6f} {mesh_max[1]:.6f} {mesh_max[2]:.6f})  // 6
    ({mesh_min[0]:.6f} {mesh_max[1]:.6f} {mesh_max[2]:.6f})  // 7
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
    domain
    {{
        type patch;
        faces
        (
            (0 4 7 3)  // x_min face
            (1 2 6 5)  // x_max face
            (0 1 5 4)  // y_min face
            (3 7 6 2)  // y_max face
            (0 3 2 1)  // z_min face
            (4 5 6 7)  // z_max face
        );
    }}
);

mergePatchPairs
(
);

// ************************************************************************* //
'''
        
        blockmesh_file = system_dir / "blockMeshDict"
        blockmesh_file.write_text(content)
        logger.info("Generated blockMeshDict")
    
    def _generate_snappy_dict(self, case_dir: Path):
        """Generate snappyHexMeshDict."""
        system_dir = case_dir / "system"
        tri_surface_dir = case_dir / "constant" / "triSurface"
        
        # Find STL files and categorize them
        stl_files = list(tri_surface_dir.glob("*.stl"))
        inlet_files = [f for f in stl_files if 'inlet' in f.name.lower()]
        outlet_files = [f for f in stl_files if 'outlet' in f.name.lower()]
        wall_files = [f for f in stl_files if f not in inlet_files + outlet_files]
        
        # Get refinement settings from config
        refinement = self.config.get('refinement', {})
        surface_levels = refinement.get('surface_levels', [1, 1])
        feature_angle = refinement.get('feature_angle', {}).get('init', 40)
        
        # Get layer settings
        layers = self.config.get('layers', {})
        add_layers = layers.get('n', 0) > 0
        
        # Find interior point (simple center point)
        mesh_min, mesh_max, _ = self._calculate_bbox_and_cell_size(stl_files)
        interior_point = (mesh_min + mesh_max) / 2
        
        logger.info(f"Interior point: {interior_point}")
        logger.info(f"Surface levels: {surface_levels}, Add layers: {add_layers}")
        
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

// ************************************************************************* //
'''
        
        snappy_file = system_dir / "snappyHexMeshDict"
        snappy_file.write_text(content)
        logger.info("Generated snappyHexMeshDict")
    
    def _generate_control_dict(self, case_dir: Path):
        """Generate controlDict."""
        system_dir = case_dir / "system"
        
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
    location    "system";
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     snappyHexMesh;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;

writeControl    timeStep;
writeInterval   1;
purgeWrite      0;

writeFormat     ascii;
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

// ************************************************************************* //
'''
        
        control_file = system_dir / "controlDict"
        control_file.write_text(content)
        logger.info("Generated controlDict")
    
    def _run_meshing_commands(self, case_dir: Path):
        """Run OpenFOAM meshing commands."""
        import subprocess
        
        # Get OpenFOAM environment
        openfoam_env = self.config.get('openfoam_env_path', 'source /opt/openfoam12/etc/bashrc')
        
        commands = [
            "blockMesh",
            "snappyHexMesh -overwrite"
        ]
        
        for cmd in commands:
            full_cmd = f"bash -c '{openfoam_env} && {cmd}'"
            logger.info(f"Running: {cmd} in {case_dir}")
            
            try:
                result = subprocess.run(
                    full_cmd,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=str(case_dir)
                )
                logger.info(f"✅ {cmd} completed successfully")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ {cmd} failed:")
                logger.error(f"Return code: {e.returncode}")
                logger.error(f"stdout: {e.stdout}")
                logger.error(f"stderr: {e.stderr}")
                raise RuntimeError(f"OpenFOAM command failed: {cmd}")


def create_simple_config_template(output_path: Path):
    """Create a simple configuration template."""
    template = {
        "description": "Simple mesh generation configuration",
        "openfoam_env_path": "source /opt/openfoam12/etc/bashrc",
        
        "SCALING": {
            "scale_m": 0.001
        },
        
        "mesh": {
            "base_size_mode": "diameter",
            "cells_per_diameter": 20
        },
        
        "refinement": {
            "surface_levels": [1, 2],
            "feature_angle": {
                "init": 40
            }
        },
        
        "layers": {
            "n": 4,
            "expansion": 1.2,
            "relativeSizes": True,
            "minThickness": 0.1,
            "first_layer": {
                "t1_rel": 0.5
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    logger.info(f"Created config template: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simple SnappyHexMesh Automation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate mesh from STL files
  python simple_mesh_generator.py --geometry tutorial/patient1/ --config simple_config.json --output output/

  # Create configuration template
  python simple_mesh_generator.py --create-config simple_config.json
        '''
    )
    
    parser.add_argument('--geometry', help='Path to directory containing STL geometry files')
    parser.add_argument('--config', help='Path to JSON configuration file')
    parser.add_argument('--output', help='Output directory for mesh case')
    parser.add_argument('--create-config', help='Create a configuration template file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.create_config:
            create_simple_config_template(Path(args.create_config))
            return 0
        
        if not all([args.geometry, args.config, args.output]):
            parser.error("--geometry, --config, and --output are required")
        
        # Generate mesh
        generator = SimpleMeshGenerator(
            geometry_dir=args.geometry,
            config_path=args.config,
            output_dir=args.output
        )
        
        case_dir = generator.generate_mesh()
        logger.info(f"✅ Mesh generation completed: {case_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())