"""OpenFOAM dictionary generation module for Stage 1 mesh optimization.

Handles creation of blockMeshDict, snappyHexMeshDict, surfaceFeaturesDict,
and other OpenFOAM configuration files needed for mesh generation.
"""

import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from .constants import (
    EXPANSION_RATIO_DEFAULT,
    N_SURFACE_LAYERS_DEFAULT,
    FEATURE_ANGLE_LAYERS,
    SLIP_FEATURE_ANGLE,
    MIN_THICKNESS_ABSOLUTE,
    MIN_THICKNESS_FRACTION,
    FIRST_LAYER_DEFAULT
)
from .utils import safe_float, safe_int


class OpenFOAMDictGenerator:
    """Generates OpenFOAM dictionary files for mesh generation."""
    
    def __init__(self, config: dict, wall_name: str, logger: logging.Logger):
        self.config = config
        self.wall_name = wall_name
        self.logger = logger
        
        # Cache commonly used config sections
        self.stage1 = config.get("STAGE1", {})
        self.snappy = config.get("SNAPPY", {})
        self.layers = config.get("LAYERS", {})
        self.blockmesh = config.get("BLOCKMESH", {})
    
    def generate_blockmesh_dict(self, iter_dir: Path, bbox_info: dict, dx_base: float):
        """Generate blockMeshDict for background mesh."""
        # Get bounding box info
        md = bbox_info["mesh_domain"]
        x_min, x_max = md["x_min"], md["x_max"]
        y_min, y_max = md["y_min"], md["y_max"] 
        z_min, z_max = md["z_min"], md["z_max"]
        
        bbox_size = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
        
        # Compute divisions with hierarchical override control
        divisions = self._compute_block_divisions(bbox_size, dx_base)
        
        # Generate blockMeshDict content
        dict_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v12                                   |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
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
    ({x_min:.6f} {y_min:.6f} {z_min:.6f})  // 0
    ({x_max:.6f} {y_min:.6f} {z_min:.6f})  // 1
    ({x_max:.6f} {y_max:.6f} {z_min:.6f})  // 2
    ({x_min:.6f} {y_max:.6f} {z_min:.6f})  // 3
    ({x_min:.6f} {y_min:.6f} {z_max:.6f})  // 4
    ({x_max:.6f} {y_min:.6f} {z_max:.6f})  // 5
    ({x_max:.6f} {y_max:.6f} {z_max:.6f})  // 6
    ({x_min:.6f} {y_max:.6f} {z_max:.6f})  // 7
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
        
        # Write blockMeshDict
        system_dir = iter_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        
        blockmesh_file = system_dir / "blockMeshDict"
        blockmesh_file.write_text(dict_content)
        
        total_cells = int(np.prod(divisions))
        self.logger.info(f"Generated blockMeshDict: {divisions} → {total_cells:,} background cells")
        
        # Generate controlDict for the iteration
        self._generate_control_dict(iter_dir)

    def _compute_block_divisions(self, bbox_size: np.ndarray, dx_base: float) -> np.ndarray:
        """Compute blockMesh divisions with hierarchical override control.
        
        Precedence: divisions > cell_size_m > resolution > geometry-aware
        """
        # Handle null/None values in config properly
        min_per_axis_config = self.blockmesh.get("min_per_axis", [10, 10, 10])
        if min_per_axis_config is None:
            min_per_axis_config = [10, 10, 10]
        mins = np.array(min_per_axis_config, dtype=int)

        if "divisions" in self.blockmesh and self.blockmesh["divisions"] is not None:
            divs = np.array(self.blockmesh["divisions"], dtype=int)
            return np.maximum(divs, mins)

        if "cell_size_m" in self.blockmesh and self.blockmesh["cell_size_m"] is not None:
            dx = float(self.blockmesh["cell_size_m"])
            divs = np.ceil(bbox_size / max(dx, 1e-9)).astype(int)
            return np.maximum(divs, mins)

        if "resolution" in self.blockmesh:
            R = int(self.blockmesh["resolution"])
            Lmax = float(max(bbox_size))
            scale = R / max(Lmax, 1e-12)  # cells per meter
            divs = np.ceil(bbox_size * scale).astype(int)
            return np.maximum(divs, mins)

        # default: geometry-aware from dx_base
        if dx_base is None:
            self.logger.error("dx_base is None in _compute_block_divisions - this should not happen")
            dx_base = 1e-3  # 1mm fallback
        divs = np.ceil(bbox_size / max(dx_base, 1e-9)).astype(int)
        return np.maximum(divs, mins)

    def generate_surface_features_dict(self, iter_dir: Path, surface_list: List[str]):
        """Generate surfaceFeaturesDict for feature extraction."""
        system_dir = iter_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        
        # Use proper includedAngle for surfaceFeatures (NOT resolveFeatureAngle!)
        # includedAngle: edges with angle < this value are marked as features
        # Typical values: 120-150 degrees for capturing sharp edges
        included_angle = self.config.get("SURFACE_FEATURES", {}).get("includedAngle", 160)
        
        dict_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v12                                   |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeaturesDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

'''
        
        for surface in surface_list:
            surface_name = surface.replace('.stl', '')
            dict_content += f'''
{surface_name}
{{
    surfaces
    (
        "{surface}"
    );

    // Mark edges whose adjacent surface normals are at an angle less
    // than includedAngle as features
    // - 0  : selects no edges  
    // - 180: selects all edges
    includedAngle           {included_angle};

    // Do not mark region edges
    geometricTestOnly       yes;

    writeObj                yes;
    writeSurfaceFeatures    yes;
}}
'''
        
        dict_content += "\n// ************************************************************************* //\n"
        
        features_file = system_dir / "surfaceFeaturesDict"
        features_file.write_text(dict_content)
        
        self.logger.info(f"Generated surfaceFeaturesDict: includedAngle={included_angle}° for {len(surface_list)} surfaces")

    def generate_snappy_dict(self, iter_dir: Path, outlet_names: List[str], 
                           internal_point: np.ndarray, dx_base: float, 
                           phase: str) -> Path:
        """Generate snappyHexMeshDict for specified phase.
        
        Args:
            iter_dir: Iteration directory
            outlet_names: List of outlet patch names  
            internal_point: Seed point for mesh generation
            dx_base: Base cell size for refinement calculations
            phase: Either "no_layers" or "layers"
            
        Returns:
            Path to generated dictionary file
        """
        if phase == "no_layers":
            return self._generate_snappy_no_layers_dict(iter_dir, outlet_names, internal_point, dx_base)
        elif phase == "layers":
            return self._generate_snappy_layers_dict(iter_dir, outlet_names, internal_point, dx_base)
        else:
            raise ValueError(f"Unknown snappy phase: {phase}")

    def _generate_snappy_no_layers_dict(self, iter_dir: Path, outlet_names: List[str],
                                      internal_point: np.ndarray, dx_base: float) -> Path:
        """Generate snappyHexMeshDict without boundary layers."""
        system_dir = iter_dir / "system"
        
        # Calculate refinement settings
        surface_levels = self._get_current_surface_levels()
        refinement_bands = self._calculate_refinement_bands(dx_base)
        
        # Get mesh quality settings
        quality_settings = self._get_mesh_quality_settings()
        
        dict_content = f'''/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v12                                   |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
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
addLayers       false;

geometry
{{
'''
        
        # Add surface geometries
        for outlet_name in outlet_names:
            dict_content += f'''    {outlet_name}.stl {{ type triSurfaceMesh; name {outlet_name}; file "{outlet_name}.stl"; }}
'''
        dict_content += f'''    inlet.stl {{ type triSurfaceMesh; name inlet; file "inlet.stl"; }}
    {self.wall_name}.stl {{ type triSurfaceMesh; name {self.wall_name}; file "{self.wall_name}.stl"; }}
}};

castellatedMeshControls
{{
    maxLocalCells {self.snappy.get('maxLocalCells', 2000000)};
    maxGlobalCells {self.snappy.get('maxGlobalCells', 8000000)};
    minRefinementCells {self.snappy.get('minRefinementCells', 10)};
    nCellsBetweenLevels {self.snappy.get('nCellsBetweenLevels', 3)};
    
    features
    (
'''
        
        # Add feature files
        for outlet_name in outlet_names:
            dict_content += f'''        {{ file "{outlet_name}.eMesh"; level {surface_levels[0]}; }}
'''
        dict_content += f'''        {{ file "inlet.eMesh"; level {surface_levels[0]}; }}
        {{ file "{self.wall_name}.eMesh"; level {surface_levels[1]}; }}
    );
    
    refinementSurfaces
    {{
'''
        
        # Add surface refinement
        for outlet_name in outlet_names:
            dict_content += f'''        {outlet_name} {{ level ({surface_levels[0]} {surface_levels[0]}); patchInfo {{ type patch; }} }}
'''
        dict_content += f'''        inlet {{ level ({surface_levels[0]} {surface_levels[0]}); patchInfo {{ type patch; }} }}
        {self.wall_name} {{ level ({surface_levels[0]} {surface_levels[1]}); patchInfo {{ type wall; }} }}
    }};
    
    refinementRegions
    {{
'''
        
        # Add refinement bands if configured
        if refinement_bands:
            dict_content += refinement_bands
        
        dict_content += f'''    }};
    
    locationInMesh ({internal_point[0]:.6f} {internal_point[1]:.6f} {internal_point[2]:.6f});
    allowFreeStandingZoneFaces false;
    
    resolveFeatureAngle {self.snappy.get('resolveFeatureAngle', 32)};
    tolerance           {self.snappy.get('tolerance', 4.0)};
    nSmoothPatch        {self.snappy.get('nSmoothPatch', 3)};
    nSolveIter          {self.snappy.get('nSolveIter', 30)};
    nRelaxIter          {self.snappy.get('nRelaxIter', 5)};
}};

snapControls
{{
    nSmoothPatch            {self.snappy.get('nSmoothPatch', 3)};
    tolerance               {self.snappy.get('snapTolerance', 2.0)};
    nSolveIter              {self.snappy.get('nSolveIter', 30)};
    nRelaxIter              {self.snappy.get('nRelaxIter', 5)};
    nFeatureSnapIter        {self._get_dynamic_feature_snap_iter()};
    implicitFeatureSnap     {str(self.snappy.get('implicitFeatureSnap', 'false')).lower()};
    explicitFeatureSnap     {str(self.snappy.get('explicitFeatureSnap', 'true')).lower()};
    multiRegionFeatureSnap  {str(self.snappy.get('multiRegionFeatureSnap', 'false')).lower()};
}};

addLayersControls
{{
    // Disabled for no-layers phase
}};

meshQualityControls
{{
{quality_settings}
}};

writeFlags
(
    scalarLevels
    layerSets
    layerFields
);

mergeTolerance {self.snappy.get('mergeTolerance', 1e-6)};

// ************************************************************************* //
'''
        
        # Write file
        dict_file = system_dir / "snappyHexMeshDict.noLayer"
        dict_file.write_text(dict_content)
        
        self.logger.debug(f"Generated snappyHexMeshDict (no layers): levels {surface_levels}")
        return dict_file

    def _generate_snappy_layers_dict(self, iter_dir: Path, outlet_names: List[str],
                                   internal_point: np.ndarray, dx_base: float) -> Path:
        """Generate snappyHexMeshDict with boundary layers - enhanced version."""
        system_dir = iter_dir / "system"
        
        # Get layer configuration and apply consistent minThickness policy
        layer_config = self._calculate_layer_configuration(dx_base)
        L = layer_config.copy()  # Make a copy to avoid modifying config
        
        # Apply consistent minThickness policy before generating dict
        self._apply_consistent_minThickness_policy(L)
        
        # Generate dictionary with correct keys for relative vs absolute modes
        is_rel = bool(L.get("relativeSizes", False))
        if is_rel:
            first_line = f"    firstLayerThickness {L.get('firstLayerThickness', 0.2):.3f};"
            final_line = f"    finalLayerThickness {L.get('finalLayerThickness', 0.75):.3f};"
            min_line = f"    minThickness {L.get('minThickness', 0.15):.3f};"
        else:
            first_line = f"    firstLayerThickness {L.get('firstLayerThickness_abs', 50e-6):.2e};"
            min_line = f"    minThickness {L.get('minThickness_abs', 20e-6):.2e};"
            final_line = ""  # not used in absolute mode

        # Generate the dictionary content
        content = self._generate_enhanced_layers_content(outlet_names, internal_point, L, is_rel, first_line, final_line, min_line)
        
        # Write layers file
        dict_file = system_dir / "snappyHexMeshDict.layers"
        dict_file.write_text(content)
        
        self.logger.debug(f"Generated snappyHexMeshDict (layers): {L['nSurfaceLayers']} layers, "
                         f"{'relative' if is_rel else 'absolute'} sizing")
        return dict_file

    def _generate_enhanced_layers_content(self, outlet_names: List[str], internal_point: np.ndarray,
                                        L: dict, is_rel: bool, first_line: str, final_line: str, min_line: str) -> str:
        """Generate enhanced layers dictionary content with proper relative/absolute handling."""
        # Ensure all numeric values are properly typed to prevent string division errors
        # Using centralized safe conversion functions from utils
        # Convert all layer configuration values to proper types
        expansion_ratio = safe_float(L.get("expansionRatio", 1.2))
        feature_angle = safe_float(L.get("featureAngle", 60))
        max_face_thickness_ratio = safe_float(L.get("maxFaceThicknessRatio", 0.5))
        max_thickness_to_medial_ratio = safe_float(L.get("maxThicknessToMedialRatio", 0.3))
        min_median_axis_angle = safe_float(L.get("minMedianAxisAngle", 70))
        
        n_surface_layers = safe_int(L.get("nSurfaceLayers", 5))
        n_grow = safe_int(L.get("nGrow", 0))
        n_relax_iter = safe_int(L.get("nRelaxIter", 5))
        n_smooth_surface_normals = safe_int(L.get("nSmoothSurfaceNormals", 3))
        n_layer_iter = safe_int(L.get("nLayerIter", 50))
        n_relaxed_iter = safe_int(L.get("nRelaxedIter", 20))
        n_buffer_cells_no_extrude = safe_int(L.get("nBufferCellsNoExtrude", 0))
        return f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

castellatedMesh false;
snap            false;
addLayers       true;

geometry
{{
    inlet.stl       {{ type triSurfaceMesh; name inlet;       file "inlet.stl"; }}
{chr(10).join(f'    {n}.stl        {{ type triSurfaceMesh; name {n};           file "{n}.stl"; }}' for n in outlet_names)}
    {self.wall_name}.stl  {{ type triSurfaceMesh; name {self.wall_name};  file "{self.wall_name}.stl"; }}
}};

castellatedMeshControls
{{
    maxLocalCells 100000;
    maxGlobalCells 2000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 1;
    features ();
    refinementSurfaces {{}};
    resolveFeatureAngle 30;
    refinementRegions {{}};
    locationInMesh ({internal_point[0]:.6f} {internal_point[1]:.6f} {internal_point[2]:.6f});
    allowFreeStandingZoneFaces true;
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
    relativeSizes {str(is_rel).lower()};
{('    ' + final_line) if is_rel else ''}
    layers
    {{
        "{self.wall_name}"
        {{
            nSurfaceLayers {n_surface_layers};
        }}
    }}

{first_line}
    expansionRatio {expansion_ratio};
{min_line}
    nGrow {n_grow};
    featureAngle {min(feature_angle, 90)};
    nRelaxIter {n_relax_iter};
    nSmoothSurfaceNormals {n_smooth_surface_normals};
    nSmoothNormals 5;
    nSmoothThickness 10;
    maxFaceThicknessRatio {max_face_thickness_ratio};
    maxThicknessToMedialRatio {min(max_thickness_to_medial_ratio, 0.6)};
    minMedianAxisAngle {max(min_median_axis_angle, 70)};
    nBufferCellsNoExtrude {n_buffer_cells_no_extrude};
    nLayerIter {n_layer_iter};
    nRelaxedIter {n_relaxed_iter};
    additionalReporting false;
}};

meshQualityControls
{{
    maxNonOrtho {self.config.get('accept', {}).get('maxNonOrtho', 75)};
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
    }}
}};

writeFlags
(
    scalarLevels
    layerFields
);

mergeTolerance 1e-6;
"""

    def _generate_control_dict(self, iter_dir: Path):
        """Generate OpenFOAM controlDict for mesh generation."""
        system_dir = iter_dir / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        
        control_dict_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Version:  12
     \\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}
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
"""
        
        control_dict_file = system_dir / "controlDict"
        control_dict_file.write_text(control_dict_content)
        self.logger.debug(f"Generated controlDict: {control_dict_file}")

    def _calculate_layer_configuration(self, dx_base: float) -> dict:
        """Calculate boundary layer configuration parameters."""
        config = {}
        
        # Get layer settings from config
        config['nSurfaceLayers'] = self.layers.get('nSurfaceLayers', N_SURFACE_LAYERS_DEFAULT)
        config['expansionRatio'] = self.layers.get('expansionRatio', EXPANSION_RATIO_DEFAULT)
        config['relativeSizes'] = self.layers.get('relativeSizes', False)
        
        # Layer thickness calculation
        if config['relativeSizes']:
            # Relative sizing (fraction of local cell size)
            config['firstLayerThickness'] = self.layers.get('firstLayerThickness', 0.2)
            config['minThickness'] = MIN_THICKNESS_FRACTION * config['firstLayerThickness']
        else:
            # Absolute sizing (meters)
            config['firstLayerThickness'] = self.layers.get('firstLayerThickness_abs', FIRST_LAYER_DEFAULT)
            config['minThickness'] = max(
                MIN_THICKNESS_ABSOLUTE,
                MIN_THICKNESS_FRACTION * config['firstLayerThickness']
            )
        
        # Advanced layer controls from config
        config.update({
            'nGrow': self.layers.get('nGrow', 0),
            'featureAngle': self.layers.get('featureAngle', FEATURE_ANGLE_LAYERS),
            'slipFeatureAngle': self.layers.get('slipFeatureAngle', SLIP_FEATURE_ANGLE),
            'nRelaxIter': self.layers.get('nRelaxIter', 5),
            'nSmoothSurfaceNormals': self.layers.get('nSmoothSurfaceNormals', 1),
            'nSmoothThickness': self.layers.get('nSmoothThickness', 10),
            'maxFaceThicknessRatio': self.layers.get('maxFaceThicknessRatio', 0.5),
            'maxThicknessToMedialRatio': self.layers.get('maxThicknessToMedialRatio', 0.3),
            'minMedianAxisAngle': self.layers.get('minMedianAxisAngle', 90),
            'nBufferCellsNoExtrude': self.layers.get('nBufferCellsNoExtrude', 0),
            'nLayerIter': self.layers.get('nLayerIter', 50),
            'nRelaxedIter': self.layers.get('nRelaxedIter', 20)
        })
        
        return config

    def _get_current_surface_levels(self) -> List[int]:
        """Get current surface refinement levels."""
        # This would be set by the optimizer during iterations
        return getattr(self, 'surface_levels', [1, 1])
    
    def _get_dynamic_feature_snap_iter(self) -> int:
        """Calculate nFeatureSnapIter based on resolveFeatureAngle (starter kit recommendation)."""
        resolve_angle = self.snappy.get('resolveFeatureAngle', 32)
        # Dynamic calculation: more iterations for smaller angles
        n_iter = max(10, int(180 / resolve_angle))
        # Allow override from config
        return self.snappy.get('nFeatureSnapIter', n_iter)

    def _calculate_refinement_bands(self, dx_base: float) -> str:
        """Calculate refinement regions based on distance from walls with override support."""
        stage1 = self.config.get("STAGE1", {})
        
        # Check if refinement bands are enabled
        if not stage1.get("use_refinement_bands", True):
            return ""
        
        # Check if wall geometry exists
        if not self.wall_name:
            self.logger.warning("No wall geometry found, skipping refinement bands")
            return ""
        
        # Use band override system
        near_dist, far_dist = self._calculate_refinement_distances(dx_base)
        
        # Use the actual wall geometry object for distance-based refinement
        # This references the wall_aorta geometry defined in the geometry section
        bands_config = f'''        {self.wall_name}
        {{
            mode distance;
            levels
            (
                (0.0 2)          // within 0.0 m refinement level 2
                ({near_dist:.6f} 1)  // within {near_dist*1000:.1f}mm refinement level 1
                ({far_dist:.6f} 0)   // within {far_dist*1000:.1f}mm refinement level 0
            );
        }}'''
        
        return bands_config

    def _calculate_refinement_distances(self, dx_base: float) -> tuple:
        """Calculate refinement distances with band override support."""
        # Check for band override first
        bo = self.config.get("STAGE1", {}).get("band_override", {})
        if bo and bo.get("enabled", False):
            # Priority: absolute > cells
            if "near_m" in bo and "far_m" in bo:
                near_dist = float(bo["near_m"])
                far_dist = float(bo["far_m"])
                self.logger.info(f"📏 BAND OVERRIDE: near={near_dist*1e3:.2f}mm, far={far_dist*1e3:.2f}mm (abs m)")
            else:
                near_cells = float(bo.get("near_cells", 8))
                far_cells = float(bo.get("far_cells", 18))
                near_dist = near_cells * dx_base
                far_dist = far_cells * dx_base
                self.logger.info(f"📏 BAND OVERRIDE: near={near_dist*1e3:.2f}mm, far={far_dist*1e3:.2f}mm (cells×Δx)")

            # Guardrail: ensure far > near
            if far_dist <= near_dist:
                far_dist = max(near_dist * 1.5, near_dist + 5.0 * dx_base)
        else:
            # Standard calculation from configuration
            stage1 = self.config.get("STAGE1", {})
            near_cells = stage1.get("near_band_cells", 4)
            far_cells = stage1.get("far_band_cells", 10)
            near_dist = near_cells * dx_base
            far_dist = far_cells * dx_base
        
        return near_dist, far_dist

    def _apply_consistent_minThickness_policy(self, L: dict):
        """Apply consistent minThickness policy for layer generation."""
        is_rel = bool(L.get("relativeSizes", False))
        
        if is_rel:
            # Relative mode: use starter kit recommendation of 0.15 as default
            # This prevents layer collapse while maintaining quality
            min_thickness = L.get("minThickness", 0.15)
            L["minThickness"] = min_thickness
        else:
            # Absolute mode: ensure minThickness is reasonable fraction of absolute thickness
            first_thickness_abs = L.get("firstLayerThickness_abs", 50e-6)
            min_thickness_abs = max(10e-6, min(30e-6, first_thickness_abs * 0.4))  # 40% of first layer, clipped to [10μm, 30μm]
            L["minThickness_abs"] = min_thickness_abs

    def _get_mesh_quality_settings(self) -> str:
        """Get mesh quality control settings."""
        quality = self.config.get("MESH_QUALITY", {})
        
        return f'''    maxNonOrtho             {quality.get('maxNonOrtho', 65)};
    maxBoundarySkewness     {quality.get('maxBoundarySkewness', 20)};
    maxInternalSkewness     {quality.get('maxInternalSkewness', 4)};
    maxConcave              {quality.get('maxConcave', 80)};
    minVol                  {quality.get('minVol', '1e-13')};
    minTetQuality           {quality.get('minTetQuality', '1e-30')};
    minArea                 {quality.get('minArea', '-1')};
    minTwist                {quality.get('minTwist', 0.02)};
    minDeterminant          {quality.get('minDeterminant', 0.001)};
    minFaceWeight           {quality.get('minFaceWeight', 0.02)};
    minVolRatio             {quality.get('minVolRatio', 0.01)};
    minTriangleTwist        {quality.get('minTriangleTwist', '-1')};
    nSmoothScale            {quality.get('nSmoothScale', 4)};
    errorReduction          {quality.get('errorReduction', 0.75)};'''

    def create_foam_file(self, iter_dir: Path):
        """Create .foam file for ParaView visualization."""
        foam_file = iter_dir / f"{iter_dir.name}.foam"
        foam_file.touch()
        self.logger.debug(f"Created .foam file: {foam_file}")

    def update_surface_levels(self, surface_levels: List[int]):
        """Update surface refinement levels for dictionary generation."""
        self.surface_levels = surface_levels