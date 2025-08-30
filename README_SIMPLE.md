# Simple SnappyHexMesh Generator

A streamlined mesh generation tool that removes all optimization, iteration, and quality checking loops from the original AortaCFD-Mesh system. This provides pure snappyHexMesh automation based on JSON configuration.

## What Was Removed

✅ **Removed optimization components:**
- `iteration_manager.py` - optimization loops  
- `mesh_orchestrator.py` - optimization orchestration
- `micro_tuning.py` - micro-optimization logic
- `layer_diagnostics.py` - layer optimization analysis  
- `mesh_quality.py` - quality-based iteration decisions
- `error_detection.py` - retry/recovery logic
- `stage1_mesh.py` - optimization wrapper
- All iteration loops, quality checking, retry mechanisms

✅ **What remains:**
- Single-pass snappyHexMesh execution
- JSON configuration parsing
- STL geometry processing  
- OpenFOAM dictionary generation
- Basic mesh quality reporting (post-generation)

## Usage

### Basic mesh generation:
```bash
python simple_mesh_generator.py \
  --geometry tutorial/patient1/ \
  --config simple_config_example.json \
  --output output_mesh/
```

### Create configuration template:
```bash
python simple_mesh_generator.py --create-config my_config.json
```

## Configuration

The tool uses a simplified JSON configuration format:

```json
{
  "description": "Simple mesh configuration",
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
    "relativeSizes": true,
    "minThickness": 0.10,
    "first_layer": {
      "t1_rel": 0.50
    }
  }
}
```

## Key Features

- **Single-pass execution**: No optimization loops or iterations
- **Binary STL support**: Reads binary STL files correctly  
- **Automatic cell sizing**: Calculates from geometry and config
- **Layer support**: Generates boundary layers if configured
- **Proper OpenFOAM**: Generates valid dictionaries for v12
- **Quality reporting**: Shows mesh quality metrics after generation

## Output

The tool creates a complete OpenFOAM case with:
- `constant/polyMesh/` - Generated mesh files
- `system/` - OpenFOAM dictionaries (blockMeshDict, snappyHexMeshDict, controlDict)
- `constant/triSurface/` - Copied STL geometry files
- `*.foam` file for ParaView visualization

## Example Results

With the patient1 geometry and simple config:
- **Background cells**: 34,200 (19×30×60)
- **Final mesh**: ~65,000 cells with boundary layers
- **Mesh quality**: Non-orthogonality <51°, Skewness <2.8
- **Generation time**: ~2 minutes

## Differences from Original System

| Original AortaCFD-Mesh | Simple Generator |
|------------------------|------------------|
| Multiple iterations with quality checking | Single-pass execution |
| Complex optimization loops | Direct snappyHexMesh run |
| Automatic parameter tuning | Fixed parameters from config |
| Error detection and recovery | Fail-fast on errors |
| Detailed diagnostics | Basic quality check |
| ~15 Python modules | 1 self-contained script |

## Requirements

- OpenFOAM v12 (installed and sourced)
- Python 3.8+ with numpy
- STL geometry files

## Validation

Successfully tested with:
- ✅ Binary STL files (patient1 aorta geometry)
- ✅ Multiple outlets (inlet + 4 outlets + wall)
- ✅ Boundary layer generation (4 layers)
- ✅ Surface refinement (levels 1-2)
- ✅ Mesh quality validation (checkMesh passes)

This simplified tool provides reliable mesh generation without the complexity of optimization algorithms.