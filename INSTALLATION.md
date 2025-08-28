# Installation Guide - AortaCFD-Mesh

## Quick Installation

### 1. Prerequisites

#### OpenFOAM Version Compatibility
AortaCFD-Mesh automatically detects your OpenFOAM version and adapts to the correct directory structure:
- **OpenFOAM v12+**: Uses `constant/geometry/` directory (recommended)
- **OpenFOAM v11 and earlier**: Uses `constant/triSurface/` directory  

```bash
# Install OpenFOAM 12 (Foundation version) - Recommended
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt-get update
sudo apt-get install openfoam12

# Alternative: OpenFOAM 11 (also supported)
# sudo apt-get install openfoam11

# Install Python 3.8+
sudo apt-get install python3 python3-pip python3-dev

# Source OpenFOAM (add to ~/.bashrc for permanent)
source /opt/openfoam12/etc/bashrc
```

### 2. Install AortaCFD-Mesh

#### Method A: Virtual Environment Setup (Recommended)
```bash
git clone https://github.com/YourUsername/AortaCFD-Mesh.git
cd AortaCFD-Mesh

# Automated setup
./setup_venv.sh

# Daily usage
source activate.sh
```

#### Method B: System Installation
```bash
git clone https://github.com/YourUsername/AortaCFD-Mesh.git
cd AortaCFD-Mesh
pip install -r requirements.txt
```

### 3. Test Installation
```bash
# Test CLI interface
python -m mesh_optim stage1 --help

# Test with tutorial data
python -m mesh_optim stage1 --geometry tutorial/patient1 --max-iterations 1
```

## Verify Installation

### OpenFOAM Check
```bash
which blockMesh                    # Should show: /opt/openfoam12/bin/blockMesh
which snappyHexMesh               # Should show: /opt/openfoam12/bin/snappyHexMesh
echo $FOAM_VERSION                # Should show: 12
```

### Python Dependencies
```bash
python -c "import numpy, scipy; print('Dependencies OK')"
```

### Tutorial Data
```bash
ls tutorial/patient1/             # Should show: inlet.stl, outlet*.stl, wall_aorta.stl, BPM75.csv, config.json
```

## System Requirements

- **Operating System**: Ubuntu 18.04+, CentOS 7+, macOS 10.15+
- **Memory**: 8 GB RAM minimum, 16 GB recommended
- **Storage**: 5 GB free space for temporary files
- **Processors**: Multi-core recommended for large meshes

## Virtual Environment Setup

### Manual Virtual Environment Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Source OpenFOAM
source /opt/openfoam12/etc/bashrc

# Test installation
python -m mesh_optim --help

# Deactivate when done
deactivate
```

### Virtual Environment Benefits
- **Isolation**: No conflicts with system Python packages
- **Reproducibility**: Exact package versions across different machines
- **Clean uninstall**: Just delete the `venv` directory
- **Multiple projects**: Different environments for different projects

## Troubleshooting

### Virtual Environment Issues
```bash
# If venv creation fails
sudo apt-get install python3-venv python3-pip

# If setup_venv.sh permission denied
chmod +x setup_venv.sh

# If activate.sh doesn't work
source venv/bin/activate
source /opt/openfoam12/etc/bashrc
```

### OpenFOAM Issues
```bash
# If blockMesh not found
echo $PATH | grep foam            # Should show OpenFOAM paths
source /opt/openfoam12/etc/bashrc # Re-source if needed

# If version conflicts
which foam                        # Check OpenFOAM installation
foam                             # Should start OpenFOAM shell
```

### Python Issues
```bash
# If "No module named 'psutil'" error
source venv/bin/activate          # Activate venv first
pip install -r requirements.txt

# If import errors
pip install --upgrade numpy scipy psutil

# If permission errors (system install)
pip install --user -r requirements.txt
```

### File Permission Issues
```bash
# Fix script permissions
chmod +x scripts/*

# Fix output directory permissions
chmod -R 755 output/
```

## Development Installation

For development and testing:
```bash
git clone https://github.com/YourUsername/AortaCFD-Mesh.git
cd AortaCFD-Mesh
pip install -e .                 # Editable install
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests
python -m pytest tests/

# Format code
black mesh_optim/

# Lint code
flake8 mesh_optim/
```

## Next Steps

After installation:
1. **Read**: [README.md](README.md) for usage guide
2. **Try**: Tutorial example in `tutorial/patient1/`
3. **Learn**: Review configuration files in `mesh_optim/configs/`
4. **Optimize**: Your patient data with Stage 1 or Stage 2

## Support

- **Installation Issues**: [GitHub Issues](https://github.com/YourUsername/AortaCFD-Mesh/issues)
- **OpenFOAM Help**: [OpenFOAM Documentation](https://doc.openfoam.org/)
- **Python Help**: [Python Package Installation](https://packaging.python.org/tutorials/installing-packages/)