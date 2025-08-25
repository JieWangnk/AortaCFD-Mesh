# Installation Guide - AortaCFD-Snappy

## Quick Installation

### 1. Prerequisites
```bash
# Install OpenFOAM 12 (Foundation version)
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt-get update
sudo apt-get install openfoam12

# Install Python 3.8+
sudo apt-get install python3 python3-pip python3-dev

# Source OpenFOAM (add to ~/.bashrc for permanent)
source /opt/openfoam12/etc/bashrc
```

### 2. Install AortaCFD-Snappy
```bash
git clone https://github.com/YourUsername/AortaCFD-Snappy.git
cd AortaCFD-Snappy
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

## Troubleshooting

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
# If import errors
pip install --upgrade numpy scipy

# If permission errors
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
git clone https://github.com/YourUsername/AortaCFD-Snappy.git
cd AortaCFD-Snappy
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

- **Installation Issues**: [GitHub Issues](https://github.com/YourUsername/AortaCFD-Snappy/issues)
- **OpenFOAM Help**: [OpenFOAM Documentation](https://doc.openfoam.org/)
- **Python Help**: [Python Package Installation](https://packaging.python.org/tutorials/installing-packages/)