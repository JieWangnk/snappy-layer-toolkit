# snappyHexMesh Boundary Layer Toolkit

Automated boundary layer meshing for OpenFOAM with 90% fewer parameters needed.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-12-blue.svg)](https://openfoam.org/)

## Quick Start

```bash
# 1. Place your STL files in constant/geometry/
#    - wall_aorta.stl (or wall.stl) 
#    - inlet.stl
#    - outlet1.stl, outlet2.stl, etc.

# 2. Run with just 2 parameters!
python auto_layers.py --T-rel 0.35 --n-layers 5
```

**That's it!** Everything else is auto-detected:
- ✅ Wall and inlet patch names
- ✅ Interior point via ray-casting
- ✅ Base cell size from blockMeshDict  
- ✅ Outlet patches automatically

##  Key Benefits

- **90% Parameter Reduction**: From 10+ parameters to just 2
- **Auto-Detection**: No manual patch names or interior points
- **Validated Results**: 87-95% layer coverage consistently
- **Quick Iteration**: Reuse surface mesh for fast tuning

##  Validated Parameter Ranges

| Layers | Best T_rel | Expected Coverage |
|--------|------------|-------------------|
| 2-3    | 0.30-0.40  | 90-95% ✅        |
| 4-5    | 0.30-0.35  | 87-90% ✅        |
| 8-10   | 0.20-0.25  | 78-84% ⚠️        |

*Based on systematic testing with complex vascular geometry*

##  Installation

### Prerequisites
```bash
# OpenFOAM 12
which snappyHexMesh

# Python 3.7+ with NumPy
pip install numpy
```

### Usage
```bash
# Clone and run
git clone https://github.com/yourusername/snappy-layer-toolkit.git
cd snappy-layer-toolkit

# Basic usage (everything auto-detected)
python auto_layers.py --T-rel 0.35 --n-layers 5

# Parameter optimization
python auto_layers.py --sweep --T-rel-range 0.2 0.5 0.05 --n-layers 5

# Quick iteration (reuse surface mesh)
python auto_layers.py --T-rel 0.30 --n-layers 5 --quick

# Manual dictionary generation only
python scripts/generate_dicts.py --T-rel 0.35 --n-layers 5
```

##  How It Works

1. **Auto-Detection**: Scans `constant/geometry/` for STL files
2. **Ray-Casting**: Computes interior point automatically  
3. **Smart Defaults**: Uses empirically-validated parameters
4. **Two-Phase Meshing**: Surface mesh → Layer addition
5. **Honest Metrics**: Reports actual effective layers achieved

## 📁 Your Geometry Setup

```
your-case/
├── constant/geometry/
│   ├── wall.stl        # Main wall (auto-detected)
│   ├── inlet.stl       # Inlet boundary  
│   ├── outlet1.stl     # Outlets (auto-discovered)
│   └── outlet2.stl
├── system/
│   └── blockMeshDict   # Background mesh
└── # Run toolkit here
```

## Components

- `auto_layers.py` - Main automation script
- `scripts/generate_dicts.py` - Dictionary generator with auto-detection
- `scripts/find_interior_point.py` - Ray-casting point finder
- `scripts/metrics_logger.py` - Coverage analyzer
- `system/` - OpenFOAM template files

## Examples

### For New Users
```bash
# Conservative settings (best success rate)
python auto_layers.py --T-rel 0.35 --n-layers 3
```

### For Production
```bash
# Balanced performance
python auto_layers.py --T-rel 0.30 --n-layers 5 --levels 2 3
```

### For Research  
```bash
# High resolution (challenging)
python auto_layers.py --T-rel 0.20 --n-layers 10 --levels 2 3
```

##  Troubleshooting

**Auto-detection fails?**
```bash
# Override with your patch names
python auto_layers.py --T-rel 0.35 --n-layers 5 \
    --wall-name my_wall --inlet-name my_inlet
```

**Poor layer coverage?**
- Try lower T_rel (reduce by 20-30%)
- Use fewer layers with higher T_rel
- Increase surface refinement levels


## 🙏 Acknowledgments

- Validation geometry: [Vascular Model Repository](https://www.vascularmodel.com/)
- Built for the [OpenFOAM](https://openfoam.org/) community

## Citation

```bibtex
@software{snappy_layer_toolkit_2025,
  title={snappyHexMesh Boundary Layer Toolkit},
  author={Jie Wang}, 
  year={2025},
  url={https://github.com/JieWangnk/snappy-layer-toolkit}
}
```

---
**Transform your boundary layer meshing from hours to minutes!** 
