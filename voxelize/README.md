# NeRF to Voxel Grid Converter

Week 1 prototype for converting NeRF scene representations into dense semantic voxel grids for Minecraft export.

## Installation

### Requirements
- Python 3.8+
- NumPy
- Matplotlib

### Setup
```bash
# Clone the repository (or create project directory)
mkdir worldcraft_voxel
cd worldcraft_voxel

# Install dependencies
pip install -r requirements.txt

# Copy nerf_to_voxel.py to this directory
```

## Usage

### Basic Usage (with default parameters)
```bash
python nerf_to_voxel.py
```

This will create outputs in `outputs/scene_001/` with default settings:
- Voxel size: 0.15m
- Bounding box: [-10, 10] x [-10, 10] x [0, 20] meters
- Uses dummy NeRF data for testing

### Custom Parameters
```bash
python nerf_to_voxel.py \
  --voxel_size 0.20 \
  --bbox -5 5 -5 5 0 15 \
  --out outputs/test_scene
```

### Arguments
- `--voxel_size`: Size of each voxel in meters (default: 0.15)
- `--bbox`: Bounding box as 6 floats: xmin xmax ymin ymax zmin zmax (default: -10 10 -10 10 0 20)
- `--out`: Output directory path (default: outputs/scene_001)

## Output Format

The script generates a structured directory containing:

```
outputs/scene_001/
├── occupancy.npy          # Boolean array (X, Y, Z) - True if voxel is solid
├── rgb.npy                # uint8 array (X, Y, Z, 3) - RGB color [0-255]
├── semantic_id.npy        # int32 array (X, Y, Z) - Semantic class ID
├── meta.json              # Metadata (grid spec, transforms, labels)
└── slices_visualization.png  # Quick visual check (XY, XZ, YZ slices)
```

### Loading Outputs

```python
import numpy as np
import json

# Load voxel data
occupancy = np.load('outputs/scene_001/occupancy.npy')
rgb = np.load('outputs/scene_001/rgb.npy')
semantic_id = np.load('outputs/scene_001/semantic_id.npy')

# Load metadata
with open('outputs/scene_001/meta.json') as f:
    meta = json.load(f)

print(f"Grid size: {meta['grid_size']}")
print(f"Voxel size: {meta['voxel_size_m']}m")
```

## Coordinate System

- **Origin**: Bounding box minimum corner
- **Axes**: ENU (East-North-Up) convention
  - X-axis: East (right)
  - Y-axis: North (forward)
  - Z-axis: Up
- **Handedness**: Right-handed coordinate system
- **Units**: Meters

### Coordinate Transformations

World to Voxel Index:
```
voxel_index = floor((world_coord - bbox_min) / voxel_size)
```

Voxel Index to World (center):
```
world_coord = bbox_min + (voxel_index + 0.5) * voxel_size
```

## Semantic Labels (Week 1)

Current label set (placeholder for testing):
- `0`: air/void
- `1`: building
- `2`: vegetation

This will be updated when real Semantic-NeRF data is available.

## Current Limitations (Week 1 Prototype)

1. **Dummy NeRF**: Uses synthetic test data instead of real NeRF model
2. **No Chunking**: Processes entire grid in memory (works for small scenes <100x100x100 voxels)
3. **Simple Semantics**: 3 placeholder classes only
4. **Fixed Threshold**: Occupancy threshold hardcoded to 0.5

## Next Steps

- [ ] Integrate with actual Semantic-NeRF checkpoint loading
- [ ] Implement chunked processing for large scenes
- [ ] Add CLI for NeRF checkpoint path and semantic label map
- [ ] Optimize memory usage
- [ ] Add progress bars for long-running operations

## Integration with Export Team

The Export team can use the outputs as follows:

```python
# Example: Read voxel grid and convert to Minecraft
import numpy as np

occupancy = np.load('outputs/scene_001/occupancy.npy')
rgb = np.load('outputs/scene_001/rgb.npy')
semantic_id = np.load('outputs/scene_001/semantic_id.npy')

# Iterate through occupied voxels
for x in range(occupancy.shape[0]):
    for y in range(occupancy.shape[1]):
        for z in range(occupancy.shape[2]):
            if occupancy[x, y, z]:
                color = rgb[x, y, z]  # RGB values
                block_type = semantic_id[x, y, z]  # Semantic class
                # Convert to Minecraft block...
```

## Troubleshooting

**Memory Error**: Reduce voxel size or bounding box dimensions
```bash
python nerf_to_voxel.py --voxel_size 0.20 --bbox -5 5 -5 5 0 10
```

**Import Error**: Install dependencies
```bash
pip install numpy matplotlib
```

## Contact

For questions or issues, contact the Voxelization subteam during weekly meetings.

## Version

- **Version**: v0.1
- **Date**: Week 1 Prototype
- **Status**: Testing with dummy data