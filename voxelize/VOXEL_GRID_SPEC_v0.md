# Voxel Grid Specification v0

**Project**: WorldCraft - NeRF to Minecraft Converter  
**Component**: Voxelization Module  
**Version**: 0.1 (Week 1 Prototype)  
**Date**: October 2025  
**Author**: Voxelization Team

---

## 1. Overview

This document specifies the coordinate system, transformation rules, and data format for the dense semantic voxel grid that serves as the intermediate representation between NeRF scene reconstruction and Minecraft export.

---

## 2. Coordinate System Definition

### 2.1 World Coordinate Frame

**Axes Convention**: ENU (East-North-Up)
- **X-axis**: Points East (right when facing north)
- **Y-axis**: Points North (forward)
- **Z-axis**: Points Up (vertical)

**Handedness**: Right-handed coordinate system
- Cross product X × Y = Z
- Follows standard geospatial and robotics conventions
- Compatible with most NeRF implementations

**Units**: Meters (SI standard)

**Origin**: Defined per-scene at bounding box minimum corner
- Origin placement at bbox_min simplifies indexing (all voxel indices are non-negative)
- Alternative: scene center (may be used in future versions if needed)

### 2.2 Voxel Index Space

Voxel indices are discrete integer coordinates:
- **Range**: `[0, grid_size[i])` for each dimension i ∈ {0, 1, 2}
- **Indexing Order**: (X, Y, Z) following world axes
- **Storage**: NumPy arrays with shape `(X, Y, Z, ...)` using C-contiguous memory layout

---

## 3. Voxel Size and Resolution

### 3.1 Week 1 Parameters

**Voxel Size**: 0.10 - 0.20 meters per voxel
- **Week 1 Default**: 0.15 m
- **Rationale**: 
  - Smaller (0.10m): Higher fidelity but 2x memory/compute
  - Larger (0.20m): Faster but may lose architectural details
  - 0.15m provides good balance for prototype testing

### 3.2 Resolution Trade-offs

| Voxel Size | Voxels per 100m³ | Memory (approx) | Detail Level |
|------------|------------------|-----------------|--------------|
| 0.10 m     | 1,000,000        | ~8 MB           | High         |
| 0.15 m     | ~296,000         | ~2.4 MB         | Medium       |
| 0.20 m     | 125,000          | ~1 MB           | Low          |

### 3.3 Future Scaling

**Open Question**: Absolute scale consistency
- NeRF training often uses normalized scene coordinates
- Semantic-NeRF team must provide scale factor: `NeRF_units → meters`
- **Action Required**: Coordinate with upstream team to establish consistent scale

---

## 4. Bounding Box Strategy

### 4.1 Week 1 Approach: Fixed Cuboid

**Definition**: Axis-aligned bounding box (AABB) defined by:
```
bbox = [x_min, x_max, y_min, y_max, z_min, z_max]
```

**Week 1 Default**: `[-10, 10, -10, 10, 0, 20]`
- Creates 20m × 20m × 20m scene volume
- Suitable for single building prototyping

**Advantages**:
- Simple to specify and debug
- Predictable memory requirements
- Easy to visualize

**Disadvantages**:
- May include empty space
- Requires manual tuning per scene

### 4.2 Future Enhancement: Camera Frustum Union

**Concept**: Compute bbox from camera viewing frustums
- Parse camera intrinsics/extrinsics from NeRF training data
- Compute union of all camera frustums (or convex hull of camera origins)
- Automatically fits bbox to scene content

**Benefits**: Eliminates manual bbox specification, reduces wasted voxels

**Status**: Not implemented in v0 (add in future iteration)

---

## 5. Coordinate Transformations

### 5.1 World to Voxel Index

**Forward Transform**:
```
voxel_index[i] = floor((world_coord[i] - bbox_min[i]) / voxel_size)
```

Where:
- `world_coord` is 3D position in meters
- `bbox_min` is `[x_min, y_min, z_min]`
- `voxel_size` is scalar in meters
- `floor()` rounds down to nearest integer

**Example**:
```
world_coord = [2.3, 5.7, 8.1] m
bbox_min = [0, 0, 0] m
voxel_size = 0.15 m

voxel_index = floor([2.3/0.15, 5.7/0.15, 8.1/0.15])
            = floor([15.33, 38.0, 54.0])
            = [15, 38, 54]
```

### 5.2 Voxel Index to World (Center)

**Inverse Transform** (to voxel center):
```
world_coord[i] = bbox_min[i] + (voxel_index[i] + 0.5) * voxel_size
```

The `+0.5` offset places the coordinate at the voxel center.

**Example**:
```
voxel_index = [10, 20, 30]
bbox_min = [0, 0, 0] m
voxel_size = 0.15 m

world_coord = [0, 0, 0] + ([10, 20, 30] + 0.5) * 0.15
            = [1.575, 3.075, 4.575] m
```

### 5.3 Homogeneous Transformation Matrix

**4×4 World-to-Voxel Transform**:
```
T = [1/s   0     0     -x_min/s]
    [0     1/s   0     -y_min/s]
    [0     0     1/s   -z_min/s]
    [0     0     0     1        ]
```

Where `s = voxel_size`.

**Usage** (with homogeneous coordinates):
```python
world_homo = [x, y, z, 1]
voxel_homo = T @ world_homo
voxel_index = floor(voxel_homo[:3])
```

---

## 6. Data Format Specification

### 6.1 Output Files

All outputs stored in structured directory:

```
<output_dir>/
├── occupancy.npy       # Boolean grid
├── rgb.npy             # Color grid
├── semantic_id.npy     # Semantic class grid
└── meta.json           # Metadata
```

### 6.2 Array Specifications

**occupancy.npy**:
- **Shape**: `(X, Y, Z)`
- **Dtype**: `bool` (1 byte per voxel)
- **Meaning**: `True` if voxel is solid/occupied, `False` if empty

**rgb.npy**:
- **Shape**: `(X, Y, Z, 3)`
- **Dtype**: `uint8` (3 bytes per voxel)
- **Range**: [0, 255] for each channel
- **Order**: RGB (Red, Green, Blue)

**semantic_id.npy**:
- **Shape**: `(X, Y, Z)`
- **Dtype**: `int32` (4 bytes per voxel)
- **Range**: `[0, num_classes)`
- **Meaning**: Integer ID corresponding to semantic class (see label_set in meta.json)

### 6.3 Metadata Format (meta.json)

```json
{
  "scene_id": "string",
  "voxel_size_m": float,
  "bbox_world": {
    "min": [float, float, float],
    "max": [float, float, float]
  },
  "grid_size": [int, int, int],
  "world_to_voxel_transform": [[float, ...], ...],
  "coordinate_system": {
    "origin": "bbox_min",
    "axes": "ENU",
    "handedness": "right",
    "units": "meters"
  },
  "label_set": {
    "0": "class_name",
    ...
  },
  "color_encoding": "uint8_rgb",
  "density_threshold": float,
  "creation_date": "ISO8601 timestamp",
  "version": "string",
  "notes": "string"
}
```

---

## 7. Semantic Classes

### 7.1 Week 1 Label Set (Placeholder)

| ID | Label       | Description                    |
|----|-------------|--------------------------------|
| 0  | air/void    | Empty space                    |
| 1  | building    | Solid building structure       |
| 2  | vegetation  | Trees, grass, foliage          |

### 7.2 Future Expansion

Expected classes for UC Berkeley campus:
- Wall (brick, concrete, glass)
- Roof
- Window
- Door
- Ground/pavement
- Vegetation (trees, grass)
- Sky/void

**Action Required**: Coordinate with Semantic-NeRF team on final taxonomy

---

## 8. Semantic Decision Rules

### 8.1 Occupancy Threshold

**Method**: Density thresholding
```
occupancy = (density > threshold)
```

**Week 1 Threshold**: 0.5
- NeRF density typically in range [0, ∞)
- Threshold of 0.5 determined empirically for dummy data
- **Must be tuned** for real NeRF model

### 8.2 Semantic Class Selection

**Method**: Argmax over semantic logits

For a single sampling point at voxel center:
```python
semantic_id = argmax(semantic_logits)
```

Where `semantic_logits` is a K-dimensional vector (K = number of classes).

**Future Enhancement**: Multi-sample voting
- Sample multiple points within each voxel
- Use majority vote or max-probability aggregation
- More robust but computationally expensive

---

## 9. Memory and Performance

### 9.1 Memory Estimation

Per-voxel storage:
- Occupancy: 1 byte (bool)
- RGB: 3 bytes (uint8 × 3)
- Semantic: 4 bytes (int32)
- **Total**: 8 bytes/voxel

**Example Scenes**:

| Scene Dimensions | Voxels     | Memory  |
|------------------|------------|---------|
| 20×20×20 m @ 0.15m | 133³ ≈ 2.4M | ~19 MB  |
| 50×50×50 m @ 0.15m | 333³ ≈ 37M  | ~296 MB |
| 100×100×50 m @ 0.20m | 500×500×250 | ~500 MB |

### 9.2 Chunked Processing (Future)

For large scenes exceeding available RAM:
- Process grid in chunks (e.g., 32³ voxels at a time)
- Save chunks to disk as temp files
- Merge chunks in final step

**Not implemented in Week 1** (scenes kept small for prototype)

---

## 10. Open Questions and Assumptions

### 10.1 Scale Consistency ⚠️

**Critical Open Issue**: 
- NeRF models often trained in normalized coordinates (e.g., scene fits in [-1, 1]³)
- Real-world scale must be provided by Semantic-NeRF team
- **Assumption**: 1 NeRF unit = 1 meter (to be validated)

**Action**: Schedule meeting with Semantic-NeRF team to align on scale

### 10.2 Semantic-NeRF Integration

**Assumptions**:
- Semantic-NeRF outputs per-point semantic logits (K-dimensional vector)
- Semantics available at query time (not post-processed)
- Label set is consistent across all scenes

**Action**: Validate assumptions when real Semantic-NeRF checkpoint is available

### 10.3 Coordinate Frame Alignment

**Assumption**: 
- NeRF training uses same ENU coordinate frame
- If NeRF uses different convention (e.g., OpenGL: right-up-back), rotation matrix needed

**Action**: Inspect NeRF camera parameters to verify alignment

---

## 11. Validation and Testing

### 11.1 Week 1 Validation

**Visual Inspection**:
- Generate XY, XZ, YZ slice PNGs
- Verify occupancy and RGB patterns match expected geometry
- Check semantic labels are reasonable

**Sanity Checks**:
- Grid size matches expected (derived from bbox and voxel_size)
- Occupancy rate in reasonable range (not 0% or 100%)
- RGB values have variation (not all black or all white)
- meta.json loads without errors

### 11.2 Future Testing

- Quantitative metrics: IoU with ground truth (if available)
- Visual comparison with NeRF rendering from same viewpoint
- End-to-end test: voxelize → export → import to Minecraft

---

## 12. Version History

- **v0.1** (Week 1): Initial specification with dummy NeRF data

---

## 13. References

- NeRF paper: Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", ECCV 2020
- Coordinate conventions: REP 103 (ROS Enhancement Proposal for coordinate frames)
- NumPy array format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html