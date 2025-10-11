"""
NeRF to Voxel Grid Converter
Converts a NeRF scene representation into a dense semantic voxel grid.
"""

import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


class VoxelGrid:
    """Represents a 3D voxel grid with associated metadata."""
    
    def __init__(self, voxel_size, bbox_min, bbox_max):
        """
        Initialize voxel grid parameters.
        
        Args:
            voxel_size: Size of each voxel in meters
            bbox_min: [x_min, y_min, z_min] in world coordinates
            bbox_max: [x_max, y_max, z_max] in world coordinates
        """
        self.voxel_size = voxel_size
        self.bbox_min = np.array(bbox_min)
        self.bbox_max = np.array(bbox_max)
        
        # Calculate grid dimensions
        self.grid_size = self._compute_grid_size()
        
        print(f"Grid size: {self.grid_size}")
        print(f"Total voxels: {np.prod(self.grid_size):,}")
        print(f"Memory estimate: ~{self._estimate_memory_mb():.1f} MB")
    
    def _compute_grid_size(self):
        """Compute number of voxels needed in each dimension."""
        lengths = self.bbox_max - self.bbox_min
        grid_size = np.ceil(lengths / self.voxel_size).astype(int)
        return grid_size
    
    def _estimate_memory_mb(self):
        """Estimate memory usage for all arrays."""
        n_voxels = np.prod(self.grid_size)
        # occupancy (bool=1) + rgb (3*uint8) + semantic (int32=4) = 8 bytes per voxel
        return n_voxels * 8 / (1024 ** 2)
    
    def world_to_voxel(self, coords):
        """
        Convert world coordinates to voxel indices.
        
        Args:
            coords: [..., 3] array of world coordinates
        
        Returns:
            [..., 3] array of voxel indices
        """
        indices = np.floor((coords - self.bbox_min) / self.voxel_size).astype(int)
        return indices
    
    def voxel_to_world(self, indices):
        """
        Convert voxel indices to world coordinates (voxel centers).
        
        Args:
            indices: [..., 3] array of voxel indices
        
        Returns:
            [..., 3] array of world coordinates
        """
        coords = self.bbox_min + (indices + 0.5) * self.voxel_size
        return coords
    
    def get_transform_matrix(self):
        """
        Get 4x4 world-to-voxel transformation matrix.
        
        Returns:
            4x4 numpy array (homogeneous coordinates)
        """
        scale = 1.0 / self.voxel_size
        tx, ty, tz = -self.bbox_min * scale
        
        matrix = np.array([
            [scale, 0,     0,     tx],
            [0,     scale, 0,     ty],
            [0,     0,     scale, tz],
            [0,     0,     0,     1]
        ])
        return matrix
    
    def create_sampling_grid(self):
        """
        Create a 3D grid of world coordinates for sampling.
        
        Returns:
            Grid of shape (X, Y, Z, 3) with world coordinates at each voxel center
        """
        x_indices = np.arange(self.grid_size[0])
        y_indices = np.arange(self.grid_size[1])
        z_indices = np.arange(self.grid_size[2])
        
        # Create meshgrid
        grid_x, grid_y, grid_z = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        
        # Stack into (X, Y, Z, 3)
        indices = np.stack([grid_x, grid_y, grid_z], axis=-1)
        
        # Convert to world coordinates
        coords = self.voxel_to_world(indices)
        
        return coords


class DummyNeRF:
    """Dummy NeRF model for testing (replace with real model later)."""
    
    def __init__(self, center=None):
        """
        Initialize dummy NeRF.
        
        Args:
            center: [x, y, z] center point for test geometry
        """
        self.center = np.array(center if center is not None else [0, 0, 10])
    
    def query(self, coords):
        """
        Query NeRF at given coordinates.
        
        Args:
            coords: [..., 3] array of world coordinates
        
        Returns:
            density: [...] array of density values
            rgb: [..., 3] array of RGB colors [0, 1]
            semantic_logits: [..., K] array of semantic logits
        """
        shape = coords.shape[:-1]
        
        # Create a sphere centered at self.center
        distances = np.linalg.norm(coords - self.center, axis=-1)
        radius = 5.0
        
        # Density: gaussian falloff
        density = 3.0 * np.exp(-((distances / radius) ** 2))
        
        # RGB: position-based gradient
        rgb = np.zeros((*shape, 3))
        rgb[..., 0] = np.clip((coords[..., 0] + 10) / 20, 0, 1)  # R: x gradient
        rgb[..., 1] = np.clip((coords[..., 1] + 10) / 20, 0, 1)  # G: y gradient
        rgb[..., 2] = np.clip((coords[..., 2]) / 20, 0, 1)       # B: z gradient
        
        # Semantic: simple rule (0=air, 1=building, 2=vegetation)
        semantic_logits = np.zeros((*shape, 3))
        semantic_logits[..., 0] = 5.0  # default to air
        semantic_logits[..., 1] = np.where(density > 1.0, 10.0, 0.0)  # building if dense
        semantic_logits[..., 2] = np.where((density > 0.5) & (density <= 1.0), 8.0, 0.0)  # vegetation
        
        return density, rgb, semantic_logits


class Voxelizer:
    """Main voxelization engine."""
    
    def __init__(self, grid, nerf_model, density_threshold=0.5):
        """
        Initialize voxelizer.
        
        Args:
            grid: VoxelGrid instance
            nerf_model: NeRF model with query() method
            density_threshold: Threshold for occupancy determination
        """
        self.grid = grid
        self.nerf_model = nerf_model
        self.density_threshold = density_threshold
    
    def voxelize(self, chunk_size=None):
        """
        Perform voxelization.
        
        Args:
            chunk_size: If specified, process in chunks to avoid OOM
        
        Returns:
            occupancy: (X, Y, Z) boolean array
            rgb: (X, Y, Z, 3) uint8 array
            semantic_id: (X, Y, Z) int array
        """
        if chunk_size is None:
            # Process entire grid at once
            return self._voxelize_full()
        else:
            # Process in chunks
            return self._voxelize_chunked(chunk_size)
    
    def _voxelize_full(self):
        """Voxelize entire grid at once."""
        print("Creating sampling grid...")
        coords = self.grid.create_sampling_grid()
        
        print("Querying NeRF...")
        density, rgb, semantic_logits = self.nerf_model.query(coords)
        
        print("Processing results...")
        # Determine occupancy
        occupancy = density > self.density_threshold
        
        # Convert RGB to uint8
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        
        # Determine semantic class (argmax)
        semantic_id = np.argmax(semantic_logits, axis=-1).astype(np.int32)
        
        return occupancy, rgb_uint8, semantic_id
    
    def _voxelize_chunked(self, chunk_size):
        """Voxelize in chunks to reduce memory usage."""
        # Not implemented for week 1 prototype
        raise NotImplementedError("Chunked processing will be added if needed")


def save_outputs(output_dir, grid, occupancy, rgb, semantic_id, label_map):
    """
    Save voxelization outputs to disk.
    
    Args:
        output_dir: Output directory path
        grid: VoxelGrid instance
        occupancy: Occupancy array
        rgb: RGB array
        semantic_id: Semantic ID array
        label_map: Dictionary mapping IDs to labels
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_path}...")
    
    # Save numpy arrays
    np.save(output_path / "occupancy.npy", occupancy)
    np.save(output_path / "rgb.npy", rgb)
    np.save(output_path / "semantic_id.npy", semantic_id)
    print("✓ Saved .npy files")
    
    # Create metadata
    meta = {
        "scene_id": output_path.name,
        "voxel_size_m": float(grid.voxel_size),
        "bbox_world": {
            "min": grid.bbox_min.tolist(),
            "max": grid.bbox_max.tolist()
        },
        "grid_size": grid.grid_size.tolist(),
        "world_to_voxel_transform": grid.get_transform_matrix().tolist(),
        "coordinate_system": {
            "origin": "bbox_min",
            "axes": "ENU (East-North-Up)",
            "handedness": "right",
            "units": "meters"
        },
        "label_set": label_map,
        "color_encoding": "uint8_rgb",
        "density_threshold": 0.5,
        "creation_date": datetime.now().isoformat(),
        "version": "v0.1_week1_prototype",
        "notes": "Week 1 prototype using dummy NeRF data"
    }
    
    with open(output_path / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print("✓ Saved meta.json")
    
    # Statistics
    occupied_count = occupancy.sum()
    total_count = occupancy.size
    print(f"\nStatistics:")
    print(f"  Occupied voxels: {occupied_count:,} / {total_count:,}")
    print(f"  Occupancy rate: {100 * occupied_count / total_count:.2f}%")


def visualize_slices(output_dir, grid, occupancy, rgb, semantic_id, label_map):
    """
    Generate slice visualizations for quick inspection.
    
    Args:
        output_dir: Output directory path
        grid: VoxelGrid instance
        occupancy: Occupancy array
        rgb: RGB array
        semantic_id: Semantic ID array
        label_map: Dictionary mapping IDs to labels
    """
    output_path = Path(output_dir)
    
    print("\nGenerating visualizations...")
    
    # Select middle slices
    mid_x = grid.grid_size[0] // 2
    mid_y = grid.grid_size[1] // 2
    mid_z = grid.grid_size[2] // 2
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # XY plane (top view)
    axes[0, 0].imshow(occupancy[:, :, mid_z].T, cmap='gray', origin='lower')
    axes[0, 0].set_title(f'Occupancy XY (z={mid_z})')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    
    axes[0, 1].imshow(rgb[:, :, mid_z, :].transpose(1, 0, 2), origin='lower')
    axes[0, 1].set_title(f'RGB XY (z={mid_z})')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    
    axes[0, 2].imshow(semantic_id[:, :, mid_z].T, cmap='tab10', origin='lower', vmin=0, vmax=9)
    axes[0, 2].set_title(f'Semantic XY (z={mid_z})')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    
    # XZ plane (side view)
    axes[1, 0].imshow(occupancy[:, mid_y, :].T, cmap='gray', origin='lower')
    axes[1, 0].set_title(f'Occupancy XZ (y={mid_y})')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Z')
    
    axes[1, 1].imshow(rgb[:, mid_y, :, :].transpose(1, 0, 2), origin='lower')
    axes[1, 1].set_title(f'RGB XZ (y={mid_y})')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Z')
    
    axes[1, 2].imshow(semantic_id[:, mid_y, :].T, cmap='tab10', origin='lower', vmin=0, vmax=9)
    axes[1, 2].set_title(f'Semantic XZ (y={mid_y})')
    axes[1, 2].set_xlabel('X')
    axes[1, 2].set_ylabel('Z')
    
    # YZ plane (front view)
    axes[2, 0].imshow(occupancy[mid_x, :, :].T, cmap='gray', origin='lower')
    axes[2, 0].set_title(f'Occupancy YZ (x={mid_x})')
    axes[2, 0].set_xlabel('Y')
    axes[2, 0].set_ylabel('Z')
    
    axes[2, 1].imshow(rgb[mid_x, :, :, :].transpose(1, 0, 2), origin='lower')
    axes[2, 1].set_title(f'RGB YZ (x={mid_x})')
    axes[2, 1].set_xlabel('Y')
    axes[2, 1].set_ylabel('Z')
    
    axes[2, 2].imshow(semantic_id[mid_x, :, :].T, cmap='tab10', origin='lower', vmin=0, vmax=9)
    axes[2, 2].set_title(f'Semantic YZ (x={mid_x})')
    axes[2, 2].set_xlabel('Y')
    axes[2, 2].set_ylabel('Z')
    
    plt.tight_layout()
    plt.savefig(output_path / "slices_visualization.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved slices_visualization.png")
    plt.close()


def main():
    """Main execution function."""
    
    # Parse arguments (simplified for week 1)
    parser = argparse.ArgumentParser(description='Convert NeRF to voxel grid')
    parser.add_argument('--voxel_size', type=float, default=0.15,
                        help='Voxel size in meters (default: 0.15)')
    parser.add_argument('--bbox', type=float, nargs=6, 
                        default=[-10, 10, -10, 10, 0, 20],
                        help='Bounding box: xmin xmax ymin ymax zmin zmax')
    parser.add_argument('--out', type=str, default='outputs/scene_001',
                        help='Output directory')
    
    args = parser.parse_args()
    
    print("="*70)
    print("NeRF to Voxel Grid Converter - Week 1 Prototype")
    print("="*70)
    
    # Setup label map
    label_map = {
        "0": "air/void",
        "1": "building",
        "2": "vegetation"
    }
    
    # Initialize grid
    bbox_min = args.bbox[::2]  # [xmin, ymin, zmin]
    bbox_max = args.bbox[1::2]  # [xmax, ymax, zmax]
    grid = VoxelGrid(args.voxel_size, bbox_min, bbox_max)
    
    # Initialize dummy NeRF
    nerf_model = DummyNeRF(center=[0, 0, 10])
    
    # Initialize voxelizer
    voxelizer = Voxelizer(grid, nerf_model, density_threshold=0.5)
    
    # Run voxelization
    occupancy, rgb, semantic_id = voxelizer.voxelize()
    
    # Save outputs
    save_outputs(args.out, grid, occupancy, rgb, semantic_id, label_map)
    
    # Generate visualizations
    visualize_slices(args.out, grid, occupancy, rgb, semantic_id, label_map)
    
    print("\n" + "="*70)
    print("✓ Complete! Check outputs in:", args.out)
    print("="*70)


if __name__ == "__main__":
    main()