#!/usr/bin/env python3
"""
Standalone interior point finder using ray-casting algorithm.

This script finds a point inside complex geometries using a robust ray-casting
approach and caches the result to avoid repeated computation.

Usage:
    # Find point and save to cache
    python scripts/find_interior_point.py --wall-name wall_aorta --inlet-name inlet
    
    # Use cached result
    python scripts/find_interior_point.py --use-cached
    
    # Force recalculation
    python scripts/find_interior_point.py --wall-name wall_aorta --inlet-name inlet --force
"""

import argparse
import json
import re
import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict


def read_stl_triangles(stl_path: Path) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Read STL triangles and return list of (normal, v1, v2, v3) tuples.
    Handles both ASCII and binary STL formats.
    """
    triangles = []
    
    try:
        # Try ASCII first
        with open(stl_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            if not line.lower().strip().startswith('solid'):
                raise UnicodeDecodeError("", b"", 0, 0, "not ascii")
            
            f.seek(0)
            current_normal = None
            vertices = []
            
            for line in f:
                line = line.strip().lower()
                
                if line.startswith('facet normal'):
                    parts = line.split()
                    try:
                        current_normal = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                    except (IndexError, ValueError):
                        current_normal = None
                        
                elif line.startswith('vertex'):
                    parts = line.split()
                    try:
                        vertex = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                        vertices.append(vertex)
                    except (IndexError, ValueError):
                        continue
                        
                elif line.startswith('endfacet'):
                    if len(vertices) == 3:
                        v1, v2, v3 = vertices
                        
                        # Compute normal if not provided or zero
                        if current_normal is None or np.linalg.norm(current_normal) < 1e-12:
                            edge1 = v2 - v1
                            edge2 = v3 - v1
                            computed_normal = np.cross(edge1, edge2)
                            norm = np.linalg.norm(computed_normal)
                            if norm > 1e-12:
                                current_normal = computed_normal / norm
                            else:
                                current_normal = np.array([0.0, 0.0, 1.0])
                        
                        triangles.append((current_normal, v1, v2, v3))
                    
                    current_normal = None
                    vertices = []
                    
    except UnicodeDecodeError:
        # Binary STL
        triangles = []
        with open(stl_path, 'rb') as f:
            # Skip header
            f.seek(80)
            
            # Read number of triangles
            n_triangles = struct.unpack('<I', f.read(4))[0]
            
            for _ in range(n_triangles):
                # Read normal
                normal = np.array(struct.unpack('<3f', f.read(12)))
                
                # Read vertices
                v1 = np.array(struct.unpack('<3f', f.read(12)))
                v2 = np.array(struct.unpack('<3f', f.read(12)))
                v3 = np.array(struct.unpack('<3f', f.read(12)))
                
                # Skip attribute byte count
                f.read(2)
                
                # Compute normal if zero
                if np.linalg.norm(normal) < 1e-12:
                    edge1 = v2 - v1
                    edge2 = v3 - v1
                    computed = np.cross(edge1, edge2)
                    norm = np.linalg.norm(computed)
                    if norm > 1e-12:
                        normal = computed / norm
                    else:
                        normal = np.array([0.0, 0.0, 1.0])
                
                triangles.append((normal, v1, v2, v3))
    
    return triangles


def discover_outlets(geometry_dir: Path) -> List[Tuple[str, str]]:
    """Discover outlet STL files automatically."""
    outlets = []
    if not geometry_dir.is_dir():
        return outlets
    for entry in sorted(geometry_dir.iterdir()):
        if entry.is_file() and entry.name.lower().endswith(".stl"):
            m = re.match(r"outlet(\d*)\.stl", entry.name, re.IGNORECASE)
            if m:
                idx_str = m.group(1) or ""
                patch_name = f"outlet{idx_str}" if idx_str else "outlet"
                outlets.append((patch_name, entry.name))
    return outlets


def count_ray_intersections(origin: np.ndarray, direction: np.ndarray, 
                           triangles: List[Tuple]) -> int:
    """Count intersections using Möller-Trumbore algorithm."""
    intersection_count = 0
    epsilon = 1e-10
    
    for triangle in triangles:
        _, v1, v2, v3 = triangle
        
        # Möller-Trumbore algorithm
        edge1 = v2 - v1
        edge2 = v3 - v1
        h = np.cross(direction, edge2)
        a = np.dot(edge1, h)
        
        if abs(a) < epsilon:
            continue
            
        f = 1.0 / a
        s = origin - v1
        u = f * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            continue
            
        q = np.cross(s, edge1)
        v = f * np.dot(direction, q)
        
        if v < 0.0 or u + v > 1.0:
            continue
            
        t = f * np.dot(edge2, q)
        
        if t > epsilon:
            intersection_count += 1
            
    return intersection_count


def is_point_inside(point: np.ndarray, triangles: List[Tuple], 
                    ray_directions: int = 3) -> bool:
    """Test if point is inside using ray-casting with consensus."""
    directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
        np.array([-1.0, 1.0, 0.0]) / np.sqrt(2),
    ]
    
    inside_votes = 0
    
    for i in range(min(ray_directions, len(directions))):
        direction = directions[i]
        intersection_count = count_ray_intersections(point, direction, triangles)
        
        if intersection_count % 2 == 1:
            inside_votes += 1
            
    return inside_votes > ray_directions // 2


def find_interior_point(geometry_dir: Path, wall_name: str, inlet_name: str,
                       verbose: bool = True) -> Dict:
    """Find interior point and return metadata."""
    if verbose:
        print("\nFinding interior point using ray-casting algorithm...")
    
    # Discover outlets
    outlets = discover_outlets(geometry_dir)
    
    # Load all STL files
    all_triangles = []
    bbox_min = np.array([float('inf')] * 3)
    bbox_max = np.array([float('-inf')] * 3)
    
    stl_files = []
    
    # Wall
    wall_path = geometry_dir / f"{wall_name}.stl"
    if wall_path.exists():
        stl_files.append(wall_path)
    
    # Inlet
    inlet_path = geometry_dir / f"{inlet_name}.stl"
    if inlet_path.exists():
        stl_files.append(inlet_path)
    
    # Outlets
    for patch_name, filename in outlets:
        outlet_path = geometry_dir / filename
        if outlet_path.exists():
            stl_files.append(outlet_path)
    
    if not stl_files:
        raise ValueError(f"No STL files found in {geometry_dir}")
    
    # Load triangles and compute bounding box
    for stl_path in stl_files:
        if verbose:
            print(f"  Loading {stl_path.name}...")
        triangles = read_stl_triangles(stl_path)
        all_triangles.extend(triangles)
        
        # Update bounding box
        for _, v1, v2, v3 in triangles:
            for vertex in [v1, v2, v3]:
                bbox_min = np.minimum(bbox_min, vertex)
                bbox_max = np.maximum(bbox_max, vertex)
    
    if verbose:
        print(f"  Loaded {len(all_triangles)} triangles total")
        print(f"  Bounding box: [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}] to "
              f"[{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]")
    
    # Define sampling region
    margin = 0.1 * (bbox_max - bbox_min)
    sample_min = bbox_min + margin
    sample_max = bbox_max - margin
    
    # Try random candidates
    max_candidates = 100
    for i in range(max_candidates):
        candidate = np.random.uniform(sample_min, sample_max)
        
        if is_point_inside(candidate, all_triangles):
            if verbose:
                print(f"  Interior point found after {i+1} candidates: "
                      f"({candidate[0]:.6f}, {candidate[1]:.6f}, {candidate[2]:.6f})")
            
            return {
                'point': candidate.tolist(),
                'bounding_box': {
                    'min': bbox_min.tolist(),
                    'max': bbox_max.tolist()
                },
                'geometry_files': [str(p.name) for p in stl_files],
                'triangle_count': len(all_triangles),
                'attempts': i + 1,
                'method': 'random_sampling'
            }
    
    # Fallback: grid search
    if verbose:
        print("  Random sampling failed, trying grid search...")
    
    grid_size = 20
    x_vals = np.linspace(sample_min[0], sample_max[0], grid_size)
    y_vals = np.linspace(sample_min[1], sample_max[1], grid_size)
    z_vals = np.linspace(sample_min[2], sample_max[2], grid_size)
    
    # Center-out search
    def center_indices(n):
        center = n // 2
        indices = [center]
        for i in range(1, n):
            if center + i < n:
                indices.append(center + i)
            if center - i >= 0:
                indices.append(center - i)
        return indices
    
    x_indices = center_indices(len(x_vals))
    y_indices = center_indices(len(y_vals))
    z_indices = center_indices(len(z_vals))
    
    candidates_tested = 0
    for xi in x_indices:
        for yi in y_indices:
            for zi in z_indices:
                candidate = np.array([x_vals[xi], y_vals[yi], z_vals[zi]])
                candidates_tested += 1
                
                if is_point_inside(candidate, all_triangles):
                    if verbose:
                        print(f"  Grid search found point after {candidates_tested}/{grid_size**3} candidates: "
                              f"({candidate[0]:.6f}, {candidate[1]:.6f}, {candidate[2]:.6f})")
                    
                    return {
                        'point': candidate.tolist(),
                        'bounding_box': {
                            'min': bbox_min.tolist(),
                            'max': bbox_max.tolist()
                        },
                        'geometry_files': [str(p.name) for p in stl_files],
                        'triangle_count': len(all_triangles),
                        'attempts': candidates_tested,
                        'method': 'grid_search'
                    }
    
    # Final fallback
    center = (sample_min + sample_max) / 2.0
    if verbose:
        print(f"  Warning: Using center point as fallback: "
              f"({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
    
    return {
        'point': center.tolist(),
        'bounding_box': {
            'min': bbox_min.tolist(),
            'max': bbox_max.tolist()
        },
        'geometry_files': [str(p.name) for p in stl_files],
        'triangle_count': len(all_triangles),
        'attempts': max_candidates + grid_size**3,
        'method': 'fallback_center'
    }


def save_cache(result: Dict, cache_path: Path):
    """Save result to cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(result, f, indent=2)


def load_cache(cache_path: Path) -> Optional[Dict]:
    """Load cached result if available."""
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def validate_cache(result: Dict, geometry_dir: Path, wall_name: str, inlet_name: str) -> bool:
    """Validate cached result against current geometry."""
    # Check if geometry files still exist
    expected_files = {f"{wall_name}.stl", f"{inlet_name}.stl"}
    
    # Add outlet files
    outlets = discover_outlets(geometry_dir)
    for _, filename in outlets:
        expected_files.add(filename)
    
    # Check against cached files
    cached_files = set(result.get('geometry_files', []))
    
    return expected_files == cached_files


def auto_detect_patch_names(geometry_dir: Path) -> tuple[str, str]:
    """Auto-detect wall and inlet patch names by scanning geometry directory."""
    if not geometry_dir.is_dir():
        return ('wall_aorta', 'inlet')
    
    stl_files = [f.name for f in geometry_dir.iterdir() 
                 if f.is_file() and f.name.lower().endswith('.stl')]
    
    wall_name = 'wall_aorta'  # Default fallback
    inlet_name = 'inlet'      # Default fallback
    
    # Look for common wall patterns
    wall_patterns = ['wall_aorta.stl', 'wall.stl', 'aorta.stl', 'vessel.stl']
    for pattern in wall_patterns:
        if pattern in stl_files:
            wall_name = pattern[:-4]  # Remove .stl extension
            break
    
    # Look for common inlet patterns  
    inlet_patterns = ['inlet.stl', 'inflow.stl', 'entrance.stl']
    for pattern in inlet_patterns:
        if pattern in stl_files:
            inlet_name = pattern[:-4]  # Remove .stl extension
            break
    
    return (wall_name, inlet_name)


def main():
    parser = argparse.ArgumentParser(description="Find interior point using ray-casting algorithm.")
    parser.add_argument('--wall-name', type=str,
                       help='Wall patch name (auto-detected if not specified)')
    parser.add_argument('--inlet-name', type=str,
                       help='Inlet patch name (auto-detected if not specified)')
    parser.add_argument('--case-dir', type=str, default='.',
                       help='OpenFOAM case directory (default: .)')
    parser.add_argument('--use-cached', action='store_true',
                       help='Use cached result if available')
    parser.add_argument('--force', action='store_true',
                       help='Force recalculation even if cache exists')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    parser.add_argument('--output-format', choices=['point', 'json'], default='point',
                       help='Output format (default: point)')
    
    args = parser.parse_args()
    
    geometry_dir = Path(args.case_dir) / 'constant' / 'geometry'
    
    # Auto-detect patch names if not provided
    if args.wall_name is None or args.inlet_name is None:
        detected_wall, detected_inlet = auto_detect_patch_names(geometry_dir)
        if args.wall_name is None:
            args.wall_name = detected_wall
            if not args.quiet:
                print(f"Auto-detected wall patch: {args.wall_name}")
        if args.inlet_name is None:
            args.inlet_name = detected_inlet
            if not args.quiet:
                print(f"Auto-detected inlet patch: {args.inlet_name}")
    
    cache_dir = Path(args.case_dir) / '.snappymesh_cache'
    cache_file = cache_dir / 'interior_point.json'
    
    # Try to use cache first
    result = None
    if not args.force and cache_file.exists():
        result = load_cache(cache_file)
        if result and validate_cache(result, geometry_dir, args.wall_name, args.inlet_name):
            if not args.quiet:
                print("Using cached interior point:")
                print(f"  Point: ({result['point'][0]:.6f}, {result['point'][1]:.6f}, {result['point'][2]:.6f})")
                print(f"  Method: {result['method']}")
                print(f"  Triangles: {result['triangle_count']}")
        else:
            result = None  # Invalid cache
    
    # Calculate if no valid cache
    if result is None:
        if args.use_cached:
            print("Error: No valid cache found. Run without --use-cached first.")
            return 1
        
        try:
            result = find_interior_point(geometry_dir, args.wall_name, args.inlet_name, 
                                       verbose=not args.quiet)
            save_cache(result, cache_file)
            
            if not args.quiet:
                print(f"\nResult cached to: {cache_file}")
        
        except Exception as e:
            print(f"Error finding interior point: {e}")
            return 1
    
    # Output result
    if args.output_format == 'json':
        print(json.dumps(result, indent=2))
    else:
        # Just the point coordinates
        point = result['point']
        print(f"{point[0]} {point[1]} {point[2]}")
    
    return 0


if __name__ == '__main__':
    exit(main())