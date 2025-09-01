#!/usr/bin/env python3
"""
Generate snappyHexMesh dictionaries for complex vascular geometries.

This script takes a small set of high‑level parameters and writes
`system/snappyHexMeshDict.noLayers` (castellation + snap only) and
`system/snappyHexMeshDict.layers` (with layer controls).  It
automatically discovers outlet STL files, scales distance bands and
layer thickness with the base cell size, and fills in sensible
defaults for mesh quality controls.

Features:
- Auto-detects wall and inlet patch names from STL files in constant/geometry/
- Auto-calculates dx-base from blockMeshDict if not provided
- Auto-computes interior point using cached ray-casting results
- Discovers outlet patches automatically (outlet.stl, outlet1.stl, etc.)

Usage examples:

    # Minimal command - all parameters auto-detected
    python scripts/generate_dicts.py --T-rel 0.35 --n-layers 3

    # With custom refinement levels
    python scripts/generate_dicts.py --T-rel 0.30 --n-layers 4 --levels 2 3

    # Override auto-detection if needed
    python scripts/generate_dicts.py --T-rel 0.35 --n-layers 3 \
        --wall-name custom_wall --inlet-name custom_inlet

The script should be run from your case directory; it writes files
into the `system` subfolder.  If the folder does not exist it will
create it.
"""

import argparse
import json
import subprocess
import re
from pathlib import Path


def parse_bool(value: str) -> bool:
    """Return True if the string value represents a truthy value."""
    if isinstance(value, bool):
        return value
    value_lower = value.strip().lower()
    return value_lower in ("true", "t", "yes", "y", "1")


def discover_outlets(trisurface_dir: Path) -> list[tuple[str, str]]:
    """
    Scan the given directory for STL files matching the pattern
    'outlet*.stl'.  Returns a list of (patch_name, filename) tuples.
    (For OpenFOAM 12, use 'constant/geometry' instead of 'constant/triSurface'.)
    """
    outlets = []
    if not trisurface_dir.is_dir():
        return outlets
    for entry in sorted(trisurface_dir.iterdir()):
        if entry.is_file() and entry.name.lower().endswith(".stl"):
            # use a simple regex to detect outlet prefixes
            m = re.match(r"outlet(\d*)\.stl", entry.name, re.IGNORECASE)
            if m:
                idx_str = m.group(1) or ""
                patch_name = f"outlet{idx_str}" if idx_str else "outlet"
                outlets.append((patch_name, entry.name))
    return outlets


def auto_detect_patch_names(geometry_dir: Path) -> tuple[str, str]:
    """
    Auto-detect wall and inlet patch names by scanning geometry directory.
    Returns (wall_name, inlet_name) tuple with detected names.
    Falls back to common defaults if files not found.
    """
    if not geometry_dir.is_dir():
        return ('wall', 'inlet')
    
    stl_files = [f.name for f in geometry_dir.iterdir() 
                 if f.is_file() and f.name.lower().endswith('.stl')]
    
    wall_name = 'wall'
    inlet_name = 'inlet'
    
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


def compute_thickness_values(args) -> dict[str, float]:
    """
    Determine thickness parameters for addLayersControls.  In relative
    mode the dictionary uses fractions of Δx; in absolute mode it uses
    absolute metres.  Returns a mapping with keys:

    * finalLayerThickness
    * thickness
    * firstLayerThickness
    * minThickness
    """
    dx = args.dx_base
    if args.relative:
        return {
            "relativeSizes": True,
            "finalLayerThickness": args.T_rel,
            "thickness": args.T_rel * dx,
            "firstLayerThickness": args.t1_rel,
            "minThickness": args.min_rel,
        }
    else:
        # absolute sizes: derive from provided values or fall back to
        # relative fractions times Δx.
        T_abs = args.T_abs if args.T_abs is not None else args.T_rel * dx
        t1_abs = args.t1_abs if args.t1_abs is not None else args.t1_rel * dx
        min_abs = args.min_abs if args.min_abs is not None else args.min_rel * dx
        return {
            "relativeSizes": False,
            "finalLayerThickness": T_abs,  # not used when relativeSizes=false
            "thickness": T_abs,
            "firstLayerThickness": t1_abs,
            "minThickness": min_abs,
        }


def generate_snappy_dict(args, with_layers: bool, outlets: list[tuple[str, str]]) -> str:
    """
    Produce the contents of snappyHexMeshDict for either the no‑layer
    or layer version.

    :param args: parsed command line arguments
    :param with_layers: if False, disables layer addition (noLayers
                        version); if True, includes addLayersControls
    :param outlets: list of (patch_name, filename) pairs for outlet
                    patches discovered in constant/triSurface.
    :return: a string containing the dictionary file contents.
    """
    # Determine refinement levels
    min_level, max_level = args.levels

    # Determine band distances scaled with Δx and max refinement level
    max_level_factor = max_level if max_level > 0 else 1
    near_dist = args.near_band * args.dx_base * max_level_factor
    far_dist = args.far_band * args.dx_base * max_level_factor

    # Compute thickness values
    thickness_dict = compute_thickness_values(args)

    # Compose geometry section
    geo_entries = []
    # wall
    geo_entries.append(f"    {args.wall_name}\n    {{\n        type triSurfaceMesh;\n        file \"{args.wall_name}.stl\";\n    }}\n")
    # inlet
    geo_entries.append(f"    {args.inlet_name}\n    {{\n        type triSurfaceMesh;\n        file \"{args.inlet_name}.stl\";\n    }}\n")
    # outlets
    for patch_name, fname in outlets:
        geo_entries.append(f"    {patch_name}\n    {{\n        type triSurfaceMesh;\n        file \"{fname}\";\n    }}\n")

    geometry_block = "geometry\n{\n" + "".join(geo_entries) + "}\n"

    # Compose refinementSurfaces entries
    # Each surface is refined to [min max] levels.
    surf_entries = []
    for patch_name, _ in [(args.wall_name, None), (args.inlet_name, None)] + outlets:
        surf_entries.append(
            f"        {patch_name}\n        {{\n            level ({min_level} {max_level});\n        }}\n"
        )

    refinement_surfaces_block = (
        "    refinementSurfaces\n    {\n" + "".join(surf_entries) + "    }\n"
    )

    # Compose castellation controls
    if with_layers:
        # For layers file: only run addLayers phase
        castellated_block = "castellatedMesh false;\nsnap false;\naddLayers true;\n\n"
    else:
        # For noLayers file: only run castellation and snap phases
        castellated_block = "castellatedMesh true;\nsnap true;\naddLayers false;\n\n"

    # Castellated mesh controls
    castellated_block += "castellatedMeshControls\n{\n"
    castellated_block += "    maxLocalCells 1000000;\n"
    castellated_block += "    maxGlobalCells 50000000;\n"
    castellated_block += "    minRefinementCells 10;\n"
    castellated_block += "    nCellsBetweenLevels 3;\n"
    castellated_block += f"    resolveFeatureAngle {args.resolve_feature_angle};\n"
    castellated_block += "    features\n    (\n"
    # Add eMesh for wall
    castellated_block += f"        {{\n            file \"{args.wall_name}.eMesh\";\n            level 0;\n        }}\n"
    # Add eMesh for inlet
    castellated_block += f"        {{\n            file \"{args.inlet_name}.eMesh\";\n            level 0;\n        }}\n"
    # Add eMesh for outlets
    for patch_name, _ in outlets:
        castellated_block += f"        {{\n            file \"{patch_name}.eMesh\";\n            level 0;\n        }}\n"
    castellated_block += "    );\n"
    castellated_block += refinement_surfaces_block
    # refinementRegions left empty
    castellated_block += "    refinementRegions\n    {\n    }\n"
    castellated_block += f"    insidePoint ({args.location_in_mesh[0]} {args.location_in_mesh[1]} {args.location_in_mesh[2]});\n"
    castellated_block += "    allowFreeStandingZoneFaces true;\n"
    castellated_block += "}\n\n"

    # Snap controls
    snap_block = (
        "snapControls\n{\n"
        "    nSmoothPatch 3;\n"
        "    tolerance 2.0;\n"
        "    nSolveIter 30;\n"
        "    nRelaxIter 5;\n"
        "    nFeatureSnapIter 15;\n"
        "    implicitFeatureSnap true;\n"
        "    explicitFeatureSnap false;\n"
        "    multiRegionSnap false;\n"
        "}\n\n"
    )

    # Add layer controls (only if with_layers)
    layers_block = ""
    if with_layers:
        layers_block += "addLayersControls\n{\n"
        # Which surfaces to add layers to
        layers_block += "    layers\n    {\n"
        layers_block += f"        {args.wall_name}\n        {{\n            nSurfaceLayers {args.n_layers};\n        }}\n"
        layers_block += "    }\n"
        # Relative or absolute sizing
        layers_block += f"    relativeSizes {'true' if thickness_dict['relativeSizes'] else 'false'};\n"
        layers_block += f"    expansionRatio {args.er};\n"
        if thickness_dict['relativeSizes']:
            layers_block += f"    finalLayerThickness {thickness_dict['finalLayerThickness']};\n"
            layers_block += f"    minThickness {thickness_dict['minThickness']};\n"
        else:
            layers_block += f"    firstLayerThickness {thickness_dict['firstLayerThickness']};\n"
            layers_block += f"    minThickness {thickness_dict['minThickness']};\n"
        # Distances for near/far transition
        layers_block += f"    nearWallOffset {near_dist};\n"
        layers_block += f"    farWallOffset {far_dist};\n"
        # Feature angle for layers
        layers_block += f"    featureAngle {args.feature_angle};\n"
        layers_block += f"    slipFeatureAngle {args.slip_angle};\n"
        # Surface smoothing - enhanced for better layer formation
        layers_block += "    nSmoothSurfaceNormals 3;\n"
        layers_block += "    nSmoothThickness 15;\n"
        # Growth iterations
        layers_block += f"    nLayerIter {args.n_layer_iter};\n"
        layers_block += f"    nRelaxedIter {args.n_relaxed_iter};\n"
        # Quality vetoes
        layers_block += f"    maxThicknessToMedialRatio {args.maxThicknessToMedialRatio};\n"
        layers_block += f"    minMedianAxisAngle {args.minMedianAxisAngle};\n"
        layers_block += "    nSmoothNormals 3;\n"
        layers_block += f"    maxFaceThicknessRatio 1.0;\n"
        layers_block += f"    maxFaceThicknessVariation 0.3;\n"
        layers_block += "    nGrow 0;\n"
        layers_block += "    nBufferCellsNoExtrude 0;\n"
        layers_block += "    nRelaxIter 5;\n"
        layers_block += "}\n\n"

    # Mesh quality controls - relaxed for better layer formation
    quality_block = (
        "meshQualityControls\n{\n"
        f"    maxNonOrtho 75;\n"
        f"    maxBoundarySkewness {args.maxBoundarySkewness};\n"
        f"    maxInternalSkewness {args.maxInternalSkewness};\n"
        "    maxConcave 85;\n"
        "    minVol 1e-15;\n"
        "    minTetQuality 1e-12;\n"
        "    minArea -1;\n"
        "    minTwist 0.01;\n"
        "    minTriangleTwist -1;\n"
        "    minFaceWeight 0.01;\n"
        "    minVolRatio 0.005;\n"
        "    minDeterminant 0.0005;\n"
        "    nSmoothScale 6;\n"
        "    errorReduction 0.5;\n"
        "    relaxed\n    {\n"
        "        maxNonOrtho 85;\n"
        "        maxBoundarySkewness 35;\n"
        "        maxInternalSkewness 8;\n"
        "    }\n"
        "}\n\n"
        "mergeTolerance 1e-6;\n"
    )

    # Compose full dictionary
    dict_str = (
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "| =========                 |                                                 |\n"
        "| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox          |\n"
        "|  \\    /   O peration     | Version:  v2012                                |\n"
        "|   \\  /    A nd           | Web:      www.OpenFOAM.com                     |\n"
        "|    \\/     M anipulation  |                                                 |\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n" + "{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object snappyHexMeshDict;\n}\n\n"
        + castellated_block
        + geometry_block
        + snap_block
        + layers_block
        + quality_block
    )
    return dict_str


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write(contents)


def get_interior_point(case_dir: Path, wall_name: str, inlet_name: str) -> tuple[float, float, float]:
    """
    Get interior point, using cache if available or computing via find_interior_point.py.
    Returns tuple of (x, y, z) coordinates.
    """
    cache_file = case_dir / '.snappymesh_cache' / 'interior_point.json'
    
    # Try to load from cache first
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                result = json.load(f)
            point = result.get('point')
            if point and len(point) == 3:
                print(f"Using cached interior point: ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")
                return tuple(point)
        except (json.JSONDecodeError, IOError, KeyError):
            pass
    
    # Cache not available or invalid, compute using find_interior_point script
    print("Computing interior point using ray-casting...")
    try:
        script_path = case_dir / 'scripts' / 'find_interior_point.py'
        result = subprocess.run([
            'python', str(script_path),
            '--wall-name', wall_name,
            '--inlet-name', inlet_name,
            '--case-dir', str(case_dir),
            '--quiet'
        ], capture_output=True, text=True, cwd=case_dir)
        
        if result.returncode == 0:
            # Parse output (should be "x y z")
            coords = result.stdout.strip().split()
            if len(coords) == 3:
                point = [float(coords[0]), float(coords[1]), float(coords[2])]
                print(f"Computed interior point: ({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")
                return tuple(point)
    except Exception as e:
        print(f"Warning: Could not compute interior point: {e}")
    
    # Fallback - use a default point that's likely to work for many geometries
    print("Warning: Using fallback interior point (0, 0, 0)")
    print("Consider running: python scripts/find_interior_point.py --wall-name {} --inlet-name {}".format(wall_name, inlet_name))
    return (0.0, 0.0, 0.0)


def calculate_dx_from_blockmesh(blockmesh_path: str = "system/blockMeshDict") -> float:
    """
    Parse blockMeshDict and calculate average cell size (dx-base).
    
    Returns the average of dx, dy, dz calculated from domain size and cell counts.
    """
    try:
        with open(blockmesh_path, 'r') as f:
            content = f.read()
        
        # Extract vertices (assuming standard blockMesh format with 8 vertices)
        import re
        
        # Find vertices section
        vertices_match = re.search(r'vertices\s*\(\s*(.*?)\s*\);', content, re.DOTALL)
        if not vertices_match:
            raise ValueError("Could not find vertices in blockMeshDict")
        
        vertices_text = vertices_match.group(1)
        # Extract coordinates - look for patterns like (x y z)
        coords = re.findall(r'\(\s*([-\d.e+-]+)\s+([-\d.e+-]+)\s+([-\d.e+-]+)\s*\)', vertices_text)
        
        if len(coords) < 8:
            raise ValueError(f"Found {len(coords)} vertices, expected 8")
        
        # Convert to floats
        vertices = [(float(x), float(y), float(z)) for x, y, z in coords[:8]]
        
        # Calculate domain dimensions (assuming axis-aligned box)
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]  
        z_coords = [v[2] for v in vertices]
        
        dx_domain = max(x_coords) - min(x_coords)
        dy_domain = max(y_coords) - min(y_coords)
        dz_domain = max(z_coords) - min(z_coords)
        
        # Extract cell counts from blocks section
        blocks_match = re.search(r'blocks\s*\(\s*hex.*?\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)', content, re.DOTALL)
        if not blocks_match:
            raise ValueError("Could not find cell counts in blocks section")
        
        nx, ny, nz = map(int, blocks_match.groups())
        
        # Calculate cell sizes
        dx = dx_domain / nx
        dy = dy_domain / ny  
        dz = dz_domain / nz
        
        # Return average cell size
        avg_dx = (dx + dy + dz) / 3
        
        print(f"Auto-calculated dx-base from blockMeshDict:")
        print(f"  Domain: {dx_domain:.3f} × {dy_domain:.3f} × {dz_domain:.3f} m")
        print(f"  Cells: {nx} × {ny} × {nz}")
        print(f"  Cell sizes: {dx:.4f} × {dy:.4f} × {dz:.4f} m")
        print(f"  Average dx-base: {avg_dx:.4f} m")
        
        return avg_dx
        
    except Exception as e:
        raise ValueError(f"Failed to calculate dx-base from blockMeshDict: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate snappyHexMesh dictionaries.")
    parser.add_argument('--dx-base', type=float, help='Base cell size (Δx) in metres. If not specified, auto-calculated from blockMeshDict.')
    parser.add_argument('--levels', type=int, nargs=2, default=[1, 1], metavar=('min', 'max'), help='Surface refinement levels (min max).')
    parser.add_argument('--location-in-mesh', type=float, nargs=3, metavar=('x', 'y', 'z'), help='Point inside the mesh volume. Auto-detected if not provided.')
    parser.add_argument('--wall-name', type=str, help='Patch name for the wall STL file (auto-detected if not specified).')
    parser.add_argument('--inlet-name', type=str, help='Patch name for the inlet STL file (auto-detected if not specified).')
    parser.add_argument('--included-angle', type=float, default=160.0, help='Included angle for merging sharp edges (deg).')
    parser.add_argument('--resolve-feature-angle', type=float, default=30.0, help='Feature extraction angle (deg).')
    parser.add_argument('--relative', type=parse_bool, default=True, help='Use relative layer sizing (true/false).')
    parser.add_argument('--t1-rel', type=float, default=0.15, help='First layer thickness as multiple of Δx (relative mode).')
    parser.add_argument('--T-rel', type=float, default=0.75, help='Total layer thickness as multiple of Δx (relative mode).')
    parser.add_argument('--er', type=float, default=1.10, help='Layer expansion ratio.')
    parser.add_argument('--min-rel', type=float, default=0.10, help='Minimum allowed layer thickness as multiple of Δx (relative mode).')
    parser.add_argument('--t1-abs', type=float, help='First layer thickness in metres (absolute mode).')
    parser.add_argument('--T-abs', type=float, help='Total layer thickness in metres (absolute mode).')
    parser.add_argument('--min-abs', type=float, help='Minimum allowed layer thickness in metres (absolute mode).')
    parser.add_argument('--n-layers', type=int, default=5, help='Number of prismatic layers to attempt.')
    parser.add_argument('--near-band', type=float, default=7.0, help='Near‑wall distance band in multiples of Δx.')
    parser.add_argument('--far-band', type=float, default=14.0, help='Far‑field distance band in multiples of Δx.')
    parser.add_argument('--slip-angle', type=float, default=45.0, help='Slip feature angle for sharp corners (deg).')
    parser.add_argument('--feature-angle', type=float, default=75.0, help='Feature angle for layer growth (deg).')
    parser.add_argument('--maxThicknessToMedialRatio', type=float, default=0.65, help='Maximum thickness to medial ratio.')
    parser.add_argument('--minMedianAxisAngle', type=float, default=55.0, help='Minimum median axis angle (deg).')
    parser.add_argument('--maxBoundarySkewness', type=float, default=25.0, help='Maximum boundary skewness for mesh quality controls.')
    parser.add_argument('--maxInternalSkewness', type=float, default=5.0, help='Maximum internal skewness for mesh quality controls.')
    parser.add_argument('--n-layer-iter', type=int, default=100, help='Maximum iterations for adding layers.')
    parser.add_argument('--n-relaxed-iter', type=int, default=50, help='Number of relaxed iterations for adding layers.')
    parser.add_argument('--case-dir', type=str, default='.', help='Path to the OpenFOAM case directory containing constant/triSurface.')
    args = parser.parse_args()

    # Auto-calculate dx-base if not provided
    if args.dx_base is None:
        blockmesh_path = Path(args.case_dir) / 'system' / 'blockMeshDict'
        if blockmesh_path.exists():
            args.dx_base = calculate_dx_from_blockmesh(str(blockmesh_path))
        else:
            parser.error('--dx-base is required when system/blockMeshDict does not exist.')

    # Ensure lists for levels
    if len(args.levels) != 2:
        parser.error('--levels requires exactly two integers (min max).')

    # Auto-detect patch names if not provided
    geometry_dir = Path(args.case_dir) / 'constant' / 'geometry'
    if args.wall_name is None or args.inlet_name is None:
        detected_wall, detected_inlet = auto_detect_patch_names(geometry_dir)
        if args.wall_name is None:
            args.wall_name = detected_wall
            print(f"Auto-detected wall patch: {args.wall_name}")
        if args.inlet_name is None:
            args.inlet_name = detected_inlet
            print(f"Auto-detected inlet patch: {args.inlet_name}")

    # Get or compute location-in-mesh if not provided
    if args.location_in_mesh is None:
        case_path = Path(args.case_dir)
        args.location_in_mesh = get_interior_point(case_path, args.wall_name, args.inlet_name)

    # Discover outlets
    outlets = discover_outlets(geometry_dir)

    # Generate dictionaries
    no_layers_dict = generate_snappy_dict(args, with_layers=False, outlets=outlets)
    layers_dict = generate_snappy_dict(args, with_layers=True, outlets=outlets)

    # Write files
    system_dir = Path(args.case_dir) / 'system'
    write_file(system_dir / 'snappyHexMeshDict.noLayers', no_layers_dict)
    write_file(system_dir / 'snappyHexMeshDict.layers', layers_dict)

    # Create a minimal surfaceFeaturesDict placeholder if absent
    sfd_path = system_dir / 'surfaceFeaturesDict'
    if not sfd_path.exists():
        placeholder = (
            "/* Extract surface features for snappyHexMesh */\n\n"
            "surfaceFeatureExtractDict\n{\n"
            "    // See OpenFOAM user guide for details\n"
            "    // For OpenFOAM 12, STL files should be placed in 'constant/geometry'\n"
            "}\n"
        )
        write_file(sfd_path, placeholder)

    print(f"Generated {system_dir/'snappyHexMeshDict.noLayers'} and {system_dir/'snappyHexMeshDict.layers'}")


if __name__ == '__main__':
    main()