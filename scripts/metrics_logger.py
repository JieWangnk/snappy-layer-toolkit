#!/usr/bin/env python3
"""
Compute effective layer count from snappyHexMesh results.

This simple tool calculates how many effective layers were achieved based on
the thickness fraction reported by snappyHexMesh. It can also work without a
log file to show theoretical calculations.

Usage:
    # With log file (searches for layer info)
    python metrics_logger.py --thickness-fraction 0.65
    
    # Manual entry (if you know the fraction)
    python metrics_logger.py --thickness-fraction 0.65
    
    # Show interpretation guide (no log needed)
    python metrics_logger.py --show-guide
"""

import argparse
import math
import re
from pathlib import Path


def compute_effective_layers(thickness_fraction: float, expansion_ratio: float = 1.15, n_requested: int = 5) -> float:
    """
    Calculate effective number of layers using geometric series formula.
    
    N_eff = log(1 + (ER-1) * thickness_fraction * sum) / log(ER)
    where sum = (ER^N - 1)/(ER - 1) for the original stack
    
    Simplified: When thickness_fraction of the total is achieved,
    how many continuous layers does that represent?
    """
    if expansion_ratio <= 1.0 or thickness_fraction <= 0:
        return thickness_fraction * n_requested  # Linear fallback
    
    try:
        # For a geometric series, if we achieve X% of total thickness,
        # we can back-calculate the effective number of layers
        term = 1 + (expansion_ratio**n_requested - 1) * thickness_fraction
        n_eff = math.log(term) / math.log(expansion_ratio)
        return max(0, n_eff)
    except (ValueError, ZeroDivisionError):
        return 0.0


def detect_geometry_complexity(log_file: Path) -> str:
    """
    Detect geometry complexity from the log file.
    Returns: 'simple', 'moderate', or 'complex'
    """
    if not log_file.exists():
        return 'unknown'
    
    with log_file.open('r') as f:
        content = f.read()
    
    # Look for indicators of complexity
    n_outlets = content.count('outlet')
    has_branching = 'branch' in content.lower() or n_outlets > 2
    has_curvature = 'curv' in content.lower() or 'bend' in content.lower()
    
    # Check for multiple patches
    patch_count = 0
    for line in content.split('\n'):
        if 'patch' in line.lower() and 'faces' in line.lower():
            patch_count += 1
    
    # Determine complexity
    if n_outlets > 3 or (has_branching and has_curvature):
        return 'complex'
    elif n_outlets > 1 or has_branching or has_curvature:
        return 'moderate'
    else:
        return 'simple'

def find_layer_info_in_log(log_file: Path) -> dict:
    """
    Search for layer information in snappyHexMesh log.
    Looks for patterns like:
    - "patch      faces    layers avg thickness[m]"
    - "wall_aorta 2675     5      0.0555    0.363"
    
    And also the final summary:
    - "patch      faces    layers   overall thickness"
    - "wall_aorta 2675     0.0135   0.000287  0.078"
    """
    if not log_file.exists():
        return {}
    
    results = {}
    with log_file.open('r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # First look for requested layers from the log
    requested_layers = None
    for line in lines:
        # Look for lines like "Layers: 2", "requested layers: 5", etc.
        if 'layers' in line.lower() and ':' in line:
            match = re.search(r'layers\s*[=:]\s*(\d+)', line, re.IGNORECASE)
            if match:
                requested_layers = int(match.group(1))
                break
    
    # Look for both layer addition summaries
    for i, line in enumerate(lines):
        # Pattern 1: Initial layer addition summary
        if 'patch' in line and 'faces' in line and 'layers' in line and 'thickness[m]' in line:
            # Skip header and separator lines
            j = i + 3
            while j < len(lines) and lines[j].strip():
                parts = lines[j].split()
                if len(parts) >= 5 and not parts[0].startswith('-'):
                    try:
                        patch = parts[0]
                        faces = int(parts[1])
                        layers = float(parts[2])
                        thickness_near = float(parts[3])
                        thickness_total = float(parts[4])
                        
                        results[patch] = {
                            'faces': faces,
                            'layers': layers,
                            'thickness_near': thickness_near,
                            'thickness_total': thickness_total,
                            'thickness_fraction': None,  # Will try to find this
                            'requested_layers': requested_layers or layers  # Store requested layers
                        }
                    except (ValueError, IndexError):
                        pass
                j += 1
        
        # Pattern 2: Final summary with percentage  
        elif 'overall thickness' in line:
            # Look for the format:
            # patch      faces    layers   overall thickness  
            #                             [m]       [%]
            # -----      -----    ------   ---       ---
            # wall_aorta 2675     0.0135   0.000287  0.078
            
            # Skip to the line with [m] [%]
            j = i + 1
            if j < len(lines) and '[m]' in lines[j] and '[%]' in lines[j]:
                # Skip separator line
                j += 1
                if j < len(lines) and '---' in lines[j]:
                    j += 1
                    
                # Now parse data lines
                while j < len(lines) and lines[j].strip():
                    line_content = lines[j].strip()
                    if line_content and not line_content.startswith('-'):
                        parts = line_content.split()
                        if len(parts) >= 5:
                            try:
                                patch = parts[0]
                                faces = int(parts[1])
                                layers_achieved = float(parts[2])
                                thickness_m = float(parts[3])
                                thickness_percent = float(parts[4])
                                
                                # Update existing entry or create new one
                                if patch in results:
                                    results[patch]['layers_achieved'] = layers_achieved
                                    results[patch]['thickness_fraction'] = thickness_percent / 100.0
                                else:
                                    results[patch] = {
                                        'faces': faces,
                                        'layers': layers_achieved,
                                        'thickness_near': thickness_m,
                                        'thickness_total': thickness_m,
                                        'thickness_fraction': thickness_percent / 100.0,
                                        'requested_layers': requested_layers  # Store requested layers
                                    }
                            except (ValueError, IndexError):
                                pass
                    j += 1
    
    return results


def print_interpretation_guide():
    """Print a guide for interpreting thickness fractions."""
    print("\n=== Layer Achievement Interpretation Guide ===\n")
    print("Based on empirical testing across 2-10 layers:")
    print("-" * 50)
    
    print("\n**Target Thickness Fractions by Layer Count:**")
    print("  n ≤ 3 layers:  ≥90% (easily achievable with T_rel=0.15-0.50)")
    print("  n = 4-7 layers: ≥85% (achievable with T_rel=0.30-0.40)")
    print("  n ≥ 8 layers:  ≥80% (requires careful tuning of T_rel)")
    
    print("\n**Thickness-Layers Trade-off:**")
    print("  • Fewer layers (2-3) → can use higher T_rel (0.30-0.50)")
    print("  • More layers (8-10) → need lower T_rel (0.15-0.25)")
    print("  • Surface refinement helps: level 2-3 improves preservation")
    
    print("\n**Effective Tuning Parameters:**")
    print("  1. T_rel (relative thickness): Primary control")
    print("  2. ER (expansion ratio): Keep at 1.10-1.12 for stability")
    print("  3. Surface refinement level: Higher levels → better layers")
    print("  4. Band widths: Match surface max level (3 → bands of 3)")
    
    print("\n**Quick Fixes by Fraction Achieved:**")
    print("  <30%: Very thin - reduce T_rel by 30-40%, use ER=1.10")
    print("  30-50%: Thin - reduce T_rel by 20%, try ER=1.10-1.12")
    print("  50-70%: Below target - reduce T_rel by 10%, tune bands")
    print("  70-85%: Good but improvable - minor T_rel reduction")
    print("  >85%: Excellent - settings are well-tuned")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze snappyHexMesh layer results simply.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Simple inputs
    parser.add_argument('--thickness-fraction', '-f', type=float,
                        help='Thickness fraction achieved (0-1 or 0-100 if >1)')
    parser.add_argument('--er', type=float, default=1.15,
                        help='Expansion ratio (default: 1.15)')
    parser.add_argument('--n-layers', type=int, default=5,
                        help='Number of layers requested (default: 5)')
    
    # Optional log file
    parser.add_argument('--log-file', type=str,
                        help='Path to snappyHexMesh log (optional)')
    
    # Show guide
    parser.add_argument('--show-guide', action='store_true',
                        help='Show interpretation guide')
    
    args = parser.parse_args()
    
    # Show guide if requested
    if args.show_guide:
        print_interpretation_guide()
        return
    
    # Try to get thickness fraction from log or command line
    thickness_fraction = None
    
    if args.log_file:
        log_path = Path(args.log_file)
        layer_info = find_layer_info_in_log(log_path)
        geometry_complexity = detect_geometry_complexity(log_path)
        
        if layer_info:
            print("\n=== Layer Summary from Log ===")
            if geometry_complexity != 'unknown':
                print(f"Geometry complexity: {geometry_complexity}")
                if geometry_complexity == 'complex':
                    print("⚠️  Complex branching detected - use conservative settings")
                print()
            print(f"{'Patch':<15} {'Faces':>8} {'Layers':>8} {'Thick[m]':>10} {'Fraction':>10}")
            print("-" * 65)
            
            for patch, info in layer_info.items():
                frac = info.get('thickness_fraction')
                frac_str = f"{frac:.1%}" if frac is not None else "N/A"
                
                print(f"{patch:<15} {info['faces']:>8} {info['layers']:>8.1f} "
                      f"{info['thickness_total']:>10.4f} {frac_str:>10}")
                
                # If we have thickness fraction, compute effective layers
                if frac is not None:
                    # Use requested layers from log if available, otherwise use command line arg
                    n_req = info.get('requested_layers') or args.n_layers
                    n_eff = compute_effective_layers(frac, args.er, n_req)
                    print(f"{'':>15} {'':>8} {'':>8} {'':>10} → {n_eff:.1f} eff layers (of {n_req} requested)")
                    
                    # Set target based on requested layer count
                    if n_req <= 3:
                        target = 0.90
                    elif n_req <= 7:
                        target = 0.85
                    else:
                        target = 0.80
                    
                    # Add recommendations based on performance
                    if frac < 0.30:
                        print(f"{'':>15} ❌ Very thin ({frac:.0%}) - heavy pruning occurred")
                        print(f"{'':>15} → Reduce T_rel by 30-40%, use ER≈1.10, widen bands")
                        if n_req >= 8:
                            print(f"{'':>15} → Alternative: Try 5 layers with T_rel=0.30-0.35")
                    elif frac < 0.50:
                        print(f"{'':>15} ⚠️  Thin layers ({frac:.0%}) - significant pruning")
                        print(f"{'':>15} → Reduce T_rel by 20%, try ER=1.10-1.12")
                        print(f"{'':>15} → Surface refinement: increase to level 2-3")
                        if n_req >= 8:
                            print(f"{'':>15} → Alternative: Reduce to {n_req//2} layers with higher T_rel")
                    elif frac < target:
                        print(f"{'':>15} ✓ Usable ({frac:.0%}) but below target ({target:.0%} for n={n_req})")
                        print(f"{'':>15} → Reduce T_rel by 10%, ensure bands match surface level")
                        print(f"{'':>15} → Surface refinement: try level 2-3 for 95%+ coverage")
                        if n_req >= 5 and frac > 0.70:
                            # Good coverage but not meeting high layer count target
                            alt_layers = max(3, n_req - 3)
                            print(f"{'':>15} → Trade-off: {alt_layers} layers may achieve {min(95, frac*100+10):.0f}% with current T_rel")
                    else:
                        print(f"{'':>15} ✅ Excellent ({frac:.0%})! Meets target for n={n_req} layers")
                        if n_req <= 5 and frac > 0.95:
                            print(f"{'':>15} → Could try {n_req+2} layers with slightly lower T_rel")
            print()
        else:
            print(f"Warning: Could not find layer information in {log_path}")
    
    if args.thickness_fraction is not None:
        # Handle percentage input (>1 means percentage)
        if args.thickness_fraction > 1:
            thickness_fraction = args.thickness_fraction / 100.0
        else:
            thickness_fraction = args.thickness_fraction
        
        # Calculate effective layers
        n_req = args.n_layers
        n_eff = compute_effective_layers(thickness_fraction, args.er, n_req)
        
        print(f"\n=== Effective Layer Analysis ===")
        print(f"Thickness fraction: {thickness_fraction:.1%}")
        print(f"Expansion ratio: {args.er}")
        print(f"Requested layers: {n_req}")
        print(f"**Effective layers: {n_eff:.2f}**")
        
        # Set target based on requested layer count
        if n_req <= 3:
            target = 0.90
        elif n_req <= 7:
            target = 0.85
        else:
            target = 0.80
        
        print(f"Target fraction for n={n_req}: ≥{target:.0%}")
        
        # Provide interpretation based on actual fraction and target
        if thickness_fraction < 0.30:
            print("\n❌ Very thin layers - heavy pruning occurred")
            print("  → Reduce T_rel by 30-40%, use ER≈1.10")
            print("  → Widen bands to match surface refinement level")
            if n_req >= 8:
                print(f"  → Alternative: Try 5 layers with T_rel=0.30-0.35 for better coverage")
        elif thickness_fraction < 0.50:
            print("\n⚠️  Thin layers - significant pruning")
            print("  → Reduce T_rel by 20%, try ER=1.10-1.12")
            print("  → Surface refinement: increase to level 2-3")
            if n_req >= 8:
                print(f"  → Trade-off: Reduce to {n_req//2} layers with T_rel=0.30")
        elif thickness_fraction < target:
            print(f"\n✓ Usable but below target ({target:.0%})")
            print("  → Reduce T_rel by 10%")
            print("  → Surface refinement: try level 2-3 for 95%+ coverage")
            print("  → Ensure bands scale with surface max level")
            if n_req >= 5 and thickness_fraction > 0.70:
                alt_layers = max(3, n_req - 3)
                print(f"  → Thickness-layers trade-off: {alt_layers} layers may achieve {min(95, thickness_fraction*100+10):.0f}%")
        else:
            print(f"\n✅ Excellent! Meets target for n={n_req} layers")
            print("  → Settings are well-tuned")
            if n_req <= 5 and thickness_fraction > 0.95:
                print(f"  → Could try {n_req+2} layers with slightly lower T_rel")
    
    elif not args.log_file:
        print("Provide either --thickness-fraction or --log-file")
        print("Or use --show-guide to see the interpretation guide")


if __name__ == '__main__':
    main()