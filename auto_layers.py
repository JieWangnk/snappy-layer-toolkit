#!/usr/bin/env python3
"""
Automated layer optimization wrapper for snappyHexMesh.

This script automates the layer tuning process by:
1. Running generate_dicts.py with auto-detected parameters
2. Executing snappyHexMesh for layer addition
3. Analyzing results with metrics_logger.py
4. Logging results to CSV for analysis

Auto-Detection Features:
- Wall and inlet patch names detected from STL files in constant/geometry/
- Interior point computed automatically using ray-casting (cached)
- Base cell size calculated from blockMeshDict
- Outlet patches discovered automatically

Usage:
    # Minimal single run - everything auto-detected!
    python auto_layers.py --T-rel 0.35 --n-layers 5
    
    # Parameter sweep with auto-detection
    python auto_layers.py --sweep --T-rel-range 0.2 0.5 0.05 --n-layers 5
    
    # Quick iteration mode (reuse surface mesh)
    python auto_layers.py --T-rel 0.35 --n-layers 5 --quick
    
    # Override auto-detection for custom names
    python auto_layers.py --T-rel 0.35 --n-layers 5 \
        --wall-name custom_wall --inlet-name custom_inlet
"""

import subprocess
import argparse
import csv
import json
import re
from pathlib import Path
from datetime import datetime
import shutil
import sys


class LayerOptimizer:
    """Automated layer optimization for snappyHexMesh."""
    
    def __init__(self, args):
        self.args = args
        self.results = []
        self.csv_file = args.csv_file or f"layer_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
    def run_command(self, cmd, description="", capture_output=True):
        """Execute a shell command and optionally capture output."""
        print(f"\n{'='*60}")
        print(f"Running: {description or cmd}")
        print(f"{'='*60}")
        
        try:
            if capture_output:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
                return result.stdout
            else:
                subprocess.run(cmd, shell=True, check=True)
                return None
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            if capture_output:
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
            return None
    
    def parse_layer_results(self, log_file="log.addlayer"):
        """Parse snappyHexMesh log to extract layer coverage metrics."""
        if not Path(log_file).exists():
            print(f"Warning: Log file {log_file} not found")
            return None, None
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for the final layer summary with percentage
        fraction = None
        pattern = r'wall_aorta\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)'
        matches = re.findall(pattern, content)
        if matches:
            # Get the last match (final summary)
            try:
                fraction = float(matches[-1]) / 100.0
            except (ValueError, IndexError):
                pass
        
        # Also look for explicit percentage format
        percent_pattern = r'wall_aorta.*?(\d+\.?\d*)%'
        percent_matches = re.findall(percent_pattern, content)
        if percent_matches and fraction is None:
            try:
                fraction = float(percent_matches[-1]) / 100.0
            except ValueError:
                pass
        
        # Calculate effective layers if we have fraction
        n_eff = None
        if fraction is not None:
            n_eff = self.compute_effective_layers(fraction, self.args.er, self.args.n_layers)
        
        return fraction, n_eff
    
    def compute_effective_layers(self, thickness_fraction, expansion_ratio, n_requested):
        """Calculate effective number of layers achieved."""
        import math
        
        if expansion_ratio <= 1.0 or thickness_fraction <= 0:
            return thickness_fraction * n_requested
        
        try:
            term = 1 + (expansion_ratio**n_requested - 1) * thickness_fraction
            n_eff = math.log(term) / math.log(expansion_ratio)
            return max(0, n_eff)
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def run_single_optimization(self, T_rel, n_layers, er, quick_mode=False):
        """Run a single layer optimization iteration."""
        print(f"\n{'#'*70}")
        print(f"# Testing: n_layers={n_layers}, T_rel={T_rel:.3f}, ER={er:.2f}")
        print(f"{'#'*70}")
        
        # Step 1: Generate dictionaries with auto-detection
        generate_cmd = f"python scripts/generate_dicts.py --T-rel {T_rel} --er {er} --n-layers {n_layers}"
        
        # Only add optional parameters if explicitly provided
        if self.args.dx_base != 0.1:  # Only if different from auto-calculated
            generate_cmd += f" --dx-base {self.args.dx_base}"
            
        if self.args.levels != [1, 1]:  # Only if different from default
            generate_cmd += f" --levels {' '.join(map(str, self.args.levels))}"
            
        if self.args.wall_name != 'wall_aorta':  # Only if overriding auto-detection
            generate_cmd += f" --wall-name {self.args.wall_name}"
            
        if self.args.inlet_name != 'inlet':  # Only if overriding auto-detection
            generate_cmd += f" --inlet-name {self.args.inlet_name}"
            
        if self.args.resolve_feature_angle != 30:  # Only if different from default
            generate_cmd += f" --resolve-feature-angle {self.args.resolve_feature_angle}"
        
        # Add location-in-mesh only if provided (now optional due to auto-detection)
        if hasattr(self.args, 'location_in_mesh') and self.args.location_in_mesh is not None:
            generate_cmd += f" --location-in-mesh {' '.join(map(str, self.args.location_in_mesh))}"
        
        if self.args.relative is not None:
            generate_cmd += f" --relative {self.args.relative}"
        
        self.run_command(generate_cmd, "Generating snappyHexMesh dictionaries")
        
        # Step 2: Run meshing
        if quick_mode and Path("constant/polyMesh_copy").exists():
            # Quick mode: reuse surface mesh
            print("\nQuick mode: Reusing existing surface mesh")
            self.run_command("rm -rf constant/polyMesh", "Removing old mesh")
            self.run_command("cp -r constant/polyMesh_copy constant/polyMesh", "Restoring surface mesh")
        else:
            # Full meshing workflow
            if not Path("constant/polyMesh").exists() or not quick_mode:
                # Need to create base mesh
                print("\nCreating base mesh...")
                self.run_command("blockMesh > log.block", "Running blockMesh")
                
                # Extract features if needed
                if not any(Path("constant/geometry").glob("*.eMesh")):
                    self.run_command("surfaceFeatures", "Extracting surface features")
                
                # Castellate and snap
                self.run_command(
                    "cp system/snappyHexMeshDict.noLayers system/snappyHexMeshDict",
                    "Setting up for castellation/snapping"
                )
                self.run_command("snappyHexMesh -overwrite > log.snap", "Running castellation and snapping")
                
                # Save surface mesh for quick iterations
                self.run_command("rm -rf constant/polyMesh_copy", "Removing old surface mesh copy")
                self.run_command("cp -r constant/polyMesh constant/polyMesh_copy", "Saving surface mesh")
        
        # Step 3: Add layers
        self.run_command(
            "cp system/snappyHexMeshDict.layers system/snappyHexMeshDict",
            "Setting up for layer addition"
        )
        self.run_command("snappyHexMesh -overwrite > log.addlayer", "Adding layers")
        
        # Step 4: Analyze results
        print("\nAnalyzing layer results...")
        fraction, n_eff = self.parse_layer_results("log.addlayer")
        
        # Run metrics_logger for detailed analysis
        if fraction is not None:
            metrics_cmd = f"python scripts/metrics_logger.py --log-file log.addlayer --n-layers {n_layers}"
            output = self.run_command(metrics_cmd, "Running metrics analysis", capture_output=True)
            if output:
                print(output)
        
        # Store results
        result = {
            'n_layers': n_layers,
            'ER': er,
            't1_rel': None,  # Not used in relative mode
            'T_rel': T_rel,
            'fraction': fraction,
            'n_eff': n_eff,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        self.write_csv()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Requested layers: {n_layers}")
        print(f"  T_rel: {T_rel:.3f}")
        print(f"  Expansion ratio: {er:.2f}")
        if fraction is not None:
            print(f"  Coverage achieved: {fraction:.1%}")
            print(f"  Effective layers: {n_eff:.2f}")
            
            # Provide recommendation
            target = 0.90 if n_layers <= 3 else (0.85 if n_layers <= 7 else 0.80)
            if fraction < target:
                print(f"  Status: Below target ({target:.0%})")
                print(f"  Recommendation: Reduce T_rel by {int((1-fraction/target)*30)}%")
            else:
                print(f"  Status: ✓ Meets target ({target:.0%})")
        else:
            print(f"  Status: Failed to extract metrics")
        print(f"{'='*60}")
        
        return fraction, n_eff
    
    def run_parameter_sweep(self):
        """Run a parameter sweep over T_rel values."""
        T_rel_start, T_rel_end, T_rel_step = self.args.T_rel_range
        T_rel_values = []
        current = T_rel_start
        while current <= T_rel_end + 1e-6:
            T_rel_values.append(round(current, 3))
            current += T_rel_step
        
        print(f"\nRunning parameter sweep:")
        print(f"  T_rel values: {T_rel_values}")
        print(f"  n_layers: {self.args.n_layers}")
        print(f"  ER: {self.args.er}")
        
        best_result = None
        best_fraction = 0
        
        for i, T_rel in enumerate(T_rel_values):
            print(f"\n[{i+1}/{len(T_rel_values)}] Testing T_rel = {T_rel:.3f}")
            
            # Use quick mode after first iteration
            quick = i > 0 and self.args.quick
            fraction, n_eff = self.run_single_optimization(
                T_rel, self.args.n_layers, self.args.er, quick_mode=quick
            )
            
            if fraction and fraction > best_fraction:
                best_fraction = fraction
                best_result = (T_rel, fraction, n_eff)
        
        # Print sweep summary
        print(f"\n{'#'*70}")
        print(f"# PARAMETER SWEEP COMPLETE")
        print(f"{'#'*70}")
        print(f"\nResults saved to: {self.csv_file}")
        
        if best_result:
            T_rel, fraction, n_eff = best_result
            print(f"\nBest configuration:")
            print(f"  T_rel: {T_rel:.3f}")
            print(f"  Coverage: {fraction:.1%}")
            print(f"  Effective layers: {n_eff:.2f}")
        
        self.print_results_table()
    
    def print_results_table(self):
        """Print a formatted table of all results."""
        if not self.results:
            return
        
        print(f"\n{'='*70}")
        print(f"{'n_layers':<10} {'T_rel':<10} {'ER':<10} {'Coverage':<12} {'n_eff':<10}")
        print(f"{'-'*70}")
        
        for r in self.results:
            if r['fraction'] is not None:
                print(f"{r['n_layers']:<10} {r['T_rel']:<10.3f} {r['ER']:<10.2f} "
                      f"{r['fraction']:<12.1%} {r['n_eff']:<10.2f}")
            else:
                print(f"{r['n_layers']:<10} {r['T_rel']:<10.3f} {r['ER']:<10.2f} "
                      f"{'FAILED':<12} {'N/A':<10}")
        print(f"{'='*70}")
    
    def write_csv(self):
        """Write results to CSV file."""
        if not self.results:
            return
        
        with open(self.csv_file, 'w', newline='') as f:
            fieldnames = ['n_layers', 'ER', 't1_rel', 'T_rel', 'fraction', 'n_eff', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        
        print(f"Results written to: {self.csv_file}")
    
    def run(self):
        """Main execution method."""
        if self.args.sweep:
            self.run_parameter_sweep()
        else:
            # Single run
            fraction, n_eff = self.run_single_optimization(
                self.args.T_rel, self.args.n_layers, self.args.er, 
                quick_mode=self.args.quick
            )
            
            print(f"\nResults saved to: {self.csv_file}")
            self.print_results_table()


def main():
    parser = argparse.ArgumentParser(
        description='Automated layer optimization for snappyHexMesh',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single optimization run
  python auto_layers.py --T-rel 0.35 --n-layers 5 --er 1.10
  
  # Parameter sweep over T_rel range
  python auto_layers.py --sweep --T-rel-range 0.2 0.5 0.05 --n-layers 5
  
  # Quick mode (reuse surface mesh)
  python auto_layers.py --T-rel 0.35 --n-layers 5 --quick
  
  # Custom geometry parameters
  python auto_layers.py --T-rel 0.35 --n-layers 5 --wall-name wall_aorta --levels 2 3
        """
    )
    
    # Layer parameters
    parser.add_argument('--T-rel', type=float, default=0.35,
                        help='Relative total thickness (default: 0.35)')
    parser.add_argument('--n-layers', type=int, default=5,
                        help='Number of layers (default: 5)')
    parser.add_argument('--er', type=float, default=1.10,
                        help='Expansion ratio (default: 1.10)')
    
    # Sweep mode
    parser.add_argument('--sweep', action='store_true',
                        help='Run parameter sweep mode')
    parser.add_argument('--T-rel-range', type=float, nargs=3,
                        metavar=('START', 'END', 'STEP'),
                        default=[0.2, 0.5, 0.05],
                        help='T_rel sweep range for sweep mode (default: 0.2 0.5 0.05)')
    
    # Mesh parameters - mostly auto-detected now
    parser.add_argument('--dx-base', type=float, default=0.1,
                        help='Base cell size (auto-calculated from blockMeshDict if not specified)')
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 1],
                        help='Refinement levels (default: 1 1)')
    parser.add_argument('--location-in-mesh', type=float, nargs=3,
                        default=None,
                        help='Location in mesh (auto-detected using ray-casting if not provided)')
    
    # Geometry parameters - auto-detected from STL files
    parser.add_argument('--wall-name', type=str, default='wall_aorta',
                        help='Wall patch name (auto-detected from wall_aorta.stl, wall.stl, etc.)')
    parser.add_argument('--inlet-name', type=str, default='inlet',
                        help='Inlet patch name (auto-detected from inlet.stl, inflow.stl, etc.)')
    parser.add_argument('--resolve-feature-angle', type=float, default=30,
                        help='Feature resolution angle (default: 30)')
    
    # Mode options
    parser.add_argument('--relative', type=bool, default=None,
                        help='Use relative sizing mode (default: true)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode - reuse surface mesh if available')
    
    # Output options
    parser.add_argument('--csv-file', type=str,
                        help='Output CSV filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Validation
    if args.sweep and args.T_rel != 0.35:
        print("Warning: --T-rel is ignored in sweep mode, using --T-rel-range instead")
    
    # Check for required scripts
    required_scripts = [
        'scripts/generate_dicts.py',
        'scripts/metrics_logger.py'
    ]
    for script in required_scripts:
        if not Path(script).exists():
            print(f"Error: Required script not found: {script}")
            sys.exit(1)
    
    # Run optimization
    optimizer = LayerOptimizer(args)
    optimizer.run()


if __name__ == '__main__':
    main()