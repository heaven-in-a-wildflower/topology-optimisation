#!/usr/bin/env python3
"""
Thermal Performance Analysis Tool for Heat Exchanger Design

Extracts and visualizes key thermal performance metrics from MFEM solutions
to aid in heat exchanger optimization.

Usage: python thermal_analysis.py <solution_dir> [--output plots/]
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re

class ThermalAnalyzer:
    """Analyze thermal solutions for heat exchanger performance"""
    
    def __init__(self, solution_file, mesh_file=None):
        self.solution_file = Path(solution_file)
        self.mesh_file = Path(mesh_file) if mesh_file else None
        self.temperatures = None
        self.mesh_data = None
        self.design_name = solution_file.stem.replace('_steady_solution', '')
        
    def load_solution(self):
        """Load temperature field from MFEM solution file"""
        with open(self.solution_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines until we reach the data
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('FiniteElementSpace') \
               and not line.startswith('FiniteElementCollection') \
               and not line.startswith('VDim') \
               and not line.startswith('Ordering'):
                data_start = i
                break
        
        # Read temperature values
        temps = []
        for line in lines[data_start:]:
            line = line.strip()
            if line:
                try:
                    temps.append(float(line))
                except ValueError:
                    continue
        
        self.temperatures = np.array(temps)
        print(f"Loaded {len(self.temperatures)} temperature values from {self.solution_file.name}")
        
    def load_mesh(self):
        """Load mesh data for volume calculations"""
        if self.mesh_file is None:
            return
            
        with open(self.mesh_file, 'r') as f:
            content = f.read()
        
        # Extract number of elements
        elements_match = re.search(r'elements\s+(\d+)', content)
        if elements_match:
            n_elements = int(elements_match.group(1))
            self.mesh_data = {'n_elements': n_elements}
            print(f"Mesh has {n_elements} elements")
    
    def compute_statistics(self):
        """Compute basic thermal statistics"""
        T = self.temperatures
        
        stats = {
            'T_min': np.min(T),
            'T_max': np.max(T),
            'T_mean': np.mean(T),
            'T_median': np.median(T),
        }
        
        return stats
    
    def print_report(self, stats):
        """Print formatted thermal performance report"""
        print("\n" + "="*70)
        print(f"THERMAL PERFORMANCE REPORT: {self.design_name}")
        print("="*70)
        
        print("\n1. BASIC TEMPERATURE STATISTICS")
        print(f"   Min Temperature:        {stats['T_min']:8.6f} °C")
        print(f"   Max Temperature:        {stats['T_max']:8.6f} °C")
        print(f"   Mean Temperature:       {stats['T_mean']:8.6f} °C")
        print(f"   Median Temperature:     {stats['T_median']:8.6f} °C")
        print(f"   Std Deviation:          {stats['T_std']:8.6f} °C")
        
        print("\n2. HEAT EXCHANGER PERFORMANCE METRICS")
        print(f"   Temperature Range:      {stats['T_range']:8.6f} °C  ← Lower is better")
        print(f"   Uniformity Index:       {stats['T_uniformity']:8.4f}     ← Higher is better (0-1)")
        print(f"   Coeff. of Variation:    {stats['coefficient_of_variation']:8.4f}     ← Lower is better")
        
        print("\n3. HOT SPOT ANALYSIS")
        print(f"   99th Percentile Temp:   {stats['T_p99']:8.6f} °C")
        print(f"   95th Percentile Temp:   {stats['T_p95']:8.6f} °C")
        print(f"   Hot Spot Fraction:      {stats['hot_spot_fraction']*100:8.6f} %  ← % above 95th percentile")
        
        print("\n4. TEMPERATURE DISTRIBUTION")
        print(f"   5th Percentile:         {stats['T_p5']:8.6f} °C")
        print(f"   95th Percentile:        {stats['T_p95']:8.6f} °C")
        print(f"   P90-P10 Range:          {stats['T_p90_p10_range']:8.6f} °C  ← Narrower is better")
        
        print("\n5. DISTRIBUTION CHARACTERISTICS")
        print(f"   Skewness:               {stats['skewness']:8.4f}     ← 0 = symmetric")
        print(f"   Kurtosis:               {stats['kurtosis']:8.4f}     ← 0 = normal distribution")
        
        print("\n" + "="*70)
        
        # Key performance indicator
        score = self._compute_performance_score(stats)
        print(f"\nPERFORMANCE SCORE: {score:.6f}/100")
        print("  (Higher is better - based on uniformity and low max temp)")
        print("="*70 + "\n")
    
    def _compute_performance_score(self, stats):
        """Compute overall performance score (0-100)"""
        # Score components (each 0-1)
        uniformity_score = stats['T_uniformity']
        
        # Lower CoV is better
        cv_score = max(0, 1 - stats['coefficient_of_variation'] * 10)
        
        # Lower max temperature is better (assuming target is ~100°C)
        max_temp_score = max(0, 1 - (stats['T_max'] - 100) / 100)
        
        # Lower hot spot fraction is better
        hotspot_score = 1 - min(1, stats['hot_spot_fraction'] * 5)
        
        # Weighted average
        score = (0.3 * uniformity_score + 
                0.2 * cv_score + 
                0.3 * max_temp_score + 
                0.2 * hotspot_score) * 100
        
        return score
    
    def plot_distributions(self, output_dir=None):
        """Create visualization plots"""
        T = self.temperatures
        stats = self.compute_statistics()
        
        fig = plt.figure(figsize=(15, 5))
        
        # 1. Temperature histogram
        ax1 = plt.subplot(1, 3, 1)
        n, bins, patches = ax1.hist(T, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(stats['T_mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['T_mean']:.6f}°C")
        ax1.axvline(stats['T_median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['T_median']:.6f}°C")
        ax1.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Temperature Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative distribution
        ax2 = plt.subplot(1, 3, 2)
        sorted_T = np.sort(T)
        cdf = np.arange(1, len(T) + 1) / len(T)
        ax2.plot(sorted_T, cdf * 100, linewidth=2, color='darkblue')
        ax2.axhline(50, color='gray', linestyle=':', alpha=0.5)
        ax2.axhline(95, color='red', linestyle=':', alpha=0.5, label='95th percentile')
        ax2.axhline(5, color='blue', linestyle=':', alpha=0.5, label='5th percentile')
        ax2.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Box plot with percentiles
        ax3 = plt.subplot(1, 3, 3)
        bp = ax3.boxplot([T], vert=True, patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_edgecolor('darkblue')
        bp['medians'][0].set_color('red')
        bp['medians'][0].set_linewidth(2)
        ax3.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax3.set_title('Temperature Box Plot', fontsize=14, fontweight='bold')
        ax3.set_xticklabels(['All Points'])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add annotations
        ax3.text(1.3, stats['T_max'], f"Max: {stats['T_max']:.6f}°C", 
                va='center', fontsize=10, color='red', fontweight='bold')
        ax3.text(1.3, stats['T_min'], f"Min: {stats['T_min']:.6f}°C", 
                va='center', fontsize=10, color='blue', fontweight='bold')
        
        plt.suptitle(f'Thermal Analysis: {self.design_name}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / f'{self.design_name}_analysis.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        
        return fig

def batch_analyze(solution_dir, output_dir='thermal_analysis_plots'):
    """Analyze all solutions in a directory"""
    solution_dir = Path(solution_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all solution files
    solution_files = list(solution_dir.glob('*steady_solution.000000'))
    
    if not solution_files:
        print(f"No solution files found in {solution_dir}")
        return
    
    print(f"\nFound {len(solution_files)} solution files to analyze\n")
    
    for sol_file in sorted(solution_files):
        # Find corresponding mesh file
        mesh_file = sol_file.parent / sol_file.name.replace('steady_solution', 'mfem_mesh')
        if not mesh_file.exists():
            mesh_file = None
        
        analyzer = ThermalAnalyzer(sol_file, mesh_file)
        analyzer.load_solution()
        if mesh_file:
            analyzer.load_mesh()
        
        # Create plots
        analyzer.plot_distributions(output_dir)
        plt.close()
        
        print()

def create_comparison_plot(results, output_dir):
    """Create comparison plots across all designs"""
    pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python thermal_analysis.py <solution_directory> [output_directory]")
        print("\nExample: python thermal_analysis.py heat_cond_steady_state_solns/ plots/")
        sys.exit(1)
    
    solution_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'thermal_analysis_plots'
    
    batch_analyze(solution_dir, output_dir)
    
    print("\nAnalysis complete!")