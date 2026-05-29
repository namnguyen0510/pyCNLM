import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import math
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import random
import sys
import os
import json
from datetime import datetime
import networkx as nx
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

# D-Wave topology imports
try:
    import dwave_networkx as dnx
    from minorminer import find_embedding
    DWAVE_NX_AVAILABLE = True
except ImportError:
    DWAVE_NX_AVAILABLE = False
    print("Warning: dwave-networkx not installed. Install with: pip install dwave-networkx minorminer")

# D-Wave imports
try:
    from dimod import BinaryQuadraticModel
    from neal import SimulatedAnnealingSampler
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    print("Warning: D-Wave Ocean SDK not installed. Install with: pip install dwave-ocean-sdk")

from pycnlm.core.AdaptCNLM import *
from pycnlm.utils.dataloader import *


def create_output_directory(cnf_filename: str) -> str:
    """Create output directory based on CNF filename."""
    base_name = os.path.splitext(os.path.basename(cnf_filename))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{base_name}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "topology_graphs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "analysis_plots"), exist_ok=True)
    
    print(f"\nOutput directory created: {output_dir}")
    return output_dir


def save_results_json(output_dir: str, results: dict):
    """Save all results to JSON file."""
    json_path = os.path.join(output_dir, "complete_results.json")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        else:
            return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to: {json_path}")
    return json_path


def plot_comparison_results(output_dir: str, results: dict):
    """Create comprehensive comparison plots."""
    
    # Extract data
    methods = ['Non-Reduced', 'Clique-Based', 'Cluster-Based']
    qubits = [
        results['non_reduced']['num_qubits'],
        results['clique_based']['num_qubits'],
        results['cluster_based']['num_qubits']
    ]
    sat_scores = [
        results['non_reduced']['sat_score'],
        results['clique_based']['sat_score'],
        results['cluster_based']['sat_score']
    ]
    unsat_clauses = [
        results['non_reduced']['unsatisfied_clauses'],
        results['clique_based']['unsatisfied_clauses'],
        results['cluster_based']['unsatisfied_clauses']
    ]
    
    # Create comparison figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Qubit Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    bars1 = ax1.bar(methods, qubits, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Number of Qubits', fontsize=12, fontweight='bold')
    ax1.set_title('Qubit Usage Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: SAT Score Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(methods, sat_scores, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('SAT Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Satisfaction Rate Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Perfect SAT')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Unsatisfied Clauses
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(methods, unsat_clauses, color=colors, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Unsatisfied Clauses', fontsize=12, fontweight='bold')
    ax3.set_title('Unsatisfied Clauses Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Compression Rate
    ax4 = fig.add_subplot(gs[1, 0])
    original_vars = results['problem_info']['num_variables']
    compression_rates = [(1 - q/original_vars)*100 for q in qubits]
    bars4 = ax4.bar(methods, compression_rates, color=colors, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Compression Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Qubit Compression Rate', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Embedding Success (Chimera)
    ax5 = fig.add_subplot(gs[1, 1])
    chimera_success = []
    for method in ['non_reduced', 'clique_based', 'cluster_based']:
        emb = results[method].get('embeddings', {}).get('chimera', {})
        chimera_success.append(100 if emb.get('success', False) else 0)
    bars5 = ax5.bar(methods, chimera_success, color=colors, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Embedding Success (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Chimera Embedding Success', fontsize=14, fontweight='bold')
    ax5.set_ylim([0, 105])
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars5, chimera_success):
        ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                'Success' if val == 100 else 'Failed', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Embedding Success (Pegasus)
    ax6 = fig.add_subplot(gs[1, 2])
    pegasus_success = []
    for method in ['non_reduced', 'clique_based', 'cluster_based']:
        emb = results[method].get('embeddings', {}).get('pegasus', {})
        pegasus_success.append(100 if emb.get('success', False) else 0)
    bars6 = ax6.bar(methods, pegasus_success, color=colors, edgecolor='black', linewidth=2)
    ax6.set_ylabel('Embedding Success (%)', fontsize=12, fontweight='bold')
    ax6.set_title('Pegasus Embedding Success', fontsize=14, fontweight='bold')
    ax6.set_ylim([0, 105])
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars6, pegasus_success):
        ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                'Success' if val == 100 else 'Failed', ha='center', va='bottom', fontweight='bold')
    
    # Plot 7: BitFlip Performance
    ax7 = fig.add_subplot(gs[2, :])
    for i, (method_key, method_name) in enumerate(zip(['non_reduced', 'clique_based', 'cluster_based'], methods)):
        bitflip_data = results[method_key].get('bitflip', {})
        if 'sat_history' in bitflip_data:
            iterations = range(len(bitflip_data['sat_history']))
            ax7.plot(iterations, bitflip_data['sat_history'], 
                    label=method_name, color=colors[i], linewidth=2, alpha=0.7)
    
    ax7.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax7.set_ylabel('SAT Score (%)', fontsize=12, fontweight='bold')
    ax7.set_title('BitFlip Post-Processing Performance', fontsize=14, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.suptitle('SAT Solver Methods Comparison', fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = os.path.join(output_dir, 'analysis_plots', 'comprehensive_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {output_path}")


def main():
    random.seed(42)
    np.random.seed(42)
    
    print("="*70)
    print("ADVANCED SAT SOLVER WITH COMPREHENSIVE ANALYSIS")
    print("="*70)
    print(f"D-Wave Ocean SDK: {'Available' if DWAVE_AVAILABLE else 'NOT Available'}")
    print(f"D-Wave NetworkX: {'Available' if DWAVE_NX_AVAILABLE else 'NOT Available'}")
    
    # Parse command line arguments or get user input
    if len(sys.argv) > 1:
        cnf_file = sys.argv[1]
    else:
        print("\nUsage: python main_fixed.py <filename.cnf>")
        print("Or enter CNF filename:")
        cnf_file = input("> ").strip()
        if not cnf_file:
            print("Error: No filename provided")
            sys.exit(1)
    
    # Create output directory
    output_dir = create_output_directory(cnf_file)
    
    # Initialize results dictionary
    results = {
        'problem_info': {},
        'non_reduced': {},
        'clique_based': {},
        'cluster_based': {},
        'metadata': {
            'cnf_file': cnf_file,
            'timestamp': datetime.now().isoformat(),
            'seed': 42
        }
    }
    
    # 1. Load SAT instance
    print("\n" + "="*70)
    print("STEP 1: LOADING CNF FILE")
    print("="*70)
    sat = parse_cnf_file(cnf_file)
    print(f"\n{sat}")
    
    results['problem_info'] = {
        'num_variables': sat.num_vars,
        'num_clauses': sat.num_clauses,
        'clauses': [[int(lit) for lit in clause] for clause in sat.clauses]
    }
    
    # 2. Preprocessing: Symmetry Detection
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING - SYMMETRY DETECTION")
    print("="*70)
    sym = SymmetryDetector(sat)
    orbits = sym.find_orbits()
    print(f"\nSymmetry Detection:")
    print(f"  Reduced {sat.num_vars} variables to {len(orbits)} orbits.")
    
    results['problem_info']['orbits'] = {
        'num_orbits': len(orbits),
        'orbit_groups': [[int(v+1) for v in orbit] for orbit in orbits]
    }
    
    for i, orbit in enumerate(orbits):
        if len(orbit) > 1:
            print(f"  Orbit {i}: {sorted([v+1 for v in orbit])}")
    
    # Apply simplification
    sat_simplified = sat.simplify()
    print(f"  After simplification: {sat_simplified.num_clauses} clauses")
    
    # 3. Three different encoding methods
    print("\n" + "="*70)
    print("STEP 3: ENCODING METHODS")
    print("="*70)
    
    # Confidence parameter for energy scaling
    # ε ∈ [0, 1]: controls the strength of clause satisfaction signals
    # ε = 1.0: full logical enforcement (default)
    # ε → 0: weak, uncertain reasoning
    confidence = 1.0
    
    # 3.1 Non-Reduced (Direct Encoding)
    print("\n--- Method 1: Non-Reduced (Direct) ---")
    print(f"Confidence parameter ε = {confidence}")
    non_reduced_encoder = OrbitBasedEncoder(sat, orbits, confidence=confidence)
    nr_qubit_map, nr_num_qubits = non_reduced_encoder.allocate_qubits()
    
    results['non_reduced']['num_qubits'] = nr_num_qubits
    results['non_reduced']['encoding_method'] = 'direct'
    results['non_reduced']['confidence'] = confidence
    
    # 3.2 Clique-Based Encoding
    print("\n--- Method 2: Clique-Based ---")
    print(f"Confidence parameter ε = {confidence}")
    clique_encoder = CliqueBasedEncoder(sat, orbits, confidence=confidence)
    clique_qubit_map, clique_num_qubits = clique_encoder.allocate_qubits()
    
    results['clique_based']['num_qubits'] = clique_num_qubits
    results['clique_based']['encoding_method'] = 'clique'
    results['clique_based']['num_cliques'] = len(clique_encoder.clique_groups)
    results['clique_based']['confidence'] = confidence
    
    # 3.3 Cluster-Based Encoding
    print("\n--- Method 3: Cluster-Based ---")
    print(f"Confidence parameter ε = {confidence}")
    cluster_encoder = ClusterBasedEncoder(sat, orbits, confidence=confidence)
    cluster_qubit_map, cluster_num_qubits = cluster_encoder.allocate_qubits()
    
    results['cluster_based']['num_qubits'] = cluster_num_qubits
    results['cluster_based']['encoding_method'] = 'cluster'
    results['cluster_based']['num_clusters'] = len(cluster_encoder.clusters)
    results['cluster_based']['confidence'] = confidence
    
    print(f"\nEncoding Comparison:")
    print(f"  Non-Reduced:    {nr_num_qubits} qubits")
    print(f"  Clique-Based:   {clique_num_qubits} qubits ({(1-clique_num_qubits/nr_num_qubits)*100:.1f}% reduction)")
    print(f"  Cluster-Based:  {cluster_num_qubits} qubits ({(1-cluster_num_qubits/nr_num_qubits)*100:.1f}% reduction)")
    
    # 4. D-Wave Topology Embedding
    print("\n" + "="*70)
    print("STEP 4: D-WAVE TOPOLOGY EMBEDDING")
    print("="*70)
    
    topologies = ['pegasus']
    encoders = {
        'non_reduced': non_reduced_encoder,
        'clique_based': clique_encoder,
        'cluster_based': cluster_encoder
    }
    
    for method_name, encoder in encoders.items():
        print(f"\n--- {method_name.replace('_', ' ').title()} ---")
        results[method_name]['embeddings'] = {}
        
        for topology in topologies:
            print(f"\n  Embedding to {topology.upper()}...")
            try:
                embedding_result = encoder.embed_to_dwave_topology(
                    topology=topology,
                    output_dir=output_dir,
                    method_name=method_name
                )
                
                results[method_name]['embeddings'][topology] = embedding_result
                
                if embedding_result['success']:
                    print(f"Success! Chain length: {embedding_result.get('max_chain_length', 'N/A')}")
                else:
                    print(f"Failed")
            except Exception as e:
                print(f"Error: {e}")
                results[method_name]['embeddings'][topology] = {
                    'success': False,
                    'error': str(e)
                }
    
    # 5. Solve with each method (including adaptive confidence)
    print("\n" + "="*70)
    print("STEP 5: SOLVING WITH ADAPTIVE CONFIDENCE")
    print("="*70)
    
    for method_name, encoder in encoders.items():
        print(f"\n--- Solving with {method_name.replace('_', ' ').title()} ---")
        
        # First try standard solver
        print("  Standard solver...")
        if DWAVE_AVAILABLE:
            assignment_std, energy_std = encoder.solve_with_dwave(num_reads=2000, num_sweeps=20000)
        else:
            assignment_std, energy_std = encoder.solve(steps=30000, restarts=30)
        
        valid_std, unsat_std = sat.evaluate(assignment_std)
        sat_score_std = (1 - len(unsat_std)/sat.num_clauses) * 100
        
        print(f"    SAT Score: {sat_score_std:.2f}%")
        print(f"    Unsatisfied: {len(unsat_std)}/{sat.num_clauses}")
        
        # Try adaptive confidence solver
        print("  Adaptive confidence solver...")
        try:
            assignment_adapt, energy_adapt, adapt_stats = encoder.solve_adaptive_confidence(
                steps=50000, 
                initial_conf=0.5,
                verbose=False
            )
            
            valid_adapt, unsat_adapt = sat.evaluate(assignment_adapt)
            sat_score_adapt = (1 - len(unsat_adapt)/sat.num_clauses) * 100
            
            print(f"    SAT Score: {sat_score_adapt:.2f}%")
            print(f"    Unsatisfied: {len(unsat_adapt)}/{sat.num_clauses}")
            print(f"    Improvements: {adapt_stats['improvements']}")
            print(f"    Stagnations: {adapt_stats['stagnation']}")
            
            # Use best solution
            if sat_score_adapt > sat_score_std:
                print(f"  ✓ Adaptive solver is better! (+{sat_score_adapt - sat_score_std:.2f}%)")
                assignment = assignment_adapt
                energy = energy_adapt
                valid = valid_adapt
                unsat = unsat_adapt
                sat_score = sat_score_adapt
                best_method = 'adaptive'
            else:
                print(f"  Standard solver is better")
                assignment = assignment_std
                energy = energy_std
                valid = valid_std
                unsat = unsat_std
                sat_score = sat_score_std
                best_method = 'standard'
            
            # Store both results
            results[method_name]['solution_adaptive'] = {
                'assignment': [int(x) for x in assignment_adapt],
                'energy': float(energy_adapt),
                'valid': bool(valid_adapt),
                'unsatisfied_clauses': int(len(unsat_adapt)),
                'sat_score': float(sat_score_adapt),
                'stats': {
                    'improvements': adapt_stats['improvements'],
                    'stagnation': adapt_stats['stagnation'],
                    'final_confidence': adapt_stats['confidence_history'][-1] if adapt_stats['confidence_history'] else 0.5
                }
            }
            
        except Exception as e:
            print(f"  Adaptive solver failed: {e}")
            assignment = assignment_std
            energy = energy_std
            valid = valid_std
            unsat = unsat_std
            sat_score = sat_score_std
            best_method = 'standard'
        
        # Store standard results
        results[method_name]['solution'] = {
            'assignment': [int(x) for x in assignment],
            'energy': float(energy),
            'valid': bool(valid),
            'unsatisfied_clauses': int(len(unsat)),
            'unsatisfied_clause_ids': [int(x) for x in unsat],
            'sat_score': float(sat_score),
            'best_solver': best_method
        }
        
        results[method_name]['sat_score'] = sat_score
        results[method_name]['unsatisfied_clauses'] = len(unsat)
        
        print(f"  SAT Score: {sat_score:.2f}%")
        print(f"  Unsatisfied: {len(unsat)}/{sat.num_clauses}")
        print(f"  Valid: {valid}")
    
    # 6. BitFlip Post-Processing
    print("\n" + "="*70)
    print("STEP 6: BITFLIP POST-PROCESSING")
    print("="*70)
    
    bitflip_methods = {
        'non_reduced': 'orbit',
        'clique_based': 'clique',
        'cluster_based': 'cluster'
    }
    
    for method_name, bitflip_type in bitflip_methods.items():
        print(f"\n--- BitFlip for {method_name.replace('_', ' ').title()} (by {bitflip_type}) ---")
        encoder = encoders[method_name]
        
        bf_assignment, sat_history, unsat_history = encoder.bitflip_search(
            num_iterations=10000,
            flip_prob=0.3,
            flip_type=bitflip_type
        )
        
        valid_bf, unsat_bf = sat.evaluate(bf_assignment)
        sat_score_bf = (1 - len(unsat_bf)/sat.num_clauses) * 100
        
        results[method_name]['bitflip'] = {
            'flip_type': bitflip_type,
            'assignment': [int(x) for x in bf_assignment],
            'valid': bool(valid_bf),
            'unsatisfied_clauses': int(len(unsat_bf)),
            'sat_score': float(sat_score_bf),
            'best_sat_score': float(max(sat_history)),
            'mean_sat_score': float(np.mean(sat_history)),
            'std_sat_score': float(np.std(sat_history)),
            'sat_history': [float(x) for x in sat_history],
            'unsat_history': [int(x) for x in unsat_history]
        }
        
        print(f"  Best SAT Score: {max(sat_history):.2f}%")
        print(f"  Final SAT Score: {sat_score_bf:.2f}%")
        print(f"  Mean SAT Score: {np.mean(sat_history):.2f}%")
        
        # Update if bitflip found better solution
        if sat_score_bf > results[method_name]['sat_score']:
            print(f"BitFlip improved solution!")
            results[method_name]['sat_score'] = sat_score_bf
            results[method_name]['unsatisfied_clauses'] = len(unsat_bf)
            results[method_name]['best_solution'] = 'bitflip'
        else:
            results[method_name]['best_solution'] = 'solver'
    
    # 7. Generate all plots
    print("\n" + "="*70)
    print("STEP 7: GENERATING VISUALIZATIONS")
    print("="*70)
    
    print("\nGenerating comparison plots...")
    plot_comparison_results(output_dir, results)
    
    # 8. Save results JSON
    print("\n" + "="*70)
    print("STEP 8: SAVING RESULTS")
    print("="*70)
    
    json_path = save_results_json(output_dir, results)
    
    # 9. Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print(f"\nProblem: {sat.num_vars} variables, {sat.num_clauses} clauses")
    print(f"\nBest Results:")
    
    best_method = max(results.keys() - {'problem_info', 'metadata'}, 
                     key=lambda k: results[k]['sat_score'])
    
    print(f"  Winner: {best_method.replace('_', ' ').title()}")
    print(f"  SAT Score: {results[best_method]['sat_score']:.2f}%")
    print(f"  Qubits Used: {results[best_method]['num_qubits']}")
    print(f"  Unsatisfied: {results[best_method]['unsatisfied_clauses']}/{sat.num_clauses}")
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"  - JSON results: complete_results.json")
    print(f"  - Topology graphs: topology_graphs/")
    print(f"  - Analysis plots: analysis_plots/")
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
