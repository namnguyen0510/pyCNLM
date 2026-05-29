from typing import List, Tuple, Dict, Set
from collections import defaultdict
from pycnlm.utils.dataloader import SATInstance
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import random
import math
import os

# D-Wave topology imports
try:
    import dwave_networkx as dnx
    from minorminer import find_embedding
    DWAVE_NX_AVAILABLE = True
except ImportError:
    DWAVE_NX_AVAILABLE = False

# D-Wave imports
try:
    from dimod import BinaryQuadraticModel
    from neal import SimulatedAnnealingSampler
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False


class SymmetryDetector:
    """Detects structural symmetries to reduce problem size."""
    
    def __init__(self, sat: SATInstance):
        self.sat = sat
        
    def find_orbits(self) -> List[Set[int]]:
        """
        Enhanced symmetry detection based on variable participation patterns.
        Returns groups of variables (orbits) that are structurally interchangeable.
        """
        var_signatures = defaultdict(list)
        
        for var in range(1, self.sat.num_vars + 1):
            sig_parts = []
            for i, clause in enumerate(self.sat.clauses):
                clause_size = len(clause)
                if var in clause:
                    sig_parts.append(f"P{clause_size}_{i}")
                if -var in clause:
                    sig_parts.append(f"N{clause_size}_{i}")
            
            co_occurs = set()
            for clause in self.sat.clauses:
                if var in clause or -var in clause:
                    for lit in clause:
                        if abs(lit) != var:
                            co_occurs.add(abs(lit))
            
            sig_parts.append(f"CO{len(co_occurs)}")
            var_signatures[tuple(sorted(sig_parts))].append(var - 1)
            
        orbits = []
        assigned = set()
        
        for sig_vars in var_signatures.values():
            if not sig_vars:
                continue
            orbits.append(set(sig_vars))
            assigned.update(sig_vars)
            
        for i in range(self.sat.num_vars):
            if i not in assigned:
                orbits.append({i})
                
        return orbits


class OrbitBasedEncoder:
    """Direct encoding without compression - one qubit per orbit."""
    def __init__(self, sat: SATInstance, orbits: List[Set[int]], confidence: float = 1.0):
        """
        Initialize encoder with confidence-scaled energy.
        
        Args:
            sat: SAT instance
            orbits: List of orbit groups
            confidence: Confidence parameter ε ∈ [0, 1] for gradient scaling
                       - ε = 1.0: full logical enforcement (default)
                       - ε → 0: weak, uncertain reasoning
        """
        self.sat = sat
        self.orbits = list(orbits)
        self.orbit_map = {var: i for i, group in enumerate(self.orbits) for var in group}
        self.num_orbits = len(self.orbits)
        self.qubit_map = {}
        self.total_qubits = 0
        self.confidence = confidence  # ε parameter for energy scaling
        
        # Enhanced: Clause-specific confidences
        self.clause_confidences = self._compute_clause_confidences()
        
    def _compute_clause_confidences(self) -> Dict[int, float]:
        """
        Compute clause-specific confidence values based on structural analysis.
        
        Smaller, more constrained clauses get higher confidence (they're harder).
        """
        confidences = {}
        
        # Collect variable statistics
        var_stats = defaultdict(lambda: {'count': 0, 'clauses': []})
        for i, clause in enumerate(self.sat.clauses):
            for lit in clause:
                var = abs(lit) - 1
                var_stats[var]['count'] += 1
                var_stats[var]['clauses'].append(i)
        
        # Compute confidence for each clause
        for i, clause in enumerate(self.sat.clauses):
            clause_size = len(clause)
            
            # Factor 1: Size-based confidence (smaller = harder = higher confidence)
            size_conf = 1.0 / (1.0 + 0.2 * (clause_size - 1))
            
            # Factor 2: Variable rarity (clauses with rare variables are important)
            var_rarities = []
            for lit in clause:
                var = abs(lit) - 1
                if var in var_stats:
                    freq = var_stats[var]['count']
                    rarity = 1.0 / (1.0 + 0.1 * freq)
                    var_rarities.append(rarity)
            
            rarity_conf = np.mean(var_rarities) if var_rarities else 0.5
            
            # Combine factors
            confidence = 0.7 * size_conf + 0.3 * rarity_conf
            
            # Clamp to reasonable range
            confidence = max(0.4, min(1.0, confidence))
            confidences[i] = confidence
        
        return confidences
        
    def allocate_qubits(self) -> Tuple[Dict, int]:
        """Allocate one qubit per orbit (direct mapping)."""
        self.total_qubits = self.num_orbits
        
        for i in range(self.num_orbits):
            self.qubit_map[i] = {
                'q_index': i,
                'orbit_id': i
            }
        
        print(f"Non-Reduced Encoding: {self.total_qubits} qubits (1 per orbit)")
        return self.qubit_map, self.total_qubits
    
    def embed_to_dwave_topology(self, topology='chimera', output_dir=None, method_name='non_reduced'):
        """Embed the problem graph to D-Wave topology using actual qubit-level structure."""
        if not DWAVE_NX_AVAILABLE:
            return {'success': False, 'error': 'dwave-networkx not available'}
        
        # Create problem graph at QUBIT level (not orbit level)
        problem_graph = nx.Graph()
        for i in range(self.total_qubits):
            problem_graph.add_node(i)
        
        # For non-reduced: simple qubit-to-qubit interactions based on clauses
        # Map orbits to their qubit indices
        for clause in self.sat.clauses:
            qubit_indices = []
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in self.orbit_map:
                    oid = self.orbit_map[var_idx]
                    # Get the qubit index for this orbit
                    q_idx = self.qubit_map[oid]['q_index']
                    if q_idx not in qubit_indices:
                        qubit_indices.append(q_idx)
            
            # Add edges between all qubits in the same clause
            for i in range(len(qubit_indices)):
                for j in range(i + 1, len(qubit_indices)):
                    problem_graph.add_edge(qubit_indices[i], qubit_indices[j])
        
        # Create hardware graph
        if topology == 'chimera':
            hardware_graph = dnx.chimera_graph(16, 16, 4)
        elif topology == 'pegasus':
            hardware_graph = dnx.pegasus_graph(16)
        else:
            return {'success': False, 'error': f'Unknown topology: {topology}'}
        
        # Find embedding
        try:
            embedding = find_embedding(problem_graph.edges(), hardware_graph.edges(), 
                                      random_seed=42, timeout=30)
            
            if not embedding:
                return {'success': False, 'error': 'Embedding not found'}
            
            # Calculate statistics
            chain_lengths = [len(chain) for chain in embedding.values()]
            max_chain = max(chain_lengths) if chain_lengths else 0
            avg_chain = np.mean(chain_lengths) if chain_lengths else 0
            
            result = {
                'success': True,
                'topology': topology,
                'num_qubits': self.total_qubits,
                'num_hardware_qubits': len(hardware_graph.nodes()),
                'embedding_size': len(embedding),
                'max_chain_length': int(max_chain),
                'avg_chain_length': float(avg_chain),
                'total_chains': len(embedding),
                'problem_edges': problem_graph.number_of_edges(),
                'encoding_type': 'non_reduced'
            }
            
            # Visualize
            if output_dir:
                self._visualize_dwave_embedding(problem_graph, hardware_graph, embedding, 
                                               topology, output_dir, method_name)
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        # Create hardware graph
        if topology == 'chimera':
            hardware_graph = dnx.chimera_graph(16, 16, 4)
        elif topology == 'pegasus':
            hardware_graph = dnx.pegasus_graph(16)
        else:
            return {'success': False, 'error': f'Unknown topology: {topology}'}
        
        # Find embedding
        try:
            embedding = find_embedding(problem_graph.edges(), hardware_graph.edges(), 
                                      random_seed=42, timeout=30)
            
            if not embedding:
                return {'success': False, 'error': 'Embedding not found'}
            
            # Calculate statistics
            chain_lengths = [len(chain) for chain in embedding.values()]
            max_chain = max(chain_lengths) if chain_lengths else 0
            avg_chain = np.mean(chain_lengths) if chain_lengths else 0
            
            result = {
                'success': True,
                'topology': topology,
                'num_qubits': self.total_qubits,
                'num_hardware_qubits': len(hardware_graph.nodes()),
                'embedding_size': len(embedding),
                'max_chain_length': int(max_chain),
                'avg_chain_length': float(avg_chain),
                'total_chains': len(embedding)
            }
            
            # Visualize
            if output_dir:
                self._visualize_dwave_embedding(problem_graph, hardware_graph, embedding, 
                                               topology, output_dir, method_name)
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _visualize_dwave_embedding(self, problem_graph, hardware_graph, embedding, 
                                   topology, output_dir, method_name):
        """Visualize the embedding on D-Wave hardware."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot problem graph
        pos_problem = nx.spring_layout(problem_graph, k=2, iterations=50, seed=42)
        ax1.set_title(f'Problem Graph - {method_name.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold')
        nx.draw_networkx_nodes(problem_graph, pos_problem, node_color='lightblue',
                              node_size=500, ax=ax1, edgecolors='black', linewidths=2)
        nx.draw_networkx_edges(problem_graph, pos_problem, edge_color='gray',
                              width=2, alpha=0.6, ax=ax1)
        nx.draw_networkx_labels(problem_graph, pos_problem, font_size=8, 
                               font_weight='bold', ax=ax1)
        ax1.axis('off')
        
        # Plot hardware graph with embedding
        if topology == 'chimera':
            pos_hardware = dnx.chimera_layout(hardware_graph)
        elif topology == 'pegasus':
            pos_hardware = dnx.pegasus_layout(hardware_graph)
        
        ax2.set_title(f'{topology.capitalize()} Hardware Graph with Embedding',
                     fontsize=14, fontweight='bold')
        
        # Draw hardware nodes
        nx.draw_networkx_nodes(hardware_graph, pos_hardware, node_color='lightgray',
                              node_size=20, ax=ax2, alpha=0.3)
        nx.draw_networkx_edges(hardware_graph, pos_hardware, edge_color='lightgray',
                              width=0.5, alpha=0.2, ax=ax2)
        
        # Draw embedded chains
        colors = plt.cm.tab20(np.linspace(0, 1, len(embedding)))
        for idx, (logical_qubit, chain) in enumerate(embedding.items()):
            chain_pos = {node: pos_hardware[node] for node in chain}
            nx.draw_networkx_nodes(hardware_graph, chain_pos, nodelist=chain,
                                  node_color=[colors[idx]], node_size=50, ax=ax2,
                                  edgecolors='black', linewidths=1)
        
        ax2.axis('off')
        
        # Save
        output_path = os.path.join(output_dir, 'topology_graphs', 
                                  f'{method_name}_{topology}_embedding.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  Topology graph saved: {output_path}")
    
    def build_bqm(self) -> 'BinaryQuadraticModel':
        """
        Build Binary Quadratic Model for D-Wave with confidence-scaled energy.
        
        The confidence parameter ε modulates the strength of clause satisfaction signals:
            penalty_ε = ε · penalty_base
            
        This implements the modified free energy:
            F_ε(x) = -Σ_{m,j} log(1 + exp(ε · h · φ_mj(x)))
        """
        if not DWAVE_AVAILABLE:
            raise ImportError("D-Wave Ocean SDK not available")
        
        bqm = BinaryQuadraticModel('BINARY')
        
        # Add variables for each qubit
        for i in range(self.total_qubits):
            bqm.add_variable(f'q{i}', 0.0)
        
        # Add clause constraints with confidence scaling
        for clause in self.sat.clauses:
            qubit_lits = []
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in self.orbit_map:
                    oid = self.orbit_map[var_idx]
                    q_idx = self.qubit_map[oid]['q_index']
                    qubit_lits.append((q_idx, lit > 0))
            
            # Base penalty scaled by confidence parameter
            penalty = self.confidence * 1.0
            
            # Linear terms: reward satisfying the clause
            for q_idx, sign in qubit_lits:
                var_name = f'q{q_idx}'
                if sign:
                    # Positive literal: reward when qubit is 1
                    bqm.add_variable(var_name, -penalty)
                else:
                    # Negative literal: reward when qubit is 0
                    bqm.add_variable(var_name, penalty)
            
            # Quadratic terms: prevent conflicts (also scaled by confidence)
            for i in range(len(qubit_lits)):
                for j in range(i + 1, len(qubit_lits)):
                    q1, sign1 = qubit_lits[i]
                    q2, sign2 = qubit_lits[j]
                    
                    # Penalize if both negative literals are true
                    if not sign1 and not sign2:
                        bqm.add_interaction(f'q{q1}', f'q{q2}', penalty * 0.5)
        
        return bqm
    
    def solve_with_dwave(self, num_reads=2000, num_sweeps=20000):
        """Solve using D-Wave simulated annealing - Non-Reduced version."""
        if not DWAVE_AVAILABLE:
            return self.solve()
        
        bqm = self.build_bqm()
        sampler = SimulatedAnnealingSampler()
        
        sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps,
                                  beta_range=[0.1, 10.0])
        
        best_sample = sampleset.first.sample
        
        # Decode to assignment - for non-reduced, direct mapping
        assignment = [False] * self.sat.num_vars
        for oid in range(self.num_orbits):
            q_idx = self.qubit_map[oid]['q_index']
            val = best_sample.get(f'q{q_idx}', 0)
            for var in self.orbits[oid]:
                assignment[var] = bool(val)
        
        _, unsat = self.sat.evaluate(assignment)
        return assignment, len(unsat)
    
    def solve(self, steps=30000, restarts=30):
        """Custom solver with simulated annealing."""
        best_energy = float('inf')
        best_assignment = None
        
        for restart in range(restarts):
            state = np.random.choice([0, 1], size=self.total_qubits)
            curr_energy = self._compute_energy(state)
            
            T = 10.0
            decay = 0.995
            
            for step in range(steps):
                idx = random.randint(0, self.total_qubits - 1)
                state[idx] = 1 - state[idx]
                
                new_energy = self._compute_energy(state)
                delta = new_energy - curr_energy
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    curr_energy = new_energy
                else:
                    state[idx] = 1 - state[idx]
                
                T *= decay
                
                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_assignment = state.copy()
                    
                    if best_energy == 0:
                        break
            
            if best_energy == 0:
                break
        
        # Decode
        assignment = [False] * self.sat.num_vars
        for oid in range(self.num_orbits):
            val = best_assignment[oid]
            for var in self.orbits[oid]:
                assignment[var] = bool(val)
        
        return assignment, best_energy
    
    def solve_adaptive_confidence(self, steps=50000, initial_conf=0.5, verbose=False):
        """
        Enhanced solver with adaptive confidence scheduling.
        
        Strategy:
        1. Start with low confidence (exploration)
        2. Gradually increase confidence (exploitation)
        3. Adapt based on improvement rate
        4. Use clause-specific confidences
        
        Returns:
            (assignment, energy, statistics)
        """
        # Save original confidence
        original_conf = self.confidence
        
        # Initialize
        state = np.random.choice([0, 1], size=self.total_qubits)
        best_state = state.copy()
        best_energy = float('inf')
        
        # Adaptive parameters
        self.confidence = initial_conf
        target_conf = 1.0
        adaptation_rate = (target_conf - initial_conf) / steps
        
        # Temperature schedule
        T = 10.0
        T_min = 0.01
        cooling_rate = (T_min / T) ** (1.0 / steps)
        
        # Track statistics
        stats = {
            'energy_history': [],
            'confidence_history': [],
            'improvements': 0,
            'stagnation': 0
        }
        
        last_improvement_step = 0
        
        for step in range(steps):
            # Adaptive confidence update
            if step - last_improvement_step > 1000:
                # Stagnating: reduce confidence for more exploration
                self.confidence = max(0.3, self.confidence - 0.1)
                last_improvement_step = step
                stats['stagnation'] += 1
            else:
                # Progressing: increase confidence
                self.confidence = min(target_conf, initial_conf + step * adaptation_rate)
            
            stats['confidence_history'].append(self.confidence)
            
            # Select variable to flip (weighted by clause confidences)
            if random.random() < 0.7:
                # Confidence-guided selection
                idx = self._select_variable_smart(state)
            else:
                # Random selection
                idx = random.randint(0, self.total_qubits - 1)
            
            # Flip
            state[idx] = 1 - state[idx]
            
            # Compute energy with clause-specific confidences
            new_energy = self._compute_energy(state, use_clause_confidences=True)
            
            # Current energy for comparison
            state[idx] = 1 - state[idx]
            curr_energy = self._compute_energy(state, use_clause_confidences=True)
            state[idx] = 1 - state[idx]
            
            delta = new_energy - curr_energy
            
            # Accept or reject
            if delta < 0 or random.random() < math.exp(-delta / T):
                curr_energy = new_energy
                
                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_state = state.copy()
                    last_improvement_step = step
                    stats['improvements'] += 1
                    
                    if verbose and step % 5000 == 0:
                        sat_score = self._compute_sat_score_from_state(best_state)
                        print(f"  Step {step}: Energy={best_energy:.2f}, SAT={sat_score:.2f}%, ε={self.confidence:.3f}")
                    
                    if best_energy == 0:
                        break
            else:
                state[idx] = 1 - state[idx]
            
            # Cool down
            T *= cooling_rate
            
            stats['energy_history'].append(best_energy)
        
        # Decode best solution
        assignment = [False] * self.sat.num_vars
        for oid in range(self.num_orbits):
            val = best_state[oid]
            for var in self.orbits[oid]:
                assignment[var] = bool(val)
        
        # Restore original confidence
        self.confidence = original_conf
        
        stats['final_energy'] = best_energy
        stats['final_sat_score'] = self._compute_sat_score_from_state(best_state)
        
        return assignment, best_energy, stats
    
    def _select_variable_smart(self, state) -> int:
        """
        Select variable to flip based on unsatisfied clause analysis.
        
        Prioritizes variables that appear in high-confidence unsatisfied clauses.
        """
        active_orbits = set(i for i in range(self.total_qubits) if state[i] == 1)
        
        # Score each variable
        var_scores = defaultdict(float)
        
        for i, clause in enumerate(self.sat.clauses):
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in self.orbit_map:
                    oid = self.orbit_map[var_idx]
                    is_true = oid in active_orbits
                    if (lit > 0 and is_true) or (lit < 0 and not is_true):
                        satisfied = True
                        break
            
            # If unsatisfied, variables in it are important
            if not satisfied:
                clause_conf = self.clause_confidences.get(i, 1.0)
                for lit in clause:
                    var_idx = abs(lit) - 1
                    if var_idx in self.orbit_map:
                        oid = self.orbit_map[var_idx]
                        var_scores[oid] += clause_conf
        
        if not var_scores:
            return random.randint(0, self.total_qubits - 1)
        
        # Probabilistic selection weighted by scores
        vars_list = list(var_scores.keys())
        weights = [var_scores[v] for v in vars_list]
        total = sum(weights)
        
        if total > 0:
            probs = [w / total for w in weights]
            return np.random.choice(vars_list, p=probs)
        else:
            return random.choice(vars_list)
    
    def _compute_sat_score_from_state(self, state) -> float:
        """Compute SAT score from qubit state."""
        active_orbits = set(i for i in range(self.total_qubits) if state[i] == 1)
        
        unsatisfied = 0
        for clause in self.sat.clauses:
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in self.orbit_map:
                    oid = self.orbit_map[var_idx]
                    is_true = oid in active_orbits
                    if (lit > 0 and is_true) or (lit < 0 and not is_true):
                        satisfied = True
                        break
            if not satisfied:
                unsatisfied += 1
        
        return (1 - unsatisfied / self.sat.num_clauses) * 100
    
    def _compute_energy(self, state, use_clause_confidences: bool = True):
        """
        Compute confidence-scaled energy with optional clause-specific weights.
        
        Energy formulation:
            E_ε(x, h) = Σ_m Σ_j ε · ε_clause_j · h_mj · φ_mj(x)
        
        where:
            - ε is the global confidence parameter
            - ε_clause_j is the clause-specific confidence (if enabled)
            - h_mj represents hidden clause-selector variables (active orbits)
            - φ_mj(x) represents clause satisfaction signals
            
        For SAT problems, this translates to:
            E_ε = ε · Σ_j (ε_clause_j · unsat_j)
            
        Args:
            state: Current qubit state
            use_clause_confidences: Whether to use clause-specific confidences
        """
        active_orbits = set(i for i in range(self.total_qubits) if state[i] == 1)
        
        total_energy = 0.0
        
        for i, clause in enumerate(self.sat.clauses):
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in self.orbit_map:
                    oid = self.orbit_map[var_idx]
                    is_true = oid in active_orbits
                    if (lit > 0 and is_true) or (lit < 0 and not is_true):
                        satisfied = True
                        break
            
            if not satisfied:
                if use_clause_confidences and i in self.clause_confidences:
                    # Use clause-specific confidence
                    clause_conf = self.clause_confidences[i]
                    total_energy += self.confidence * clause_conf
                else:
                    # Use only global confidence
                    total_energy += self.confidence
        
        return total_energy
    
    def bitflip_search(self, num_iterations=10000, flip_prob=0.3, flip_type='orbit'):
        """BitFlip post-processing search."""
        # Initialize random assignment
        assignment = [random.choice([True, False]) for _ in range(self.sat.num_vars)]
        
        sat_history = []
        unsat_history = []
        
        best_assignment = assignment[:]
        best_unsat = self.sat.num_clauses
        
        for iteration in range(num_iterations):
            # Flip based on orbit
            if random.random() < flip_prob:
                oid = random.randint(0, self.num_orbits - 1)
                for var in self.orbits[oid]:
                    assignment[var] = not assignment[var]
            
            # Evaluate
            valid, unsat = self.sat.evaluate(assignment)
            num_unsat = len(unsat)
            sat_score = (1 - num_unsat / self.sat.num_clauses) * 100
            
            sat_history.append(sat_score)
            unsat_history.append(num_unsat)
            
            if num_unsat < best_unsat:
                best_unsat = num_unsat
                best_assignment = assignment[:]
        
        return best_assignment, sat_history, unsat_history


class CliqueBasedEncoder(OrbitBasedEncoder):
    """Clique-based compression encoder with confidence-scaled energy."""
    
    def __init__(self, sat: SATInstance, orbits: List[Set[int]], confidence: float = 1.0):
        super().__init__(sat, orbits, confidence)
        self.clique_groups = []
        
    def allocate_qubits(self) -> Tuple[Dict, int]:
        """Allocate qubits using clique-based compression."""
        adj = self.build_conflict_graph()
        self.clique_groups = self.partition_into_cliques(adj)
        
        self.total_qubits = 0
        self.qubit_map = {}
        
        for group in self.clique_groups:
            if not group:
                continue
            
            bits_needed = math.ceil(math.log2(len(group) + 1)) if len(group) > 1 else 1
            
            for oid in group:
                self.qubit_map[oid] = {
                    'q_start': self.total_qubits,
                    'q_count': bits_needed,
                    'group': group,
                    'group_size': len(group)
                }
            
            self.total_qubits += bits_needed
        
        print(f"Clique-Based Encoding: {self.total_qubits} qubits from {len(self.clique_groups)} cliques")
        return self.qubit_map, self.total_qubits
    
    def build_conflict_graph(self) -> Dict[int, Set[int]]:
        """Build conflict graph."""
        adj = defaultdict(set)
        
        for clause in self.sat.clauses:
            orbit_lits = []
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in self.orbit_map:
                    oid = self.orbit_map[var_idx]
                    orbit_lits.append((oid, lit > 0))
            
            if len(orbit_lits) == 2:
                (o1, s1), (o2, s2) = orbit_lits
                if not s1 and not s2:
                    adj[o1].add(o2)
                    adj[o2].add(o1)
            
            if all(not sign for _, sign in orbit_lits):
                for i in range(len(orbit_lits)):
                    for j in range(i + 1, len(orbit_lits)):
                        o1, o2 = orbit_lits[i][0], orbit_lits[j][0]
                        if o1 != o2:
                            adj[o1].add(o2)
                            adj[o2].add(o1)
        
        return adj
    
    def partition_into_cliques(self, adj: Dict[int, Set[int]]) -> List[List[int]]:
        """Partition into cliques."""
        nodes = set(range(self.num_orbits))
        cliques = []
        used = set()
        
        # Simple greedy clique finding
        while nodes - used:
            remaining = nodes - used
            if not remaining:
                break
            
            # Start with a random node
            seed = min(remaining)
            clique = {seed}
            
            # Greedily add nodes that are connected to all in clique
            for node in remaining:
                if node == seed:
                    continue
                if all(node in adj[c] or node == c for c in clique):
                    clique.add(node)
            
            cliques.append(sorted(list(clique)))
            used.update(clique)
        
        # Add singletons for uncovered
        for node in nodes - used:
            cliques.append([node])
        
        return cliques
    
    def _compute_energy(self, state, use_clause_confidences: bool = True):
        """
        Compute confidence-scaled energy with optional clause-specific weights.
        
        Energy formulation:
            E_ε(x, h) = Σ_m Σ_j ε · ε_clause_j · h_mj · φ_mj(x)
        
        where:
            - ε is the global confidence parameter
            - ε_clause_j is the clause-specific confidence (if enabled)
            - h_mj represents hidden clause-selector variables (active orbits)
            - φ_mj(x) represents clause satisfaction signals
            
        For SAT problems, this translates to:
            E_ε = ε · Σ_j (ε_clause_j · unsat_j)
            
        Args:
            state: Current qubit state
            use_clause_confidences: Whether to use clause-specific confidences
        """
        active_orbits = set()
        
        for group in self.clique_groups:
            if not group:
                continue
            
            sample_oid = group[0]
            info = self.qubit_map[sample_oid]
            start = info['q_start']
            count = info['q_count']
            
            val = 0
            for i in range(count):
                if state[start + i] == 1:
                    val += (1 << i)
            
            if val > 0 and val <= len(group):
                active_orbits.add(group[val - 1])
        
        total_energy = 0.0
        
        for i, clause in enumerate(self.sat.clauses):
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in self.orbit_map:
                    oid = self.orbit_map[var_idx]
                    is_true = oid in active_orbits
                    if (lit > 0 and is_true) or (lit < 0 and not is_true):
                        satisfied = True
                        break
            
            if not satisfied:
                if use_clause_confidences and i in self.clause_confidences:
                    # Use clause-specific confidence
                    clause_conf = self.clause_confidences[i]
                    total_energy += self.confidence * clause_conf
                else:
                    # Use only global confidence
                    total_energy += self.confidence
        
        return total_energy
    
    def bitflip_search(self, num_iterations=10000, flip_prob=0.3, flip_type='clique'):
        """BitFlip by clique groups - flips a percentage of bits within the selected group."""
        assignment = [random.choice([True, False]) for _ in range(self.sat.num_vars)]
        
        sat_history = []
        unsat_history = []
        best_assignment = assignment[:]
        best_unsat = self.sat.num_clauses
        
        for iteration in range(num_iterations):
            if random.random() < flip_prob:
                # Select a random clique group
                group_idx = random.randint(0, len(self.clique_groups) - 1)
                group = self.clique_groups[group_idx]
                
                # Flip bits within the group with probability flip_prob
                for oid in group:
                    if random.random() < flip_prob:
                        for var in self.orbits[oid]:
                            assignment[var] = not assignment[var]
            
            valid, unsat = self.sat.evaluate(assignment)
            num_unsat = len(unsat)
            sat_score = (1 - num_unsat / self.sat.num_clauses) * 100
            
            sat_history.append(sat_score)
            unsat_history.append(num_unsat)
            
            if num_unsat < best_unsat:
                best_unsat = num_unsat
                best_assignment = assignment[:]
        
        return best_assignment, sat_history, unsat_history
    
    def build_bqm(self) -> 'BinaryQuadraticModel':
        """
        Build Binary Quadratic Model for D-Wave - Clique-Based version with confidence scaling.
        
        Penalties are scaled by confidence parameter ε.
        """
        if not DWAVE_AVAILABLE:
            raise ImportError("D-Wave Ocean SDK not available")
        
        bqm = BinaryQuadraticModel('BINARY')
        
        # Add variables for each qubit
        for i in range(self.total_qubits):
            bqm.add_variable(f'q{i}', 0.0)
        
        # Add one-hot constraints for each clique
        # Only one orbit in each clique should be active
        for group in self.clique_groups:
            if not group or len(group) <= 1:
                continue
            
            sample_oid = group[0]
            info = self.qubit_map[sample_oid]
            q_start = info['q_start']
            q_count = info['q_count']
            
            # One-hot encoding constraint: exactly one configuration
            # Keep this unscaled to maintain structural constraints
            one_hot_penalty = 2.0
            
            # Penalize having multiple bits set (should represent one value)
            for i in range(q_count):
                for j in range(i + 1, q_count):
                    bqm.add_interaction(f'q{q_start + i}', f'q{q_start + j}', one_hot_penalty)
        
        # Add clause constraints with confidence scaling
        for clause in self.sat.clauses:
            # Penalty scaled by confidence
            penalty = self.confidence * 1.0
            
            orbit_lits = []
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in self.orbit_map:
                    oid = self.orbit_map[var_idx]
                    orbit_lits.append((oid, lit > 0))
            
            # Map each orbit to its qubit encoding
            for oid, sign in orbit_lits:
                info = self.qubit_map[oid]
                q_start = info['q_start']
                q_count = info['q_count']
                group = info['group']
                
                # Find position of this orbit in its group
                try:
                    pos = group.index(oid)
                    # The binary value pos+1 should activate this orbit
                    target_value = pos + 1
                    
                    # Reward this encoding if sign is positive
                    if sign:
                        # Positive literal: reward when this orbit is active
                        for i in range(q_count):
                            if target_value & (1 << i):
                                bqm.add_variable(f'q{q_start + i}', -penalty / q_count)
                    else:
                        # Negative literal: penalize when this orbit is active
                        for i in range(q_count):
                            if target_value & (1 << i):
                                bqm.add_variable(f'q{q_start + i}', penalty / q_count)
                except ValueError:
                    pass
        
        return bqm
    
    def solve(self, steps=30000, restarts=30):
        """Custom solver with simulated annealing for clique-based encoding."""
        best_energy = float('inf')
        best_assignment = None
        
        for restart in range(restarts):
            state = np.random.choice([0, 1], size=self.total_qubits)
            curr_energy = self._compute_energy(state)
            
            T = 10.0
            decay = 0.995
            
            for step in range(steps):
                idx = random.randint(0, self.total_qubits - 1)
                state[idx] = 1 - state[idx]
                
                new_energy = self._compute_energy(state)
                delta = new_energy - curr_energy
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    curr_energy = new_energy
                else:
                    state[idx] = 1 - state[idx]
                
                T *= decay
                
                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_assignment = state.copy()
                    
                    if best_energy == 0:
                        break
            
            if best_energy == 0:
                break
        
        # Decode clique encoding to variable assignment
        assignment = [False] * self.sat.num_vars
        orbit_values = {}
        
        for group in self.clique_groups:
            if not group:
                continue
            
            sample_oid = group[0]
            info = self.qubit_map[sample_oid]
            q_start = info['q_start']
            q_count = info['q_count']
            
            # Read the binary encoding
            val = 0
            for i in range(q_count):
                if best_assignment[q_start + i] == 1:
                    val += (1 << i)
            
            # Set all orbits in group to false first
            for oid in group:
                orbit_values[oid] = False
            
            # Activate the selected orbit if valid
            if val > 0 and val <= len(group):
                active_oid = group[val - 1]
                orbit_values[active_oid] = True
        
        # Expand to full variable assignment
        for oid, val in orbit_values.items():
            for var in self.orbits[oid]:
                assignment[var] = val
        
        return assignment, best_energy
    
    def solve_adaptive_confidence(self, steps=50000, initial_conf=0.5, verbose=False):
        """Enhanced solver with adaptive confidence scheduling for clique-based encoding."""
        # Save original confidence
        original_conf = self.confidence
        
        # Initialize
        state = np.random.choice([0, 1], size=self.total_qubits)
        best_state = state.copy()
        best_energy = float('inf')
        
        # Adaptive parameters
        self.confidence = initial_conf
        target_conf = 1.0
        adaptation_rate = (target_conf - initial_conf) / steps
        
        # Temperature schedule
        T = 10.0
        T_min = 0.01
        cooling_rate = (T_min / T) ** (1.0 / steps)
        
        # Track statistics
        stats = {
            'energy_history': [],
            'confidence_history': [],
            'improvements': 0,
            'stagnation': 0
        }
        
        last_improvement_step = 0
        
        for step in range(steps):
            # Adaptive confidence update
            if step - last_improvement_step > 1000:
                # Stagnating: reduce confidence for more exploration
                self.confidence = max(0.3, self.confidence - 0.1)
                last_improvement_step = step
                stats['stagnation'] += 1
            else:
                # Progressing: increase confidence
                self.confidence = min(target_conf, initial_conf + step * adaptation_rate)
            
            stats['confidence_history'].append(self.confidence)
            
            # Random selection (smart selection would need modification for clique encoding)
            idx = random.randint(0, self.total_qubits - 1)
            
            # Flip
            state[idx] = 1 - state[idx]
            
            # Compute energy with clause-specific confidences
            new_energy = self._compute_energy(state, use_clause_confidences=True)
            
            # Current energy for comparison
            state[idx] = 1 - state[idx]
            curr_energy = self._compute_energy(state, use_clause_confidences=True)
            state[idx] = 1 - state[idx]
            
            delta = new_energy - curr_energy
            
            # Accept or reject
            if delta < 0 or random.random() < math.exp(-delta / T):
                curr_energy = new_energy
                
                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_state = state.copy()
                    last_improvement_step = step
                    stats['improvements'] += 1
                    
                    if verbose and step % 5000 == 0:
                        sat_score = self._compute_sat_score_from_state(best_state)
                        print(f"  Step {step}: Energy={best_energy:.2f}, SAT={sat_score:.2f}%, ε={self.confidence:.3f}")
                    
                    if best_energy == 0:
                        break
            else:
                state[idx] = 1 - state[idx]
            
            # Cool down
            T *= cooling_rate
            
            stats['energy_history'].append(best_energy)
        
        # Decode clique encoding to variable assignment
        assignment = [False] * self.sat.num_vars
        orbit_values = {}
        
        for group in self.clique_groups:
            if not group:
                continue
            
            sample_oid = group[0]
            info = self.qubit_map[sample_oid]
            q_start = info['q_start']
            q_count = info['q_count']
            
            # Read the binary encoding
            val = 0
            for i in range(q_count):
                if best_state[q_start + i] == 1:
                    val += (1 << i)
            
            # Set all orbits in group to false first
            for oid in group:
                orbit_values[oid] = False
            
            # Activate the selected orbit if valid
            if val > 0 and val <= len(group):
                active_oid = group[val - 1]
                orbit_values[active_oid] = True
        
        # Expand to full variable assignment
        for oid, val in orbit_values.items():
            for var in self.orbits[oid]:
                assignment[var] = val
        
        # Restore original confidence
        self.confidence = original_conf
        
        stats['final_energy'] = best_energy
        stats['final_sat_score'] = self._compute_sat_score_from_state(best_state)
        
        return assignment, best_energy, stats
    
    def solve_with_dwave(self, num_reads=2000, num_sweeps=20000):
        """Solve using D-Wave simulated annealing - Clique-Based version."""
        if not DWAVE_AVAILABLE:
            return self.solve()
        
        bqm = self.build_bqm()
        sampler = SimulatedAnnealingSampler()
        
        sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps,
                                  beta_range=[0.1, 10.0])
        
        best_sample = sampleset.first.sample
        
        # Decode clique encoding to orbit values
        orbit_values = {}
        for group in self.clique_groups:
            if not group:
                continue
            
            sample_oid = group[0]
            info = self.qubit_map[sample_oid]
            q_start = info['q_start']
            q_count = info['q_count']
            
            # Read the binary encoding
            val = 0
            for i in range(q_count):
                if best_sample.get(f'q{q_start + i}', 0) == 1:
                    val += (1 << i)
            
            # Set all orbits in group to false first
            for oid in group:
                orbit_values[oid] = False
            
            # Activate the selected orbit if valid
            if val > 0 and val <= len(group):
                active_oid = group[val - 1]
                orbit_values[active_oid] = True
        
        # Expand to full variable assignment
        assignment = [False] * self.sat.num_vars
        for oid, val in orbit_values.items():
            for var in self.orbits[oid]:
                assignment[var] = val
        
        _, unsat = self.sat.evaluate(assignment)
        return assignment, len(unsat)
    
    def embed_to_dwave_topology(self, topology='chimera', output_dir=None, method_name='clique_based'):
        """Embed the clique-compressed graph to D-Wave topology."""
        if not DWAVE_NX_AVAILABLE:
            return {'success': False, 'error': 'dwave-networkx not available'}
        
        # Create problem graph at QUBIT level for clique encoding
        problem_graph = nx.Graph()
        for i in range(self.total_qubits):
            problem_graph.add_node(i)
        
        # For clique-based: qubits within same clique group are connected
        # Also connect qubits from different cliques that interact via clauses
        
        # 1. Connect qubits within each clique (they share the encoding)
        for group in self.clique_groups:
            if not group or len(group) == 0:
                continue
            sample_oid = group[0]
            info = self.qubit_map[sample_oid]
            q_start = info['q_start']
            q_count = info['q_count']
            
            # All qubits in this clique's encoding are interconnected
            for i in range(q_count):
                for j in range(i + 1, q_count):
                    problem_graph.add_edge(q_start + i, q_start + j)
        
        # 2. Connect cliques that interact via clauses
        for clause in self.sat.clauses:
            # Get all clique groups involved in this clause
            clique_qubit_ranges = set()
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in self.orbit_map:
                    oid = self.orbit_map[var_idx]
                    info = self.qubit_map[oid]
                    # Store the range of qubits for this clique
                    clique_qubit_ranges.add((info['q_start'], info['q_count']))
            
            # Connect qubits from different cliques
            clique_list = list(clique_qubit_ranges)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    start1, count1 = clique_list[i]
                    start2, count2 = clique_list[j]
                    
                    # Connect representative qubits from each clique
                    # Use the first qubit of each encoding as representative
                    problem_graph.add_edge(start1, start2)
                    
                    # Also connect to ensure full coupling
                    for qi in range(count1):
                        for qj in range(count2):
                            problem_graph.add_edge(start1 + qi, start2 + qj)
        
        # Create hardware graph
        if topology == 'chimera':
            hardware_graph = dnx.chimera_graph(16, 16, 4)
        elif topology == 'pegasus':
            hardware_graph = dnx.pegasus_graph(16)
        else:
            return {'success': False, 'error': f'Unknown topology: {topology}'}
        
        # Find embedding
        try:
            embedding = find_embedding(problem_graph.edges(), hardware_graph.edges(), 
                                      random_seed=42, timeout=30)
            
            if not embedding:
                return {'success': False, 'error': 'Embedding not found'}
            
            # Calculate statistics
            chain_lengths = [len(chain) for chain in embedding.values()]
            max_chain = max(chain_lengths) if chain_lengths else 0
            avg_chain = np.mean(chain_lengths) if chain_lengths else 0
            
            result = {
                'success': True,
                'topology': topology,
                'num_qubits': self.total_qubits,
                'num_hardware_qubits': len(hardware_graph.nodes()),
                'embedding_size': len(embedding),
                'max_chain_length': int(max_chain),
                'avg_chain_length': float(avg_chain),
                'total_chains': len(embedding),
                'problem_edges': problem_graph.number_of_edges(),
                'encoding_type': 'clique_based'
            }
            
            # Visualize
            if output_dir:
                self._visualize_dwave_embedding(problem_graph, hardware_graph, embedding, 
                                               topology, output_dir, method_name)
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


class ClusterBasedEncoder(OrbitBasedEncoder):
    """Cluster-based compression encoder with confidence-scaled energy."""
    
    def __init__(self, sat: SATInstance, orbits: List[Set[int]], confidence: float = 1.0):
        super().__init__(sat, orbits, confidence)
        self.clusters = []
        
    def allocate_qubits(self) -> Tuple[Dict, int]:
        """Allocate qubits using cluster-based compression."""
        # Build interaction graph
        G = nx.Graph()
        for i in range(self.num_orbits):
            G.add_node(i)
        
        for clause in self.sat.clauses:
            orbit_ids = []
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in self.orbit_map:
                    oid = self.orbit_map[var_idx]
                    if oid not in orbit_ids:
                        orbit_ids.append(oid)
            
            for i in range(len(orbit_ids)):
                for j in range(i + 1, len(orbit_ids)):
                    if G.has_edge(orbit_ids[i], orbit_ids[j]):
                        G[orbit_ids[i]][orbit_ids[j]]['weight'] += 1
                    else:
                        G.add_edge(orbit_ids[i], orbit_ids[j], weight=1)
        
        # Community detection using Louvain
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G)
        except:
            # Fallback: simple clustering by connected components
            partition = {}
            for idx, component in enumerate(nx.connected_components(G)):
                for node in component:
                    partition[node] = idx
        
        # Group by cluster
        cluster_map = defaultdict(list)
        for node, cluster_id in partition.items():
            cluster_map[cluster_id].append(node)
        
        self.clusters = list(cluster_map.values())
        
        # Allocate qubits per cluster
        self.total_qubits = 0
        self.qubit_map = {}
        
        for cluster in self.clusters:
            bits_needed = math.ceil(math.log2(len(cluster) + 1)) if len(cluster) > 1 else 1
            
            for oid in cluster:
                self.qubit_map[oid] = {
                    'q_start': self.total_qubits,
                    'q_count': bits_needed,
                    'cluster': cluster,
                    'cluster_size': len(cluster)
                }
            
            self.total_qubits += bits_needed
        
        print(f"Cluster-Based Encoding: {self.total_qubits} qubits from {len(self.clusters)} clusters")
        return self.qubit_map, self.total_qubits
    
    def _compute_energy(self, state, use_clause_confidences: bool = True):
        """
        Compute confidence-scaled energy with optional clause-specific weights.
        
        Energy formulation:
            E_ε(x, h) = Σ_m Σ_j ε · ε_clause_j · h_mj · φ_mj(x)
        
        where:
            - ε is the global confidence parameter
            - ε_clause_j is the clause-specific confidence (if enabled)
            - h_mj represents hidden clause-selector variables (active orbits)
            - φ_mj(x) represents clause satisfaction signals
            
        For SAT problems, this translates to:
            E_ε = ε · Σ_j (ε_clause_j · unsat_j)
            
        Args:
            state: Current qubit state
            use_clause_confidences: Whether to use clause-specific confidences
        """
        active_orbits = set()
        
        for cluster in self.clusters:
            if not cluster:
                continue
            
            sample_oid = cluster[0]
            info = self.qubit_map[sample_oid]
            start = info['q_start']
            count = info['q_count']
            
            val = 0
            for i in range(count):
                if state[start + i] == 1:
                    val += (1 << i)
            
            if val > 0 and val <= len(cluster):
                active_orbits.add(cluster[val - 1])
        
        total_energy = 0.0
        
        for i, clause in enumerate(self.sat.clauses):
            satisfied = False
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in self.orbit_map:
                    oid = self.orbit_map[var_idx]
                    is_true = oid in active_orbits
                    if (lit > 0 and is_true) or (lit < 0 and not is_true):
                        satisfied = True
                        break
            
            if not satisfied:
                if use_clause_confidences and i in self.clause_confidences:
                    # Use clause-specific confidence
                    clause_conf = self.clause_confidences[i]
                    total_energy += self.confidence * clause_conf
                else:
                    # Use only global confidence
                    total_energy += self.confidence
        
        return total_energy
    
    def bitflip_search(self, num_iterations=10000, flip_prob=0.3, flip_type='cluster'):
        """BitFlip by cluster groups - flips a percentage of bits within the selected cluster."""
        assignment = [random.choice([True, False]) for _ in range(self.sat.num_vars)]
        
        sat_history = []
        unsat_history = []
        best_assignment = assignment[:]
        best_unsat = self.sat.num_clauses
        
        for iteration in range(num_iterations):
            if random.random() < flip_prob:
                # Select a random cluster
                cluster_idx = random.randint(0, len(self.clusters) - 1)
                cluster = self.clusters[cluster_idx]
                
                # Flip bits within the cluster with probability flip_prob
                for oid in cluster:
                    if random.random() < flip_prob:
                        for var in self.orbits[oid]:
                            assignment[var] = not assignment[var]
            
            valid, unsat = self.sat.evaluate(assignment)
            num_unsat = len(unsat)
            sat_score = (1 - num_unsat / self.sat.num_clauses) * 100
            
            sat_history.append(sat_score)
            unsat_history.append(num_unsat)
            
            if num_unsat < best_unsat:
                best_unsat = num_unsat
                best_assignment = assignment[:]
        
        return best_assignment, sat_history, unsat_history
    
    def build_bqm(self) -> 'BinaryQuadraticModel':
        """
        Build Binary Quadratic Model for D-Wave - Cluster-Based version with confidence scaling.
        
        Penalties are scaled by confidence parameter ε.
        """
        if not DWAVE_AVAILABLE:
            raise ImportError("D-Wave Ocean SDK not available")
        
        bqm = BinaryQuadraticModel('BINARY')
        
        # Add variables for each qubit
        for i in range(self.total_qubits):
            bqm.add_variable(f'q{i}', 0.0)
        
        # Add one-hot constraints for each cluster
        # Only one orbit in each cluster should be active
        for cluster in self.clusters:
            if not cluster or len(cluster) <= 1:
                continue
            
            sample_oid = cluster[0]
            info = self.qubit_map[sample_oid]
            q_start = info['q_start']
            q_count = info['q_count']
            
            # One-hot encoding constraint: exactly one configuration
            # Keep this unscaled to maintain structural constraints
            one_hot_penalty = 2.0
            
            # Penalize having multiple bits set (should represent one value)
            for i in range(q_count):
                for j in range(i + 1, q_count):
                    bqm.add_interaction(f'q{q_start + i}', f'q{q_start + j}', one_hot_penalty)
        
        # Add clause constraints with confidence scaling
        for clause in self.sat.clauses:
            # Penalty scaled by confidence
            penalty = self.confidence * 1.0
            
            orbit_lits = []
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in self.orbit_map:
                    oid = self.orbit_map[var_idx]
                    orbit_lits.append((oid, lit > 0))
            
            # Map each orbit to its qubit encoding
            for oid, sign in orbit_lits:
                info = self.qubit_map[oid]
                q_start = info['q_start']
                q_count = info['q_count']
                cluster = info['cluster']
                
                # Find position of this orbit in its cluster
                try:
                    pos = cluster.index(oid)
                    # The binary value pos+1 should activate this orbit
                    target_value = pos + 1
                    
                    # Reward this encoding if sign is positive
                    if sign:
                        # Positive literal: reward when this orbit is active
                        for i in range(q_count):
                            if target_value & (1 << i):
                                bqm.add_variable(f'q{q_start + i}', -penalty / q_count)
                    else:
                        # Negative literal: penalize when this orbit is active
                        for i in range(q_count):
                            if target_value & (1 << i):
                                bqm.add_variable(f'q{q_start + i}', penalty / q_count)
                except ValueError:
                    pass
        
        return bqm
    
    def solve(self, steps=30000, restarts=30):
        """Custom solver with simulated annealing for cluster-based encoding."""
        best_energy = float('inf')
        best_assignment = None
        
        for restart in range(restarts):
            state = np.random.choice([0, 1], size=self.total_qubits)
            curr_energy = self._compute_energy(state)
            
            T = 10.0
            decay = 0.995
            
            for step in range(steps):
                idx = random.randint(0, self.total_qubits - 1)
                state[idx] = 1 - state[idx]
                
                new_energy = self._compute_energy(state)
                delta = new_energy - curr_energy
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    curr_energy = new_energy
                else:
                    state[idx] = 1 - state[idx]
                
                T *= decay
                
                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_assignment = state.copy()
                    
                    if best_energy == 0:
                        break
            
            if best_energy == 0:
                break
        
        # Decode cluster encoding to variable assignment
        assignment = [False] * self.sat.num_vars
        orbit_values = {}
        
        for cluster in self.clusters:
            if not cluster:
                continue
            
            sample_oid = cluster[0]
            info = self.qubit_map[sample_oid]
            q_start = info['q_start']
            q_count = info['q_count']
            
            # Read the binary encoding
            val = 0
            for i in range(q_count):
                if best_assignment[q_start + i] == 1:
                    val += (1 << i)
            
            # Set all orbits in cluster to false first
            for oid in cluster:
                orbit_values[oid] = False
            
            # Activate the selected orbit if valid
            if val > 0 and val <= len(cluster):
                active_oid = cluster[val - 1]
                orbit_values[active_oid] = True
        
        # Expand to full variable assignment
        for oid, val in orbit_values.items():
            for var in self.orbits[oid]:
                assignment[var] = val
        
        return assignment, best_energy
    
    def solve_adaptive_confidence(self, steps=50000, initial_conf=0.5, verbose=False):
        """Enhanced solver with adaptive confidence scheduling for cluster-based encoding."""
        # Save original confidence
        original_conf = self.confidence
        
        # Initialize
        state = np.random.choice([0, 1], size=self.total_qubits)
        best_state = state.copy()
        best_energy = float('inf')
        
        # Adaptive parameters
        self.confidence = initial_conf
        target_conf = 1.0
        adaptation_rate = (target_conf - initial_conf) / steps
        
        # Temperature schedule
        T = 10.0
        T_min = 0.01
        cooling_rate = (T_min / T) ** (1.0 / steps)
        
        # Track statistics
        stats = {
            'energy_history': [],
            'confidence_history': [],
            'improvements': 0,
            'stagnation': 0
        }
        
        last_improvement_step = 0
        
        for step in range(steps):
            # Adaptive confidence update
            if step - last_improvement_step > 1000:
                # Stagnating: reduce confidence for more exploration
                self.confidence = max(0.3, self.confidence - 0.1)
                last_improvement_step = step
                stats['stagnation'] += 1
            else:
                # Progressing: increase confidence
                self.confidence = min(target_conf, initial_conf + step * adaptation_rate)
            
            stats['confidence_history'].append(self.confidence)
            
            # Random selection (smart selection would need modification for cluster encoding)
            idx = random.randint(0, self.total_qubits - 1)
            
            # Flip
            state[idx] = 1 - state[idx]
            
            # Compute energy with clause-specific confidences
            new_energy = self._compute_energy(state, use_clause_confidences=True)
            
            # Current energy for comparison
            state[idx] = 1 - state[idx]
            curr_energy = self._compute_energy(state, use_clause_confidences=True)
            state[idx] = 1 - state[idx]
            
            delta = new_energy - curr_energy
            
            # Accept or reject
            if delta < 0 or random.random() < math.exp(-delta / T):
                curr_energy = new_energy
                
                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_state = state.copy()
                    last_improvement_step = step
                    stats['improvements'] += 1
                    
                    if verbose and step % 5000 == 0:
                        sat_score = self._compute_sat_score_from_state(best_state)
                        print(f"  Step {step}: Energy={best_energy:.2f}, SAT={sat_score:.2f}%, ε={self.confidence:.3f}")
                    
                    if best_energy == 0:
                        break
            else:
                state[idx] = 1 - state[idx]
            
            # Cool down
            T *= cooling_rate
            
            stats['energy_history'].append(best_energy)
        
        # Decode cluster encoding to variable assignment
        assignment = [False] * self.sat.num_vars
        orbit_values = {}
        
        for cluster in self.clusters:
            if not cluster:
                continue
            
            sample_oid = cluster[0]
            info = self.qubit_map[sample_oid]
            q_start = info['q_start']
            q_count = info['q_count']
            
            # Read the binary encoding
            val = 0
            for i in range(q_count):
                if best_state[q_start + i] == 1:
                    val += (1 << i)
            
            # Set all orbits in cluster to false first
            for oid in cluster:
                orbit_values[oid] = False
            
            # Activate the selected orbit if valid
            if val > 0 and val <= len(cluster):
                active_oid = cluster[val - 1]
                orbit_values[active_oid] = True
        
        # Expand to full variable assignment
        for oid, val in orbit_values.items():
            for var in self.orbits[oid]:
                assignment[var] = val
        
        # Restore original confidence
        self.confidence = original_conf
        
        stats['final_energy'] = best_energy
        stats['final_sat_score'] = self._compute_sat_score_from_state(best_state)
        
        return assignment, best_energy, stats
    
    def solve_with_dwave(self, num_reads=2000, num_sweeps=20000):
        """Solve using D-Wave simulated annealing - Cluster-Based version."""
        if not DWAVE_AVAILABLE:
            return self.solve()
        
        bqm = self.build_bqm()
        sampler = SimulatedAnnealingSampler()
        
        sampleset = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps,
                                  beta_range=[0.1, 10.0])
        
        best_sample = sampleset.first.sample
        
        # Decode cluster encoding to orbit values
        orbit_values = {}
        for cluster in self.clusters:
            if not cluster:
                continue
            
            sample_oid = cluster[0]
            info = self.qubit_map[sample_oid]
            q_start = info['q_start']
            q_count = info['q_count']
            
            # Read the binary encoding
            val = 0
            for i in range(q_count):
                if best_sample.get(f'q{q_start + i}', 0) == 1:
                    val += (1 << i)
            
            # Set all orbits in cluster to false first
            for oid in cluster:
                orbit_values[oid] = False
            
            # Activate the selected orbit if valid
            if val > 0 and val <= len(cluster):
                active_oid = cluster[val - 1]
                orbit_values[active_oid] = True
        
        # Expand to full variable assignment
        assignment = [False] * self.sat.num_vars
        for oid, val in orbit_values.items():
            for var in self.orbits[oid]:
                assignment[var] = val
        
        _, unsat = self.sat.evaluate(assignment)
        return assignment, len(unsat)
    
    def embed_to_dwave_topology(self, topology='chimera', output_dir=None, method_name='cluster_based'):
        """Embed the cluster-compressed graph to D-Wave topology."""
        if not DWAVE_NX_AVAILABLE:
            return {'success': False, 'error': 'dwave-networkx not available'}
        
        # Create problem graph at QUBIT level for cluster encoding
        problem_graph = nx.Graph()
        for i in range(self.total_qubits):
            problem_graph.add_node(i)
        
        # For cluster-based: qubits within same cluster are connected
        # Also connect qubits from different clusters that interact via clauses
        
        # 1. Connect qubits within each cluster (they share the encoding)
        for cluster in self.clusters:
            if not cluster or len(cluster) == 0:
                continue
            sample_oid = cluster[0]
            info = self.qubit_map[sample_oid]
            q_start = info['q_start']
            q_count = info['q_count']
            
            # All qubits in this cluster's encoding are interconnected
            for i in range(q_count):
                for j in range(i + 1, q_count):
                    problem_graph.add_edge(q_start + i, q_start + j)
        
        # 2. Connect clusters that interact via clauses
        for clause in self.sat.clauses:
            # Get all cluster groups involved in this clause
            cluster_qubit_ranges = set()
            for lit in clause:
                var_idx = abs(lit) - 1
                if var_idx in self.orbit_map:
                    oid = self.orbit_map[var_idx]
                    info = self.qubit_map[oid]
                    # Store the range of qubits for this cluster
                    cluster_qubit_ranges.add((info['q_start'], info['q_count']))
            
            # Connect qubits from different clusters
            cluster_list = list(cluster_qubit_ranges)
            for i in range(len(cluster_list)):
                for j in range(i + 1, len(cluster_list)):
                    start1, count1 = cluster_list[i]
                    start2, count2 = cluster_list[j]
                    
                    # Connect representative qubits from each cluster
                    # Use the first qubit of each encoding as representative
                    problem_graph.add_edge(start1, start2)
                    
                    # Also connect to ensure full coupling for small clusters
                    if count1 <= 2 and count2 <= 2:
                        for qi in range(count1):
                            for qj in range(count2):
                                problem_graph.add_edge(start1 + qi, start2 + qj)
        
        # Create hardware graph
        if topology == 'chimera':
            hardware_graph = dnx.chimera_graph(16, 16, 4)
        elif topology == 'pegasus':
            hardware_graph = dnx.pegasus_graph(16)
        else:
            return {'success': False, 'error': f'Unknown topology: {topology}'}
        
        # Find embedding
        try:
            embedding = find_embedding(problem_graph.edges(), hardware_graph.edges(), 
                                      random_seed=42, timeout=30)
            
            if not embedding:
                return {'success': False, 'error': 'Embedding not found'}
            
            # Calculate statistics
            chain_lengths = [len(chain) for chain in embedding.values()]
            max_chain = max(chain_lengths) if chain_lengths else 0
            avg_chain = np.mean(chain_lengths) if chain_lengths else 0
            
            result = {
                'success': True,
                'topology': topology,
                'num_qubits': self.total_qubits,
                'num_hardware_qubits': len(hardware_graph.nodes()),
                'embedding_size': len(embedding),
                'max_chain_length': int(max_chain),
                'avg_chain_length': float(avg_chain),
                'total_chains': len(embedding),
                'problem_edges': problem_graph.number_of_edges(),
                'encoding_type': 'cluster_based'
            }
            
            # Visualize
            if output_dir:
                self._visualize_dwave_embedding(problem_graph, hardware_graph, embedding, 
                                               topology, output_dir, method_name)
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}