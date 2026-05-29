from typing import List, Tuple, Dict, Set
from collections import defaultdict
import sys

class SATInstance:
    """Represents a SAT problem in CNF."""
    
    def __init__(self, num_vars: int, clauses: List[List[int]]):
        self.num_vars = num_vars
        self.clauses = clauses
        self.num_clauses = len(clauses)
        
    def evaluate(self, assignment: List[bool]) -> Tuple[bool, List[int]]:
        """Evaluate the SAT formula with given assignment."""
        unsatisfied = []
        for i, clause in enumerate(self.clauses):
            clause_satisfied = False
            for literal in clause:
                var_idx = abs(literal) - 1
                if var_idx < len(assignment):
                    var_value = assignment[var_idx]
                    if (literal > 0 and var_value) or (literal < 0 and not var_value):
                        clause_satisfied = True
                        break
            if not clause_satisfied:
                unsatisfied.append(i + 1)
        return len(unsatisfied) == 0, unsatisfied

    def simplify(self) -> 'SATInstance':
        """Apply preprocessing: unit propagation and pure literal elimination."""
        clauses = [c[:] for c in self.clauses]
        assignment = [None] * self.num_vars
        
        changed = True
        while changed:
            changed = False
            
            # Unit propagation
            for clause in clauses[:]:
                if len(clause) == 1:
                    lit = clause[0]
                    var_idx = abs(lit) - 1
                    val = lit > 0
                    
                    if assignment[var_idx] is None:
                        assignment[var_idx] = val
                        changed = True
                        
                        # Remove satisfied clauses and literals
                        new_clauses = []
                        for c in clauses:
                            if lit in c:
                                continue  # Clause satisfied
                            new_c = [l for l in c if l != -lit]
                            if new_c:
                                new_clauses.append(new_c)
                        clauses = new_clauses
            
            # Pure literal elimination
            lit_counts = defaultdict(lambda: [0, 0])
            for clause in clauses:
                for lit in clause:
                    idx = 0 if lit > 0 else 1
                    lit_counts[abs(lit)][idx] += 1
            
            for var, (pos, neg) in lit_counts.items():
                if pos > 0 and neg == 0:
                    assignment[var - 1] = True
                    clauses = [c for c in clauses if var not in c]
                    changed = True
                elif neg > 0 and pos == 0:
                    assignment[var - 1] = False
                    clauses = [c for c in clauses if -var not in c]
                    changed = True
        
        return SATInstance(self.num_vars, clauses)

    def __str__(self):
        return f"SAT Instance: {self.num_vars} variables, {self.num_clauses} clauses"
    

def parse_cnf_file(filename: str) -> SATInstance:
    """
    Parse a DIMACS CNF file and return a SATInstance.
    
    Args:
        filename: Path to the .cnf file
        
    Returns:
        SATInstance object
    """
    num_vars = 0
    clauses = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('c'):
                    continue
                
                # Parse problem line
                if line.startswith('p'):
                    parts = line.split()
                    if len(parts) >= 4 and parts[1] == 'cnf':
                        num_vars = int(parts[2])
                        # num_clauses = int(parts[3])  # We'll count actual clauses
                    continue
                
                # Parse clause
                literals = list(map(int, line.split()))
                
                # Remove trailing 0 if present
                if literals and literals[-1] == 0:
                    literals = literals[:-1]
                
                # Only add non-empty clauses
                if literals:
                    clauses.append(literals)
        
        if num_vars == 0:
            # If no problem line found, infer from clauses
            num_vars = max(abs(lit) for clause in clauses for lit in clause) if clauses else 0
        
        print(f"Loaded CNF file: {filename}")
        print(f"  Variables: {num_vars}")
        print(f"  Clauses: {len(clauses)}")
        
        return SATInstance(num_vars, clauses)
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing CNF file: {e}")
        sys.exit(1)
