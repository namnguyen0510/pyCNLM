"""
reducer/ferq/precompute_coefficients.py
Pre-compute FERQ alpha coefficients and Fermat quotient lookup tables.
Saves as feather files for ultra-fast loading.

Usage:
    python -m reducer.ferq.precompute_coefficients --max-degree 50
    python -m reducer.ferq.precompute_coefficients --max-degree 100 --output-dir ./ferq_data
"""
import os
import argparse
import numpy as np
from fractions import Fraction
from math import factorial
from functools import lru_cache
from typing import Dict, List, Tuple
import time
import json

# Try to import pyarrow for feather format
try:
    import pyarrow as pa
    import pyarrow.feather as feather
    _FEATHER_AVAILABLE = True
except ImportError:
    _FEATHER_AVAILABLE = False
    print("⚠ Warning: pyarrow not installed. Install with: pip install pyarrow")
    print("  Will fall back to numpy .npy format")


# ═══════════════════════════════════════════════════════════════════════════
# NUMBER-THEORETIC PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=None)
def get_primes(n: int) -> List[int]:
    """Return the first n primes."""
    primes: List[int] = []
    c = 2
    while len(primes) < n:
        if all(c % p != 0 for p in primes):
            primes.append(c)
        c += 1
    return primes


@lru_cache(maxsize=None)
def stirling2(n: int, k: int) -> int:
    """Stirling number of the second kind S(n, k)."""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0 or k > n:
        return 0
    
    # Use iterative DP for large values
    if n > 50:
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                dp[i][j] = j * dp[i-1][j] + dp[i-1][j-1]
        return dp[n][k]
    
    return k * stirling2(n - 1, k) + stirling2(n - 1, k - 1)


def surjection_number(p: int, q: int) -> int:
    """N(p, q) = q! · S(p, q)."""
    return factorial(q) * stirling2(p, q)


def fermat_quotient_fast(a: int, p: int) -> float:
    """δ_p(a) = (a - a^p) / p with integer arithmetic."""
    if a == 0 or a == 1:
        return 0.0
    return (a - pow(a, p)) / p


# ═══════════════════════════════════════════════════════════════════════════
# MATRIX ALGEBRA
# ═══════════════════════════════════════════════════════════════════════════

def build_M_matrix(d: int, primes: List[int]) -> np.ndarray:
    """Build M_d matrix using numpy."""
    size = d - 1
    M = np.zeros((size, size), dtype=np.float64)
    
    for k in range(size):
        pk = primes[k]
        for q in range(size):
            deg = q + 2
            if deg <= min(pk, d):
                M[k, q] = surjection_number(pk, deg) / pk
    
    return M


def compute_alpha(d: int, primes: List[int]) -> np.ndarray:
    """Compute alpha coefficients for degree d."""
    if d < 2:
        return np.array([])
    
    M = build_M_matrix(d, primes)
    
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        # Fallback to rational arithmetic
        M_frac = build_M_matrix_rational(d, primes)
        M_inv_frac = invert_rational_matrix(M_frac)
        M_inv = np.array([[float(x) for x in row] for row in M_inv_frac])
    
    last_row = -M_inv[d - 2, :]
    return last_row


def build_M_matrix_rational(d: int, primes: List[int]) -> List[List[Fraction]]:
    """Build M_d matrix using rational arithmetic (fallback)."""
    size = d - 1
    M = [[Fraction(0)] * size for _ in range(size)]
    for k in range(size):
        pk = primes[k]
        for q in range(size):
            deg = q + 2
            if deg <= min(pk, d):
                M[k][q] = Fraction(surjection_number(pk, deg), pk)
    return M


def invert_rational_matrix(M: List[List[Fraction]]) -> List[List[Fraction]]:
    """Gauss-Jordan inversion over ℚ."""
    n = len(M)
    aug = [
        [Fraction(M[i][j]) for j in range(n)] + [Fraction(int(i == j)) for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            raise ValueError("Matrix M_d is singular.")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        f = aug[col][col]
        aug[col] = [x / f for x in aug[col]]
        for row in range(n):
            if row != col and aug[row][col] != 0:
                g = aug[row][col]
                aug[row] = [aug[row][j] - g * aug[col][j] for j in range(2 * n)]
    return [[aug[i][n + j] for j in range(n)] for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
# PRE-COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def precompute_degree(d: int) -> Dict:
    """Pre-compute all data for a specific degree d."""
    primes = get_primes(d - 1)
    alphas = compute_alpha(d, primes)
    
    # Pre-compute Fermat quotient lookup table
    # fq_table[k, a] = δ_{p_k}(a) for a ∈ {0, 1, ..., d}
    fq_table = np.zeros((len(primes), d + 1), dtype=np.float64)
    for k, p in enumerate(primes):
        for a in range(d + 1):
            fq_table[k, a] = fermat_quotient_fast(a, p)
    
    return {
        'degree': d,
        'primes': np.array(primes, dtype=np.int32),
        'alphas': alphas.astype(np.float64),
        'fq_table': fq_table,
        'n_primes': len(primes)
    }


def precompute_all(max_degree: int = 50, verbose: bool = True) -> Dict[int, Dict]:
    """Pre-compute all degrees from 2 to max_degree."""
    if verbose:
        print(f"Pre-computing FERQ coefficients for degrees 2 to {max_degree}...")
    
    start_time = time.time()
    all_data = {}
    
    for d in range(2, max_degree + 1):
        if verbose:
            print(f"  Degree {d:3d}...", end=" ", flush=True)
        
        data = precompute_degree(d)
        all_data[d] = data
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"✓ ({elapsed:.2f}s total)")
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n✓ Pre-computation complete in {total_time:.2f}s")
        print(f"  Degrees: 2 to {max_degree}")
        print(f"  Total entries: {len(all_data)}")
    
    return all_data


# ═══════════════════════════════════════════════════════════════════════════
# SAVE/LOAD FUNCTIONS (FIXED)
# ═══════════════════════════════════════════════════════════════════════════

def save_to_feather(all_data: Dict[int, Dict], output_path: str, verbose: bool = True):
    """Save pre-computed data to feather file with consistent array lengths."""
    if not _FEATHER_AVAILABLE:
        print("⚠ pyarrow not available, saving to numpy format instead")
        return save_to_numpy(all_data, output_path.replace('.feather', '.npz'), verbose)
    
    if verbose:
        print(f"\nSaving to feather: {output_path}")
    
    # Build arrays with consistent lengths
    # Each row represents one (degree, prime_index) pair
    all_degrees = []
    all_prime_indices = []
    all_primes = []
    all_alphas = []
    all_n_primes = []
    all_fq_tables = []  # Will store as flattened arrays
    
    for d in sorted(all_data.keys()):
        data = all_data[d]
        n_primes = data['n_primes']
        
        for k in range(n_primes):
            all_degrees.append(d)
            all_prime_indices.append(k)
            all_primes.append(data['primes'][k])
            all_alphas.append(data['alphas'][k])
            all_n_primes.append(n_primes)
            # Store entire fq_table row for this prime (length d+1)
            all_fq_tables.append(data['fq_table'][k, :])
    
    # Convert to numpy arrays
    n_rows = len(all_degrees)
    max_d = max(all_data.keys())
    
    # Create feather-compatible table
    table_dict = {
        'degree': np.array(all_degrees, dtype=np.int32),
        'prime_index': np.array(all_prime_indices, dtype=np.int32),
        'primes': np.array(all_primes, dtype=np.int32),
        'alphas': np.array(all_alphas, dtype=np.float64),
        'n_primes': np.array(all_n_primes, dtype=np.int32),
    }
    
    # Add fq_table columns (one for each possible a value up to max_degree)
    for a in range(max_d + 1):
        fq_col = np.array([fq[a] if a < len(fq) else 0.0 for fq in all_fq_tables], dtype=np.float64)
        table_dict[f'fq_a{a}'] = fq_col
    
    # Save as feather
    table = pa.table(table_dict)
    feather.write_feather(table, output_path)
    
    # Also save metadata as JSON
    metadata_path = output_path.replace('.feather', '_metadata.json')
    metadata = {
        'max_degree': max(all_data.keys()) if all_data else 0,
        'n_degrees': len(all_data),
        'n_rows': n_rows,
        'format': 'feather',
        'timestamp': time.time()
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        file_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"  ✓ Saved {file_size:.2f} MB to {output_path}")
        print(f"  ✓ Metadata saved to {metadata_path}")


def save_to_numpy(all_data: Dict[int, Dict], output_path: str, verbose: bool = True):
    """Fallback: Save to numpy .npz format."""
    if verbose:
        print(f"\nSaving to numpy: {output_path}")
    
    # Flatten all data for numpy storage
    save_dict = {}
    for d, data in all_data.items():
        save_dict[f'degree_{d}_primes'] = data['primes']
        save_dict[f'degree_{d}_alphas'] = data['alphas']
        save_dict[f'degree_{d}_fq_table'] = data['fq_table']
        save_dict[f'degree_{d}_n_primes'] = np.array([data['n_primes']])
    
    np.savez_compressed(output_path, **save_dict)
    
    if verbose:
        file_size = os.path.getsize(output_path) / 1024 / 1024
        print(f"  ✓ Saved {file_size:.2f} MB to {output_path}")


def load_from_feather(input_path: str, verbose: bool = True) -> Dict[int, Dict]:
    """Load pre-computed data from feather file."""
    if not _FEATHER_AVAILABLE:
        return load_from_numpy(input_path.replace('.feather', '.npz'), verbose)
    
    if verbose:
        print(f"Loading from feather: {input_path}")
    
    start_time = time.time()
    
    # Read feather file
    table = feather.read_table(input_path)
    
    # Reconstruct data by degree
    all_data = {}
    degrees = np.unique(table['degree'].to_numpy())
    
    for d in degrees:
        mask = table['degree'].to_numpy() == d
        n_primes = int(table['n_primes'].to_numpy()[mask][0])
        
        primes = table['primes'].to_numpy()[mask]
        alphas = table['alphas'].to_numpy()[mask]
        
        # Reconstruct fq_table from columns
        fq_table = np.zeros((n_primes, d + 1), dtype=np.float64)
        for a in range(d + 1):
            fq_table[:, a] = table[f'fq_a{a}'].to_numpy()[mask]
        
        all_data[int(d)] = {
            'degree': int(d),
            'primes': primes,
            'alphas': alphas,
            'fq_table': fq_table,
            'n_primes': n_primes
        }
    
    load_time = time.time() - start_time
    
    if verbose:
        print(f"  ✓ Loaded {len(all_data)} degrees in {load_time:.4f}s")
    
    return all_data


def load_from_numpy(input_path: str, verbose: bool = True) -> Dict[int, Dict]:
    """Load from numpy .npz format."""
    if verbose:
        print(f"Loading from numpy: {input_path}")
    
    start_time = time.time()
    
    data = np.load(input_path)
    all_data = {}
    
    # Extract degree information from keys
    degree_keys = [k for k in data.files if k.startswith('degree_')]
    degrees = sorted(set(int(k.split('_')[1]) for k in degree_keys))
    
    for d in degrees:
        primes = data[f'degree_{d}_primes']
        alphas = data[f'degree_{d}_alphas']
        fq_table = data[f'degree_{d}_fq_table']
        n_primes = int(data[f'degree_{d}_n_primes'][0])
        
        all_data[d] = {
            'degree': d,
            'primes': primes,
            'alphas': alphas,
            'fq_table': fq_table,
            'n_primes': n_primes
        }
    
    load_time = time.time() - start_time
    
    if verbose:
        print(f"  ✓ Loaded {len(all_data)} degrees in {load_time:.4f}s")
    
    return all_data


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute FERQ coefficients and save to feather file'
    )
    parser.add_argument(
        '--max-degree', '-d',
        type=int,
        default=43,
        help='Maximum degree to pre-compute (default: 50)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./reducers/ferq/data',
        help='Output directory (default: ./reducers/ferq/data)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['feather', 'numpy'],
        default='feather' if _FEATHER_AVAILABLE else 'numpy',
        help='Output format (default: feather if available)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Pre-compute
    verbose = not args.quiet
    all_data = precompute_all(max_degree=args.max_degree, verbose=verbose)
    
    # Save
    if args.format == 'feather' and _FEATHER_AVAILABLE:
        output_path = os.path.join(args.output_dir, f'ferq_coeffs_d{args.max_degree}.feather')
        save_to_feather(all_data, output_path, verbose)
    else:
        output_path = os.path.join(args.output_dir, f'ferq_coeffs_d{args.max_degree}.npz')
        save_to_numpy(all_data, output_path, verbose)
    
    # Test load speed
    if verbose:
        print("\nTesting load speed...")
        start = time.time()
        loaded = load_from_feather(output_path, verbose=False) if args.format == 'feather' else load_from_numpy(output_path, verbose=False)
        load_time = time.time() - start
        print(f"  Load time: {load_time:.4f}s ({len(loaded)} degrees)")
    
    print("\n✓ Pre-computation complete!")
    print(f"  Use: FERQ(max_degree={args.max_degree}, precomputed_path='{output_path}')")


if __name__ == "__main__":
    main()