

import sys
import os

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from cliquecoloc import (
    GeneratorParams,
    generate_synthetic,
    run_pipeline,
    SpatialDataset
)
import time
import multiprocessing

def verify():
    print("Generating synthetic dataset...")
    # Small dataset for correctness check
    params = GeneratorParams(
        P=5, I=50, D=500, F=5, Q=2, m=1000, min_dist=30
    )
    ds = generate_synthetic(params, seed=42)
    
    print(f"Dataset: {len(ds.instances)} instances.")

    # Serial Run
    print("\n--- Running Serial (n_workers=1) ---")
    start = time.time()
    # Assuming run_pipeline will be updated to accept n_workers
    cliques_s, chash_s, patterns_s = run_pipeline(ds, min_dist=30, min_prev=0.2, schema="nds", n_workers=1)
    end = time.time()
    print(f"Serial took {end - start:.4f}s. Cliques: {len(cliques_s)}, Patterns: {len(patterns_s)}")

    # Parallel Run
    # Use fewer workers for test if cpu_count is huge, but user has 32 CPUs.
    n_workers = min(4, multiprocessing.cpu_count())
    print(f"\n--- Running Parallel (n_workers={n_workers}) ---")
    start = time.time()
    cliques_p, chash_p, patterns_p = run_pipeline(ds, min_dist=30, min_prev=0.2, schema="nds", n_workers=n_workers)
    end = time.time()
    print(f"Parallel took {end - start:.4f}s. Cliques: {len(cliques_p)}, Patterns: {len(patterns_p)}")

    # Verification
    print("\n--- Verifying Results ---")
    # Convert lists to sets for comparison (order might differ)
    cliques_s_set = set(tuple(sorted(c)) for c in cliques_s)
    cliques_p_set = set(tuple(sorted(c)) for c in cliques_p)
    
    assert cliques_s_set == cliques_p_set, f"Clique mismatch! Serial: {len(cliques_s_set)}, Parallel: {len(cliques_p_set)}"
    print("Cliques match!")
    
    assert patterns_s == patterns_p, "Patterns mismatch!"
    print("Patterns match!")
    
    print("\nALL CHECKS PASSED.")

if __name__ == "__main__":
    verify()
