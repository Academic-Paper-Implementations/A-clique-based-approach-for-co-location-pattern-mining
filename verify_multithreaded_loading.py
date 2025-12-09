from cliquecoloc.data import load_csv, SpatialDataset, Instance, save_csv
from pathlib import Path
import time
import os

def create_dummy_csv(path, n_rows=10000):
    print(f"Creating dummy dataset with {n_rows} rows at {path}...")
    instances = []
    for i in range(n_rows):
        instances.append(Instance(f"A", i, float(i), float(i)))
    ds = SpatialDataset(instances)
    save_csv(ds, path)

def verify():
    path = Path("test_multithread_data.csv")
    if not path.exists():
        create_dummy_csv(path, n_rows=100000) # 100k rows
        
    print("Testing single-threaded load...")
    start = time.time()
    ds1 = load_csv(path, workers=1)
    end = time.time()
    print(f"Single-threaded time: {end - start:.4f}s. Loaded {len(ds1.instances)} instances.")
    
    print("Testing multi-threaded load (workers=4)...")
    start = time.time()
    ds4 = load_csv(path, workers=4)
    end = time.time()
    print(f"Multi-threaded time: {end - start:.4f}s. Loaded {len(ds4.instances)} instances.")
    
    # Verification
    assert len(ds1.instances) == len(ds4.instances), "Instance count mismatch!"
    # Features might be in different order in list but set should be same
    assert ds1.features == ds4.features, "Features mismatch!"
    
    # Check a few random items
    s1_set = set((i.feature, i.idx, i.x, i.y) for i in ds1.instances)
    s4_set = set((i.feature, i.idx, i.x, i.y) for i in ds4.instances)
    assert s1_set == s4_set, "Content mismatch!"
    
    print("Verification PASSED!")
    
    # Clean up
    if path.exists():
        os.remove(path)

if __name__ == "__main__":
    verify()
