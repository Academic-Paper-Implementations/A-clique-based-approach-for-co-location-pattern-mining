from cliquecoloc.data import load_csv
from cliquecoloc.generator import GeneratorParams, generate_synthetic
from cliquecoloc.__init__ import run_pipeline
from pathlib import Path

csv_path = Path("e:/A clique-based approach for co-location pattern mining/data/testDataset.csv")
ds_real = load_csv(csv_path)

cliques, chash, patterns = run_pipeline(
    ds_real,
    min_dist=5.0,
    min_prev=0.6,
    schema="ids",
)

print("Real data:")
print("  #cliques:", len(cliques))
print("  #candidate keys in C-hash:", len(chash.candidates))
print("  #prevalent co-locations:", len(patterns))

# in thử vài pattern
sorted_patterns = sorted(patterns.items(), key=lambda x: (len(x[0]), sorted(list(x[0]))))
for k, v in sorted_patterns:
    print(k, "PI =", v)
