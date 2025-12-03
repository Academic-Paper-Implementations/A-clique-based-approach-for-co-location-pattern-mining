# Clique-based Co-location Pattern Mining

Implementation of algorithms from the paper:
**"A clique-based approach for co-location pattern mining"** by Bao et al. (2019)

## Overview

This repository implements all 6 core algorithms from the paper for mining prevalent spatial co-location patterns:

1. **Algorithm 1**: Grid-based neighborhood materialization
2. **Algorithm 2**: IDS (Instance Driven Schema) clique generation
3. **Algorithm 3**: NDS (Neighborhood Driven Schema) clique generation
4. **Algorithm 4**: C-hash (Compressed clique hash) structure
5. **Algorithm 5**: Prevalent co-location filtering
6. **Algorithm 6**: PI (Participation Index) value calculation

## Structure

```
colocation/
├── __init__.py          # Main API (mine_with_ids, mine_with_nds)
├── data.py              # Core data structures (Instance, SpatialDataset)
├── neighborhood.py      # Algorithm 1: Neighborhood materialization
├── ids.py               # Algorithm 2: IDS clique generation
├── nds.py               # Algorithm 3: NDS clique generation
├── chash.py             # Algorithm 4: C-hash structure
└── prevalence.py        # Algorithms 5 & 6: Prevalence mining

examples/
├── run_example.py       # Simple usage example
└── demo.ipynb           # Interactive demonstration

data/
└── sample_data.csv      # Sample dataset
```

## Key Concepts

### Data Structures

- **Instance**: `(feature, idx, x, y)` - A spatial object with type and location
- **SpatialDataset**: Collection of instances representing spatial features
- **NeighborhoodList**: Stores `Ns(s)`, `SNs(s)`, `BNs(s)` for all instances

### Algorithms

#### Algorithm 1: Neighborhood Materialization
- Divides space into grid cells
- Computes three types of neighborhoods for each instance:
  - `Ns(s)`: all neighbors
  - `SNs(s)`: smaller neighbors (instances `s'` where `s' < s`)
  - `BNs(s)`: bigger neighbors (instances `s'` where `s < s'`)

#### Algorithm 2: IDS (Instance Driven Schema)
- Builds I-tree structure using BNs and right-sibling relationships
- Generates maximal cliques through breadth-first traversal
- Efficient for datasets with uniform feature distribution

#### Algorithm 3: NDS (Neighborhood Driven Schema)
- Builds N-tree structure using SNs and BNs
- Uses 4 expansion strategies:
  1. No relation → create new body
  2. Relation ⊇ body → extend existing body
  3. Relation ⊂ body → create new body from relation
  4. Intersection not empty → create new body from intersection
- Efficient for datasets with skewed feature distribution

#### Algorithm 4: C-hash
- Compressed storage of cliques
- Key: set of features in clique
- Value: instances grouped by feature
- Enables efficient PI calculation

#### Algorithms 5 & 6: Prevalence Mining
- Filters candidate patterns by minimum prevalence threshold
- Computes Participation Index (PI) for each pattern
- PI = min{PR(c, f) | f ∈ c} where PR = participation ratio

## Usage

### Basic Example

```python
from colocation import SpatialDataset, mine_with_ids, mine_with_nds

# Create dataset
data = [
    ("A", 1, 0.0, 0.0),
    ("B", 1, 0.1, 0.0),
    ("C", 1, 0.05, 0.08),
    # ... more instances
]
dataset = SpatialDataset.from_list(data)

# Run IDS pipeline
patterns_ids = mine_with_ids(
    dataset=dataset,
    min_dist=0.2,    # neighbor distance threshold
    min_prev=0.5     # minimum prevalence threshold
)

# Run NDS pipeline
patterns_nds = mine_with_nds(
    dataset=dataset,
    min_dist=0.2,
    min_prev=0.5
)

# Display results
for pattern, pi in patterns_ids.items():
    print(f"{set(pattern)}: PI = {pi:.3f}")
```

### Step-by-Step Pipeline

```python
from colocation import (
    SpatialDataset,
    materialize_neighborhoods,
    mine_cliques_ids,
    mine_cliques_nds,
    CHash,
    mine_prevalent_patterns
)

# 1. Create dataset
dataset = SpatialDataset.from_list(data)

# 2. Materialize neighborhoods (Algorithm 1)
nbs = materialize_neighborhoods(dataset, min_dist=0.2)

# 3a. Generate cliques with IDS (Algorithm 2)
cliques = mine_cliques_ids(dataset, nbs)

# OR 3b. Generate cliques with NDS (Algorithm 3)
cliques = mine_cliques_nds(dataset, nbs)

# 4. Build C-hash (Algorithm 4)
chash = CHash()
for clique in cliques:
    chash.add_clique(clique)

# 5. Mine prevalent patterns (Algorithms 5 & 6)
patterns = mine_prevalent_patterns(dataset, chash, min_prev=0.5)
```

## Running Examples

### Command Line

```bash
# Run the example script
python examples/run_example.py
```

### Jupyter Notebook

```bash
# Launch Jupyter and open demo.ipynb
jupyter notebook examples/demo.ipynb
```

## Implementation Notes

### Adherence to Paper

This implementation follows the paper's algorithms as closely as possible:

- **Exact ordering**: Instances ordered by (feature, idx) as specified
- **Grid-based materialization**: Follows Algorithm 1 step-by-step
- **I-tree structure**: Implements BNs, RS relationships per Definition 6 & Lemma 3
- **N-tree structure**: Implements 4 strategies from Example 6 & Section 3.2.2
- **C-hash**: Follows Definition 9 exactly
- **PI calculation**: Implements Lemma 10 and Definition 4

### Design Decisions

1. **Tree reset**: After processing each head node, trees are reset to save memory (doesn't affect correctness)
2. **Subset removal**: NDS performs explicit duplicate/subset removal to ensure Lemma 6-7 properties
3. **Grid coordinates**: Uses floor division for consistent grid assignment
4. **Float comparisons**: Uses squared distances to avoid sqrt overhead

## Performance

Both IDS and NDS produce identical results but have different performance characteristics:

- **IDS**: Better for uniform feature distributions
- **NDS**: Better for skewed feature distributions

Choose based on your data characteristics (see Section 4 of the paper).

## Testing

The implementation can be verified by:

1. Running `examples/run_example.py` - should show identical results for IDS and NDS
2. Comparing with paper's examples (Example 6 shows expected cliques)
3. Checking that PI values satisfy: 0 ≤ PI ≤ 1 and monotonicity properties

## References

Bao, X., Wang, L., et al. (2019). "A clique-based approach for co-location pattern mining." *Information Sciences*, 490, 244-264.

## License

MIT License - Academic implementation for research and educational purposes.
