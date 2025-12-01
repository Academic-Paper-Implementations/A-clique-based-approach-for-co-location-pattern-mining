# A Clique-Based Approach for Co-Location Pattern Mining

Implementation of clique-based co-location pattern mining algorithm using Instance-Data-Structure (IDS) and Neighbor-Data-Structure (NDS).

## 📋 Requirements

- Python 3.7+
- Libraries: numpy, scipy (will be installed automatically)

## 🚀 Installation

### Step 1: Clone repository

```bash
git clone https://github.com/Academic-Paper-Implementations/A-clique-based-approach-for-co-location-pattern-mining.git
cd "A clique-based approach for co-location pattern mining"
```

### Step 2: Create virtual environment (recommended)

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install numpy scipy
```

## 💻 Usage

### Run simple demo

```bash
python examples/demo.py
```

### Run with Jupyter Notebook

```bash
pip install jupyter
jupyter notebook examples/demo.ipynb
```

## 📖 Documentation

### Basic example

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from colocation.synthetic import GeneratorParams, SyntheticSpatialGenerator
from colocation.miner import CoLocationMiner

# Generate synthetic data
params = GeneratorParams(
    P=20,        # Number of prevalent patterns
    I=500,       # Number of instances per feature
    D=10000.0,   # Space dimension
    F=20,        # Number of features
    Q=5,         # Pattern size
    m=50000,     # Total number of instances
    min_dist=50.0,   # Minimum distance threshold
    clumpy=1     # Clumpiness level (1, 2, 3,...)
)

gen = SyntheticSpatialGenerator(params, seed=42)
dataset = gen.generate()

# Mine co-location patterns
miner = CoLocationMiner(
    dataset=dataset,
    min_dist=params.min_dist,
    min_prev=0.2  # Minimum prevalence threshold
)

# Run algorithms
cliques_ids, prev_ids = miner.run_ids()  # Using IDS
cliques_nds, prev_nds = miner.run_nds()  # Using NDS

# View results
print(f"Number of cliques (IDS): {len(cliques_ids)}")
print(f"Number of prevalent patterns (IDS): {len(prev_ids)}")
print(f"Number of cliques (NDS): {len(cliques_nds)}")
print(f"Number of prevalent patterns (NDS): {len(prev_nds)}")
```

## 📁 Project Structure

```
.
├── colocation/          # Main module
│   ├── synthetic.py     # Synthetic data generation
│   ├── miner.py         # Mining algorithms
│   ├── itree_ids.py     # IDS structure
│   ├── ntree_nds.py     # NDS structure
│   ├── neighbors.py     # Neighbor search
│   ├── prevalent.py     # Prevalence calculation
│   └── chash.py         # Clique hashing
├── data/                # Data module
├── examples/            # Examples
│   ├── demo.py          # Python demo
│   └── demo.ipynb       # Notebook demo
└── readme.md            # This file
```

## ⚙️ Configuration Parameters

### GeneratorParams
- `P`: Number of prevalent patterns to generate
- `I`: Number of instances per feature
- `D`: Space dimension (D×D)
- `F`: Number of spatial features
- `Q`: Size of each pattern
- `m`: Total number of instances
- `min_dist`: Minimum neighbor distance threshold
- `clumpy`: Clumpiness level (1=sparse, 2-3=denser)

### CoLocationMiner
- `dataset`: Input dataset
- `min_dist`: Neighbor distance threshold
- `min_prev`: Minimum prevalence threshold (0-1)

## 📊 Output

- **Cliques**: Set of spatial feature combinations satisfying neighbor conditions
- **Prevalent patterns**: Cliques with participation index ≥ `min_prev`

## 🔧 Troubleshooting

### Error "ModuleNotFoundError: No module named 'colocation'"

Make sure to add the parent directory path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Missing dependencies

```bash
pip install numpy scipy
```

## 📚 References

Implementation based on research paper about clique-based co-location pattern mining.

## 📝 License

MIT License

## 👥 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
