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
python run_example.py
```

### Run with Jupyter Notebook

```bash
pip install jupyter
jupyter notebook examples/demo.ipynb
```

## 📖 Documentation

### Basic example

```python
from pathlib import Path
from cliquecoloc import (
    run_pipeline,
    GeneratorParams,
    generate_synthetic,
    save_csv,
)

DATA_DIR = Path("data")

# Generate synthetic data
params = GeneratorParams(
    P=20,        # Number of prevalent patterns
    I=500,       # Number of instances per feature
    D=5000.0,    # Space dimension
    F=20,        # Number of features
    Q=5,         # Pattern size
    m=50000,     # Total number of instances
    min_dist=50.0 # Minimum distance threshold
)

ds_syn = generate_synthetic(params, seed=42)

# Mine co-location patterns
# schema can be 'ids' or 'nds'
cliques, chash, patterns = run_pipeline(ds_syn, min_dist=50, min_prev=0.2, schema="nds")

print(f"Number of cliques: {len(cliques)}")
print(f"Number of prevalent patterns: {len(patterns)}")
```

## 📁 Project Structure

```
.
├── cliquecoloc/         # Main module
│   ├── synthetic.py     # Synthetic data generation
│   ├── miner.py         # Mining algorithms
│   ├── ids.py           # IDS algorithm
│   ├── nds.py           # NDS algorithm
│   ├── neighborhood.py  # Neighbor search
│   ├── prevalence.py    # Prevalence calculation
│   └── chash.py         # Clique hashing
├── data/                # Data module
├── examples/            # Examples
│   └── demo.ipynb       # Notebook demo
├── run_example.py       # Python demo
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

### run_pipeline
- `dataset`: Input dataset
- `min_dist`: Neighbor distance threshold
- `min_prev`: Minimum prevalence threshold (0-1)
- `schema`: Algorithm variant ('ids' or 'nds')

## 📊 Output

- **Cliques**: Set of spatial feature combinations satisfying neighbor conditions
- **Prevalent patterns**: Cliques with participation index ≥ `min_prev`

## 🔧 Troubleshooting

### Error "ModuleNotFoundError: No module named 'cliquecoloc'"

Make sure you are running the script from the root directory of the project, which contains the `cliquecoloc` folder.

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
