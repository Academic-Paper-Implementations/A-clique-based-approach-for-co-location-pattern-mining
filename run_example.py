from pathlib import Path
from cliquecoloc import (
    load_csv,
    run_pipeline,
    GeneratorParams,
    generate_synthetic,
    save_csv,
)

DATA_DIR = Path("data")


def main():
    # 1. Test với synthetic
    params = GeneratorParams(
        P=20,
        I=500,
        D=5000,
        F=20,
        Q=5,
        m=50000,
        min_dist=50,
    )
    ds_syn = generate_synthetic(params, seed=42)
    save_csv(ds_syn, DATA_DIR / "synthetic.csv")
    print("Synthetic dataset saved to data/synthetic.csv")

    cliques, _, patterns = run_pipeline(ds_syn, min_dist=50, min_prev=0.2, schema="nds")
    print(f"Synthetic: #cliques={len(cliques)}, #patterns={len(patterns)}")

    # 2. Test với data.csv nếu có
    csv_path = DATA_DIR / "data.csv"
    if csv_path.exists():
        ds = load_csv(csv_path)
        cliques2, _, patterns2 = run_pipeline(ds, min_dist=50, min_prev=0.2, schema="nds")
        print(f"CSV data: #cliques={len(cliques2)}, #patterns={len(patterns2)}")
    else:
        print("data/data.csv không tồn tại – bỏ qua phần test CSV.")


if __name__ == "__main__":
    main()
