import sys
from pathlib import Path

# Add parent directory to path so we can import colocation module
sys.path.insert(0, str(Path(__file__).parent.parent))

from colocation.synthetic import GeneratorParams, SyntheticSpatialGenerator
from colocation.miner import CoLocationMiner

def main():
    params = GeneratorParams(
        P=20,
        I=500,
        D=10000.0,
        F=20,
        Q=5,
        m=50000,
        min_dist=50.0,
        clumpy=1,   # bạn đổi clumpy=2,3,... để test mật độ như paper
    )

    gen = SyntheticSpatialGenerator(params, seed=42)
    ds = gen.generate()

    miner = CoLocationMiner(dataset=ds,
                            min_dist=params.min_dist,
                            min_prev=0.2)

    cliques_ids, prev_ids = miner.run_ids()
    cliques_nds, prev_nds = miner.run_nds()

    print("Số instance:", len(ds.instances))
    print("Số clique IDS:", len(cliques_ids))
    print("Số clique NDS:", len(cliques_nds))
    print("Số prevalent pattern (IDS):", len(prev_ids))
    print("Số prevalent pattern (NDS):", len(prev_nds))

if __name__ == "__main__":
    main()
