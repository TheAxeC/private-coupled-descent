"""Single entry point: reproduce the paper's SYN-1 synthetic-validation numbers.

Run from this directory:
    python paper.py

Runs the SYN-1 theory validation (deterministic from fixed seeds; no external data, no download) and
prints the five manuscript macros. The theorems themselves are in the manuscript appendices; this only
checks their quantitative predictions (if a prediction fails, a proof has a bug).
"""
from syn1_theory_validation import run


def main():
    r = run()
    print("SYN-1 theory-validation results (20 seeds):")
    print(f"  synFloorSlope = {r['synFloorSlope']:.3f}   (Thm 2 predicts 1)")
    print(f"  synDimFlat    = {r['synDimFlat']:.3f}   (Cor predicts 1)")
    print(f"  synEpsSlope   = {r['synEpsSlope']:.3f}   (Prop predicts 1)")
    print(f"  synGap        = {r['synGap']:.3f}   (predicts O(1))")
    print(f"  synSeeds      = {r['synSeeds']}")
    print("\n% macros for the manuscript:")
    print(f"\\newcommand{{\\synFloorSlope}}{{{r['synFloorSlope']:.2f}}}")
    print(f"\\newcommand{{\\synDimFlat}}{{{r['synDimFlat']:.2f}}}")
    print(f"\\newcommand{{\\synEpsSlope}}{{{r['synEpsSlope']:.2f}}}")
    print(f"\\newcommand{{\\synGap}}{{{r['synGap']:.1f}}}")
    print(f"\\newcommand{{\\synSeeds}}{{{r['synSeeds']}}}")


if __name__ == "__main__":
    main()
