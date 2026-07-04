# Private Coupled Descent - synthetic validation

Synthetic validation code for the theory paper *"Private Coupled Descent: Dimension-Free Privacy for Vertical Federated Learning by Privatizing the Coupling Variable"* (Faes, van den Berg, Amir Haeri; IEEE TSP).

This is a **theory** paper: the contribution is the theorems (no-go, convergence to an O(sigma) neighborhood under DP, dimension-free prediction risk, a matching lower bound). There is **no external dataset**; the code generates vertical block-term data in-script from fixed seeds and checks the theorems' own quantitative predictions. The COVERT method core lives in the applied companion paper's code, not here; this repository validates the theory only.

## What it computes

`syn1_theory_validation.py` generates vertical block-term regression data and confirms three predicted scalings, reported over 20 seeds (deterministic given the seeds):

| macro | what it measures | theory predicts |
|---|---|---|
| `synFloorSlope` | log-log slope of the stationarity floor vs sigma^2 | 1 |
| `synDimFlat` | excess prediction-risk ratio across a 100x ambient-dimension increase | 1 |
| `synEpsSlope` | log-log slope of the prediction-risk floor vs 1/eps^2 | 1 |
| `synGap` | measured floor / lower-bound prediction | O(1) |

The measured values are 1.00 / 1.03 / 0.97 / 1.1, all within seed variability of the predicted exponents.

## Run

```bash
pip install -r requirements.txt        # numpy only
python paper.py                        # runs SYN-1, prints the 5 macros (deterministic)
```
`paper.py` prints both the values and the `\newcommand` macros to paste into the manuscript.

## Layout

- `syn1_theory_validation.py` - the SYN-1 validation: the profiled (Rayleigh-quotient) prediction risk, the proximal-gradient stationarity floor with DP noise on the coupling score, and the three scaling fits.
- `paper.py` - the single entry point (runs SYN-1, prints the macros).
- `requirements.txt`, `LICENSE` (MIT).

The theorems themselves are proved in the manuscript appendices; this code only checks their numbers.
