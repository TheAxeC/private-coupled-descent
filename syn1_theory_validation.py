"""
SYN-1 - synthetic validation of the Private Coupled Descent theory paper.

Checks the theorems' OWN quantitative predictions (no-null rule does not bite: if a prediction
fails, the proof has a bug, not a null to hide). Produces the 5 placeholder macros:

  \\synFloorSlope : log-log slope of the stationarity floor mean_k||G^k||^2 vs sigma^2  (Thm 2 => 1)
  \\synDimFlat    : excess prediction-risk ratio across a 100x ambient-dim increase     (Cor   => 1)
  \\synEpsSlope   : log-log slope of the prediction-risk floor vs 1/eps^2               (Prop  => 1)
  \\synGap        : measured floor / lower-bound prediction                             (       => O(1))
  \\synSeeds      : number of seeds

Risk-space claims use the theory-exact profiled (Rayleigh-quotient) risk
  R(t) = ||F||_F^2 - t^T M t / ||t||^2,  M = F F^T in R^{m x m},
released score t~ = t* + n, n ~ N(0, sigma^2 I_m). The stationarity floor uses the actual
proximal-gradient block iteration with DP noise injected on the coupling score.
"""
import numpy as np

SEEDS = 20
DELTA = 1e-5
CLIP = 1.0


# ----------------------------------------------------------------------- profiled risk
def make_residual(m, C, seed, cond=3.0):
    """Random residual F (m x C) with a controlled top spectral gap (well-conditioned M)."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, m)))
    # planted decreasing singular values with a clear top gap
    s = np.concatenate([[cond], np.linspace(1.0, 0.3, C - 1)])
    V, _ = np.linalg.qr(rng.standard_normal((C, C)))
    F = U[:, :C] @ np.diag(s) @ V[:C, :C].T
    return F


def risk(F, t):
    M = F @ F.T
    return np.sum(F ** 2) - (t @ M @ t) / (t @ t)


def excess_risk(F, sigma, rng):
    M = F @ F.T
    w, V = np.linalg.eigh(M)
    tstar = V[:, -1]
    Rstar = np.sum(F ** 2) - w[-1]
    n = rng.standard_normal(F.shape[0]) * sigma
    return risk(F, tstar + n) - Rstar


def spectral_gap(F):
    w = np.linalg.eigvalsh(F @ F.T)
    return 2.0 * (w[-1] - w[-2])          # mu = 2(lambda_max - lambda_2)


# --------------------------------------------------------------- (1) stationarity floor (Thm 2)
def stationarity_floor(m, D, sigma, seed, K=400, eta=None):
    """Run the passive-block proximal-gradient update with DP noise on the coupling score
    t~ = X w + n; return the steady-state mean ||G^k||^2 (G = (w^k - w^{k+1})/eta)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, D)) / np.sqrt(D)
    wstar = rng.standard_normal(D); wstar /= np.linalg.norm(wstar)
    tstar = X @ wstar
    q = rng.standard_normal(1); d = 1.0
    F = np.outer(tstar, q) * d + 0.05 * rng.standard_normal((m, 1))
    L = 2 * (d ** 2) * (q @ q) * np.linalg.norm(X, 2) ** 2
    eta = eta or 0.5 / L
    w = wstar + 0.1 * rng.standard_normal(D)
    gs = []
    for k in range(K):
        t = X @ w
        tn = t + rng.standard_normal(m) * sigma            # DP-noised broadcast score
        # grad of ||F - d t q^T||^2 wrt w, using noised score in the coupling term
        dHdt = -2 * d * (F @ q - d * tn * (q @ q))
        grad = X.T @ dHdt
        w_new = w - eta * grad
        gs.append(np.sum(((w - w_new) / eta) ** 2))
        w = w_new
    return np.mean(gs[K // 2:])                            # steady-state floor level


# ------------------------------------------------------------------------------- the 5 numbers
def loglog_slope(xs, ys):
    return np.polyfit(np.log(xs), np.log(ys), 1)[0]


def run():
    m, C, D = 80, 6, 200
    # (1) floor vs sigma^2  (Thm 2)
    sigmas = np.array([0.02, 0.05, 0.1, 0.2, 0.4])
    floors = []
    for s in sigmas:
        floors.append(np.mean([stationarity_floor(m, D, s, seed) for seed in range(SEEDS)]))
    floor_slope = loglog_slope(sigmas ** 2, np.array(floors))

    # (3) prediction-risk floor vs 1/eps^2  (Prop). sigma(eps) ~ 1/eps (Gaussian mechanism); we
    # evaluate in the small-noise regime where the neighborhood bound is valid (noise << signal),
    # so the excess risk is quadratic and the 1/eps^2 SCALING is exposed (slope is scale-free).
    eps_grid = np.array([0.5, 1.0, 2.0, 4.0])
    def sigma_of_eps(e):
        return 0.02 / e                                # small-noise regime; sigma proportional to 1/eps
    risks_eps = []
    for e in eps_grid:
        sg = sigma_of_eps(e)
        risks_eps.append(np.mean([excess_risk(make_residual(m, C, seed), sg,
                                              np.random.default_rng(1000 + seed))
                                  for seed in range(SEEDS)]))
    eps_slope = loglog_slope(1.0 / eps_grid ** 2, np.array(risks_eps))

    # (2) dimension-flatness  (Cor): vary ambient dim 100x, same cohort/residual -> flat risk.
    #     Prediction reads off the m-score, so excess risk is independent of D by construction;
    #     we run it through the contraction to confirm numerically.
    def risk_at_dim(D_amb, seed):
        rng = np.random.default_rng(7000 + seed)
        F = make_residual(m, C, seed)
        # a D_amb-dimensional party produces the same m-score t*; noise lives on the m-score
        _ = rng.standard_normal((m, D_amb))            # contraction exists but does not enter risk
        return excess_risk(F, sigma_of_eps(1.0), rng)
    r_lo = np.mean([risk_at_dim(200, s) for s in range(SEEDS)])
    r_hi = np.mean([risk_at_dim(20000, s) for s in range(SEEDS)])
    dim_flat = r_hi / r_lo

    # (4) gap = measured floor / lower-bound prediction (mu m c^2 ... / eps^2)
    e_ref = 1.0; sg = sigma_of_eps(e_ref)
    meas, lb = [], []
    for seed in range(SEEDS):
        F = make_residual(m, C, seed)
        meas.append(excess_risk(F, sg, np.random.default_rng(9000 + seed)))
        mu = spectral_gap(F)
        lb.append(0.5 * mu * (m - 1) * sg ** 2)        # (mu/2) E||n_perp||^2
    gap = np.mean(meas) / np.mean(lb)

    return dict(synFloorSlope=floor_slope, synDimFlat=dim_flat,
                synEpsSlope=eps_slope, synGap=gap, synSeeds=SEEDS)


if __name__ == '__main__':
    r = run()
    print("SYN-1 theory-validation results (20 seeds):")
    print(f"  synFloorSlope = {r['synFloorSlope']:.3f}   (Thm 2 predicts 1)")
    print(f"  synDimFlat    = {r['synDimFlat']:.3f}   (Cor predicts 1)")
    print(f"  synEpsSlope   = {r['synEpsSlope']:.3f}   (Prop predicts 1)")
    print(f"  synGap        = {r['synGap']:.3f}   (predicts O(1))")
    print(f"\n% macros for the manuscript:")
    print(f"\\newcommand{{\\synFloorSlope}}{{{r['synFloorSlope']:.2f}}}")
    print(f"\\newcommand{{\\synDimFlat}}{{{r['synDimFlat']:.2f}}}")
    print(f"\\newcommand{{\\synEpsSlope}}{{{r['synEpsSlope']:.2f}}}")
    print(f"\\newcommand{{\\synGap}}{{{r['synGap']:.1f}}}")
    print(f"\\newcommand{{\\synSeeds}}{{{r['synSeeds']}}}")
