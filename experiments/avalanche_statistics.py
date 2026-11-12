
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.special import zeta
from scipy.optimize import minimize_scalar


# ==========================================
# 1. PROJECT PATHS AND PARAMETERS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULT_DIR = BASE_DIR / "results"
FIGURE_DIR = BASE_DIR / "figures"

RESULT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

LATTICE_SIZES = [20, 40, 60]

# Require enough observations in the fitted tail
MIN_TAIL_SAMPLES = 50


# ==========================================
# 2. LOAD AVALANCHE DATA
# ==========================================

def load_data(size):

    filename = RESULT_DIR / f"avalanches_L{size}.csv"

    data = pd.read_csv(filename)["avalanche_size"].values

    # Remove zero-size avalanches
    data = data[data > 0]

    return data.astype(int)


# ==========================================
# 3. CALCULATE EMPIRICAL CCDF
# ==========================================

def calculate_ccdf(data):

    values, counts = np.unique(
        data, return_counts=True
    )

    survival = np.cumsum(counts[::-1])[::-1]

    ccdf = survival / len(data)

    return values, ccdf


# ==========================================
# 4. DISCRETE POWER-LAW MLE
# ==========================================

def estimate_alpha(data, xmin):

    tail = data[data >= xmin]

    n = len(tail)

    sum_log = np.sum(np.log(tail))

    # Discrete power-law negative log likelihood
    def negative_log_likelihood(alpha):

        return (
            alpha * sum_log
            + n * np.log(zeta(alpha, xmin))
        )

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(1.01, 5.0),
        method="bounded"
    )

    return result.x


# ==========================================
# 5. THEORETICAL DISCRETE CCDF
# ==========================================

def theoretical_ccdf(values, alpha, xmin):

    return (
        zeta(alpha, values)
        / zeta(alpha, xmin)
    )


# ==========================================
# 6. SELECT xmin USING KS DISTANCE
# ==========================================

def fit_power_law(data):

    unique_values = np.unique(data)

    best_fit = None
    minimum_ks = np.inf

    for xmin in unique_values:

        tail = data[data >= xmin]

        if len(tail) < MIN_TAIL_SAMPLES:
            continue

        alpha = estimate_alpha(data, xmin)

        # Compare empirical and theoretical CDFs
        values, counts = np.unique(
            tail, return_counts=True
        )

        empirical_cdf = (
            np.cumsum(counts) / len(tail)
        )

        theoretical_survival = theoretical_ccdf(
            values, alpha, xmin
        )

        theoretical_cdf = 1 - theoretical_survival

        ks_distance = np.max(
            np.abs(empirical_cdf - theoretical_cdf)
        )

        if ks_distance < minimum_ks:

            minimum_ks = ks_distance

            best_fit = {
                "alpha": alpha,
                "xmin": int(xmin),
                "ks": minimum_ks,
                "n_tail": len(tail)
            }

    return best_fit


# ==========================================
# 7. ANALYSE EACH LATTICE SIZE
# ==========================================

all_data = {}
fit_results = {}

for size in LATTICE_SIZES:

    print(f"\nAnalysing lattice size {size} x {size}")

    data = load_data(size)

    all_data[size] = data

    fit = fit_power_law(data)

    if fit is None:

        print("Not enough tail observations for fitting.")
        continue

    fit_results[size] = fit

    print("Positive avalanches:", len(data))
    print("Estimated alpha:", round(fit["alpha"], 4))
    print("Estimated xmin:", fit["xmin"])
    print("Observations in fitted tail:", fit["n_tail"])
    print("KS distance:", round(fit["ks"], 5))


# ==========================================
# 8. PLOT EMPIRICAL CCDF AND FITTED MODEL
# ==========================================

plt.figure(figsize=(10, 7))

for size in LATTICE_SIZES:

    if size not in fit_results:
        continue

    data = all_data[size]
    fit = fit_results[size]

    alpha = fit["alpha"]
    xmin = fit["xmin"]

    values, ccdf = calculate_ccdf(data)

    # Empirical distribution
    plt.loglog(
        values,
        ccdf,
        marker="o",
        markersize=3,
        linestyle="none",
        label=f"L = {size} (empirical)"
    )

    # Fitted theoretical tail
    fit_values = values[values >= xmin]

    fitted_ccdf = theoretical_ccdf(
        fit_values, alpha, xmin
    )

    plt.loglog(
        fit_values,
        fitted_ccdf,
        linestyle="--",
        linewidth=2,
        label=f"L = {size} (MLE fit)"
    )

plt.xlabel("Avalanche size (topplings)")
plt.ylabel("CCDF: P(S >= s)")

plt.title("Avalanche CCDF and Discrete Power-Law Fits")

plt.legend()
plt.grid(alpha=0.25, which="both")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "07_avalanche_ccdf.png",
    dpi=300
)

plt.show()


# ==========================================
# 9. SAVE FITTING RESULTS
# ==========================================

if fit_results:

    results_table = pd.DataFrame([
        {
            "Lattice size": size,
            "Alpha": fit["alpha"],
            "CCDF exponent (approx.)": fit["alpha"] - 1,
            "xmin": fit["xmin"],
            "Tail observations": fit["n_tail"],
            "KS distance": fit["ks"]
        }
        for size, fit in fit_results.items()
    ])

    results_table.to_csv(
        RESULT_DIR / "power_law_fitting_results.csv",
        index=False
    )

    print("\nPOWER-LAW FITTING SUMMARY")
    print(results_table.to_string(index=False))

print("\nAnalysis completed!")
print("CCDF graph saved in figures/")
print("Fitting results saved in results/")