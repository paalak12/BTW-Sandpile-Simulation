
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path

# ==========================================
# 1. PARAMETERS
# ==========================================

LATTICE_SIZES = [20, 40, 60]

GRAIN_COUNTS = {
    20: 20000,
    40: 100000,
    60: 200000
}

CRITICAL_HEIGHT = 4
SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = BASE_DIR / "figures"
RESULT_DIR = BASE_DIR / "results"

FIGURE_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. TOPPLING ALGORITHM
# ==========================================

def topple(grid, start_i, start_j):

    rows, cols = grid.shape

    queue = deque([(start_i, start_j)])

    in_queue = np.zeros(grid.shape, dtype=bool)
    in_queue[start_i, start_j] = True

    topplings = 0

    while queue:

        i, j = queue.popleft()
        in_queue[i, j] = False

        while grid[i, j] >= CRITICAL_HEIGHT:

            grid[i, j] -= 4
            topplings += 1

            neighbours = [
                (i-1, j),
                (i+1, j),
                (i, j-1),
                (i, j+1)
            ]

            for ni, nj in neighbours:

                if 0 <= ni < rows and 0 <= nj < cols:

                    grid[ni, nj] += 1

                    if (grid[ni, nj] >= CRITICAL_HEIGHT
                            and not in_queue[ni, nj]):

                        queue.append((ni, nj))
                        in_queue[ni, nj] = True

    return topplings


# ==========================================
# 3. ADD GRAIN
# ==========================================

def add_grain(grid, rng):

    i = rng.integers(0, grid.shape[0])
    j = rng.integers(0, grid.shape[1])

    grid[i, j] += 1

    return topple(grid, i, j)


# ==========================================
# 4. RUN SIMULATION
# ==========================================

def simulate(size, number_of_grains, seed):

    rng = np.random.default_rng(seed)

    grid = np.zeros((size, size), dtype=int)

    avalanche_sizes = []
    mean_heights = []

    print(f"\nRunning lattice {size} x {size}...")

    for step in range(number_of_grains):

        avalanche = add_grain(grid, rng)

        avalanche_sizes.append(avalanche)

        mean_heights.append(grid.mean())

        if (step + 1) % 20000 == 0:
            print(f"  Grains added: {step + 1}")

    return grid, np.array(avalanche_sizes), np.array(mean_heights)


# ==========================================
# 5. RUN ALL EXPERIMENTS
# ==========================================

all_results = {}

for size in LATTICE_SIZES:

    grid, avalanches, heights = simulate(
        size,
        GRAIN_COUNTS[size],
        SEED
    )

    all_results[size] = avalanches

    # Save individual numerical data
    np.savetxt(
        RESULT_DIR / f"avalanches_L{size}.csv",
        avalanches,
        delimiter=",",
        header="avalanche_size",
        comments=""
    )

    # Save mean height history
    np.savetxt(
        RESULT_DIR / f"mean_height_L{size}.csv",
        heights,
        delimiter=",",
        header="mean_height",
        comments=""
    )

    print("Mean height:", round(grid.mean(), 4))
    print("Non-zero avalanches:", np.count_nonzero(avalanches))
    print("Largest avalanche:", avalanches.max())


# ==========================================
# 6. COMPARE AVALANCHE DISTRIBUTIONS
# ==========================================

plt.figure(figsize=(9, 6))

for size, avalanches in all_results.items():

    positive = avalanches[avalanches > 0]

    if len(positive) == 0:
        continue

    bins = np.logspace(
        0,
        np.log10(positive.max() + 1),
        35
    )

    probabilities, edges = np.histogram(
        positive,
        bins=bins,
        density=True
    )

    centers = np.sqrt(edges[:-1] * edges[1:])

    valid = probabilities > 0

    plt.loglog(
        centers[valid],
        probabilities[valid],
        marker="o",
        markersize=4,
        linewidth=1.2,
        label=f"L = {size}"
    )

plt.xlabel("Avalanche size (topplings)")
plt.ylabel("Probability density")
plt.title("Avalanche Distribution for Different Lattice Sizes")

plt.legend()
plt.grid(alpha=0.25, which="both")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "05_lattice_comparison.png",
    dpi=300
)

plt.show()

print("\nAll lattice-size experiments completed!")
print("Graphs saved in figures/")
print("Data saved in results/")
