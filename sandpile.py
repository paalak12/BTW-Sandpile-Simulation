
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from collections import deque
from pathlib import Path

# ==========================================
# 1. PARAMETERS
# ==========================================

GRID_SIZE = 20
N_GRAINS = 20000
SEED = 42
CRITICAL_HEIGHT = 4

FIGURE_DIR = Path("figures")
ANIMATION_DIR = Path("animations")

FIGURE_DIR.mkdir(exist_ok=True)
ANIMATION_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)


# ==========================================
# 2. TOPPLING ALGORITHM (QUEUE BASED)
# ==========================================

def topple(grid, start_i, start_j):
    """
    Relax the sandpile using a non-recursive queue.
    Returns the total number of topplings.
    """

    rows, cols = grid.shape
    queue = deque()
    in_queue = np.zeros(grid.shape, dtype=bool)

    queue.append((start_i, start_j))
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

                # Open boundary: grains leaving the lattice are lost
                if 0 <= ni < rows and 0 <= nj < cols:

                    grid[ni, nj] += 1

                    if (grid[ni, nj] >= CRITICAL_HEIGHT
                            and not in_queue[ni, nj]):

                        queue.append((ni, nj))
                        in_queue[ni, nj] = True

    return topplings


# ==========================================
# 3. ADD ONE GRAIN
# ==========================================

def add_grain(grid, rng):

    i = rng.integers(0, grid.shape[0])
    j = rng.integers(0, grid.shape[1])

    grid[i, j] += 1

    avalanche_size = topple(grid, i, j)

    return avalanche_size


# ==========================================
# 4. RUN SIMULATION
# ==========================================

def run_simulation(size, n_grains, seed=SEED):

    local_rng = np.random.default_rng(seed)

    grid = np.zeros((size, size), dtype=int)

    avalanche_sizes = []
    mean_heights = []

    for step in range(n_grains):

        avalanche = add_grain(grid, local_rng)

        avalanche_sizes.append(avalanche)
        mean_heights.append(grid.mean())

    return grid, np.array(avalanche_sizes), np.array(mean_heights)


print("Running BTW Sandpile Simulation...")

grid, avalanche_sizes, mean_heights = run_simulation(
    GRID_SIZE, N_GRAINS
)

print("\nSIMULATION RESULTS")
print("-------------------------")
print("Grains added:", N_GRAINS)
print("Lattice size:", GRID_SIZE, "x", GRID_SIZE)
print("Mean height:", round(grid.mean(), 3))
print("Number of non-zero avalanches:",
      np.count_nonzero(avalanche_sizes))
print("Largest avalanche:", avalanche_sizes.max())


# ==========================================
# 5. FIGURE 1: FINAL LATTICE CONFIGURATION
# ==========================================

plt.figure(figsize=(7, 6))

plt.imshow(grid, cmap="plasma", origin="lower",
           vmin=0, vmax=3)

plt.colorbar(label="Number of grains")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Final Sandpile Configuration")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "01_lattice.png", dpi=300)
plt.show()


# ==========================================
# 6. FIGURE 2: MEAN HEIGHT VS TIME
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(np.arange(1, N_GRAINS + 1),
         mean_heights, linewidth=0.8)

plt.xlabel("Number of grains added")
plt.ylabel("Mean lattice height")
plt.title("Evolution of Mean Sandpile Height")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "02_mean_height.png", dpi=300)
plt.show()


# ==========================================
# 7. FIGURE 3: AVALANCHE SIZE DISTRIBUTION
# ==========================================

positive_sizes = avalanche_sizes[avalanche_sizes > 0]

if len(positive_sizes) > 0:

    bins = np.logspace(
        0,
        np.log10(positive_sizes.max() + 1),
        30
    )

    probabilities, edges = np.histogram(
        positive_sizes,
        bins=bins,
        density=True
    )

    centers = np.sqrt(edges[:-1] * edges[1:])

    valid = probabilities > 0

    plt.figure(figsize=(8, 6))

    plt.loglog(
        centers[valid],
        probabilities[valid],
        "o-",
        markersize=4,
        label="Simulation"
    )

    # Optional power-law fit.
    fit_mask = (
        valid
        & (centers >= 10)
        & (probabilities > 0)
    )

    if np.count_nonzero(fit_mask) >= 3:

        slope, intercept = np.polyfit(
            np.log10(centers[fit_mask]),
            np.log10(probabilities[fit_mask]),
            1
        )

        x_fit = centers[fit_mask]

        plt.loglog(
            x_fit,
            10**intercept * x_fit**slope,
            "--",
            label=f"Power-law fit: slope = {slope:.3f}"
        )

        print("Estimated power-law slope:", round(slope, 3))

    plt.xlabel("Avalanche size (number of topplings)")
    plt.ylabel("Probability density")
    plt.title("Avalanche Size Distribution")
    plt.legend()
    plt.grid(alpha=0.2, which="both")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "03_avalanche_distribution.png",
                dpi=300)
    plt.show()



# ==========================================
# 8. FIGURE 4: FINAL 3D SURFACE
# ==========================================

from matplotlib.cm import ScalarMappable

# Create coordinate grids
x, y = np.meshgrid(
    np.arange(GRID_SIZE),
    np.arange(GRID_SIZE)
)

# Set colour normalization from 0 to 3 grains
norm = Normalize(vmin=0, vmax=3)

# Create figure and 3D axes
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Plot the sandpile surface
surface = ax.plot_surface(
    x, y, grid,
    cmap="plasma",
    norm=norm,
    edgecolor="black",
    linewidth=0.2,
    antialiased=True
)

# Axis labels and title
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
ax.set_zlabel("Height (grains)")
ax.set_title("3D Surface of the BTW Sandpile")

# Set height limits and ticks
ax.set_zlim(0, 3)
ax.set_zticks([0, 1, 2, 3])

# Correct colour bar
fig.colorbar(
    ScalarMappable(norm=norm, cmap="plasma"),
    ax=ax,
    shrink=0.6,
    label="Height (grains)",
    ticks=[0, 1, 2, 3]
)

plt.tight_layout()

# Save high-resolution figure
plt.savefig(
    FIGURE_DIR / "04_3d_surface.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()



# ==========================================
# 9. 3D ANIMATION
# ==========================================

def create_animation(size=20, frames=150):

    animation_rng = np.random.default_rng(SEED)
    animation_grid = np.zeros((size, size), dtype=int)

    X, Y = np.meshgrid(
        np.arange(size),
        np.arange(size)
    )

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_zlim(0, 4)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_zlabel("Height")

    ax.set_title("BTW Sandpile Evolution")

    surface = [None]
    status = fig.text(0.5, 0.02, "", ha="center")

    def update(frame):

        avalanche = add_grain(
            animation_grid,
            animation_rng
        )

        if surface[0] is not None:
            surface[0].remove()

        surface[0] = ax.plot_surface(
            X, Y, animation_grid,
            cmap="plasma",
            norm=Normalize(0, 3),
            edgecolor="black",
            linewidth=0.2
        )

        status.set_text(
            f"Grain: {frame + 1} | Avalanche size: {avalanche}"
        )

        return surface[0], status

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=100,
        blit=False
    )

    ani.save(
        ANIMATION_DIR / "sandpile_animation.gif",
        writer="pillow",
        fps=8
    )

    plt.close(fig)

    print("Animation saved successfully!")


create_animation()

print("\nAll figures and animation have been saved!")
