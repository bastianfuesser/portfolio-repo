# Project 01 — Molecular Dynamics Simulation of Confined Water in Fluorohectorite

> HPC-scale molecular dynamics workflow for studying water and lithium dynamics in layered clay minerals. Includes automated SLURM job orchestration across a temperature × slit-size parameter sweep, a multi-stage GROMACS simulation pipeline, and parallelized Python analysis tools — with custom modifications to the `mdevaluate` library for spatially resolved structural analysis.

## Context & Objective

Fluorohectorite is a synthetic layered clay mineral with nanoscale interlayer spaces that confine water molecules and lithium ions. Understanding how confinement affects water dynamics and ion transport is relevant to applications in energy storage, catalysis, and geological fluid transport.

This project sets up and runs molecular dynamics simulations using the ClayFF force field in GROMACS, sweeping across **11 temperatures** (250–350 K) × **multiple slit sizes** (interlayer spacings). The goal is to characterize how confinement and temperature affect diffusion, relaxation dynamics, and local structure of the confined species.

- **Setting:** Master's thesis research
- **My role:** Sole developer of the entire simulation and analysis pipeline, self-taught. The starting point was a single 6,000-line Jupyter notebook with hardcoded absolute paths and no loops — every simulation and plot was a manual copy-paste job. I redesigned the entire workflow from scratch into the modular, automated pipeline shown here.
- **Design philosophy:** Every analysis script is deliberately self-contained (no shared utility module). This was a conscious decision — my supervisor's group had limited programming experience, and I wanted anyone inheriting this codebase to be able to open any single script and understand it without chasing imports across files.
- **Tech stack:** GROMACS, SLURM, Bash, Python, Eigen (via mdevaluate), Matplotlib, NumPy, SciPy

---

## 🏗 Project Structure

```
project-01-md-simulation/
├── scripts/
│   ├── orchestration/
│   │   ├── create_directory_temp_variation.sh   # Generates full parameter sweep from template
│   │   ├── run_directory.sh                     # Finds and launches all simulations
│   │   ├── run_cluster.sh                       # Per-job: fixes paths, injects temp, submits to SLURM
│   │   └── collect_all_data.sh                  # Parallelized post-simulation analysis pipeline
│   ├── analysis/
│   │   ├── dynamics/
│   │   │   ├── calculate_msd_isf.py             # MSD, ISF, diffusion, KWW relaxation times
│   │   │   ├── calculate_van_hove_self.py       # Van Hove self-correlation functions
│   │   │   ├── create_nojump.py                 # Unwrapped trajectory generation
│   │   │   └── ...                              # Additional dynamics modules
│   │   ├── structural/
│   │   │   ├── calculating_rdf_CN.py            # RDF and coordination numbers (modified mdevaluate)
│   │   │   ├── xy_density.py                    # 2D spatial density maps in z-slabs (fully custom)
│   │   │   └── ...                              # Additional structural analysis
│   │   └── layered_density/
│   │       ├── calculate_z_density_profile.py   # Z-density profiles
│   │       └── ...
│   └── cell_generation/                         # Simulation cell construction (not highlighted)
├── plotting/
│   ├── plotting_msd_isf.py                      # MSD and ISF visualization
│   ├── plotting_vhs.py                          # Van Hove 2D + 3D visualization
│   ├── diff_kww1e_plotting.py                   # Diffusion and relaxation time plots
│   └── ...                                      # Additional plotting modules
└── README.md
```

> Only selected scripts are included in this repository. The full analysis suite contains additional modules for z-density profiles, nojump trajectory generation, diffusion/relaxation comparison plots, and more — all following the same self-contained design pattern.

### Simulation directory layout (on the HPC cluster)

Each parameter combination is a self-contained simulation with four stages:

```
temperature_variation/
├── create_directory_temp_variation.sh
├── run_directory.sh                      ← one command launches everything
├── temp_250/
│   ├── slitsize_02p4/
│   │   ├── run_cluster.sh
│   │   ├── cluster.sh                    ← SLURM job script
│   │   ├── 00_setup/                     ← cell solvation, box adjustment
│   │   ├── 01_em/                        ← energy minimization
│   │   ├── 02_eq/                        ← equilibration + density matching
│   │   └── 03_prod/                      ← production run (data collected here)
│   ├── slitsize_04p8/
│   │   └── ...
│   └── slitsize_xxpx/
├── temp_260/
│   └── ...
└── temp_350/
    └── ...
```

---

## Key Code Deep Dives

### 1. Three-Level HPC Orchestration — `scripts/orchestration/`

**What it does:** A three-script chain that takes a single template directory and produces a fully parameterized, cluster-ready simulation sweep — then launches it all with one command.

**Why it matters:** Running 100+ simulations manually is error-prone. This system makes the entire workflow reproducible: create the sweep, launch everything, collect results — three commands total.

**Level 1 — `create_directory_temp_variation.sh`:** Generates the full parameter sweep from a template. Loops from T_min to T_max, copies the template (which already contains all slit-size subdirectories), and patches the temperature into each `run_cluster.sh`:

```bash
TMIN=250
TMAX=350
TSTEP=10
for ((i=$TMIN; i<=$TMAX; i+=TSTEP)); do
    NEW_DIRECTORY="$WORK_DIR/temp_$i"
    mkdir -p "$NEW_DIRECTORY"
    cp -a "$TEMPLATE_DIRECTORY/"* "$NEW_DIRECTORY"

    # Patch temperature into every run_cluster.sh found in this tree
    mapfile -t DIRECTORIES < <(find_directories_with_script "$NEW_DIRECTORY")
    for DIR in "${DIRECTORIES[@]}"; do
        sed -i "/^TEMP=/c \\TEMP=$i" "$DIR/run_cluster.sh"
    done
done
```

**Level 2 — `run_directory.sh`:** Recursively discovers every directory containing `run_cluster.sh` and executes it — no manual enumeration needed:

```bash
find . -type f -name "run_cluster.sh" -exec dirname {} \;
# → Finds all simulation directories, runs each one
```

**Level 3 — `run_cluster.sh`:** The per-simulation launcher. Solves the absolute-path problem on SLURM clusters by auto-detecting its own location and rewriting paths in the job script before submission:

```bash
WORK_DIR=$(realpath "$(dirname "$0")")
sed -i "/^#SBATCH --chdir=/c\\#SBATCH --chdir=$WORK_DIR" "$TARGET_SCRIPT"
sed -i "/^WORK_DIR=/c\\WORK_DIR=$WORK_DIR" "$TARGET_SCRIPT"

# Inject temperature into GROMACS .mdp files
sed -i "/^ref-t                    = /c\\ref-t                    = $TEMP" "$EQ_DIR/eq.mdp"
sed -i "/^ref-t                    = /c\\ref-t                    = $TEMP" "$PROD_DIR/prod.mdp"

sbatch $WORK_DIR/cluster.sh
```

---

### 2. Parallelized Analysis Pipeline — `scripts/orchestration/collect_all_data.sh`

**What it does:** After all simulations finish, this script runs the full analysis suite. Independent analyses run as background processes (`&`) in parallel, with dependency ordering where needed.

**Why it matters:** The analysis of 100+ simulations is computationally expensive. By parallelizing independent tasks (MSD, ISF, Van Hove in different directions) while respecting dependencies (nojump trajectories must be created first), the total wall time drops significantly.

```bash
# Nojump must complete first — all dynamics analyses depend on it
if ! $PYTHON_INTERPRETER $PYTHON_CREATE_NOJUMP --system_dir $WORK_DIR; then
    echo "Creating nojump failed! Exiting..."
    exit 1
fi

# These are all independent — run in parallel
(
    $PYTHON_INTERPRETER $PYTHON_MSD_ISF --axis xy --atom_names Li OW HW
    $PYTHON_INTERPRETER $PYTHON_MSD_ISF_PLOTTING --axis xy --atom_names Li OW HW
    $PYTHON_INTERPRETER $PYTHON_DIFF_KWW1E_PLOTTING --atom_names Li OW HW
) &

(
    $PYTHON_INTERPRETER $PYTHON_VHS --axis all --atom_names Li OW HW
    $PYTHON_INTERPRETER $PYTHON_VHS_PLOTTING
) &

(
    $PYTHON_INTERPRETER $PYTHON_VHS --axis x --atom_names Li OW HW
    ...
) &

# Statics can run alongside dynamics
(
    $PYTHON_INTERPRETER $PYTHON_ZDENSITY_PROFILE --atom_names Li OW HW
    $PYTHON_INTERPRETER $PYTHON_ZDENSITY_PROFILE_PLOTTING
)

# Sync all background jobs, then run RDF (which may need all results)
wait
bash $BASH_RDF_CN
```

---

### 3. MSD, ISF & Diffusion Analysis — `scripts/analysis/dynamics/calculate_msd_isf.py`

**What it does:** For each completed simulation, computes the mean squared displacement (MSD) and intermediate scattering function (ISF) for selected atom types. Extracts 2D diffusion coefficients from MSD and fits KWW stretched exponentials to ISF to obtain relaxation times (τ, β, mean τ).

**Why it matters:** This is the core scientific analysis — quantifying how fast atoms move (diffusion) and how quickly structural correlations decay (relaxation) under varying confinement and temperature. The multiprocessing design distributes each (atom type × analysis type) combination across CPU cores.

```python
def process_subtrajectory(sub_trjs, atm_names, collect_data_dir, meta_data):
    task_types = ["get_isf", "get_diffusion_2d"]
    runner_num = len(task_types) * len(atm_names)

    args_list = []
    for idx in range(len(sub_trjs)):
        task_meta_data = meta_data.copy()
        task_meta_data["atm_name"] = atm_names[idx]
        for task in task_types:
            args_list.append((sub_trjs[idx], collect_data_dir, task, task_meta_data))

    with Pool(processes=runner_num) as pool:
        pool.map(process_task, args_list)
```

The ISF fitting extracts physically meaningful quantities — the KWW stretched exponential parameters and mean relaxation time via the gamma function:

```python
# Fit KWW: F(q,t) = A * exp(-(t/τ)^β)
fit, cov = curve_fit(md.functions.kww, time[mask], isf[mask], maxfev=10000)
tau, beta = fit[1], fit[2]

# Mean relaxation time from KWW parameters
mean_relaxation_time = (tau / beta) * gamma(1 / beta)

# Time at which ISF = 1/e
t_1e = md.functions.kww_1e(fit[0], fit[1], fit[2])
```

---

### 4. Spatially Resolved Structural Analysis — `scripts/analysis/structural/calculating_rdf_CN.py`

**What it does:** Computes radial distribution functions (RDF) and coordination numbers between atom pairs — but with a key extension I designed: **spatial filtering** that restricts the analysis to atoms within a specified distance of the clay layer center. This required both new analysis logic and modifications to the underlying `mdevaluate` library.

**Why it matters:** In a layered clay system, the local structure near the clay surface is fundamentally different from the interlayer midplane. A bulk-averaged RDF would smear this out. The spatial resolution approach captures **layer-specific** structural information — essential for understanding confinement effects.

**What I designed from scratch:**

The `spatially_resolved_rdf()` function and the spatial filtering mechanism inside `coordination_number()` are my own work. They use the octahedral lithium (lio) positions to dynamically locate each clay layer at each frame, then select only atoms within a given z-distance:

```python
def coordination_number(frame_a, frame_b, lio_frame=None, layer_distance=None, ...):
    if lio_frame is not None:
        # Dynamically locate clay layers from octahedral Li positions
        center_pos1, center_pos2 = get_layers_centerpos(lio_frame)
        z_coords = frame_a[:, 2]
        mask = (np.abs(z_coords - center_pos1) < layer_distance) | \
               (np.abs(z_coords - center_pos2) < layer_distance)
        frame_a = frame_a[mask].copy()
    ...
```

The spatial approach also required a new time-averaging function. The original `mdevaluate` `time_average()` accepts two coordinate sets (atoms A and B). My `time_average_three_inputs()` extends this to accept a third frame — the reference layer positions (lio) needed to locate the clay sheets at each timestep:

```python
def time_average_three_inputs(function, coordinates, coordinates_b, coordinates_c, skip=0.1, segments=100):
    frame_indices = np.unique(np.int_(
        np.linspace(len(coordinates) * skip, len(coordinates) - 1, num=segments)
    ))
    result = [
        function(coordinates[i], coordinates_b[i], coordinates_c[i])
        for i in frame_indices
    ]
    # Filter out empty frames and average
    result = np.array(result)
    mask = np.all(result == np.zeros_like(result[0]), axis=tuple(range(1, result.ndim)))
    return np.mean(result[~mask], axis=0) if len(result[~mask]) > 0 else np.zeros_like(result[0])
```

**What I modified in `mdevaluate`:**

The original library functions (`rdf`, `next_neighbors`, `coordination_number`, `time_average`) assumed non-empty atom selections. When the spatial filter removes all atoms for a given frame (e.g., at very small slit sizes), the original code crashed. I modified the entire function chain to handle empty selections gracefully — returning zero arrays and filtering out empty frames before averaging.

The custom `rdf()` and `next_neighbors()` functions also handle both orthorhombic and non-orthorhombic periodic boundary conditions, using KDTree with built-in PBC for the fast path and explicit point replication as a fallback:

```python
def next_neighbors(atoms, query_atoms=None, distance_upper_bound=np.inf, ...):
    box = atoms.box
    if np.all(np.diag(np.diag(box)) == box):
        # Orthorhombic: fast KDTree with built-in PBC
        tree = KDTree(atoms % np.diag(box), boxsize=np.diag(box))
    else:
        # Non-orthorhombic: explicit PBC replication
        atoms_pbc, atoms_pbc_index = pbc_points(atoms, box, ...)
        tree = KDTree(atoms_pbc)
    ...
```

---

### 5. 2D Spatial Density Mapping — `scripts/analysis/structural/xy_density.py`

**What it does:** Computes time-averaged atom density maps in the xy-plane (parallel to the clay sheets), resolved into 0.2 Å z-slabs through the interlayer. This produces a full 3D density field showing exactly where Li, water oxygen, and water hydrogen prefer to sit within the confined space.

**Why it matters:** This is entirely my own analysis — not a wrapper around a library function. It answers a key question: do confined species form ordered structures parallel to the clay surface, and how does that ordering change with distance from the sheet? The output feeds directly into 2D heatmap visualizations that reveal adsorption sites, hydration shell structures, and preferential ion positions.

**The core algorithm:**

For each trajectory frame, the script dynamically detects the clay layer positions, then applies a wrapping trick to combine both interlayer spaces into one — doubling the sampling statistics:

```python
for idx_frame in range(sim_step_num):
    frame = sub_trj[idx_frame]

    # Detect lower layer position (fluctuates each frame)
    layer_pos = np.min(center_pos[idx_frame])

    # Fold periodic box so both interlayer spaces overlap
    frame[:, 2] = np.mod(frame[:, 2], (box[2] / 2))

    # Atoms below the layer center get wrapped on top → single combined water layer
    diff = frame[:, 2] - layer_pos
    mask_combine_water = diff <= 0
    frame[:, 2][mask_combine_water] += box[2] / 2
    frame[:, 2] = frame[:, 2] - layer_pos  # Re-center origin at clay surface

    # Bin into 3D histogram (z-slab × x × y)
    mask = frame[:, 2] <= binned_height
    data = frame[mask][:, [2, 0, 1]]
    count += np.histogramdd(data, bins=bins)[0].astype(int)
```

The supercell periodicity is exploited for the xy-binning — the full simulation box contains 8×4 repeating unit cells, so the density is folded back into a single unit cell to further improve statistics. Final normalization converts raw counts to a proper number density [atoms/Å³]:

```python
# Normalization: bin volume × number of frames × wrapping multiplier (8×4×2 periodic images)
wrapping_multiplier = 8 * 4 * 2
norm = wrapping_multiplier * bin_volume * sim_step_num
normalized_data = count.astype(float) / norm
```

---

### 6. Automated Visualization — `plotting/`

**What it does:** Generates publication-quality plots from the analysis output. Automatically adapts between single-temperature comparisons (multiple atom types on one plot) and temperature-sweep visualizations (colormap across temperatures).

**Why it matters:** With 100+ simulations producing thousands of data files, manual plotting is infeasible. These scripts auto-discover data, organize it by metadata (slit size, temperature, atom type), and produce consistent, thesis-ready figures.

Key design choices:

**Adaptive plot mode** — detects whether data spans one temperature or many, and switches layout accordingly:
```python
if len(data_dict[str_slitsize][str_atm_name]) == 1:
    # Single temperature: combined plot with multiple atom types
    # Li = red triangles, OW = blue circles, HW = black crosses
else:
    # Temperature sweep: inferno colormap with log-normalized temperature scale
    cmap = plt.get_cmap('inferno', 256)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'truncated_inferno', cmap(np.linspace(0, 0.85, 256))
    )
```

**Van Hove dual visualization** — side-by-side 2D line plot and 3D surface showing how the displacement distribution evolves with time, with axis-dependent labeling for 9 coordinate combinations (x, y, z, xy, xz, yz, all, normal, OH).

**Dual export** — all figures saved as both PDF (for the thesis) and `.agr` (Grace/xmgrace format, standard in computational physics groups) via `tudplot`.

---

## Simulation Pipeline (per parameter point)

Each (temperature, slit size) combination runs a four-stage GROMACS pipeline on the SLURM cluster (12 CPUs + 1 GPU):

| Stage | Directory | Purpose |
|-------|-----------|---------|
| Setup | `00_setup/` | Adjust box dimensions, solvate with SPC water, duplicate system in z, center |
| Energy minimization | `01_em/` | Relax bad contacts from solvation |
| Equilibration | `02_eq/` | NPT equilibration, then extract frame at mean density for optimal box size |
| Production | `03_prod/` | Production run — trajectory data collected here for analysis |

---

## Reflection

- **Starting point:** I inherited a 6,000-line Jupyter notebook where every simulation path, every plot, and every parameter was hardcoded — no loops, no functions, no abstraction. For each new simulation, the previous approach was to duplicate the notebook and manually edit paths. My first task was understanding the science; my second was recognizing that the tooling had to be rebuilt from the ground up.
- **Most valuable skill gained:** Designing reproducible HPC workflows. The template → sweep → dispatch → analyze chain means I can set up a new parameter study in minutes, not days.
- **Self-contained by design:** The duplicated utility functions across scripts (directory discovery, metadata extraction, data export) are a deliberate choice, not an oversight. My supervisor's group had limited programming experience, so I optimized for readability and independence over DRY principles — anyone should be able to open a single script, read it top-to-bottom, and understand what it does without navigating a package structure.
- **Biggest technical challenge:** The `mdevaluate` modifications for spatially resolved analysis. The original library wasn't designed for spatial filtering, so empty-frame edge cases cascaded through multiple function layers. Systematically tracing and fixing the entire call chain taught me a lot about defensive programming.
- **What I'd improve for my own future work:** In a context where I'm the long-term maintainer, I would refactor the shared utilities into a common module and add a YAML/TOML configuration layer for simulation parameters instead of hardcoded defaults.
