# Project 02 — 2D Transient Heat Equation Solver

> A finite-difference solver for the 2D transient heat equation with implicit Euler time-stepping, Newton iteration for nonlinear radiation terms, and VTK output for visualization. Built as part of a Scientific Computing course (July 2024).

## Context & Objective

This project solves the two-dimensional transient heat equation:

**∂u/∂t = α·∇²u + f(u, x, y, t)**

where *f* includes a nonlinear radiation term (βu⁴) and a manufactured source function. The domain is a unit square [0,1]² with homogeneous Dirichlet boundary conditions (u = 0 on all edges).

The goal was to integrate C++, CMake, Docker, Bash, and GitLab CI/CD into a working numerical simulation — gaining hands-on experience with the full toolchain used in scientific computing.

- **Setting:** Tutorial "Programming Tools for Scientific Computing"
- **My role:** Sole developer
- **Date:** July 2024
- **Tech stack:** C++17, Eigen, VTK 9.3, CMake, Docker, Bash

---

## Project Structure

```
project-02-pde-solver/
├── src/
│   ├── main_script.cpp          # Entry point: grid setup, time loop, solver dispatch
│   └── functions.cpp            # Math functions, VTK I/O, L2 norm computation
├── include/
│   └── functions.h              # Header with function declarations
├── scripts/
│   ├── build_process.sh         # Automated CMake build pipeline
│   └── start_sim.sh             # Interactive launcher with configurable parameters
├── CMakeLists.txt               # Build configuration (Eigen + VTK)
├── Dockerfile                   # Full environment: Ubuntu + VTK from source + Eigen
└── README.md
```

---

## How It Works

The solver operates in two modes depending on the nonlinearity coefficient **β**:

### Linear mode (β = 0)
The implicit Euler discretization yields a linear system **Au^(k+1) = u^k + Δt·q** at each timestep. The system matrix A encodes a 5-point Laplacian stencil. It is solved directly using Eigen's column-pivoting Householder QR decomposition. At the final timestep, the numerical solution is compared against the known analytic solution and the L2 error norm is computed.

### Nonlinear mode (β ≠ 0)
When the radiation term βu⁴ is active, the implicit system becomes nonlinear. The solver applies **Newton's method** at each timestep: it assembles the Jacobian (including the 4βu³ derivative), solves the linearized correction J·δu = −g(u), and iterates until the correction norm falls below a user-specified tolerance.

### Dockerized Build Pipeline
The entire environment — including VTK compiled from source — is containerized via Docker. A single `docker build && docker run` reproduces the full simulation on any machine. Bash scripts handle CMake configuration, compilation, and interactive parameter input.

---

## Key Code Deep Dives

### 1. Implicit Linear System Assembly — `main_script.cpp`

**What it does:** At each timestep in linear mode, this constructs the system matrix A for the implicit Euler scheme. The matrix encodes the 5-point Laplacian stencil: each interior grid point is coupled to its four neighbors, with the diffusion coefficient scaled by Δt·α/Δx².

**Why it matters:** This is the backbone of the solver — correctly assembling the discretized PDE as a linear system. The indexing maps the 2D grid (i, j) to a 1D vector index k = i·Ny + j, which is a standard technique in finite-difference solvers.

```cpp
Eigen::MatrixXd linear_matrix(gridpoints_x * gridpoints_y, gridpoints_x * gridpoints_y);
linear_matrix.setZero();
double temp_c = delta_t * coef_alpha / (delta_x * delta_x);

for (int i = 0; i < gridpoints_x; i++) {
    for (int j = 0; j < gridpoints_y; j++) {
        int temp_k = i * gridpoints_y + j;

        // Diagonal: 1 + 4c (implicit Euler + 5-point stencil center)
        linear_matrix(temp_k, temp_k) = 1 + 4 * temp_c;

        // Off-diagonals: -c for each neighbor (east, west, north, south)
        if (j < gridpoints_y - 1)
            linear_matrix(temp_k, temp_k + 1) = -temp_c;
        if (j > 0)
            linear_matrix(temp_k, temp_k - 1) = -temp_c;
        if (i < gridpoints_x - 1)
            linear_matrix(temp_k, temp_k + gridpoints_x) = -temp_c;
        if (i > 0)
            linear_matrix(temp_k, temp_k - gridpoints_x) = -temp_c;
    }
}
```

---

### 2. Newton Iteration for the Nonlinear System — `main_script.cpp`

**What it does:** For β ≠ 0, Newton's method iteratively linearizes the nonlinear implicit system. At each iteration, the Jacobian matrix is assembled (including the 4βu³ derivative of the radiation term on the diagonal), the residual g(u) is computed, and the correction δu is solved from J·δu = −g(u).

**Why it matters:** This is the most algorithmically complex part of the solver. The Jacobian diagonal entry `1 + 4c + 4β·Δt·u³` couples the nonlinear physics directly into the linear algebra.

```cpp
for (int n = 0; n < 500; n++) {
    // Jacobian: Laplacian stencil + nonlinear derivative on diagonal
    jacobian_matrix(temp_k, temp_k) = 1 + 4 * temp_c
        + 4 * radi_coef_beta * delta_t
        * updatedMatrix(i,j) * updatedMatrix(i,j) * updatedMatrix(i,j);

    // Residual: how far current guess is from satisfying the discretized PDE
    g_equations(i * gridpoints_y + j) = (
        updatedMatrix(i, j) - old_step(i, j)
        - delta_t * alpha * laplacian_term
        - delta_t * source_func(...)
    );

    // Solve correction and update
    Eigen::VectorXd delta_u = jacobian_matrix.colPivHouseholderQr().solve(-g_equations);
    updatedMatrix(i, j) += delta_u(i * gridpoints_y + j);

    // Converged?
    if (delta_u.norm() < tolerance) break;
}
```
*(Simplified for readability — see `main_script.cpp` for the full implementation.)*

---

### 3. Method of Manufactured Solutions — `functions.cpp`

**What it does:** The source function is back-computed from a chosen analytic solution u(x,y,t) = sin²(πx)·sin²(πy)·cos²(πγt), so that the solver's output can be rigorously verified against the known answer.

**Why it matters:** This is a standard verification technique in scientific computing. By comparing the numerical result to the exact solution, we can compute L2 error norms and confirm the solver is converging correctly — or detect bugs.

```cpp
double analytic_solution(double x, double y, double t, double gamma) {
    return sin²(πx) * sin²(πy) * cos²(πγt);
}

// Source term derived by substituting the analytic solution into the PDE
double source_func(double x, double y, double t,
                   double u, double beta, double gamma, double alpha) {
    return (
        - time_derivative_of_analytic
        - alpha * laplacian_of_analytic
        - beta * u*u*u*u   // nonlinear radiation term
    );
}
```
*(Shown in simplified notation — see `functions.cpp` for the full trigonometric expressions.)*

---

### 4. VTK Output — `functions.cpp`

**What it does:** Exports the 2D solution grid as VTK structured points files at configurable intervals, enabling visualization in ParaView.

**Why it matters:** Bridges numerical computation with visualization. The output is timestep-indexed, making it straightforward to animate the solution evolving over time.

---

## Results

### Linear Case (β = 0)
The solver produces L2 error norms between 0.005 and 0.025 across different configurations of timesteps (1–100) and frequency coefficients γ. The errors decrease with finer temporal resolution, confirming the solver converges as expected.

### Nonlinear Case (β ≠ 0)
The Newton iteration converges but requires significantly more iterations than expected (~3000 vs. the theoretical ~50). Additionally, increasing β unexpectedly *decreases* the iteration count — the opposite of what one would predict for stronger nonlinearity. This suggests a subtle bug in the Jacobian assembly or residual computation that was not resolved within the project timeline.

---

## How to Run

```bash
# Build the Docker image (compiles VTK from source — takes a while)
docker build -t pde-solver .
docker run -it -v $(pwd)/results:/app/build/results pde-solver

# The interactive script will prompt for parameters (or use defaults)
```

### Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `gridpoints_x` | Nx | 16 | Grid resolution in x |
| `gridpoints_y` | Ny | 16 | Grid resolution in y |
| `timesteps` | Nt | 10 | Number of time steps |
| `write_per_t` | — | 1 | Output write interval |
| `radi_coef_beta` | β | 0.0 | Nonlinear radiation coefficient (0 = linear) |
| `freq_coef_gamma` | γ | 1 | Frequency in analytic solution |
| `coef_alpha` | α | 1 | Diffusion coefficient |
| `tolerance` | ε | 1e-6 | Newton convergence tolerance |

---

## Reflection

- **Linear solver works well** — L2 norms confirm correct convergence against the analytic solution.
- **Newton iteration has a known issue** — excessive iterations and counterintuitive β-dependency point to a bug I didn't resolve in time. Being transparent about this: I understand *what* should happen, even though the implementation has an unresolved issue.
- **What I'd improve:** Use sparse matrices (Eigen::SparseMatrix) instead of dense — the current O(N⁴) memory scaling limits the solver to small grids. I'd also implement a proper line search to improve Newton convergence robustness.
- **Key learning:** The full pipeline experience — C++ → CMake → Docker → VTK → Git — was more valuable than any single component. Understanding how these tools connect is what makes a project reproducible.
