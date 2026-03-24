# Project 02 — 2D Nonlinear PDE Solver

> A finite-difference solver for 2D reaction-diffusion equations with implicit time-stepping, Newton iteration for nonlinear terms, and VTK output for visualization.

## 🎯 Context & Objective

This project solves a 2D partial differential equation of the form:

**∂u/∂t = α·∇²u + f(u, x, y, t)**

where *f* includes a nonlinear radiation term (βu⁴) and a known source function. The domain is a unit square [0,1]² with homogeneous Dirichlet boundary conditions (u = 0 on all edges).

- **Setting:** Academic project (Numerical Methods / Scientific Computing)
- **My role:** Sole developer — designed the numerical scheme, implemented the solver, built the Docker-based build pipeline
- **Tech stack:** C++17, Eigen, VTK, CMake, Docker, Bash

---

## 🏗 Project Structure

```
project-02-pde-solver/
├── src/
│   ├── main_script.cpp          # Entry point: grid setup, time loop, solver dispatch
│   └── functions.cpp            # Math functions, VTK I/O, error computation
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

## ⚙️ How It Works

The solver supports two modes depending on the nonlinearity coefficient **β**:

### Linear mode (β = 0)
When there is no nonlinear term, the implicit Euler discretization yields a standard linear system **Au = b** at each timestep. This is assembled as a sparse-structure dense matrix and solved directly using Eigen's column-pivoting Householder QR decomposition.

### Nonlinear mode (β ≠ 0)
When the radiation term βu⁴ is active, the implicit system becomes nonlinear. The solver uses **Newton's method** at each timestep: it assembles the Jacobian matrix (including the 4βu³ derivative of the radiation term), solves the linearized correction, and iterates until the update norm falls below a user-specified tolerance.

At the final timestep, the solver compares the numerical result against a known **analytic solution** and computes the L2 error norm — useful for convergence studies and verification.

---

## 🔬 Key Code Deep Dives

### 1. Newton Iteration for the Nonlinear System — `main_script.cpp`

**What it does:** At each timestep, this loop solves the nonlinear implicit system by repeatedly linearizing around the current guess, solving the Jacobian system, and updating the solution until convergence.

**Why it matters:** This is the core numerical algorithm. The Jacobian diagonal includes the derivative of the βu⁴ radiation term (4βu³), which couples the nonlinearity into the implicit solve. Convergence is monitored via the L2 norm of the correction vector, with an early exit once the tolerance is met.

```cpp
// Newton method
for (int n = 0; n < 500; n++) {

    // Jacobian matrix — the diagonal includes the nonlinear derivative 4βu³
    Eigen::MatrixXd jacobian_matrix(gridpoints_x * gridpoints_y, gridpoints_x * gridpoints_y);
    jacobian_matrix.setZero();
    for (int i = 0; i < gridpoints_x; i++) {
        for (int j = 0; j < gridpoints_y; j++) {
            int temp_k = i * gridpoints_y + j;
            double temp_c = delta_t * coef_alpha / (delta_x * delta_x);
            jacobian_matrix(temp_k, temp_k) = 1 + 4 * temp_c
                + 4 * radi_coef_beta * delta_t
                * updatedMatrix(i, j) * updatedMatrix(i, j) * updatedMatrix(i, j);

            // Off-diagonal entries for the 5-point Laplacian stencil
            if (j < gridpoints_x - 1)
                jacobian_matrix(temp_k, temp_k + 1) = -temp_c;
            if (j > 0)
                jacobian_matrix(temp_k, temp_k - 1) = -temp_c;
            if (i < gridpoints_x - 1)
                jacobian_matrix(temp_k, temp_k + gridpoints_x) = -temp_c;
            if (i > 0)
                jacobian_matrix(temp_k, temp_k - gridpoints_x) = -temp_c;
        }
    }

    // Residual g(u) — measures how far the current guess is from satisfying the equation
    // Includes: time derivative, Laplacian stencil, and source term
    // ...

    // Solve J · δu = -g(u) and update: u ← u + δu
    Eigen::VectorXd solved_equations = jacobian_matrix.colPivHouseholderQr().solve(-g_equations);

    // Convergence check — stop when the correction is smaller than tolerance
    if (solved_equations.norm() < tolerance) {
        total_number_iterations += n;
        break;
    }
}
```

---

### 2. Source Function with Nonlinear Radiation — `functions.cpp`

**What it does:** Computes the manufactured source term f(u, x, y, t) that ensures the PDE has a known analytic solution. The nonlinear variant includes the −βu⁴ radiation term.

**Why it matters:** This is the method of manufactured solutions — by choosing a known solution and back-computing the required source term, we can rigorously verify the solver's correctness. The split between `source_func` (nonlinear) and `source_func_linear` (β = 0) keeps the linear path clean and avoids unnecessary computation.

```cpp
double source_func(double x_value, double y_value, double t_value,
                   double prev_value, double beta, double gamma, double alpha) {
    double temp_value = (
        // Time derivative of the analytic solution
        -2 * M_PI * sin²(πx) * cos²(πy) * sin(πγt) * cos(πγt)
        // -α · ∂²u/∂x² contribution
        -2 * alpha * π² * (cos²(πx) - sin²(πx)) * sin²(πy) * cos²(πγt)
        // -α · ∂²u/∂y² contribution
        -2 * alpha * π² * (cos²(πy) - sin²(πy)) * sin²(πx) * cos²(πγt)
        // Nonlinear radiation term
        - beta * prev_value * prev_value * prev_value * prev_value
    );
    return temp_value;
}
```
*(Shown in simplified notation for readability — see `functions.cpp` for the full implementation.)*

---

### 3. VTK Output for Visualization — `functions.cpp`

**What it does:** Exports the 2D grid solution as VTK structured points files at configurable intervals, enabling visualization in ParaView or similar tools.

**Why it matters:** Scientific computing results need to be visualized to be useful. This function maps the Eigen matrix to VTK's data model and writes timestep-indexed files, making it easy to create animations of the solution evolving over time.

```cpp
void write_to_file(const Eigen::MatrixXd& matrix, int timestep, int write_interval) {
    if (timestep % write_interval != 0) return;

    vtkSmartPointer<vtkStructuredPoints> structuredPoints =
        vtkSmartPointer<vtkStructuredPoints>::New();
    structuredPoints->SetDimensions(matrix.rows(), matrix.cols(), 1);

    vtkSmartPointer<vtkDoubleArray> dataArray = vtkSmartPointer<vtkDoubleArray>::New();
    dataArray->SetNumberOfTuples(matrix.rows() * matrix.cols());

    for (int i = 0; i < matrix.rows(); ++i)
        for (int j = 0; j < matrix.cols(); ++j)
            dataArray->SetValue(i * matrix.cols() + j, matrix(i, j));

    structuredPoints->GetPointData()->SetScalars(dataArray);

    vtkSmartPointer<vtkStructuredPointsWriter> writer =
        vtkSmartPointer<vtkStructuredPointsWriter>::New();
    std::string filename = "results/results_timestep_" + std::to_string(timestep) + ".vtk";
    writer->SetFileName(filename.c_str());
    writer->SetInputData(structuredPoints);
    writer->Write();
}
```

---

### 4. Dockerized Build Pipeline — `Dockerfile` + `scripts/`

**What it does:** The Dockerfile builds a complete environment from scratch — compiling VTK 9.3 from source, installing Eigen, and setting up the C++ toolchain. The Bash scripts automate the CMake build and launch the simulation with interactive parameter input.

**Why it matters:** This ensures full reproducibility. Anyone with Docker can build and run the simulation without manually installing VTK, Eigen, or configuring compiler paths. The interactive `start_sim.sh` script makes it easy to experiment with different grid sizes, timesteps, and physical parameters.

```bash
# One command to build and run the entire simulation:
docker build -t pde-solver .
docker run -it -v $(pwd)/results:/app/build/results pde-solver
```

---

## 📐 Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `gridpoints_x` | Nx | 16 | Grid resolution in x-direction |
| `gridpoints_y` | Ny | 16 | Grid resolution in y-direction |
| `timesteps` | Nt | 10 | Number of time steps |
| `write_per_t` | — | 1 | Write output every N timesteps |
| `radi_coef_beta` | β | 0.0 | Nonlinear radiation coefficient (0 = linear mode) |
| `freq_coef_gamma` | γ | 1 | Frequency coefficient in analytic solution |
| `coef_alpha` | α | 1 | Diffusion coefficient |
| `tolerance` | ε | 1e-6 | Newton iteration convergence tolerance |

---

## 💡 Reflection

- **Most challenging:** Getting the Jacobian assembly correct for the nonlinear term — a sign error in the 4βu³ derivative caused divergence that was tricky to debug.
- **What I'd improve:** Replace dense matrix storage with sparse matrices (Eigen::SparseMatrix) to scale beyond ~100×100 grids efficiently. The current dense approach works for demonstration but is O(N⁴) in memory.
- **Key learning:** The method of manufactured solutions is invaluable for verifying numerical code — without the analytic comparison, subtle discretization bugs would be nearly impossible to catch.
