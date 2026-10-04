# qqqqqqqqqqqqQR-Stabilized Direct Genocchi Spectral Collocation Method

This repository contains the Python implementation used for the numerical experiments in the research article:

**A QR-Stabilized Direct Genocchi Spectral Collocation Method for Nonlinear Riesz Space-Fractional Reaction--Diffusion Equations**

## Description

The code implements a direct spectral collocation framework based on classical Genocchi polynomials for one- and two-dimensional Riesz space-fractional reaction--diffusion equations.

The implementation includes:

* classical Genocchi polynomial basis functions;
* analytical evaluation of the left and right Riemann--Liouville fractional derivatives of monomials;
* direct construction of the Riesz fractional operator;
* Chebyshev--Gauss--Lobatto collocation points;
* QR-based stabilization of the coefficient representation;
* Crank--Nicolson time discretization;
* Newton iteration with backtracking;
* one-dimensional and two-dimensional test problems;
* numerical error and CPU-time measurements;
* convergence and conditioning studies.

## Numerical Examples

The script contains four benchmark examples:

1. A one-dimensional nonlinear manufactured reaction--diffusion problem.
2. A two-dimensional linear Riesz diffusion benchmark.
3. A two-dimensional nonlinear Fisher-type Riesz reaction--diffusion benchmark.
4. A two-dimensional fractional Allen--Cahn benchmark.

The code also generates numerical data, figures, and LaTeX tables associated with the computational experiments.

## Requirements

The implementation requires Python 3 and the following packages:

* NumPy
* SciPy
* Matplotlib
* pandas

The numerical experiments were performed using IEEE double-precision arithmetic.

## Running the Code

Download or clone the repository and run:

```bash
python genocchi_riesz_solver.py
```

The program automatically creates an output directory named:

```text
genocchi_q1_cpu_results
```

This directory contains the numerical results, figures, and generated LaTeX tables.

## Reproducibility

The script is provided to facilitate reproduction of the numerical experiments reported in the associated research article.

The numerical parameters used in the experiments are specified directly in the Python source code.

## Author

**Mahmoud Lotfi**

Farhangian University, Tehran, Iran

## Citation

If you use this code in academic work, please cite the associated research article.

## License

The code is provided for academic and research use. Please refer to the repository license for the applicable terms.
