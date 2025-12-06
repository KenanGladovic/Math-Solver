# Kenan's IMO Math Calculator

A powerful, offline mathematical toolkit built with Python and Streamlit. This application provides step-by-step solutions for advanced optimization, linear algebra, and calculus problems based on the IMO curriculum.

## 🚀 Features

This application includes the following tools (located in the sidebar):

1.  **KKT Optimization:** Solves minimization problems with inequality constraints.
2.  **KKT Verification:** Verifies if a specific candidate point satisfies KKT conditions.
3.  **Subset Analysis:** Analyzes sets for Convexity, Closedness, Boundedness, and Compactness.
4.  **Plotting Tool:** Visualizes 1D functions, 2D contours (with constraints), and 3D surfaces.
5.  **Hessian Matrix:** Computes Gradients, Hessians, and classifies critical points.
6.  **Symmetric Diagonalization:** Performs the $B^T A B = D$ reduction step-by-step.
7.  **Fourier-Motzkin:** Eliminates variables from systems of linear inequalities.
8.  **Newton's Method:** Roots finding (1D) and Multivariable Optimization.
9.  **Least Squares:** Fits Polynomials and Circles to data points using Normal Equations.
10. **Matrix Operations:** Determinant, Inverse, Transpose, and Multiplication.
11. **Perceptron:** Visualizes the linear classification algorithm step-by-step.
12. **Calculus:** Symbolic Derivatives and Limits.
13. **Equation Solver:** Solves symbolic equations.

## 🛠️ Installation & Setup

You only need to do this once.

### 1. Install Python
Download Python from [python.org](https://www.python.org/downloads/).
> **IMPORTANT:** When installing, check the box **"Add Python to PATH"** at the bottom of the installer window.

### 2. Install Libraries
Open a terminal (Command Prompt) in this folder and run:
```bash
pip install streamlit sympy numpy pandas matplotlib scipy plotly


### How to run manually
Run this command in a terminal:
python -m streamlit run Home.py
