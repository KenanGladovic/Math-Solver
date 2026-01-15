# Kenan's IMO Math Calculator

Offline mathematical toolkit built with Python and Streamlit. This application provides step-by-step solutions for advanced optimization, linear algebra, and calculus problems based on the IMO curriculum.

## 🚀 Features

This application includes the following tools (located in the sidebar):

1.  **User Guide:** Explains syntax, common errors, and exam tips.
2.  **Latex Generator:** Generates copy-paste LaTeX code for matrices and equations.
3.  **KKT Optimization:** Generates conditions and verifies candidate points for optimality.
4.  **Subset Analysis:** Analyzes sets for Convexity, Closedness, Boundedness, and Compactness.
5.  **Plotting Tool:** Visualizes 1D functions, 2D contours (with constraints), and 3D surfaces.
6.  **Hessian Matrix:** Computes Gradients, Hessians, and classifies critical points (Min/Max/Saddle).
7.  **Symmetric Diagonalization:** Performs the $B^T A B = D$ reduction schematic procedure step-by-step.
8.  **Fourier Motzkin:** Eliminates variables from systems of linear inequalities.
9.  **Newtons Method:** Performs root finding (1D) and multivariable optimization iterations.
10. **Least Squares:** Fits Polynomials and Circles to data points using Normal Equations.
11. **Matrix Operations:** Computes Determinants, Inverses, Transposes, and Multiplication.
12. **Perceptron:** Visualizes the linear classification algorithm and weight updates.
13. **Calculus Solver:** Computes symbolic Derivatives (partial/total) and Limits.
14. **Equation Solver:** Solves symbolic algebraic equations and systems.
15. **Lagrange Interpolation:** Finds the exact polynomial passing through a specific set of points.
16. **Gradient Descent:** Visualizes the descent algorithm in 2D/3D with adjustable learning rates.
17. **Definiteness Checker:** Classifies matrices (Positive Definite, Indefinite, etc.)
18. **Support Vector Machines:** Solves for the optimal separating hyperplane.

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
