# Kenan's IMO Math Calculator

A powerful, offline mathematical toolkit built with Python and Streamlit. This application provides step-by-step solutions for advanced optimization, linear algebra, and calculus problems based on the IMO curriculum.

## 🚀 Features

This application uses a multi-page structure. Use the **Sidebar** to navigate between tools.

### 📚 Guides & Utilities
* **User Guide:** A complete guide on Python math syntax (`**` vs `^`, `log` vs `ln`), common errors, and exam strategies.
* **LaTeX Generator:** Instantly generates LaTeX code for Matrices, Systems of Equations, and Optimization Problems to copy-paste into your exam paper.

### 🧠 Analysis & Optimization
* **KKT Master Tool:** The ultimate exam tool. Generates symbolic KKT conditions ($\nabla L = 0$, etc.) and verifies if a specific candidate point is optimal.
* **Subset Analysis:** Analyzes sets defined by inequalities for Convexity, Closedness, Boundedness, and Compactness.
* **Newton's Method:**
    * *Root Finding:* Solves $f(x)=0$ (1D).
    * *Optimization:* Finds local minima for multivariable functions.
* **Gradient Descent:** 3D visualization of the descent algorithm with adjustable learning rates.
* **Fourier-Motzkin Elimination:** Solves systems of linear inequalities by eliminating variables step-by-step.

### 📐 Linear Algebra
* **Matrix Operations:** Computes Determinants, Inverses, Transposes, and performs Multiplication.
* **Symmetric Diagonalization:** Performs the $B^T A B = D$ reduction schematic procedure step-by-step.
* **Matrix Definiteness:** Classifies a matrix (PD, ND, Indefinite) using the curriculum method (Symmetric Reduction) to determine critical point types.
* **Hessian Analysis:** Calculates the Gradient and Hessian matrix of a function.

### 📉 Fitting & Regression
* **Least Squares:** Fits Polynomials and Circles to data points using Normal Equations ($A^T A x = A^T b$).
* **Lagrange Interpolation:** Finds the *exact* polynomial passing through points (Math behind Shamir's Secret Sharing).
* **Perceptron:** Visualizes the linear classification algorithm and finding separating hyperplanes.

### 🧮 General Math
* **Plotting Tool:** Visualizes 1D functions, 2D contours (with feasible regions), and 3D surfaces.
* **Calculus:** Computes Symbolic Derivatives and Limits.
* **Equation Solver:** Solves symbolic equations.

---

## 🛠️ Installation & Setup

You only need to do this once.

### 1. Install Python
Download Python from [python.org](https://www.python.org/downloads/).
> **IMPORTANT:** When installing, check the box **"Add Python to PATH"** at the bottom of the installer window.

### 2. Install Libraries
Open a terminal (Command Prompt) inside this folder and run:

```bash
pip install -r requirements.txt
(If pip is not recognized, try py -m pip install -r requirements.txt).
```

## 🛠️ Run program locally

### 1. Download
Create a folder and insert:
- app.py
- background.jpg
- requirements.txt
- utils.py
- pages (complete folder)
- Run_Calculator.bat

### 2. Run
Open Run_Calculator
