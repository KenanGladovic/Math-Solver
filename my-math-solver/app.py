import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# scipy is kept for the linear programming in existing modules if needed, 
# though curriculum specific methods are preferred.
from scipy.optimize import linprog 
import plotly.express as px
import plotly.graph_objects as go
# --- CONFIGURATION ---
st.set_page_config(page_title="Math Solver", layout="wide", page_icon="∫")

# --- CUSTOM CSS FOR LATEX & UI ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main-header {
        font-size: 2.5rem; 
        color: #4B4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
    .result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .proof-step {
        background-color: #f9f9f9;
        border-left: 4px solid #2196F3;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🔧 Library")
mode = st.sidebar.radio(
    "Choose Calculation Type:",
    [
        "Front page",
        "KKT Optimization", 
        "Subset Analysis",
        "Plotting Tool",
        "Hessian Matrix",
        "Newton's Method (Optimization)", # NEW
        "Symmetric Diagonalization (B^T A B)",
        "Fourier-Motzkin Elimination",
        "Least Squares Fitting (Normal Equations)",     # NEW
        "Perceptron Algorithm",                         # NEW
        "Matrix Operations", 
        "Calculus (Diff/Int)", 
        "Equation Solver"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Instructions:**\n"
    "1. Enter math expressions using Python syntax (e.g., `x**2` for $x^2$).\n"
    "2. The app converts them to LaTeX automatically.\n"
    "3. Hit **Solve** to see the results."
)

# --- HELPER FUNCTIONS ---
def parse_expr(input_str):
    """Safely parses a mathematical string into a sympy expression."""
    try:
        # standard variables
        x, y, z, t, a, b, c = sp.symbols('x y z t a b c')
        return sp.sympify(input_str, locals={'x': x, 'y': y, 'z': z, 't': t, 'a': a, 'b': b, 'c': c, 'pi': sp.pi, 'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp})
    except Exception as e:
        return None

def check_definiteness_curriculum(A: sp.Matrix, mat_input):
    """Checks matrix definiteness using curriculum methods (Quadratic Form / 2x2 Criterion)."""
    
    output = []
    
    # --- Basic Validation ---
    if A.rows != A.cols:
        output.append(sp.latex("\\text{Definiteness is only defined for square matrices.}"))
        return output

    if A != A.transpose():
        output.append(sp.latex("\\text{Matrix is NOT symmetric. Definiteness check requires a symmetric matrix.}"))
        return output

    n = A.rows
    
    # ----------------------------------------
    # 1. Quadratic Form (The Definition v^T A v)
    # ----------------------------------------
    v_vars = sp.symbols(f'x_{{1}}:{n+1}')
    v = sp.Matrix(n, 1, v_vars)
    qf = (v.transpose() * A * v)[0]
    
    output.append(sp.latex("\\mathbf{1. \\text{ Quadratic Form (Definition)}}"))
    output.append(sp.latex(f"v^{{T}} A v = {qf}"))

    # ----------------------------------------
    # 2. 2x2 Criterion (Exercise 3.42)
    # ----------------------------------------
    if n == 2:
        output.append(sp.latex("\\mathbf{2. } 2 \\times 2 \\text{ Criterion (Exercise 3.42)}}"))
        
        # Get symbolic representation of entries a, b, c (for c = a12)
        a_sym, b_sym, c_sym = A[0, 0], A[1, 1], A[0, 1]
        delta1 = a_sym
        delta2 = A.det()
        
        # Check if matrix contains defined symbols (like 'a', 'b', 'c')
        try:
            # Check for generic symbols like 'a', 'b'
            contains_symbol = len(qf.free_symbols) > n 
        except Exception:
            contains_symbol = False
            
        if contains_symbol:
            output.append(sp.latex("\\text{Criteria in terms of entries:}"))
            output.append(sp.latex(f"a = {a_sym}"))
            output.append(sp.latex(f"ab - c^2 = {delta2}"))
            output.append(sp.latex("\\text{The matrix is: Positive Definite if } a>0 \\text{ and } ab-c^2 > 0 \\text{.}"))
            output.append(sp.latex("\\text{The matrix is: Positive Semidefinite if } a \\geq 0 \\text{, } b \\geq 0 \\text{ and } ab-c^2 \\geq 0 \\text{.}"))
            output.append(sp.latex("\\text{The matrix is: Indefinite if } ab-c^2 < 0 \\text{.}"))

        else:
            # Numerical check using the criterion (must convert to floats/numbers)
            try:
                d1_val = float(delta1)
                d2_val = float(delta2)
                b_val = float(b_sym)
                
                if d1_val > 0 and d2_val > 0:
                    result = "Positive Definite (PD)"
                    condition = f"Criterion (Ex. 3.42): $a > 0$ ({d1_val} > 0) and $\\det(A) > 0$ ({d2_val} > 0)."
                elif d1_val >= 0 and d2_val >= 0 and b_val >= 0:
                    result = "Positive Semidefinite (PSD)"
                    condition = f"Criterion: $v^T A v \\geq 0$ verified via $a \\geq 0$, $b \\geq 0$, $\\det(A) \\geq 0$."
                elif d1_val < 0 and d2_val > 0:
                     result = "Negative Definite (ND)"
                     condition = f"Criterion: $a < 0$ ({d1_val} < 0) and $\\det(A) > 0$ ({d2_val} > 0)."
                elif d1_val <= 0 and d2_val >= 0 and b_val <= 0:
                    result = "Negative Semidefinite (NSD)"
                    condition = f"Criterion: $v^T A v \\leq 0$ verified via $a \\leq 0$, $b \\leq 0$, $\\det(A) \\geq 0$."
                elif d2_val < 0:
                    result = "Indefinite"
                    condition = f"Criterion: $\\det(A) < 0$ ({d2_val} < 0) implies Indefinite."
                else:
                    result = "Indeterminate/Borderline"
                    condition = "Requires checking the sign of the quadratic form directly, $v^T A v$."

                output.append(sp.latex(f"\\text{{Result: }} \\mathbf{{{result}}}"))
                output.append(sp.latex(f"\\text{{Condition: }} {condition}"))
            except:
                output.append(sp.latex("\\text{Numerical evaluation failed. Analyze the signs of } a \\text{ and } ab-c^2 \\text{ manually.}"))


    # ----------------------------------------
    # 3. General n x n Matrix (n > 2)
    # ----------------------------------------
    if n > 2:
        output.append(sp.latex("\\mathbf{3. \\text{ General Matrix (n > 2)}}"))
        output.append(sp.latex("\\text{The required method is to attempt Symmetric Reduction to a diagonal matrix } D = B^{T} A B \\text{ (Theorem 8.29).}"))
        output.append(sp.latex("\\text{The matrix is classified based on the signs of the entries in } D \\text{ (Exercise 8.28).}"))
        
    return output

def latex_print(prefix, expr):
    """Helper to display LaTeX nicely."""
    st.latex(f"{prefix} {sp.latex(expr)}")

# ==========================================
# 2D PLOTTING HELPER FUNCTIONS (Refactored)
# ==========================================

# --- HELPER 1: PLOTS CONTOUR LINES FOR THE OBJECTIVE FUNCTION ---
def plot_objective_contours(f_func, X, Y, ax, f_expr):
    """Draws contour lines for the 2D objective function."""
    Z = f_func(X, Y)
    # Plot Objective Contours (Level sets of f(x,y))
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_title(f"Contour Plot of ${sp.latex(f_expr)}$ with Feasible Region")

# --- HELPER 2: CALCULATES AND PLOTS FEASIBLE REGION (CONSTRAINTS) ---
def plot_constraints_region(const_str, x_sym, y_sym, X, Y, x_min, x_max, y_min, y_max, ax):
    """Parses constraints, computes feasible mask, and shades the region."""
    constraints = []
    lines = [l.strip() for l in const_str.split('\n') if l.strip()]
    
    for l in lines:
        if "<=" in l:
            lhs, rhs = l.split("<=")
            # g(x,y) <= 0
            expr = parse_expr(lhs) - parse_expr(rhs)
            if expr is not None:
                constraints.append(sp.lambdify((x_sym, y_sym), expr, 'numpy'))
        elif ">=" in l:
            lhs, rhs = l.split(">=")
            # g(x,y) >= 0 -> equivalent to -g(x,y) <= 0
            expr = parse_expr(rhs) - parse_expr(lhs)
            if expr is not None:
                constraints.append(sp.lambdify((x_sym, y_sym), expr, 'numpy'))
    
    feasible_mask = np.ones_like(X, dtype=bool)
    if constraints:
        # Check all constraints simultaneously
        for g_func in constraints:
            val = g_func(X, Y)
            if np.isscalar(val):
                if val > 0: feasible_mask[:] = False
            else:
                # Accumulate the mask for all constraints (intersection of feasible sets)
                feasible_mask &= (val <= 0)

        # Shade the feasible region (Green Area) 
        ax.imshow(feasible_mask, extent=[x_min, x_max, y_min, y_max], origin='lower', 
                  alpha=0.3, cmap='Greens', aspect='auto')
        
        # Plot boundary lines for constraints (Red Dashed Lines: where g(x,y) = 0)
        for g_func in constraints:
            g_val = g_func(X, Y)
            if not np.isscalar(g_val):
                ax.contour(X, Y, g_val, levels=[0], colors='red', linewidths=2, linestyles='dashed')

# --- MAIN COMBINED FUNCTION FOR 2D PLOTTING ---
def generate_2d_plot(func_str, const_str, x_min, x_max, y_min, y_max):
    """Main function to create and display the 2D contour and constraint plot."""
    x_sym, y_sym = sp.symbols('x y')
    f = parse_expr(func_str)
    
    if f is None:
        st.error("Invalid objective function. Check syntax or unsupported function/variable.")
        return
        
    f_func = sp.lambdify((x_sym, y_sym), f, 'numpy')
    
    # Create Meshgrid
    res = 100
    x_vals = np.linspace(x_min, x_max, res)
    y_vals = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # 1. Plot Objective
    plot_objective_contours(f_func, X, Y, ax, f)
    
    # 2. Plot Constraints
    plot_constraints_region(const_str, x_sym, y_sym, X, Y, x_min, x_max, y_min, y_max, ax)
    
    # Final plot settings
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    st.pyplot(fig)
    st.caption("**Legend:** Green shaded area = Feasible Region ($C$). Red dashed lines = Constraint Boundaries.")

# ==========================================
# 0. Front page (WELCOME PAGE)
# ==========================================
if mode == "Front page":
    # Using your custom CSS class 'main-header' for the big title
    st.markdown("<h1 class='main-header'>Welcome to Kenan's IMO calculator</h1>", unsafe_allow_html=True)
    
    # --- IMAGE SECTION (CENTERED) ---
    image_path = 'my-math-solver/background.jpg' 
    
    try:
        # Use columns to center a fixed-width image
        # Col ratio [1, 2, 1] means the center column (where the image is placed) 
        # is centered horizontally in the page.
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                image_path, 
                caption='', 
                width=600 # Set a specific width in pixels
            )
        
    except FileNotFoundError:
        st.error(f"Error: Image file not found at path: {image_path}. Please check the file name and location.")
    # --- END IMAGE SECTION ---

    # --- REST OF THE PAGE CONTENT (FLOWS NORMALLY BELOW THE IMAGE) ---
    
    # --- CHANGE 1: Version Tag ---
    st.markdown("""
    <div style='text-align: center; margin-bottom: 15px;'>
        <span style='background-color: #2196F3; color: white; padding: 4px 8px; border-radius: 15px; font-size: 0.9rem; font-weight: bold;'>
            v1.0
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # The small note centered below
    st.markdown("""
    <div style='text-align: center; color: gray; margin-top: 10px; font-style: italic;'>
        Made by my good friend
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 **Choose a tool from the menu on the left to get started.**")

    # --- CHANGE 2: Offline Guide Section ---
    st.markdown("<br>", unsafe_allow_html=True) # spacer
    with st.container():
        st.subheader("💻 Offline Access")
        st.warning(
            "**Coming Soon:** A complete guide on how to run this tool locally (offline) "
            "will be available soon. This will allow you to use the solver "
            "without an internet connection."
        )
    
# ==========================================
# 1. KKT Optimization
# ==========================================
if mode == "KKT Optimization":
    st.markdown("<h1 class='main-header'>KKT Optimization</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Define Objective")
        obj_input = st.text_input("Function to Minimize $f(x,y)$:", value="(x + y)**2 - x - y")
        
        st.subheader("2. Define Constraints ($g(x) \le 0$)")
        st.markdown("Enter inequalities. Use Python syntax (e.g., `x**2 + y**2 <= 4`).")
        
        num_constraints = st.number_input("Number of constraints", min_value=1, max_value=10, value=2)
        constraints_list = []
        for i in range(num_constraints):
            c_input = st.text_input(f"Constraint {i+1}:", value=f"Constraint {i+1} example" if i > 1 else ("x**2 + y**2 <= 4" if i==0 else "1 - x - y <= 0"))
            if c_input:
                constraints_list.append(c_input)

    with col2:
        st.subheader("3. Preview")
        f_expr = parse_expr(obj_input)
        
        if f_expr is not None:
            st.markdown("**Objective Function:**")
            st.latex(f"\\text{{Minimize }} f(x,y) = {sp.latex(f_expr)}")
            
            st.markdown("**Subject to:**")
            parsed_constraints = []
            
            # Parsing constraints logic
            valid_constraints = True
            for c_str in constraints_list:
                if "<=" in c_str:
                    lhs, rhs = c_str.split("<=")
                    g = parse_expr(lhs) - parse_expr(rhs)
                    parsed_constraints.append(g)
                    st.latex(f"{sp.latex(g)} \\le 0")
                elif ">=" in c_str:
                    lhs, rhs = c_str.split(">=")
                    # Convert g >= 0 to -g <= 0
                    g = parse_expr(rhs) - parse_expr(lhs)
                    parsed_constraints.append(g)
                    st.latex(f"{sp.latex(g)} \\le 0")
                else:
                    st.error(f"Invalid constraint format: {c_str}. Use `<=` or `>=`.")
                    valid_constraints = False
            
            if st.button("Generate KKT Conditions", type="primary"):
                if valid_constraints:
                    st.markdown("---")
                    st.subheader("📝 KKT Conditions")
                    st.caption("Based on **Example 9.31** format.")
                    
                    # Variables
                    vars_found = list(f_expr.free_symbols)
                    for g in parsed_constraints:
                        vars_found.extend(list(g.free_symbols))
                    vars_found = sorted(list(set(vars_found)), key=lambda v: v.name)
                    
                    # Lagrangian
                    lambdas = [sp.symbols(f'lambda_{i+1}') for i in range(len(parsed_constraints))]
                    L = f_expr + sum(lam * g for lam, g in zip(lambdas, parsed_constraints))
                    
                    st.markdown("#### 1. Lagrangian Function")
                    st.latex(f"\\mathcal{{L}} = {sp.latex(L)}")
                    
                    st.markdown("#### 2. Stationarity (Gradient $\\nabla \\mathcal{L} = 0$)")
                    for var in vars_found:
                        derivative = sp.diff(L, var)
                        st.latex(f"\\frac{{\\partial \\mathcal{{L}}}}{{\\partial {var.name}}} = {sp.latex(derivative)} = 0")
                        
                    st.markdown("#### 3. Primal Feasibility")
                    for g in parsed_constraints:
                        st.latex(f"{sp.latex(g)} \\le 0")
                        
                    st.markdown("#### 4. Complementary Slackness")
                    for i, (lam, g) in enumerate(zip(lambdas, parsed_constraints)):
                        st.latex(f"\\lambda_{{{i+1}}} \\cdot ({sp.latex(g)}) = 0")
                        
                    st.markdown("#### 5. Dual Feasibility")
                    lam_tex = ", ".join([f"\\lambda_{{{i+1}}}" for i in range(len(lambdas))])
                    st.latex(f"{lam_tex} \\ge 0")
                    
                    st.info("""
                    **Curriculum Note: Global Optimality**
                    If the objective function $f$ is convex and the feasible region is convex, 
                    any point satisfying these KKT conditions is a global minimum.
                    """)

# ==========================================
# 2. SUBSET ANALYSIS 
# ==========================================
elif mode == "Subset Analysis":
    st.markdown("<h1 class='main-header'>Subset Analysis & Proof Generator</h1>", unsafe_allow_html=True)
    st.info("Analyze properties of a subset $C$ (Convexity, Closedness, Boundedness, Compactness) with curriculum proofs.")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Define Subset C")
        st.write("Variables (e.g., x, y):")
        vars_input = st.text_input("Variables:", "x, y")
        
        st.write("Constraints defining C (one per line):")
        constraints_input = st.text_area(
            "Inequalities (e.g. x**2 + y**2 <= 4):", 
            "x**2 + y**2 <= 4\nx + y >= 1"
        )
    
    with col2:
        if st.button("Analyze & Generate Proof", type="primary"):
            try:
                # Parse Variables
                vars_sym = [sp.symbols(v.strip()) for v in vars_input.split(',')]
                
                # Parse Constraints
                raw_lines = [line.strip() for line in constraints_input.split('\n') if line.strip()]
                parsed_constraints = []
                
                for line in raw_lines:
                    if "<=" in line:
                        lhs, rhs = line.split("<=")
                        expr = parse_expr(lhs) - parse_expr(rhs)
                        parsed_constraints.append((expr, "<="))
                    elif ">=" in line:
                        lhs, rhs = line.split(">=")
                        # Convert g >= 0 to -g <= 0
                        expr = parse_expr(rhs) - parse_expr(lhs)
                        parsed_constraints.append((expr, "<="))
                    elif "=" in line: # Equality
                        lhs, rhs = line.split("=")
                        expr = parse_expr(lhs) - parse_expr(rhs)
                        parsed_constraints.append((expr, "="))
                    elif "<" in line:
                        lhs, rhs = line.split("<")
                        expr = parse_expr(lhs) - parse_expr(rhs)
                        parsed_constraints.append((expr, "<"))
                    elif ">" in line:
                        lhs, rhs = line.split(">")
                        expr = parse_expr(rhs) - parse_expr(lhs)
                        parsed_constraints.append((expr, "<"))

                # --- ANALYSIS LOGIC ---
                st.subheader("Formal Proof")
                
                # 1. CONVEXITY
                st.markdown("### 1. Convexity Analysis")
                st.markdown("<div class='proof-step'><b>Strategy:</b> Check Hessian of constraints (Theorem 8.23) and intersection properties (Exercise 4.23).</div>", unsafe_allow_html=True)
                
                all_convex = True
                for g, rel in parsed_constraints:
                    # Check Hessian of g
                    hessian = sp.hessian(g, vars_sym)
                    
                    # Simple check for convexity (Positive Semi-Definite Hessian)
                    try:
                        # Attempt to evaluate definiteness
                        # Linear check first
                        if hessian.is_zero_matrix:
                            st.write(f"Constraint ${sp.latex(g)} {rel} 0$ is linear.")
                            st.caption("Linear functions define convex sets (Hyperplanes/Half-spaces).")
                        else:
                            # Evaluate eigenvalues symbolically if possible or check diagonals
                            evals = hessian.eigenvals()
                            is_psd = True
                            for ev in evals:
                                if (ev.is_real and ev < 0) or (ev.is_number and ev < 0):
                                    is_psd = False
                            
                            if is_psd:
                                st.write(f"Function $g(x) = {sp.latex(g)}$ has a **Positive Semi-Definite Hessian**.")
                                st.latex(f"H_g = {sp.latex(hessian)}")
                                st.caption("By **Theorem 8.23**, $g$ is convex. By **Lemma 4.27**, the sublevel set is convex.")
                            else:
                                st.write(f"Constraint ${sp.latex(g)} {rel} 0$: Convexity check indeterminate (Hessian not clearly PSD).")
                                all_convex = False
                    except:
                        st.write(f"Constraint ${sp.latex(g)} {rel} 0$: Could not automatically verify convexity.")
                        all_convex = False

                if all_convex:
                    st.success("**Conclusion:** $C$ is the intersection of convex sets. By **Exercise 4.23**, $C$ is **CONVEX**.")
                else:
                    st.warning("**Conclusion:** Could not prove convexity for all constraints automatically.")

                # 2. CLOSEDNESS
                st.markdown("### 2. Closedness Analysis")
                st.markdown("<div class='proof-step'><b>Strategy:</b> Check inequality types using **Proposition 5.51** (Continuous Preimages).</div>", unsafe_allow_html=True)
                
                is_closed = True
                for g, rel in parsed_constraints:
                    if rel in ["<=", ">=", "="]:
                        st.write(f"Constraint ${sp.latex(g)} {rel} 0$: Defines a closed set.")
                        st.caption("Preimage of a closed interval under a continuous function is closed (**Prop 5.51**).")
                    else:
                        st.error(f"Constraint ${sp.latex(g)} {rel} 0$: Strict inequality usually defines an OPEN set.")
                        is_closed = False
                
                if is_closed:
                    st.success("**Conclusion:** $C$ is the intersection of closed sets. By **Proposition 5.39**, $C$ is **CLOSED**.")
                else:
                    st.error("**Conclusion:** $C$ is **NOT CLOSED** (contains strict inequalities).")

                # 3. BOUNDEDNESS
                st.markdown("### 3. Boundedness Analysis")
                st.markdown("<div class='proof-step'><b>Strategy:</b> Check for ball constraints $|x|^2 \le R^2$ (**Definition 5.29**).</div>", unsafe_allow_html=True)
                
                is_bounded = False
                squared_sum = sum(v**2 for v in vars_sym)
                
                for g, rel in parsed_constraints:
                    # Check if g looks like x^2 + y^2 - R
                    # We check if (g - squared_sum) is constant
                    diff = sp.simplify(g - squared_sum)
                    if diff.is_constant() and rel == "<=":
                        R_squared = -diff
                        if R_squared > 0:
                            st.write(f"Constraint ${sp.latex(g)} \le 0$ implies $|x|^2 \le {R_squared}$.")
                            st.caption(f"This subset is contained in a ball $B(0, \sqrt{{{R_squared}}})$.")
                            is_bounded = True
                
                if is_bounded:
                    st.success("**Conclusion:** $C$ is contained in a finite ball. $C$ is **BOUNDED**.")
                else:
                    st.warning("**Conclusion:** Could not explicitly find a bounding constraint (like $x^2+y^2 \le R$). Boundedness is Undetermined.")

                # 4. COMPACTNESS
                st.markdown("### 4. Compactness Analysis")
                st.markdown("<div class='proof-step'><b>Strategy:</b> Combine Closed and Bounded (**Definition 5.43**).</div>", unsafe_allow_html=True)
                
                if is_closed and is_bounded:
                    st.success("Since $C$ is both **CLOSED** and **BOUNDED**, by **Definition 5.43**, $C$ is **COMPACT**.")
                elif not is_closed:
                    st.error("Since $C$ is **NOT CLOSED**, it is **NOT COMPACT**.")
                else:
                    st.warning("Compactness is Indeterminate (Boundedness could not be proven automatically).")

            except Exception as e:
                st.error(f"Error during analysis: {e}")

# ==========================================
# 3. PLOTTING TOOL
# ==========================================
elif mode == "Plotting Tool":
    st.markdown("<h1 class='main-header'>Math Plotter & Sketcher</h1>", unsafe_allow_html=True)
    st.info("Visualizes 1D functions, 2D contours, and 3D surfaces (Interactive).")

    plot_type = st.radio(
        "Plot Type:", 
        ["1D Function (f(x))", "2D Function & Constraints", "3D Surface (Interactive)"], 
        horizontal=True
    )

    col1, col2 = st.columns([1, 1])
    
    # --- INPUT DEFINITION ---
    with col1:
        st.subheader("Setup")
        
        # Determine default values based on selected plot type
        if plot_type == "1D Function (f(x))":
            func_default = "x**2 - 2*x + 1"
            const_default = ""
        elif plot_type == "2D Function & Constraints":
            # Example 4.33 from the curriculum: Maximize x+y -> Minimize -x - y
            func_default = "-x - y"
            const_default = "2*x + y <= 1\nx + 2*y <= 1\nx >= 0\ny >= 0"
        else: # 3D Surface
            func_default = "(x + y)**2 - x - y"
            const_default = "" 
            
        func_str = st.text_input("Objective Function $f(x, y, t, ...)$:", value=func_default)
        
        if plot_type == "2D Function & Constraints":
            st.write("Constraints (for 2D shading):")
            const_str = st.text_area("Inequalities (one per line):", const_default)
        else:
            const_str = ""
        
        st.subheader("Ranges")
        c1, c2 = st.columns(2)
        x_min = c1.number_input("x min", value=-0.5 if plot_type == "2D Function & Constraints" else -3.0)
        x_max = c2.number_input("x max", value=1.5 if plot_type == "2D Function & Constraints" else 3.0)
        
        if plot_type != "1D Function (f(x))":
            y_min = c1.number_input("y min", value=-0.5 if plot_type == "2D Function & Constraints" else -3.0)
            y_max = c2.number_input("y max", value=1.5 if plot_type == "2D Function & Constraints" else 3.0)
        else:
            y_min, y_max = 0, 0 

    # --- PLOTTING LOGIC TRIGGER ---
    with col2:
        if st.button("Generate Plot", type="primary"):
            try:
                if plot_type == "1D Function (f(x))":
                    # --- 1D logic ---
                    f = parse_expr(func_str)
                    x_sym = sp.symbols('x')
                    if x_sym not in f.free_symbols and len(f.free_symbols) > 0:
                        st.error("Function must be in terms of 'x' for 1D plotting.")
                    else:
                        f_func = sp.lambdify(x_sym, f, 'numpy')
                        x_vals = np.linspace(x_min, x_max, 500)
                        y_vals = f_func(x_vals)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(x_vals, y_vals, label=f"f(x) = ${sp.latex(f)}$")
                        ax.set_xlabel('x')
                        ax.set_ylabel('f(x)')
                        ax.set_title(f"Plot of $f(x) = {sp.latex(f)}$")
                        ax.axhline(0, color='gray', linewidth=0.5)
                        ax.axvline(0, color='gray', linewidth=0.5)
                        ax.grid(True, linestyle='--', alpha=0.6)
                        ax.legend()
                        st.pyplot(fig)

                elif plot_type == "2D Function & Constraints":
                    generate_2d_plot(func_str, const_str, x_min, x_max, y_min, y_max)

                elif plot_type == "3D Surface (Interactive)":
                    # --- 3D logic ---
                    x_sym, y_sym = sp.symbols('x y')
                    f = parse_expr(func_str)
                    if f is None:
                        st.error("Invalid objective function.")
                        st.stop()
                    f_func = sp.lambdify((x_sym, y_sym), f, 'numpy')
                    
                    res = 100 
                    x_vals = np.linspace(x_min, x_max, res)
                    y_vals = np.linspace(y_min, y_max, res)
                    X, Y = np.meshgrid(x_vals, y_vals)
                    Z = f_func(X, Y)
                    
                    st.subheader("Interactive 3D Surface Plot")
                    
                    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

                    fig.update_layout(
                        scene=dict(
                            xaxis_title='X Axis',
                            yaxis_title='Y Axis',
                            zaxis_title='f(x,y)',
                            aspectratio=dict(x=1, y=1, z=0.7), 
                            camera_eye=dict(x=1.2, y=1.2, z=0.6)
                        ),
                        margin=dict(l=0, r=0, b=0, t=0), 
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Plotting Error: {e}")

# ==========================================
# 4. HESSIAN MATRIX
# ==========================================
elif mode == "Hessian Matrix":
    st.markdown("<h1 class='main-header'>Hessian Analysis</h1>", unsafe_allow_html=True)
    st.markdown("Analyze a function $f(x, y, ...)$ by calculating its Gradient and Hessian matrix.")

    col1, col2 = st.columns([1, 1])
    with col1:
        func_str = st.text_input("Function f(x, y, ...):", "x**3 + x*y + y**3")
        f = parse_expr(func_str)
        
    if f is not None:
        # Detect variables and sort them
        vars_sym = sorted(list(f.free_symbols), key=lambda s: s.name)
        
        with col2:
            st.write(f"Detected Variables: {', '.join([str(v) for v in vars_sym])}")
        
        if st.button("Calculate Hessian", type="primary"):
            st.markdown("---")
            
            # 1. Gradient
            grad = [sp.diff(f, v) for v in vars_sym]
            grad_matrix = sp.Matrix(grad)
            
            st.subheader("1. Gradient ($\\nabla f$)")
            st.caption("**Definition 7.6** (Gradient Vector)")
            st.latex(sp.latex(grad_matrix))
            
            # 2. Hessian
            hessian = sp.hessian(f, vars_sym)
            st.subheader("2. Hessian Matrix ($H_f$)")
            st.caption("**Definition 8.2** (Hessian Matrix)")
            st.latex(sp.latex(hessian))
            
            st.info("""
            **Curriculum Reference: Theorem 8.12 (Classification)**
            To classify a critical point $P$ where $\\nabla f(P) = 0$:
            * **Local Minimum:** $H_f(P)$ is Positive Definite.
            * **Local Maximum:** $H_f(P)$ is Negative Definite.
            * **Saddle Point:** $H_f(P)$ is Indefinite.
            * **Inconclusive:** $H_f(P)$ is Semi-Definite (Zero eigenvalue).
            
            **Curriculum Reference: Theorem 8.23 (Convexity)**
            * $f$ is **Convex** if $H_f$ is Positive Semi-Definite everywhere.
            * $f$ is **Strictly Convex** if $H_f$ is Positive Definite everywhere.
            """)
            
            # 3. Critical Points
            st.subheader("3. Critical Points ($\\nabla f = 0$)")
            st.caption("**Definition 7.17** (Critical Point)")
            try:
                crit_pts = sp.solve(grad, vars_sym, dict=True)
                if not crit_pts:
                    st.write("No critical points found analytically.")
                else:
                    for i, pt in enumerate(crit_pts):
                        pt_str = ", ".join([f"{v}: {pt[v]}" for v in vars_sym])
                        st.write(f"**Point {i+1}:** $({pt_str})$")
                        
                        # Show Hessian at this point
                        H_at_pt = hessian.subs(pt)
                        st.latex(f"H(P_{{{i+1}}}) = {sp.latex(H_at_pt)}")
            except Exception as e:
                st.error(f"Solver error: {e}")

# ==========================================
# 5. NEWTON'S METHOD
# ==========================================
elif mode == "Newton's Method (Optimization)":
    st.markdown("<h1 class='main-header'>Newton's Method for Optimization</h1>", unsafe_allow_html=True)
    st.info("Iteratively finds a critical point using the Hessian Inverse. (**Section 8.3, Eq 8.7**)")
    st.latex(r"v_{k+1} = v_k - [\nabla^2 F(v_k)]^{-1} \nabla F(v_k)")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        func_str = st.text_input("Objective Function $f(x, y)$:", "x**2 + 3*x*log(y) - y**3")
        start_point = st.text_input("Start Point (comma separated, e.g., 1, 1):", "1, 1")
        iterations = st.slider("Iterations:", 1, 10, 5)

    with col2:
        if st.button("Run Iterations", type="primary"):
            try:
                # Setup
                f = parse_expr(func_str)
                vars_sym = sorted(list(f.free_symbols), key=lambda s: s.name)
                
                # Symbolic Gradient and Hessian
                grad = sp.Matrix([sp.diff(f, v) for v in vars_sym])
                hessian = sp.hessian(f, vars_sym)
                
                # Initial values
                vals = [float(x.strip()) for x in start_point.split(',')]
                current_v = sp.Matrix(vals)
                
                st.subheader("Iteration Log")
                
                for k in range(iterations):
                    # Evaluate at current point
                    sub_dict = {v: val for v, val in zip(vars_sym, current_v)}
                    
                    # numeric evaluation
                    H_val = hessian.subs(sub_dict).evalf()
                    G_val = grad.subs(sub_dict).evalf()
                    
                    # Check invertibility
                    if H_val.det() == 0:
                        st.error(f"Iteration {k}: Hessian is singular. Cannot proceed.")
                        break
                        
                    # Newton Step: v_new = v_old - H_inv * G
                    # We cast to float to avoid sympy expression explosion
                    H_inv = H_val.inv()
                    step = H_inv * G_val
                    current_v = current_v - step
                    
                    # Display
                    v_formatted = [f"{float(val):.4f}" for val in current_v]
                    st.write(f"**Iter {k+1}:** $v_{{{k+1}}} \\approx {v_formatted}$")
                    
                st.success(f"Converged to approximately: {v_formatted}")
                
            except Exception as e:
                st.error(f"Computation Error: {e}")

# ==========================================
# 6. SYMMETRIC DIAGONALIZATION (B^T A B)
# ==========================================
elif mode == "Symmetric Diagonalization (B^T A B)":
    st.markdown("<h1 class='main-header'>Symmetric Matrix Diagonalization</h1>", unsafe_allow_html=True)
    st.markdown("Find an invertible matrix $B$ and a diagonal matrix $D$ such that:")
    st.latex("B^T A B = D")
    
    st.info("💡 **Method:** Symmetric row and column operations (**Sections 8.6 and 8.7**).")
    
    st.markdown("""
    **Curriculum Reference (Definiteness):**
    After finding $D$ (diagonal matrix):
    * **Positive Definite:** All diagonal entries $d_{ii} > 0$.
    * **Negative Definite:** All diagonal entries $d_{ii} < 0$.
    * **Indefinite:** Diagonal entries have mixed signs.
    """)

    # Template Selection
    template_size = st.selectbox("Select Template Size:", ["2x2", "3x3", "4x4"])
    
    default_text = "[[1, 2], [2, 8]]"
    if template_size == "3x3":
        default_text = "[[1, 2, 3], [2, 4, 5], [3, 5, 6]]"
    elif template_size == "4x4":
        default_text = "[[1, 0, 1, 0], [0, 2, 0, 1], [1, 0, 3, 0], [0, 1, 0, 4]]"

    mat_input = st.text_area("Input Symmetric Matrix A:", value=default_text, height=150)

    def diagonalize_symmetric(A_in):
        """
        Diagonalizes symmetric matrix A using symmetric row/col operations.
        Returns B and D such that B.T @ A @ B = D.
        Uses the [I // A] augmentation logic.
        """
        A = np.array(A_in, dtype=float)
        n = A.shape[0]
        
        # Validation
        if not np.allclose(A, A.T):
            return None, None, "Matrix is not symmetric!"
            
        # Augmented matrix [I // A]
        # Top n rows = B tracking (initially Identity)
        # Bottom n rows = A transforming to D
        aug = np.vstack([np.eye(n), A.copy()])
        
        for k in range(n):
            pivot = aug[n + k, k]
            
            # 1. Handle Zero Pivot (Swap or Add strategy)
            if np.isclose(pivot, 0):
                # Try to swap with a later diagonal element
                swap_idx = -1
                for j in range(k + 1, n):
                    if not np.isclose(aug[n + j, j], 0):
                        swap_idx = j
                        break
                
                if swap_idx != -1:
                    # Column Swap (Full Matrix)
                    aug[:, [k, swap_idx]] = aug[:, [swap_idx, k]]
                    # Row Swap (Bottom Only - symmetric update)
                    aug[[n + k, n + swap_idx], :] = aug[[n + swap_idx, n + k], :]
                    pivot = aug[n + k, k]
                else:
                    # Try to add a later column/row to create non-zero pivot
                    add_idx = -1
                    for j in range(k + 1, n):
                        if not np.isclose(aug[n + k, j], 0):
                            add_idx = j
                            break
                    
                    if add_idx != -1:
                        # Column Add (Full Matrix)
                        aug[:, k] += aug[:, add_idx]
                        # Row Add (Bottom Only)
                        aug[n + k, :] += aug[n + add_idx, :]
                        pivot = aug[n + k, k]

            if np.isclose(pivot, 0):
                continue

            # 2. Eliminate entries in row k and col k
            for j in range(k + 1, n):
                target = aug[n + k, j]
                if not np.isclose(target, 0):
                    m = target / pivot
                    
                    # Column Op: Cj = Cj - m * Ck (Full Matrix)
                    aug[:, j] -= m * aug[:, k]
                    
                    # Row Op: Rj = Rj - m * Rk (Bottom Only)
                    aug[n + j, :] -= m * aug[n + k, :]

        B = aug[:n, :]
        D = aug[n:, :]
        
        # Clean small floating point noise
        D[np.abs(D) < 1e-10] = 0
        B[np.abs(B) < 1e-10] = 0
        
        return B, D, None

    if st.button("Diagonalize", type="primary"):
        try:
            A_list = eval(mat_input)
            A_arr = np.array(A_list)
            
            B_res, D_res, err = diagonalize_symmetric(A_arr)
            
            if err:
                st.error(err)
            else:
                st.subheader("Results")
                
                # Show results in a clean grid
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Original Matrix A**")
                    st.latex(sp.latex(sp.Matrix(A_arr)))
                with col2:
                    st.markdown("**Transformation Matrix B**")
                    st.latex(sp.latex(sp.Matrix(B_res)))
                with col3:
                    st.markdown("**Diagonal Matrix D**")
                    st.latex(sp.latex(sp.Matrix(D_res)))
                
                st.subheader("Verification")
                LHS = B_res.T @ A_arr @ B_res
                LHS[np.abs(LHS) < 1e-10] = 0 # Clean
                
                st.write("Calculated $B^T A B$:")
                st.latex(sp.latex(sp.Matrix(LHS)))
                
                if np.allclose(LHS, D_res):
                    st.success("✅ Verification Successful: $B^T A B = D$")
                else:
                    st.error("❌ Verification Failed.")
                    
        except Exception as e:
            st.error(f"Error parsing matrix or calculation: {e}")

# ==========================================
# 7. FOURIER-MOTZKIN ELIMINATION
# ==========================================
elif mode == "Fourier-Motzkin Elimination":
    st.markdown("<h1 class='main-header'>Fourier-Motzkin Elimination</h1>", unsafe_allow_html=True)
    
    st.info("Solves a system of linear inequalities by eliminating variables one by one. (**Curriculum Section 4.5**)")
    st.markdown("""
    **Methodology:**
    1.  Rewrite inequalities as $L_i \\le x_k$ (Lower Bounds) and $x_k \\le U_j$ (Upper Bounds).
    2.  Eliminate $x_k$ by forming new inequalities $L_i \\le U_j$ for all pairs $(i, j)$.
    3.  Keep constraints that do not involve $x_k$.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input System")
        ineq_text = st.text_area(
            "Enter inequalities (one per line, e.g., 2*x + y <= 6):",
            "2*x + y <= 6\nx + 2*y <= 6\nx + 2*y >= 2\nx >= 0\ny >= 0\nz - x - y <= 0\n-z + x + y <= 0"
        )
        
        st.subheader("Elimination Settings")
        vars_input = st.text_input("Variables to Eliminate (comma separated, order matters):", "x, y")
        
    with col2:
        if st.button("Solve Elimination", type="primary"):
            try:
                # 1. Parse Inequalities
                raw_lines = [line.strip() for line in ineq_text.split('\n') if line.strip()]
                inequalities = []
                
                for line in raw_lines:
                    if "<=" in line:
                        lhs, rhs = line.split("<=")
                        inequalities.append(parse_expr(lhs) - parse_expr(rhs) <= 0)
                    elif ">=" in line:
                        lhs, rhs = line.split(">=")
                        # Convert >= to <= by flipping signs: LHS >= RHS -> RHS - LHS <= 0
                        inequalities.append(parse_expr(rhs) - parse_expr(lhs) <= 0)
                    elif "<" in line:
                        lhs, rhs = line.split("<")
                        inequalities.append(parse_expr(lhs) - parse_expr(rhs) < 0)
                    elif ">" in line:
                        lhs, rhs = line.split(">")
                        inequalities.append(parse_expr(rhs) - parse_expr(lhs) < 0)
                
                # 2. Parse Variables to Eliminate
                vars_to_elim = [sp.symbols(v.strip()) for v in vars_input.split(',')]
                
                st.subheader("Process")
                current_ineqs = inequalities
                
                for v in vars_to_elim:
                    st.markdown(f"**Eliminating variable: ${sp.latex(v)}$**")
                    
                    lower_bounds = []
                    upper_bounds = []
                    others = []
                    
                    for ineq in current_ineqs:
                        # Ensure we are working with expression <= 0
                        expr = ineq.lhs
                        
                        # Extract coefficient of v: expr = a*v + B <= 0
                        coeff = expr.coeff(v)
                        rest = expr - coeff * v
                        
                        if coeff == 0:
                            others.append(ineq)
                        elif coeff > 0:
                            # a*v + B <= 0  ->  v <= -B/a (Upper Bound)
                            upper_bounds.append(-rest / coeff)
                        elif coeff < 0:
                            # a*v + B <= 0 (a is neg) -> v >= -B/a (Lower Bound)
                            lower_bounds.append(-rest / coeff)
                    
                    # Generate new constraints: Lower <= Upper
                    new_constraints = []
                    for lb in lower_bounds:
                        for ub in upper_bounds:
                            new_constraints.append(lb <= ub)
                    
                    # Add constraints that didn't involve v
                    new_constraints.extend(others)
                    
                    # Simplify
                    simplified_constraints = []
                    for c in new_constraints:
                        simp = sp.simplify(c)
                        if simp == True: continue # Trivial 0 <= 5
                        if simp == False: 
                            st.error("System is Infeasible (e.g., 0 <= -1)")
                            current_ineqs = []
                            break
                        simplified_constraints.append(simp)
                    
                    current_ineqs = simplified_constraints
                    
                    # Show intermediate status
                    if current_ineqs:
                        st.write(f"Remaining constraints: {len(current_ineqs)}")
                    else:
                        st.warning("No constraints left (or system infeasible).")
                        break

                st.markdown("---")
                st.subheader("Resulting System")
                if not current_ineqs:
                    st.write("No constraints on remaining variables.")
                else:
                    for c in list(set(current_ineqs)): # Deduplicate
                        st.latex(sp.latex(c))
                        
            except Exception as e:
                st.error(f"Error in elimination: {e}")

# ==========================================
# 8. LEAST SQUARES (NORMAL EQUATIONS)
# ==========================================
elif mode == "Least Squares Fitting (Normal Equations)":
    st.markdown("<h1 class='main-header'>Least Squares (Normal Equations)</h1>", unsafe_allow_html=True)
    st.info("Finds the best fit by solving $(A^T A)x = A^T b$. (**Theorem 5.16**)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        data_input = st.text_area("Data Points (x, y) one per line:", "1, 2\n2, 3\n3, 5")
        degree = st.selectbox("Polynomial Degree:", [1, 2, 3])
        
    with col2:
        if st.button("Calculate Best Fit"):
            try:
                # Parse data
                points = []
                for line in data_input.split('\n'):
                    if ',' in line:
                        parts = line.split(',')
                        points.append((float(parts[0]), float(parts[1])))
                
                # Construct Matrix A (Vandermonde-like)
                # For y = a_0 + a_1*x + ... + a_n*x^n
                # A rows are [1, x, x^2...]
                
                A_rows = []
                b_vals = []
                
                for px, py in points:
                    row = [px**i for i in range(degree + 1)]
                    A_rows.append(row)
                    b_vals.append(py)
                
                A = sp.Matrix(A_rows)
                b = sp.Matrix(b_vals)
                
                st.subheader("1. Setup Matrices")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.latex(f"A = {sp.latex(A)}")
                with col_b:
                    st.latex(f"b = {sp.latex(b)}")
                
                # Compute Normal Equation terms
                ATA = A.T * A
                ATb = A.T * b
                
                st.subheader("2. Normal Equations ($(A^T A)x = A^T b$)")
                st.latex(f"{sp.latex(ATA)} \\mathbf{{x}} = {sp.latex(ATb)}")
                
                # Solve
                if ATA.det() == 0:
                    st.error("Matrix A^T A is singular. Infinite solutions or no solution.")
                else:
                    x_sol = ATA.inv() * ATb
                    
                    st.subheader("3. Solution")
                    params = [f"{float(val):.4f}" for val in x_sol]
                    
                    poly_latex = ""
                    for i, coef in enumerate(x_sol):
                        if i == 0: poly_latex += f"{float(coef):.3f}"
                        else: poly_latex += f" + {float(coef):.3f}x^{{{i}}}"
                        
                    st.latex(f"y = {poly_latex}")
                    
                    # Simple Plot
                    x_p = np.array([p[0] for p in points])
                    y_p = np.array([p[1] for p in points])
                    
                    # Generate curve
                    x_line = np.linspace(min(x_p)-1, max(x_p)+1, 100)
                    y_line = sum(float(x_sol[i]) * x_line**i for i in range(len(x_sol)))
                    
                    fig, ax = plt.subplots()
                    ax.scatter(x_p, y_p, color='red', label='Data')
                    ax.plot(x_line, y_line, color='blue', label='Least Squares Fit')
                    ax.legend()
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# 9. PERCEPTRON ALGORITHM
# ==========================================
elif mode == "Perceptron Algorithm":
    st.markdown("<h1 class='main-header'>Perceptron Learning Algorithm</h1>", unsafe_allow_html=True)
    st.info("Iteratively finds a separating hyperplane $\\alpha \\cdot v_i > 0$. (**Section 5.3.2**)")
    
    st.markdown("""
    **Algorithm:**
    1. Start with $\\alpha = 0$.
    2. If $\\exists v_i$ such that $\\alpha \\cdot v_i \\le 0$, update $\\alpha \\leftarrow \\alpha + v_i$.
    3. Repeat.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Data")
        st.write("Enter labeled points $(x, y, label)$. Label should be 1 or -1.")
        # Default example from Example 5.11 / 5.12 logic
        default_data = "0, 0, 1\n1, 1, 1\n1, -1, -1" 
        data_input = st.text_area("Points:", default_data)
        
    with col2:
        if st.button("Run Perceptron"):
            try:
                # Parse Data
                vectors = []
                for line in data_input.split('\n'):
                    parts = [float(p) for p in line.split(',')]
                    # Per curriculum eq (5.7), we transform data:
                    # \hat{v} = (label*x, label*y, label*1) for affine separation
                    label = parts[-1]
                    coords = parts[:-1]
                    # Augment with 1 for bias (Section 5.3.2 transforms 2D -> 3D)
                    augmented = np.array([label * c for c in coords] + [label])
                    vectors.append(augmented)
                
                # Algorithm
                alpha = np.zeros(len(vectors[0]))
                max_iter = 100
                history = []
                
                converged = False
                for k in range(max_iter):
                    misclassified = None
                    for v in vectors:
                        if np.dot(alpha, v) <= 0:
                            misclassified = v
                            break
                    
                    if misclassified is None:
                        converged = True
                        break
                        
                    # Update step
                    alpha = alpha + misclassified
                    history.append(f"Step {k+1}: Update with {misclassified} -> alpha = {alpha}")
                
                st.subheader("Trace")
                for step in history:
                    st.write(step)
                    
                if converged:
                    st.success("Converged!")
                    st.latex(f"\\alpha = {list(alpha)}")
                    
                    # Extract line eq: ax + by + c = 0
                    # alpha = [a, b, c]
                    a, b, c = alpha[0], alpha[1], alpha[2]
                    st.write(f"Separating Line: ${a}x + {b}y + {c} = 0$")
                else:
                    st.warning("Did not converge within iteration limit (Data might not be linearly separable).")
                    
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# 10. Matrix Operations (Final Revision - No Definiteness Check)
# ==========================================
elif mode == "Matrix Operations":
    st.markdown("<h1 class='main-header'>Matrix Operations </h1>", unsafe_allow_html=True)
    
    # Removed "Definiteness Check" from the options
    op = st.selectbox("Operation", ["Determinant", "Inverse", "Transpose", "Matrix Multiplication"], index=0)
    
    # --- 1-MATRIX OPERATIONS (Determinant, Inverse, Transpose) ---
    if op in ["Determinant", "Inverse", "Transpose"]:
        st.subheader(f"Matrix A Setup for {op}")
        
        # Template selection is simplified
        template_size = st.selectbox("Select Matrix Template Size:", ["2x2", "3x3 (IMO Jan 25 Q2)", "4x4"])
        if template_size == "2x2":
            default_text = "[[2, 1], [1, 2]]" # IMO Jan 25 Q2(a) example
        elif "3x3" in template_size:
            default_text = "[[1, 0, -1], [-2, 2, -1], [1, -1, 1]]" # Inverse example (IMO Jan 23 Q3(c))
        elif template_size == "4x4":
            default_text = "[[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]"

        st.write("Input Matrix (Python list of lists format, e.g., `[[1, 2], [3, 4]]`)")
        mat_input = st.text_area("Matrix A:", value=default_text, height=150)
        
        if st.button("Calculate", type="primary"):
            try:
                # CRITICAL FIX: Define symbols for evaluation when checking symbolic inputs
                a, b, c, x, y, z = sp.symbols('a b c x y z')
                
                # Safely evaluate the input string
                mat_list = eval(mat_input, {'__builtins__': None}, locals())
                A = sp.Matrix(mat_list)
                
                st.subheader("Input")
                st.latex(f"A = {sp.latex(A)}")
                
                st.subheader("Result")
                
                # Core Operations
                if op == "Determinant":
                    det = A.det()
                    st.latex(f"\\det(A) = {sp.latex(det)}")
                    
                elif op == "Inverse":
                    if A.rows != A.cols:
                        st.error("Matrix must be square to calculate the inverse.")
                    elif A.det() == 0:
                        st.error("Matrix is singular (Determinant is 0), cannot invert.")
                    else:
                        inv = A.inv()
                        st.latex(f"A^{{-1}} = {sp.latex(inv)}")
                        
                elif op == "Transpose":
                    trans = A.transpose()
                    st.latex(f"A^{{T}} = {sp.latex(trans)}")
                        
            except Exception as e:
                # Catches errors related to malformed matrix input
                st.error(f"Error parsing matrix: Check that every row has the same number of columns. Original error: {e}")

    # --- MATRIX MULTIPLICATION ---
    elif op == "Matrix Multiplication":
        st.subheader("Matrix A (m x n) $\\cdot$ Matrix B (n x r)")
        st.caption("Reference: **Section 3.3** (Matrix Multiplication)")
        
        col1, col2 = st.columns(2)
        with col1:
            mat_a_str = st.text_area("Matrix A:", "[[1, 2], [3, 4]]", height=150)
        with col2:
            mat_b_str = st.text_area("Matrix B:", "[[10], [20]]", height=150)
            
        if st.button("Multiply A $\\cdot$ B", type="primary"):
            try:
                A = sp.Matrix(eval(mat_a_str))
                B = sp.Matrix(eval(mat_b_str))
                
                if A.cols != B.rows:
                    st.error(f"Incompatible matrices: Matrix A has {A.cols} columns, but Matrix B has {B.rows} rows. A.cols must equal B.rows.")
                    st.stop()
                    
                res = A * B
                st.subheader("Result")
                st.latex(f"{sp.latex(A)} \\cdot {sp.latex(B)} = {sp.latex(res)}")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# 11. CALCULUS (Diff/Int)
# ==========================================
elif mode == "Calculus (Diff/Int)":
    st.markdown("<h1 class='main-header'>Calculus Assistant</h1>", unsafe_allow_html=True)
    
    calc_mode = st.radio("Operation", ["Derivative", "Limit"], horizontal=True)
    
    expr_input = st.text_input(
        "Expression (in terms of x):", 
        value="x**2 * sin(x)"
    )
    
    expr = parse_expr(expr_input)
    x = sp.symbols('x')
    
    if calc_mode == "Derivative":
        # 📝 CHANGE: Switched from st.selectbox back to st.text_input
        order_str = st.text_input(
            "Order of Derivative (Enter an integer):", 
            value="1" 
        )
        
        if st.button("Differentiate"):
            if expr is None:
                st.error("Invalid expression. Please check your syntax.")
                st.stop()
            
            try:
                # Validate input and convert to integer
                order = int(order_str)
                if order < 1:
                    st.error("Order must be a positive integer.")
                    st.stop()
                    
                res = sp.diff(expr, x, order)
                st.markdown("### Result")
                st.latex(f"\\frac{{d^{order}}}{{dx^{order}}} ({sp.latex(expr)}) = {sp.latex(res)}")
                
            except ValueError:
                st.error("Invalid input for order. Please enter an integer.")
                st.stop()
            except Exception as e:
                st.error(f"Error during differentiation: {e}")
            
    elif calc_mode == "Limit":
        target = st.text_input("Limit as x approaches:", "0")
        if st.button("Calculate Limit"):
            if expr is None:
                st.error("Invalid expression. Please check your syntax.")
                st.stop()
            else:
                try:
                    target_expr = parse_expr(target)
                    res = sp.limit(expr, x, target_expr)
                    st.markdown("### Result")
                    st.latex(f"\\lim_{{x \\to {sp.latex(target_expr)}}} ({sp.latex(expr)}) = {sp.latex(res)}")
                except Exception as e:
                    st.error(f"Error calculating limit: {e}")

# ==========================================
# 12. EQUATION SOLVER
# ==========================================
elif mode == "Equation Solver":
    st.markdown("<h1 class='main-header'>Equation Solver</h1>", unsafe_allow_html=True)
    
    eq_input = st.text_input("Enter Equation (use `==` for equality, e.g., `x**2 - 4 == 0`):", "x**2 - 5*x + 6")
    st.caption("Note: If you don't type `==`, it assumes `= 0`.")
    
    solve_for = st.text_input("Solve for variable:", "x")
    
    if st.button("Solve"):
        try:
            var_sym = sp.symbols(solve_for)
            if "==" in eq_input:
                lhs_str, rhs_str = eq_input.split("==")
                lhs = parse_expr(lhs_str)
                rhs = parse_expr(rhs_str)
                eq = sp.Eq(lhs, rhs)
            else:
                eq = parse_expr(eq_input)
            
            sol = sp.solve(eq, var_sym)
            
            st.subheader("Solutions")
            if len(sol) == 0:
                st.warning("No solutions found.")
            else:
                for s in sol:
                    st.latex(f"{solve_for} = {sp.latex(s)}")
        except Exception as e:
            st.error(f"Could not solve: {e}")