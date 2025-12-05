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

# --- CENTRALIZED DEFAULTS ---
# These keys MUST match the 'key' argument you use in the p_ wrappers later.
TOOL_DEFAULTS = {
    "KKT Optimization": {
        "kkt_obj": "(x + y)**2 - x - y",
        "kkt_constraints": "x**2 + y**2 <= 4\n1 - x - y <= 0",
        "kkt_candidate": "0.5, 0.5"
    },
    "KKT Candidate verification": {
        "kkt_obj": "x + 3*y",
        "kkt_constraints": "x**2 + 2*y**2 - 1 <= 0\nx + y - 1 <= 0\ny - x <= 0",
        "kkt_candidate": "-0.3015, -0.6396"  # Approx solution for Example 9.27
    },
    "Newton's Method (Optimization)": {
        "newton_mode": "Multivariable Optimization (Minimize f(v))",
        "newton_func": "x**2 + 3*x*log(y) - y**3",
        "newton_start": "1.0, 1.0",
        "newton_iter": 5,
        "newton_zoom": 4.0,
        "newton_root_func": "x**2 - 2",
        "newton_root_start": 1.0
    },
    "Symmetric Diagonalization (B^T A B)": {
        "sym_template": "2x2",
        "sym_matrix": "[[1, 2], [2, 8]]"
    },
    "Fourier-Motzkin Elimination": {
        "fm_ineqs": "2*x + y <= 6\nx + 2*y <= 6\nx + 2*y >= 2\nx >= 0\ny >= 0\nz - x - y <= 0\n-z + x + y <= 0"
    },
    "Least Squares Fitting (Normal Equations)": {
        "ls_type": "Polynomial Fit (y = a_0 + a_1 x + ...)",
        "ls_data": "1, 2\n2, 3\n3, 5\n4, 7",
        "ls_degree": 1
    },
    "Perceptron Algorithm": {
        "perc_preset": "Example 5.12 (Simple)",
        "perc_data": "0, 0, 1\n1, 1, 1\n1, -1, -1"
    },
    "Matrix Operations": {
        "mat_op": "Determinant",
        "mat_template": "2x2",
        "mat_a": "[[2, 1], [1, 2]]",
        "mat_b": "[[10], [20]]"
    },
    "Calculus (Diff/Int)": {
        "calc_mode": "Derivative",
        "calc_expr": "x**2 * sin(x)",
        "calc_order": "1",
        "calc_limit_target": "0"
    },
    "Equation Solver": {
        "eq_input": "x**2 - 5*x + 6",
        "eq_var": "x"
    },
    "Plotting Tool": {
        # Defaults handled nicely by code if missing
    },
    "Subset Analysis": {
        # Defaults handled nicely by code if missing
    },
    "Hessian Matrix": {
        # Defaults handled nicely by code if missing
    }
}

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .main-header { font-size: 2.5rem; color: #4B4B4B; text-align: center; margin-bottom: 1rem; }
    .block-container { padding-top: 2rem; }
    .proof-step { background-color: #f9f9f9; border-left: 4px solid #2196F3; padding: 10px; margin-bottom: 10px; }
    .result-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION & CONTROLS ---
st.sidebar.title("🔧 Library")
mode = st.sidebar.radio(
    "Choose Calculation Type:",
    [
        "Front page",
        "KKT Optimization",
        "KKT Candidate verification",
        "Subset Analysis",
        "Plotting Tool",
        "Hessian Matrix",
        "Newton's Method (Optimization)",
        "Symmetric Diagonalization (B^T A B)",
        "Fourier-Motzkin Elimination",
        "Least Squares Fitting (Normal Equations)",
        "Perceptron Algorithm",
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

def init_state(key, default):
    """
    Ensures a key exists in session state. 
    Only sets value if the key is MISSING (prevents overwriting user input).
    """
    if key not in st.session_state:
        st.session_state[key] = default

def parse_expr(input_str):
    """Safely parses a mathematical string into a sympy expression."""
    try:
        x, y, z, t, a, b, c = sp.symbols('x y z t a b c')
        return sp.sympify(input_str, locals={'x': x, 'y': y, 'z': z, 't': t, 'a': a, 'b': b, 'c': c, 'pi': sp.pi, 'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp})
    except: return None

# --- PERSISTENT WIDGET WRAPPERS ---
# These replace st.text_input, etc. They automatically initialize state
# and sync with the session_state variables.

def p_text_input(label, key, **kwargs):
    default = TOOL_DEFAULTS.get(mode, {}).get(key, "")
    init_state(key, default)
    
    # Callback: Whenever the widget changes, update the permanent storage
    def on_change(): 
        st.session_state[key] = st.session_state[f"w_{key}"]
    
    # The 'value' is pulled from storage. The 'key' is unique to the widget.
    return st.text_input(label, value=st.session_state[key], key=f"w_{key}", on_change=on_change, **kwargs)

def p_text_area(label, key, height=None, **kwargs):
    default = TOOL_DEFAULTS.get(mode, {}).get(key, "")
    init_state(key, default)
    def on_change(): 
        st.session_state[key] = st.session_state[f"w_{key}"]
    return st.text_area(label, value=st.session_state[key], key=f"w_{key}", on_change=on_change, height=height, **kwargs)

def p_number_input(label, key, **kwargs):
    default = TOOL_DEFAULTS.get(mode, {}).get(key, 0.0)
    init_state(key, default)
    def on_change(): 
        st.session_state[key] = st.session_state[f"w_{key}"]
    return st.number_input(label, value=st.session_state[key], key=f"w_{key}", on_change=on_change, **kwargs)

def p_slider(label, key, min_value, max_value, **kwargs):
    default = TOOL_DEFAULTS.get(mode, {}).get(key, min_value)
    init_state(key, default)
    def on_change(): 
        st.session_state[key] = st.session_state[f"w_{key}"]
    return st.slider(label, min_value=min_value, max_value=max_value, value=st.session_state[key], key=f"w_{key}", on_change=on_change, **kwargs)

def p_selectbox(label, options, key, **kwargs):
    default = TOOL_DEFAULTS.get(mode, {}).get(key, options[0])
    init_state(key, default)
    def on_change(): 
        st.session_state[key] = st.session_state[f"w_{key}"]
        if 'on_change' in kwargs: kwargs['on_change']()
            
    clean_kwargs = {k:v for k,v in kwargs.items() if k != 'on_change'}
    try: idx = options.index(st.session_state[key])
    except: idx = 0
    return st.selectbox(label, options, index=idx, key=f"w_{key}", on_change=on_change, **clean_kwargs)

def p_radio(label, options, key, **kwargs):
    default = TOOL_DEFAULTS.get(mode, {}).get(key, options[0])
    init_state(key, default)
    def on_change(): 
        st.session_state[key] = st.session_state[f"w_{key}"]
        if 'on_change' in kwargs: kwargs['on_change']()
    
    clean_kwargs = {k:v for k,v in kwargs.items() if k != 'on_change'}
    try: idx = options.index(st.session_state[key])
    except: idx = 0
    return st.radio(label, options, index=idx, key=f"w_{key}", on_change=on_change, **clean_kwargs)

# --- ANALYSIS HELPER (For Subset Analysis) ---
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
    
    # 1. Quadratic Form
    v_vars = sp.symbols(f'x_{{1}}:{n+1}')
    v = sp.Matrix(n, 1, v_vars)
    qf = (v.transpose() * A * v)[0]
    
    output.append(sp.latex("\\mathbf{1. \\text{ Quadratic Form (Definition)}}"))
    output.append(sp.latex(f"v^{{T}} A v = {qf}"))

    # 2. 2x2 Criterion
    if n == 2:
        output.append(sp.latex("\\mathbf{2. } 2 \\times 2 \\text{ Criterion (Exercise 3.42)}}"))
        
        a_sym, b_sym, c_sym = A[0, 0], A[1, 1], A[0, 1]
        delta1 = a_sym
        delta2 = A.det()
        
        # Check if matrix contains symbolic variables
        try:
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
                output.append(sp.latex("\\text{Numerical evaluation failed. Analyze manually.}"))

    # 3. General n x n Matrix
    if n > 2:
        output.append(sp.latex("\\mathbf{3. \\text{ General Matrix (n > 2)}}"))
        output.append(sp.latex("\\text{The required method is to attempt Symmetric Reduction to a diagonal matrix } D = B^{T} A B \\text{ (Theorem 8.29).}"))
        output.append(sp.latex("\\text{The matrix is classified based on the signs of the entries in } D \\text{ (Exercise 8.28).}"))
        
    return output

def latex_print(prefix, expr):
    st.latex(f"{prefix} {sp.latex(expr)}")

# ==========================================
# 2D PLOTTING HELPER FUNCTIONS
# ==========================================

def plot_objective_contours(f_func, X, Y, ax, f_expr):
    """Draws contour lines for the 2D objective function."""
    Z = f_func(X, Y)
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_title(f"Contour Plot of ${sp.latex(f_expr)}$ with Feasible Region")

def plot_constraints_region(const_str, x_sym, y_sym, X, Y, x_min, x_max, y_min, y_max, ax):
    """Parses constraints, computes feasible mask, and shades the region."""
    constraints = []
    lines = [l.strip() for l in const_str.split('\n') if l.strip()]
    
    for l in lines:
        if "<=" in l:
            lhs, rhs = l.split("<=")
            expr = parse_expr(lhs) - parse_expr(rhs)
            if expr is not None: constraints.append(sp.lambdify((x_sym, y_sym), expr, 'numpy'))
        elif ">=" in l:
            lhs, rhs = l.split(">=")
            expr = parse_expr(rhs) - parse_expr(lhs)
            if expr is not None: constraints.append(sp.lambdify((x_sym, y_sym), expr, 'numpy'))
    
    feasible_mask = np.ones_like(X, dtype=bool)
    if constraints:
        for g_func in constraints:
            val = g_func(X, Y)
            if np.isscalar(val):
                if val > 0: feasible_mask[:] = False
            else:
                feasible_mask &= (val <= 0)

        ax.imshow(feasible_mask, extent=[x_min, x_max, y_min, y_max], origin='lower', 
                  alpha=0.3, cmap='Greens', aspect='auto')
        
        for g_func in constraints:
            g_val = g_func(X, Y)
            if not np.isscalar(g_val):
                ax.contour(X, Y, g_val, levels=[0], colors='red', linewidths=2, linestyles='dashed')

def generate_2d_plot(func_str, const_str, x_min, x_max, y_min, y_max):
    """Main function to create and display the 2D contour and constraint plot."""
    x_sym, y_sym = sp.symbols('x y')
    f = parse_expr(func_str)
    
    if f is None:
        st.error("Invalid objective function.")
        return
        
    f_func = sp.lambdify((x_sym, y_sym), f, 'numpy')
    
    res = 100
    x_vals = np.linspace(x_min, x_max, res)
    y_vals = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plot_objective_contours(f_func, X, Y, ax, f)
    plot_constraints_region(const_str, x_sym, y_sym, X, Y, x_min, x_max, y_min, y_max, ax)
    
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
    image_path = "background.jpg" 
    
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
# 1. KKT Optimization (Updated to match Example 9.31)
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
            # Default values to match a solvable example
            def_val = "x**2 + y**2 <= 4" if i==0 else "1 - x - y <= 0"
            c_input = st.text_input(f"Constraint {i+1}:", value=def_val)
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
                    # Convert g >= 0 to -g <= 0 to match standard form g(x) <= 0
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
                    st.caption("Formatted according to **Definition 9.24** and **Example 9.31**.")
                    
                    # Variables setup
                    vars_found = list(f_expr.free_symbols)
                    for g in parsed_constraints:
                        vars_found.extend(list(g.free_symbols))
                    vars_found = sorted(list(set(vars_found)), key=lambda v: v.name)
                    
                    # Create Multipliers
                    lambdas = [sp.symbols(f'lambda_{i+1}') for i in range(len(parsed_constraints))]
                    
                    # 1. Dual Feasibility (Lambda >= 0)
                    st.markdown("#### 1. Dual Feasibility")
                    lam_tex = ", ".join([f"\\lambda_{{{i+1}}}" for i in range(len(lambdas))])
                    st.latex(f"{lam_tex} \\ge 0")

                    # 2. Primal Feasibility (g(x) <= 0)
                    st.markdown("#### 2. Primal Feasibility")
                    for g in parsed_constraints:
                        st.latex(f"{sp.latex(g)} \\le 0")

                    # 3. Complementary Slackness (lambda * g(x) = 0)
                    st.markdown("#### 3. Complementary Slackness")
                    for i, (lam, g) in enumerate(zip(lambdas, parsed_constraints)):
                        st.latex(f"\\lambda_{{{i+1}}} \\cdot ({sp.latex(g)}) = 0")

                    # 4. Stationarity (Gradient of Lagrangian = 0)
                    st.markdown("#### 4. Stationarity")
                    
                    # Construct Lagrangian L = f + sum(lambda * g)
                    L = f_expr + sum(lam * g for lam, g in zip(lambdas, parsed_constraints))
                    
                    # Calculate gradients
                    grad_eqs = []
                    for var in vars_found:
                        derivative = sp.diff(L, var)
                        st.latex(f"{sp.latex(derivative)} = 0")

                    st.info("""
                    **Solving Strategy (Section 9.5.1):**
                    Zoom in on the multipliers $\\lambda_i$. Test cases where $\\lambda_i = 0$ (inactive constraint) 
                    vs $\\lambda_i > 0$ (active constraint, implies $g_i(x) = 0$).
                    """)

# ==========================================
# 2. KKT Candidate verification
# ==========================================
elif mode == "KKT Candidate verification":
    st.markdown("<h1 class='main-header'>KKT Candidate verification</h1>", unsafe_allow_html=True)
    st.markdown("Reference: **Chapter 9.4 (The KKT Conditions)** and **Section 9.5 (Computing with KKT)**.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Problem Definition")
        obj_input = p_text_input("Minimize f(x,y):", key="kkt_obj")
        constraints_input = p_text_area("Constraints (g(x) <= 0):", key="kkt_constraints", height=100)
        
        st.markdown("---")
        st.subheader("2. Candidate Verification")
        st.markdown("Enter a point to check if it satisfies KKT.")
        candidate_str = p_text_input("Candidate Point (comma sep):", key="kkt_candidate")

    with col2:
        st.subheader("3. Analysis")
        f_expr = parse_expr(obj_input)
        
        if f_expr is not None:
            # Identify variables
            vars_sym = sorted(list(f_expr.free_symbols), key=lambda s: s.name)
            
            # Parse constraints
            raw_lines = [l.strip() for l in constraints_input.split('\n') if l.strip()]
            g_exprs = []
            for c_str in raw_lines:
                if "<=" in c_str:
                    lhs, rhs = c_str.split("<=")
                    g_exprs.append(parse_expr(lhs) - parse_expr(rhs))
                elif ">=" in c_str:
                    lhs, rhs = c_str.split(">=")
                    g_exprs.append(parse_expr(rhs) - parse_expr(lhs))
            
            # Display Problem
            st.latex(f"\\text{{Min }} f = {sp.latex(f_expr)}")
            if g_exprs:
                for i, g in enumerate(g_exprs):
                    st.latex(f"g_{{{i+1}}}: {sp.latex(g)} \\le 0")

            if st.button("Verify Candidate Point", type="primary"):
                try:
                    # Parse Candidate Point
                    vals = [float(x.strip()) for x in candidate_str.split(',')]
                    if len(vals) != len(vars_sym):
                        st.error(f"Error: Expected {len(vars_sym)} coordinates for variables {vars_sym}, got {len(vals)}.")
                    else:
                        point_map = dict(zip(vars_sym, vals))
                        
                        st.markdown("### Step 1: Primal Feasibility")
                        # Check g_i(x) <= 0
                        feasible = True
                        active_indices = []
                        
                        for i, g in enumerate(g_exprs):
                            val = float(g.subs(point_map))
                            # Using a small tolerance for floating point comparisons
                            if val > 1e-6:
                                st.markdown(f"<div class='error-box'>❌ Constraint {i+1} Violated: {val:.4f} > 0</div>", unsafe_allow_html=True)
                                feasible = False
                            else:
                                status = "Active" if abs(val) < 1e-6 else "Inactive"
                                icon = "⚠️" if abs(val) < 1e-6 else "✅"
                                st.write(f"{icon} $g_{{{i+1}}} = {val:.4f}$ ({status})")
                                if abs(val) < 1e-6:
                                    active_indices.append(i)

                        if not feasible:
                            st.error("Point is not Primal Feasible. Cannot be optimal.")
                        else:
                            st.success("Primal Feasibility: Holds")
                            
                            st.markdown("### Step 2: Stationarity & Dual Feasibility")
                            # Eq 9.24: Grad(f) + sum(lambda_i * Grad(g_i)) = 0
                            
                            # Calculate Gradients
                            grad_f = [sp.diff(f_expr, v) for v in vars_sym]
                            grad_f_val = [float(gf.subs(point_map)) for gf in grad_f]
                            
                            grad_gs_val = []
                            for g in g_exprs:
                                grad_g = [sp.diff(g, v) for v in vars_sym]
                                grad_gs_val.append([float(gg.subs(point_map)) for gg in grad_g])
                            
                            # Set up Linear System for Stationarity
                            # We only solve for lambda for ACTIVE constraints. 
                            # Inactive constraints MUST have lambda = 0 (Complementary Slackness)
                            
                            if not active_indices:
                                # No active constraints -> Gradient of f must be 0
                                if np.allclose(grad_f_val, 0, atol=1e-5):
                                    st.markdown("<div class='success-box'>✅ Unconstrained Critical Point found (Grad f = 0).</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<div class='error-box'>❌ Stationarity fails. Grad f = {grad_f_val} != 0</div>", unsafe_allow_html=True)
                            else:
                                # Solve: sum(lambda_i * grad_g_i) = -grad_f
                                # A * x = b
                                # A columns are gradients of active constraints
                                # b is negative gradient of f
                                
                                A = np.array([grad_gs_val[i] for i in active_indices]).T
                                b = -np.array(grad_f_val)
                                
                                # Solve using Least Squares (robust if overdetermined)
                                lambdas_active, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                                
                                # Reconstruct full lambda vector
                                all_lambdas = [0.0] * len(g_exprs)
                                for idx, val in zip(active_indices, lambdas_active):
                                    all_lambdas[idx] = val
                                    
                                # Verify Stationarity (check if solution actually solves the system)
                                reconstructed_grad = np.dot(A, lambdas_active)
                                error = np.linalg.norm(reconstructed_grad - b)
                                
                                st.write("Stationarity Equation:")
                                st.latex(r"\nabla f(v_0) + \sum \lambda_i \nabla g_i(v_0) = 0")
                                
                                if error > 1e-5:
                                    st.markdown(f"<div class='error-box'>❌ Stationarity Fails. Gradients are not linearly dependent. Residual: {error:.4f}</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown("<div class='success-box'>✅ Stationarity Holds.</div>", unsafe_allow_html=True)
                                    
                                    # Check Dual Feasibility (lambda >= 0)
                                    dual_feasible = True
                                    st.markdown("**Lagrange Multipliers:**")
                                    for i, lam in enumerate(all_lambdas):
                                        if lam < -1e-6:
                                            st.markdown(f"<div class='error-box'>❌ $\lambda_{{{i+1}}} = {lam:.4f}$ (Must be $\ge 0$)</div>", unsafe_allow_html=True)
                                            dual_feasible = False
                                        else:
                                            active_tag = "(Active)" if i in active_indices else "(Inactive, set to 0)"
                                            st.write(f"$\lambda_{{{i+1}}} = {lam:.4f}$ {active_tag}")
                                    
                                    if dual_feasible:
                                        st.markdown("<div class='success-box'>🎉 <b>CANDIDATE IS A KKT POINT</b></div>", unsafe_allow_html=True)
                                        st.info("Note: If the problem is convex, this is a global minimum (Theorem 9.34).")
                                    else:
                                        st.warning("Point satisfies stationarity but violates Dual Feasibility (negative multipliers).")

                except Exception as e:
                    st.error(f"Calculation Error: {e}")

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
# 5. NEWTON'S METHOD (ROOTS & OPTIMIZATION)
# ==========================================
elif mode == "Newton's Method (Optimization)":
    st.markdown("<h1 class='main-header'>Newton's Method Solver</h1>", unsafe_allow_html=True)
    
    # Selection between the two curriculum applications
    problem_type = st.radio(
        "Choose Application:", 
        ["1D Root Finding (Solve f(x) = 0)", "Multivariable Optimization (Minimize f(v))"],
        horizontal=True
    )
    
    st.markdown("---")

    # ==========================================
    # MODE A: 1D ROOT FINDING (Section 6.3.5)
    # ==========================================
    if problem_type == "1D Root Finding (Solve f(x) = 0)":
        st.info("Iteratively finds a root where $f(x) = 0$ using tangent lines. (**Section 6.3.5**)")
        st.latex(r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("1. Input")
            # Default matches your snippet
            f_str = st.text_input("Function f(x):", "x**2 - 2")
            df_str = st.text_input("Derivative f'(x) (Optional):", "")
            
            x0 = st.number_input("Start Value ($x_0$):", value=1.0, step=0.1)
            iterations = st.slider("Iterations:", 1, 10, 5)
            
            # Plot bounds
            st.write("**Plot Settings**")
            x_min_u = st.number_input("x min", value=-2.0)
            x_max_u = st.number_input("x max", value=3.0)

        with col2:
            if st.button("Run Newton-Raphson (1D)", type="primary"):
                try:
                    # Parse Function
                    x = sp.symbols('x')
                    f_expr = parse_expr(f_str)
                    
                    # Handle Derivative
                    if df_str.strip():
                        df_expr = parse_expr(df_str)
                    else:
                        df_expr = sp.diff(f_expr, x)
                        st.caption(f"Computed Derivative automatically: ${sp.latex(df_expr)}$")
                    
                    # Lambdify
                    f_num = sp.lambdify(x, f_expr, 'numpy')
                    df_num = sp.lambdify(x, df_expr, 'numpy')
                    
                    # Iteration Loop
                    x_curr = x0
                    history = [] # List of (x, f(x)) tuples
                    
                    # Store start
                    history.append((x_curr, f_num(x_curr)))
                    
                    st.subheader("Iteration Log")
                    
                    for i in range(iterations):
                        val = float(f_num(x_curr))
                        deriv = float(df_num(x_curr))
                        
                        if abs(deriv) < 1e-8:
                            st.error(f"Derivative near zero at x={x_curr:.4f}. Stopping.")
                            break
                            
                        # Newton Step
                        x_next = x_curr - val / deriv
                        
                        history.append((x_next, f_num(x_next)))
                        st.write(f"**Iter {i+1}:** $x = {x_next:.6f}$")
                        x_curr = x_next
                        
                    st.success(f"Approximated Root: **{x_curr:.6f}**")
                    
                    # --- PLOTTING (Replicating Sage Look) ---
                    fig, ax = plt.subplots(figsize=(8, 5))
                    
                    # 1. Generate Domain
                    # Ensure domain includes all history points + margins
                    all_x = [p[0] for p in history] + [x_min_u, x_max_u]
                    x_plot_min = min(all_x) - 0.5
                    x_plot_max = max(all_x) + 0.5
                    x_vals = np.linspace(x_plot_min, x_plot_max, 400)
                    y_vals = f_num(x_vals)
                    
                    # 2. Plot Function (Blue Curve)
                    ax.plot(x_vals, y_vals, 'b-', label=f'$f(x)$', linewidth=2)
                    
                    # 3. Plot Axis
                    ax.axhline(0, color='black', linewidth=1)
                    
                    # 4. Plot Steps (Red Dots & Tangent Lines)
                    # We plot the tangent lines to show the geometry (like the Sage @interact demos usually do)
                    hx = [p[0] for p in history]
                    hy = [p[1] for p in history]
                    
                    # Dots
                    ax.scatter(hx, hy, color='red', s=50, zorder=5, label='Newton Steps')
                    
                    # Visualizing the "drops" to the axis
                    for j in range(len(history)-1):
                        x_start, y_start = history[j]
                        x_end, y_end = history[j+1] # This is on the curve, but x_end is the root of tangent
                        
                        # Draw tangent line segment from (x_start, y_start) to (x_end, 0)
                        ax.plot([x_start, x_end], [y_start, 0], 'r--', alpha=0.4)
                        # Draw vertical line from (x_end, 0) to (x_end, f(x_end))
                        ax.plot([x_end, x_end], [0, f_num(x_end)], 'k:', alpha=0.3)

                    ax.set_title(f"Root Finding for ${sp.latex(f_expr)}$")
                    ax.set_xlabel("x")
                    ax.set_ylabel("f(x)")
                    ax.set_xlim(x_min_u, x_max_u)
                    # Auto-scale Y to keep plot readable
                    y_padding = max(abs(min(hy)), abs(max(hy))) * 1.5
                    ax.set_ylim(-y_padding, y_padding)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error: {e}")

    # ==========================================
    # MODE B: MULTIVARIABLE OPTIMIZATION (Section 8.3)
    # ==========================================
    else:
        # This is the existing (working) code for Optimization
        st.info("Iteratively finds a critical point (min/max/saddle) using the Hessian. (**Section 8.3**)")
        st.latex(r"v_{k+1} = v_k - [\nabla^2 F(v_k)]^{-1} \nabla F(v_k)")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("1. Problem Setup")
            func_default = "x**2 + 3*x*log(y) - y**3" 
            func_str = st.text_input("Objective Function $f(x, y)$:", value=func_default)
            start_point = st.text_input("Start Point $v_0$ (comma separated):", value="1.0, 1.0")
            iterations = st.slider("Iterations:", 1, 20, 5)

        with col2:
            st.subheader("2. Visualization")
            show_plot = st.checkbox("Show Contour Plot", value=True)
            zoom_level = st.slider("Zoom:", 1.0, 10.0, 4.0)

        if st.button("Run Newton (Optimization)", type="primary"):
            try:
                # --- MATHEMATICAL SETUP ---
                f = parse_expr(func_str)
                if f is None:
                    st.error("Could not parse function.")
                    st.stop()
                    
                vars_sym = sorted(list(f.free_symbols), key=lambda s: s.name)
                
                # Compute Gradient and Hessian
                grad = sp.Matrix([sp.diff(f, v) for v in vars_sym])
                hessian = sp.hessian(f, vars_sym)
                
                f_num = sp.lambdify(vars_sym, f, 'numpy')
                grad_num = sp.lambdify(vars_sym, grad, 'numpy')
                hess_num = sp.lambdify(vars_sym, hessian, 'numpy')

                # Parse Start Point
                current_vals = np.array([float(x.strip()) for x in start_point.split(',')])

                # --- ITERATION LOOP ---
                history = [current_vals.copy()]
                st.subheader("Iteration Log")
                
                for k in range(iterations):
                    val_grad = np.array(grad_num(*current_vals)).flatten().astype(float)
                    val_hess = np.array(hess_num(*current_vals)).astype(float)
                    
                    try:
                        H_inv = np.linalg.inv(val_hess)
                    except np.linalg.LinAlgError:
                        st.error(f"**Iteration {k}:** Hessian is singular. Stopping.")
                        break
                    
                    step = H_inv @ val_grad
                    current_vals = current_vals - step
                    history.append(current_vals.copy())
                    
                    st.write(f"**Iter {k+1}:** $v = {np.round(current_vals, 4)}$")

                # --- SIMPLIFIED PLOT (As requested previously) ---
                if show_plot and len(vars_sym) == 2:
                    history_arr = np.array(history)
                    x_center, y_center = np.mean(history_arr[:, 0]), np.mean(history_arr[:, 1])
                    span = max(np.ptp(history_arr[:, 0]), np.ptp(history_arr[:, 1]))
                    if span == 0: span = 1.0
                    margin = span * 0.5 + zoom_level
                    
                    x_vis = np.linspace(x_center - margin, x_center + margin, 100)
                    y_vis = np.linspace(y_center - margin, y_center + margin, 100)
                    X, Y = np.meshgrid(x_vis, y_vis)
                    Z = f_num(X, Y)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    cp = ax.contour(X, Y, Z, levels=15, cmap='Greys', alpha=0.5)
                    ax.plot(history_arr[:, 0], history_arr[:, 1], 'k--', linewidth=1.5, alpha=0.8)
                    ax.scatter(history_arr[0, 0], history_arr[0, 1], color='green', s=80, label='Start', zorder=5)
                    ax.scatter(history_arr[-1, 0], history_arr[-1, 1], color='red', s=120, marker='*', label='End', zorder=5)
                    ax.set_title(f"Minimization Path")
                    ax.legend()
                    ax.grid(False)
                    st.pyplot(fig)

                st.success(f"**Result:** Critical point at $v \\approx {np.round(current_vals, 5)}$")

            except Exception as e:
                st.error(f"An error occurred: {e}")

# ==========================================
# 6. SYMMETRIC DIAGONALIZATION (B^T A B)
# ==========================================
elif mode == "Symmetric Diagonalization (B^T A B)":
    st.markdown("<h1 class='main-header'>Symmetric Matrix Diagonalization</h1>", unsafe_allow_html=True)
    
    st.info("""
    **Curriculum Reference: Section 8.7 (Schematic Procedure)**
    This tool finds an invertible matrix $B$ and a diagonal matrix $D$ such that $B^T A B = D$.
    It uses the schematic procedure of transforming $\\begin{pmatrix} I \\\\ A \\end{pmatrix} \\to \\begin{pmatrix} B \\\\ D \\end{pmatrix}$.
    """)

    # Template Selection
    template_size = st.selectbox("Select Template Size:", ["2x2", "3x3", "4x4"])
    
    default_text = "[[1, 2], [2, 8]]"
    if template_size == "3x3":
        default_text = "[[1, 2, 3], [2, 8, 4], [3, 4, 16]]" # Example 8.30
    elif template_size == "4x4":
        default_text = "[[0, 0, 1, 1], [0, 0, 2, 3], [1, 2, 1, 4], [1, 3, 4, 0]]" # Example 8.31

    mat_input = st.text_area("Input Symmetric Matrix A:", value=default_text, height=150)

    # --- HELPER FUNCTIONS ---
    def clean_matrix(M):
        """Zero out elements close to machine precision for cleaner display."""
        M_clean = M.copy()
        M_clean[np.abs(M_clean) < 1e-10] = 0
        return M_clean

    def format_aug_latex(aug, n):
        """Formats the augmented matrix stack for the step-by-step log."""
        B_part = sp.Matrix(clean_matrix(aug[:n, :]))
        A_part = sp.Matrix(clean_matrix(aug[n:, :]))
        return r"\begin{pmatrix} " + sp.latex(B_part).replace(r"\left[", "").replace(r"\right]", "") + r" \\ \hline " + sp.latex(A_part).replace(r"\left[", "").replace(r"\right]", "") + r" \end{pmatrix}"

    def diagonalize_symmetric_algo(A_in):
        """Performs the diagonalization algorithm and records history."""
        A = np.array(A_in, dtype=float)
        n = A.shape[0]
        
        # Validation
        if not np.allclose(A, A.T):
            return None, None, [], "Error: Matrix is not symmetric!"
            
        history = []
        # Augmented matrix [I // A]
        aug = np.vstack([np.eye(n), A.copy()])
        
        history.append(("**Initial Setup:** Form Augmented Matrix", format_aug_latex(aug, n)))
        
        for k in range(n):
            pivot = aug[n + k, k]
            
            # --- 1. Pivot Handling (if zero) ---
            if np.isclose(pivot, 0):
                swap_idx = -1
                # Try swap
                for j in range(k + 1, n):
                    if not np.isclose(aug[n + j, j], 0):
                        swap_idx = j
                        break
                
                if swap_idx != -1:
                    desc = f"**Pivot Fix (Swap):** Swap col/row {k+1} $\\leftrightarrow$ {swap_idx+1}"
                    # Col Swap (Full)
                    aug[:, [k, swap_idx]] = aug[:, [swap_idx, k]]
                    # Row Swap (Bottom)
                    aug[[n + k, n + swap_idx], :] = aug[[n + swap_idx, n + k], :]
                    history.append((desc, format_aug_latex(aug, n)))
                    pivot = aug[n + k, k]
                else:
                    # Try add
                    add_idx = -1
                    for j in range(k + 1, n):
                        if not np.isclose(aug[n + k, j], 0):
                            add_idx = j
                            break
                    if add_idx != -1:
                        desc = f"**Pivot Fix (Add):** Col/Row {k+1} $\\leftarrow$ Col/Row {k+1} + Col/Row {add_idx+1}"
                        # Col Add (Full)
                        aug[:, k] += aug[:, add_idx]
                        # Row Add (Bottom)
                        aug[n + k, :] += aug[n + add_idx, :]
                        history.append((desc, format_aug_latex(aug, n)))
                        pivot = aug[n + k, k]

            if np.isclose(pivot, 0): continue

            # --- 2. Elimination ---
            for j in range(k + 1, n):
                target = aug[n + k, j]
                if not np.isclose(target, 0):
                    m = target / pivot
                    desc = f"**Eliminate $(A)_{{{k+1},{j+1}}}$:** $C_{j+1} - {m:.3g} C_{k+1}$ / $R_{j+1} - {m:.3g} R_{k+1}$"
                    
                    # Col Op (Full)
                    aug[:, j] -= m * aug[:, k]
                    # Row Op (Bottom)
                    aug[n + j, :] -= m * aug[n + k, :]
                    
                    history.append((desc, format_aug_latex(aug, n)))

        B = clean_matrix(aug[:n, :])
        D = clean_matrix(aug[n:, :])
        
        return B, D, history, None

    # --- MAIN EXECUTION ---
    if st.button("Diagonalize", type="primary"):
        try:
            A_list = eval(mat_input)
            A_arr = np.array(A_list)
            
            # Run Algorithm
            B_res, D_res, history, err = diagonalize_symmetric_algo(A_arr)
            
            if err:
                st.error(err)
            else:
                # ----------------------------------------
                # 1. SUMMARY RESULTS (A, B, D)
                # ----------------------------------------
                st.subheader("1. Results")
                
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown("**Original Matrix ($A$)**")
                    st.latex(sp.latex(sp.Matrix(A_arr)))
                    
                with c2:
                    st.markdown("**Transformation Matrix ($B$)**")
                    st.latex(sp.latex(sp.Matrix(B_res)))
                    
                with c3:
                    st.markdown("**Diagonal Matrix ($D$)**")
                    st.latex(sp.latex(sp.Matrix(D_res)))

                # ----------------------------------------
                # 2. VERIFICATION & CLASSIFICATION
                # ----------------------------------------
                st.subheader("2. Verification")
                
                # Compute actual B.T @ A @ B
                calculated_D = B_res.T @ A_arr @ B_res
                calculated_D = clean_matrix(calculated_D)
                
                c_ver1, c_ver2 = st.columns([2, 1])
                
                with c_ver1:
                    st.markdown("We check if $B^T A B$ equals $D$:")
                    st.latex(r"B^T A B = " + sp.latex(sp.Matrix(calculated_D)))
                    
                    if np.allclose(calculated_D, D_res):
                        st.success("✅ **Verified:** $B^T A B = D$")
                    else:
                        st.error("❌ **Verification Failed:** Result does not match D.")

                with c_ver2:
                    st.markdown("**Definiteness (from $D$)**")
                    diag = np.diag(D_res)
                    if all(d > 0 for d in diag):
                        st.info("Positive Definite (+ + +)")
                    elif all(d < 0 for d in diag):
                        st.info("Negative Definite (- - -)")
                    elif all(d >= 0 for d in diag):
                        st.info("Positive Semi-Definite (+ 0 +)")
                    elif all(d <= 0 for d in diag):
                        st.info("Negative Semi-Definite (- 0 -)")
                    else:
                        st.info("Indefinite (+ - +)")

                # ----------------------------------------
                # 3. STEP-BY-STEP LOG
                # ----------------------------------------
                st.markdown("---")
                st.subheader("3. Step-by-Step Computations")
                st.caption(f"Following **Schematic Procedure (Section 8.7)**. Top block tracks $B$, bottom block tracks $A \\to D$.")
                
                for step_num, (desc, latex) in enumerate(history):
                    with st.expander(f"Step {step_num}: {desc.split('**')[1] if '**' in desc else 'Setup'}", expanded=False):
                        st.markdown(desc)
                        st.latex(latex)

        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# 7. FOURIER-MOTZKIN ELIMINATION (Improved)
# ==========================================
elif mode == "Fourier-Motzkin Elimination":
    st.markdown("<h1 class='main-header'>Fourier-Motzkin Elimination</h1>", unsafe_allow_html=True)
    
    st.info("""
    **Curriculum Reference: Section 4.5**
    This method solves systems of linear inequalities by projecting the feasible region onto a lower-dimensional space.
    It is the primary method taught for solving Linear Optimization problems in **Chapter 4** (e.g., the Vitamin Pill Problem).
    """)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. System Setup")
        st.markdown("Enter inequalities (one per line). Use `<=`, `>=`, `<` or `>`.")
        
        # Default Example: (4.11) from the text, solving for z
        default_ineqs = "2*x + y <= 6\nx + 2*y <= 6\nx + 2*y >= 2\nx >= 0\ny >= 0\nz - x - y <= 0\n-z + x + y <= 0"
        
        ineq_text = st.text_area("System of Inequalities:", value=default_ineqs, height=200)
        
    with col2:
        st.subheader("2. Solver Settings")
        
        # Auto-detect variables from input
        dummy_expr = parse_expr("0") # Just to get access to parsing logic if needed later
        # We need to parse lines to find symbols
        detected_vars = set()
        for line in ineq_text.split('\n'):
            if line.strip():
                # Rough parse to find symbols
                for part in line.replace('<',' ').replace('>',' ').replace('=',' ').split():
                    try:
                        sym = parse_expr(part)
                        if sym: detected_vars.update(sym.free_symbols)
                    except: pass
        
        sorted_vars = sorted([s.name for s in detected_vars])
        
        if not sorted_vars:
            st.warning("No variables detected yet.")
            target_var = None
        else:
            st.markdown(f"**Detected Variables:** `{', '.join(sorted_vars)}`")
            
            # Key Change: Ask what to KEEP, not what to eliminate
            target_var = st.selectbox(
                "Which variable do you want to solve for (keep)?", 
                sorted_vars, 
                index=len(sorted_vars)-1
            )
            
            st.caption(f"The tool will automatically eliminate all other variables to find bounds for **${target_var}$**.")

    if st.button("Run Fourier-Motzkin", type="primary") and target_var:
        try:
            # 1. Parse Inequalities into Standard Form (Expr <= 0)
            raw_lines = [line.strip() for line in ineq_text.split('\n') if line.strip()]
            inequalities = []
            
            for line in raw_lines:
                if "<=" in line:
                    lhs, rhs = line.split("<=")
                    inequalities.append(parse_expr(lhs) - parse_expr(rhs))
                elif ">=" in line:
                    lhs, rhs = line.split(">=")
                    # Convert LHS >= RHS  -->  RHS - LHS <= 0
                    inequalities.append(parse_expr(rhs) - parse_expr(lhs))
                elif "<" in line:
                    lhs, rhs = line.split("<")
                    inequalities.append(parse_expr(lhs) - parse_expr(rhs)) # Treat strict as non-strict for bounding
                elif ">" in line:
                    lhs, rhs = line.split(">")
                    inequalities.append(parse_expr(rhs) - parse_expr(lhs))

            # 2. Determine Elimination Order
            # We must eliminate everything EXCEPT the target_var
            # The order generally doesn't matter for correctness, but can affect complexity.
            elimination_order = [v for v in sorted_vars if v != target_var]
            
            st.markdown("---")
            st.subheader("Step-by-Step Elimination")
            
            current_ineqs = inequalities
            
            for elim_var_name in elimination_order:
                elim_var = sp.symbols(elim_var_name)
                
                with st.expander(f"Eliminating variable: ${elim_var_name}$", expanded=True):
                    st.write(f"Current constraints: {len(current_ineqs)}")
                    
                    lower_bounds = [] # L_i <= x
                    upper_bounds = [] # x <= U_j
                    others = []       # Constraints not involving x
                    
                    for expr in current_ineqs:
                        # expr <= 0
                        # Decompose: a*x + rest <= 0
                        coeff = expr.coeff(elim_var)
                        rest = expr - coeff * elim_var
                        
                        if coeff == 0:
                            others.append(expr)
                        elif coeff > 0:
                            # a*x + rest <= 0  -->  x <= -rest/a (Upper Bound)
                            upper_bounds.append(-rest / coeff)
                        elif coeff < 0:
                            # a*x + rest <= 0 (a is neg) --> x >= -rest/a (Lower Bound)
                            lower_bounds.append(-rest / coeff)
                    
                    st.markdown(f"* **Lower Bounds ($L \\le {elim_var_name}$):** {len(lower_bounds)}")
                    if lower_bounds: st.latex(", ".join([sp.latex(b) for b in lower_bounds]) + f" \le {elim_var_name}")
                    
                    st.markdown(f"* **Upper Bounds (${elim_var_name} \\le U$):** {len(upper_bounds)}")
                    if upper_bounds: st.latex(f"{elim_var_name} \le " + ", ".join([sp.latex(b) for b in upper_bounds]))
                    
                    # Generate new constraints: L <= U  -->  L - U <= 0
                    new_constraints = []
                    
                    # Pair every Lower bound with every Upper bound (The "Explosion" mentioned in Section 4.5)
                    for lb in lower_bounds:
                        for ub in upper_bounds:
                            new_constraints.append(lb - ub)
                    
                    # Keep constraints that didn't involve the variable
                    new_constraints.extend(others)
                    
                    # Simplify and cleanup
                    simplified_constraints = []
                    for c in new_constraints:
                        simp = sp.simplify(c)
                        if simp == True or (simp.is_number and simp <= 0): 
                            continue # Trivial constraint like 0 <= 5
                        if simp == False or (simp.is_number and simp > 0):
                            st.error(f"**Contradiction Found!** Derived constraint ${sp.latex(simp)} \le 0$ is impossible.")
                            st.stop()
                        simplified_constraints.append(simp)
                    
                    # Remove duplicates
                    current_ineqs = list(set(simplified_constraints))
                    st.write(f"**Resulting constraints:** {len(current_ineqs)}")

            # 3. Final Results
            st.markdown("---")
            st.subheader(f"3. Final Bounds for ${target_var}$")
            st.caption(f"This corresponds to the inequalities derived in **(4.14)** before back-substitution.")
            
            # Separate into simple bounds for the target variable
            final_lower = []
            final_upper = []
            
            t_sym = sp.symbols(target_var)
            
            for expr in current_ineqs:
                # expr <= 0
                coeff = expr.coeff(t_sym)
                rest = expr - coeff * t_sym
                
                if coeff == 0:
                    # Check for residual contradictions
                    if rest > 0: st.error("System is Infeasible!")
                elif coeff > 0:
                    # x <= -rest/coeff
                    final_upper.append(-rest/coeff)
                elif coeff < 0:
                    # x >= -rest/coeff
                    final_lower.append(-rest/coeff)
            
            # Display nicely
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**Lower Bounds ($L \le z$)**")
                if final_lower:
                    # Find numerical max if possible
                    try:
                        nums = [float(val) for val in final_lower if val.is_number]
                        if nums: st.info(f"Tightest numerical lower bound: **{max(nums)}**")
                    except: pass
                    for val in final_lower: st.latex(f"{sp.latex(val)} \le {target_var}")
                else:
                    st.write("None ($-\infty$)")

            with col_r:
                st.markdown("**Upper Bounds ($z \le U$)**")
                if final_upper:
                    # Find numerical min if possible
                    try:
                        nums = [float(val) for val in final_upper if val.is_number]
                        if nums: st.info(f"Tightest numerical upper bound: **{min(nums)}**")
                    except: pass
                    for val in final_upper: st.latex(f"{target_var} \le {sp.latex(val)}")
                else:
                    st.write("None ($+\infty$)")

        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# 8. LEAST SQUARES METHOD - EXPLAINED
# ==========================================
elif mode == "Least Squares Method":
    st.markdown("<h1 class='main-header'>Least Squares Method</h1>", unsafe_allow_html=True)
    
    st.info("""
    **Curriculum Reference: Section 5.4 & Theorem 5.16**
    This tool finds the parameter vector $x$ that minimizes the error $|b - Ax|^2$.
    The optimal solution is found by solving the **Normal Equations**:
    """)
    st.latex(r"(A^T A)x = A^T b")
    
    # 1. Model Selection
    fit_type = st.radio(
        "Choose Model Type:", 
        ["Polynomial Fit (y = a_0 + a_1 x + ...)", "Circle Fit (Exercise 5.22)"],
        horizontal=True
    )

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Input Data")
        default_data = "1, 2\n2, 3\n3, 5\n4, 7" if "Polynomial" in fit_type else "0, 2\n0, 3\n2, 0\n3, 1"
        data_input = st.text_area("Points (x, y) one per line:", default_data, height=150)
        
        # --- EXPLANATION SECTION ---
        st.markdown("### 📘 How A and b are built")
        if "Polynomial" in fit_type:
            degree = st.slider("Polynomial Degree:", 1, 5, 1)
            st.markdown(f"""
            **Goal:** Fit $y = a_0 + a_1 x + \dots + a_n x^n$.
            
            We treat the coefficients $a_0, \dots, a_n$ as the unknowns. For each point $(x_i, y_i)$, we write one linear equation:
            $$a_0 \cdot 1 + a_1 \cdot x_i + \dots + a_n \cdot x_i^n = y_i$$
            
            **Matrix Form ($Ax=b$):**
            * **Matrix A:** The $i$-th row contains the powers of $x_i$: $[1, x_i, x_i^2, \dots]$
            * **Vector b:** The $i$-th entry is the observed value $y_i$.
            * **Unknowns x:** The vector $[a_0, a_1, \dots]^T$.
            """)
        else:
            st.markdown(r"""
            **Goal:** Fit $(x-a)^2 + (y-b)^2 = r^2$.
            
            This is non-linear in $a, b, r$. We use the trick from **Exercise 5.22**:
            Expand: $x^2 - 2ax + a^2 + y^2 - 2by + b^2 = r^2$
            Rearrange: $2ax + 2by + (r^2 - a^2 - b^2) = x^2 + y^2$
            
            Let $c = r^2 - a^2 - b^2$. Now it is linear in unknowns $a, b, c$:
            $$a(2x_i) + b(2y_i) + c(1) = x_i^2 + y_i^2$$
            
            **Matrix Form ($Ax=b$):**
            * **Matrix A:** Row $i$ is $[2x_i, 2y_i, 1]$.
            * **Vector b:** Entry $i$ is $x_i^2 + y_i^2$.
            * **Unknowns x:** $[a, b, c]^T$.
            """)

    with col2:
        if st.button("Compute Best Fit", type="primary"):
            try:
                # --- DATA PARSING ---
                points = []
                for line in data_input.split('\n'):
                    if ',' in line:
                        parts = line.split(',')
                        points.append((float(parts[0].strip()), float(parts[1].strip())))
                
                if not points:
                    st.error("No valid points found.")
                    st.stop()

                x_vals = np.array([p[0] for p in points])
                y_vals = np.array([p[1] for p in points])
                num_pts = len(points)

                # --- MATRIX CONSTRUCTION (The "A" and "b") ---
                if "Polynomial" in fit_type:
                    # Rows are [1, x, x^2, ...]
                    A_cols = [np.ones(num_pts)]
                    for d in range(1, degree + 1):
                        A_cols.append(x_vals ** d)
                    A_np = np.column_stack(A_cols)
                    b_np = y_vals
                    
                    param_names = [f"a_{i}" for i in range(degree + 1)]
                    
                else: # Circle Fit (Exercise 5.22)
                    # Matrix A rows are [2x_i, 2y_i, 1]
                    col_1 = 2 * x_vals
                    col_2 = 2 * y_vals
                    col_3 = np.ones(num_pts)
                    
                    A_np = np.column_stack([col_1, col_2, col_3])
                    # Vector b entries are x_i^2 + y_i^2
                    b_np = x_vals**2 + y_vals**2
                    
                    param_names = ["a (center x)", "b (center y)", "c (aux)"]

                # --- CALCULATIONS (Theorem 5.16) ---
                ATA = A_np.T @ A_np
                ATb = A_np.T @ b_np
                
                try:
                    x_sol = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    st.error("Matrix $A^T A$ is singular. Points might be collinear or insufficient.")
                    st.stop()

                # --- DISPLAY RESULTS ---
                st.subheader("2. The Normal Equations")
                
                c_a, c_b = st.columns(2)
                with c_a:
                    st.markdown("**Matrix $A^T A$**")
                    st.latex(sp.latex(sp.Matrix(np.round(ATA, 2))))
                with c_b:
                    st.markdown("**Vector $A^T b$**")
                    st.latex(sp.latex(sp.Matrix(np.round(ATb, 2))))

                st.subheader("3. Solution Parameters")
                # Create a nice dataframe for parameters
                df_params = pd.DataFrame([x_sol], columns=param_names)
                st.table(df_params)

                # --- INTERPRETATION & PLOTTING ---
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(x_vals, y_vals, color='red', label='Data Points', zorder=5)
                
                margin = 1.0
                x_range = np.linspace(min(x_vals)-margin, max(x_vals)+margin, 200)

                if "Polynomial" in fit_type:
                    poly_str = f"{x_sol[0]:.3f}"
                    y_plot = np.full_like(x_range, x_sol[0])
                    for i in range(1, len(x_sol)):
                        poly_str += f" + {x_sol[i]:.3f}x^{i}"
                        y_plot += x_sol[i] * (x_range ** i)
                    
                    st.success(f"**Best Fit Polynomial:** $y = {poly_str}$")
                    ax.plot(x_range, y_plot, 'b-', label=f'Poly Fit (Deg {degree})')
                    
                else: # Circle
                    a, b, c = x_sol
                    r_squared = c + a**2 + b**2
                    if r_squared < 0:
                        st.warning("Calculated radius squared is negative. No real circle fits these points well.")
                        r = 0
                    else:
                        r = np.sqrt(r_squared)
                        st.success(f"**Best Fit Circle:** Center $({a:.3f}, {b:.3f})$, Radius $r={r:.3f}$")
                        
                        circle = plt.Circle((a, b), r, color='blue', fill=False, label='Least Squares Circle')
                        ax.add_patch(circle)
                        ax.plot(a, b, 'b+', markersize=10, label='Center')
                        ax.set_aspect('equal')
                        ax.set_xlim(a - r - 1, a + r + 1)
                        ax.set_ylim(b - r - 1, b + r + 1)

                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend()
                ax.set_title("Least Squares Fit")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Computation Error: {e}")

# ==========================================
# 9. PERCEPTRON ALGORITHM (Visual & Explained)
# ==========================================
elif mode == "Perceptron Algorithm":
    st.markdown("<h1 class='main-header'>Perceptron Learning Algorithm</h1>", unsafe_allow_html=True)
    
    st.info("""
    **Curriculum Reference: Section 5.3.2**
    This algorithm finds a linear boundary (hyperplane) that separates data points into two classes (+1 and -1).
    It works by iteratively updating a normal vector $\\alpha$ until all points satisfy $\\alpha \\cdot \\hat{v}_i > 0$.
    """)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Input Data")
        
        # Presets based on Curriculum Examples
        preset = st.selectbox("Load Example:", ["Custom", "Example 5.12 (Simple)", "Exercise 5.9 (Impossible/XOR)", "Exercise 5.13"])
        
        if preset == "Example 5.12 (Simple)":
            default_data = "0, 0, 1\n1, 1, 1\n1, -1, -1"
        elif preset == "Exercise 5.9 (Impossible/XOR)":
            # Red(-1,1), Red(1,-1), Blue(-1,-1), Blue(1,1)
            default_data = "-1, 1, -1\n1, -1, -1\n-1, -1, 1\n1, 1, 1"
        elif preset == "Exercise 5.13":
            default_data = "0, 0, -1\n0, 1, 1\n1, 1, 1\n1, 0, 1"
        else:
            default_data = "1, 2, 1\n2, 1, -1"

        data_input = st.text_area("Points $(x, y, label)$:", default_data, height=150)
        st.caption("Format: `x, y, label` per line. Label must be `1` or `-1`.")
        
    with col2:
        st.subheader("2. Theory & Transformation")
        st.markdown(r"""
        **The "Augmented" Trick (Eq 5.7):**
        To handle the offset $c$ in $ax+by+c=0$, we transform 2D points into 3D vectors.
        For a point $(x, y)$ with label $\ell$:
        $$
        \hat{v} = (\ell x, \ell y, \ell)
        $$
        We then look for a vector $\alpha = (a, b, c)$ such that $\alpha \cdot \hat{v} > 0$ for all points.
        """)

    if st.button("Run Perceptron", type="primary"):
        try:
            # --- 1. PARSE DATA ---
            raw_points = [] # Stores (x, y, label) for plotting
            vectors = []    # Stores \hat{v} for calculation
            
            for line in data_input.split('\n'):
                if not line.strip(): continue
                parts = [float(p.strip()) for p in line.split(',')]
                if len(parts) != 3:
                    st.error(f"Invalid format in line: {line}")
                    st.stop()
                
                x, y, label = parts
                raw_points.append((x, y, label))
                
                # Apply Transformation (5.7)
                # v_hat = (label*x, label*y, label)
                v_hat = np.array([label * x, label * y, label])
                vectors.append(v_hat)
            
            # --- 2. THE ALGORITHM (Example 5.11 logic) ---
            # Initialize alpha = 0
            alpha = np.zeros(3) 
            max_iter = 1000
            history = []
            converged = False
            
            st.markdown("---")
            st.subheader("3. Execution Trace")
            
            # Create a dataframe to show the "Augmented Vectors"
            df_vecs = pd.DataFrame(vectors, columns=["lx", "ly", "l (bias)"])
            st.write("**Augmented Vectors ($\\hat{v}_i$):**")
            st.dataframe(df_vecs.T) # Transpose for space
            
            log_container = st.expander("Show Iteration Log", expanded=False)
            with log_container:
                for k in range(max_iter):
                    misclassified = None
                    mis_idx = -1
                    
                    # Find first misclassified point
                    for idx, v in enumerate(vectors):
                        if np.dot(alpha, v) <= 0:
                            misclassified = v
                            mis_idx = idx
                            break
                    
                    if misclassified is None:
                        converged = True
                        break
                        
                    # Update Step
                    old_alpha = alpha.copy()
                    alpha = alpha + misclassified
                    history.append(alpha.copy())
                    
                    st.write(f"**Step {k+1}:** Misclassified point {mis_idx+1} ({raw_points[mis_idx]}).")
                    st.latex(r"\alpha \leftarrow " + f"{np.round(old_alpha, 2)} + {np.round(misclassified, 2)} = {np.round(alpha, 2)}")

            # --- 3. RESULTS & VISUALIZATION ---
            if converged:
                st.success(f"**Converged in {len(history)} steps!**")
                
                a, b, c = alpha
                st.markdown(f"**Final Normal Vector:** $\\alpha = ({a}, {b}, {c})$")
                st.markdown(f"**Separating Line Equation:** ${a}x + {b}y + {c} = 0$")
                
                # --- PLOTTING ---
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # 1. Plot Points
                xs = [p[0] for p in raw_points]
                ys = [p[1] for p in raw_points]
                labels = [p[2] for p in raw_points]
                
                # Scatter with colors
                colors = ['blue' if l == 1 else 'red' for l in labels]
                ax.scatter(xs, ys, c=colors, s=100, edgecolors='k', zorder=5)
                
                # 2. Plot Separating Line: ax + by + c = 0  =>  y = (-c - ax) / b
                x_min, x_max = min(xs)-1, max(xs)+1
                y_min, y_max = min(ys)-1, max(ys)+1
                
                x_grid = np.linspace(x_min, x_max, 100)
                
                if abs(b) > 1e-5: # Non-vertical line
                    y_grid = (-c - a * x_grid) / b
                    ax.plot(x_grid, y_grid, 'g-', linewidth=2, label='Separating Line')
                    
                    # Optional: Shade the "Positive" side (where ax + by + c > 0)
                    # We pick a point far away in the normal direction
                    fill_y = y_max if b > 0 else y_min
                    ax.fill_between(x_grid, y_grid, fill_y, color='green', alpha=0.1)
                    
                else: # Vertical line x = -c/a
                    if abs(a) > 1e-5:
                        vert_x = -c / a
                        ax.axvline(vert_x, color='g', linewidth=2, label='Separating Line')
                        # Shade right or left
                        if a > 0:
                            ax.axvspan(vert_x, x_max, color='green', alpha=0.1)
                        else:
                            ax.axvspan(x_min, vert_x, color='green', alpha=0.1)
                
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.axhline(0, color='black', linewidth=0.5)
                ax.axvline(0, color='black', linewidth=0.5)
                ax.grid(True, linestyle=':')
                ax.set_title("Perceptron Classification")
                
                # Custom Legend
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Class +1'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Class -1'),
                    Line2D([0], [0], color='g', lw=2, label='Decision Boundary')
                ]
                ax.legend(handles=legend_elements)
                
                st.pyplot(fig)
                
            else:
                st.error("Algorithm did not converge within 1000 steps.")
                st.warning("""
                **Why did this happen?**
                The data is likely **not linearly separable**. 
                (See **Exercise 5.9** for an example of points that cannot be separated by a line).
                """)
                
                # Still plot the points to show WHY
                fig, ax = plt.subplots()
                xs = [p[0] for p in raw_points]
                ys = [p[1] for p in raw_points]
                labels = [p[2] for p in raw_points]
                colors = ['blue' if l == 1 else 'red' for l in labels]
                ax.scatter(xs, ys, c=colors, s=100)
                ax.set_title("Non-Separable Data")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# 10. Matrix Operations (Persistent)
# ==========================================
elif mode == "Matrix Operations":
    st.markdown("<h1 class='main-header'>Matrix Operations</h1>", unsafe_allow_html=True)
    
    # Operation Selector
    op = p_selectbox("Operation", ["Determinant", "Inverse", "Transpose", "Matrix Multiplication"], key="mat_op")
    
    # --- 1. UNARY OPERATIONS (Determinant, Inverse, Transpose) ---
    if op in ["Determinant", "Inverse", "Transpose"]:
        st.subheader(f"Matrix A Setup for {op}")
        
        # Callback to update default text based on template selection
        def update_mat_template():
            t = st.session_state.mat_template
            if t == "2x2": st.session_state.mat_a = "[[2, 1], [1, 2]]"
            elif "3x3" in t: st.session_state.mat_a = "[[1, 0, -1], [-2, 2, -1], [1, -1, 1]]"
            elif t == "4x4": st.session_state.mat_a = "[[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]"

        # Template Selector
        p_selectbox(
            "Select Matrix Template Size:", 
            ["2x2", "3x3 (IMO Jan 25 Q2)", "4x4"], 
            key="mat_template", 
            on_change=update_mat_template
        )

        st.write("Input Matrix (Python list of lists format)")
        mat_input = p_text_area("Matrix A:", key="mat_a", height=150)
        
        if st.button("Calculate", type="primary"):
            try:
                # Define symbols for evaluation context
                a, b, c, x, y, z = sp.symbols('a b c x y z')
                
                # Safely evaluate string to list
                mat_list = eval(mat_input, {'__builtins__': None}, locals())
                A = sp.Matrix(mat_list)
                
                st.subheader("Input")
                st.latex(f"A = {sp.latex(A)}")
                
                st.subheader("Result")
                
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
                st.error(f"Error parsing matrix: {e}")

    # --- 2. BINARY OPERATIONS (Matrix Multiplication) ---
    elif op == "Matrix Multiplication":
        st.subheader("Matrix A (m x n) $\\cdot$ Matrix B (n x r)")
        st.caption("Reference: **Section 3.3** (Matrix Multiplication)")
        
        col1, col2 = st.columns(2)
        with col1:
            mat_a_str = p_text_area("Matrix A:", key="mat_a", height=150)
        with col2:
            # Note: 'mat_b' is a new key added to defaults for this specific view
            mat_b_str = p_text_area("Matrix B:", key="mat_b", height=150)
            
        if st.button("Multiply A $\\cdot$ B", type="primary"):
            try:
                # Define symbols
                a, b, c, x, y, z = sp.symbols('a b c x y z')
                
                A = sp.Matrix(eval(mat_a_str, {'__builtins__': None}, locals()))
                B = sp.Matrix(eval(mat_b_str, {'__builtins__': None}, locals()))
                
                if A.cols != B.rows:
                    st.error(f"Dimension Mismatch: A has {A.cols} cols, B has {B.rows} rows.")
                    st.stop()
                    
                res = A * B
                st.subheader("Result")
                st.latex(f"{sp.latex(A)} \\cdot {sp.latex(B)} = {sp.latex(res)}")
                
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# 11. CALCULUS (Persistent)
# ==========================================
elif mode == "Calculus (Diff/Int)":
    st.markdown("<h1 class='main-header'>Calculus Assistant</h1>", unsafe_allow_html=True)
    
    # Operation Mode
    calc_mode = p_radio("Operation", ["Derivative", "Limit"], key="calc_mode", horizontal=True)
    
    # Main Expression Input
    expr_input = p_text_input("Expression (in terms of x):", key="calc_expr")
    
    expr = parse_expr(expr_input)
    x = sp.symbols('x')
    
    if calc_mode == "Derivative":
        order_str = p_text_input("Order of Derivative (Enter an integer):", key="calc_order")
        
        if st.button("Differentiate", type="primary"):
            if expr is None:
                st.error("Invalid expression.")
                st.stop()
            
            try:
                order = int(order_str)
                if order < 1: raise ValueError
                
                res = sp.diff(expr, x, order)
                st.markdown("### Result")
                st.latex(f"\\frac{{d^{order}}}{{dx^{order}}} ({sp.latex(expr)}) = {sp.latex(res)}")
                
            except ValueError:
                st.error("Order must be a positive integer.")
            except Exception as e:
                st.error(f"Error: {e}")
            
    elif calc_mode == "Limit":
        target = p_text_input("Limit as x approaches:", key="calc_limit_target")
        
        if st.button("Calculate Limit", type="primary"):
            if expr is None:
                st.error("Invalid expression.")
                st.stop()
                
            try:
                target_expr = parse_expr(target)
                res = sp.limit(expr, x, target_expr)
                st.markdown("### Result")
                st.latex(f"\\lim_{{x \\to {sp.latex(target_expr)}}} ({sp.latex(expr)}) = {sp.latex(res)}")
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# 12. EQUATION SOLVER (Persistent)
# ==========================================
elif mode == "Equation Solver":
    st.markdown("<h1 class='main-header'>Equation Solver</h1>", unsafe_allow_html=True)
    
    eq_input = p_text_input("Enter Equation (use `==` for equality, e.g. `x**2 - 4 == 0`):", key="eq_input")
    st.caption("Note: If you don't type `==`, it assumes `= 0`.")
    
    solve_for = p_text_input("Solve for variable:", key="eq_var")
    
    if st.button("Solve", type="primary"):
        try:
            var_sym = sp.symbols(solve_for)
            
            if "==" in eq_input:
                lhs_str, rhs_str = eq_input.split("==")
                lhs = parse_expr(lhs_str)
                rhs = parse_expr(rhs_str)
                eq = sp.Eq(lhs, rhs)
            else:
                eq = parse_expr(eq_input) # Assumes = 0
            
            sol = sp.solve(eq, var_sym)
            
            st.subheader("Solutions")
            if len(sol) == 0:
                st.warning("No exact solutions found.")
            else:
                for s in sol:
                    st.latex(f"{solve_for} = {sp.latex(s)}")
                    if s.is_number:
                        st.caption(f"Decimal approx: {float(s):.4f}")
                        
        except Exception as e:
            st.error(f"Could not solve: {e}")
            