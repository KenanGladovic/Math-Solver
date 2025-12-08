import streamlit as st
import sympy as sp
import numpy as np
import utils
from scipy.optimize import linprog

st.set_page_config(layout="wide")
utils.setup_page()

st.markdown("<h1 class='main-header'>Subset Analysis & Proof Steps</h1>", unsafe_allow_html=True)
st.info("Analyze subset $C$ for Convexity, Closedness, Boundedness, and Compactness based on curriculum definitions.")

# --- SIDEBAR / EXAMPLES ---
# Define preset examples from exams
examples = {
    "Custom": {
        "vars": "x, y",
        "const": "x**2 + y**2 <= 4\nx + y >= 1"
    },
    "IMO Jan 2022 Q1c (Linear 3D)": {
        "vars": "x, y, z",
        "const": "z - x <= 0\nz + x <= 0\nz - y <= 0\nz + y <= 0\n-1 - z <= 0"
    },
    "IMO Jan 2023 Q1 (Linear 2D)": {
        "vars": "x, y",
        "const": "x + y <= 10\n2*x + y <= 15\n-x <= 0\n-y <= 0"
    },
    "IMO Jan 2025 Q1 (Non-linear 2D)": {
        "vars": "x, y",
        "const": "x**2 + y**2 <= 4\n1 - x - y <= 0"
    }
}

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Define Subset C")
    
    # Selection logic
    selected_ex = st.selectbox("Load Example:", list(examples.keys()), index=1) # Default to Jan 22
    
    # Text Inputs (pre-filled)
    vars_input = st.text_input("Variables (comma separated):", value=examples[selected_ex]["vars"])
    constraints_input = st.text_area(
        "Inequalities (g(x) <= 0):", 
        value=examples[selected_ex]["const"],
        height=200
    )
    st.caption("Enter constraints in the form `lhs <= rhs` or similar.")

with col2:
    if st.button("Analyze & Generate Steps", type="primary"):
        try:
            # 1. Parse Variables
            vars_sym = [sp.symbols(v.strip()) for v in vars_input.split(',')]
            
            # 2. Parse Constraints
            raw_lines = [line.strip() for line in constraints_input.split('\n') if line.strip()]
            parsed_constraints = []
            
            # For Linear Programming check
            linear_constraints = [] # Tuples of (coeffs, constant) for A_ub * x <= b_ub
            is_fully_linear = True
            
            for i, line in enumerate(raw_lines):
                rel = ""
                if "<=" in line: rel = "<="; lhs, rhs = line.split("<=")
                elif ">=" in line: rel = ">="; lhs, rhs = line.split(">=")
                elif "=" in line: rel = "="; lhs, rhs = line.split("=")
                else: continue
                
                # Standardize to g(x) <= 0
                expr = utils.parse_expr(lhs) - utils.parse_expr(rhs)
                if rel == ">=": expr = -expr
                
                # Store for analysis
                parsed_constraints.append({
                    "expr": expr,
                    "latex": sp.latex(expr),
                    "id": i + 1,
                    "rel": rel
                })

                # Check linearity for linprog
                if expr.is_polynomial() and expr.as_poly(vars_sym).total_degree() == 1:
                    poly = expr.as_poly(vars_sym)
                    coeffs = [float(poly.coeff_monomial(v)) for v in vars_sym]
                    constant = -float(poly.coeff_monomial(1)) # Move const to RHS: ax + by <= -c
                    linear_constraints.append((coeffs, constant))
                else:
                    is_fully_linear = False

            # --- BOUNDEDNESS LOGIC (Improved) ---
            box_bounds = {v: [float('-inf'), float('inf')] for v in vars_sym}
            is_bounded = False
            
            # Method A: Explicit Sphere/Ball Check (Non-linear)
            # Looks for x^2 + y^2 + ... <= R
            sq_sum = sum(v**2 for v in vars_sym)
            for c in parsed_constraints:
                # heuristic: if g(x) = x^2 + y^2 - R <= 0
                diff = sp.simplify(c["expr"] - sq_sum)
                if diff.is_constant() and float(diff) < 0:
                    R_sq = -float(diff)
                    limit = np.sqrt(R_sq)
                    for v in vars_sym:
                        box_bounds[v] = [-limit, limit]
                    is_bounded = True
                    break

            # Method B: Linear Programming (scipy)
            # If Method A failed but we have linear constraints, try to bound every variable
            if not is_bounded and linear_constraints:
                A_ub = [x[0] for x in linear_constraints]
                b_ub = [x[1] for x in linear_constraints]
                
                lp_bounded_count = 0
                for i, v in enumerate(vars_sym):
                    # Minimize x_i
                    c_min = [0]*len(vars_sym); c_min[i] = 1
                    res_min = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method='highs')
                    
                    # Maximize x_i (Minimize -x_i)
                    c_max = [0]*len(vars_sym); c_max[i] = -1
                    res_max = linprog(c_max, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method='highs')
                    
                    if res_min.success and res_max.success:
                        box_bounds[v][0] = res_min.fun
                        box_bounds[v][1] = -res_max.fun # negate because we minimized -x
                        lp_bounded_count += 1
                
                if lp_bounded_count == len(vars_sym):
                    is_bounded = True

            # --- RENDER ANALYSIS ---
            st.markdown("---")
            st.subheader("Step-by-Step Analysis")

            # 1. Closedness
            st.markdown("### 1. Closedness (Preimages)")
            st.write("To show that $C$ is closed, we examine the preimages of the constraint functions defining the intersection:")
            
            # Build intersection string
            intersection_str = " \\cap ".join([f"\\{{ v \\mid g_{{{c['id']}}}(v) \\leq 0 \\}}" for c in parsed_constraints])
            st.latex(f"C = {intersection_str}")
            
            st.write("The preimage of the functions is:")
            preimage_str = " \\cap ".join([f"g_{{{c['id']}}}^{{-1}}((-\\infty, 0])" for c in parsed_constraints])
            st.latex(preimage_str)
            
            st.markdown("""
            * **Prop [5.42]:** Intervals of the form $(-\infty, a]$ are closed sets in $\mathbb{R}$.
            * **Prop [5.51]:** The preimage of a closed set under a continuous function is closed.
            * **Remark [5.59]:** Polynomials are continuous functions.
            * **Prop [5.39]:** The intersection of closed sets is closed.
            """)
            st.success("Conclusion: Since $C$ is an intersection of closed sets, **$C$ is Closed**.")

            # 2. Boundedness
            st.markdown("### 2. Boundedness")
            st.write("To show $C$ is bounded, we attempt to find the range of values for each variable:")
            
            max_r_sq = 0
            
            for v in vars_sym:
                low, high = box_bounds[v]
                if low == float('-inf') or high == float('inf'):
                    st.latex(f"{sp.latex(v)} \\in (-\\infty, \\infty) \\quad \\text{{(Bounds not found)}}")
                else:
                    st.latex(f"{sp.latex(v)} \\in [{low:.2f}, {high:.2f}]")
                    max_r_sq += max(abs(low), abs(high))**2
            
            if is_bounded:
                radius = np.sqrt(max_r_sq)
                st.write(f"From **Definition [5.29]**, $C$ is bounded if $C \\subseteq B(0, R)$. We calculate the maximum norm $|u|$:")
                
                sq_sum_tex = "+".join([f"|{v}|^2" for v in vars_sym])
                val_sum_tex = "+".join([f"{max(abs(b[0]), abs(b[1])):.2f}^2" for b in box_bounds.values()])
                
                st.latex(f"|u| = \\sqrt{{{sq_sum_tex}}} \\leq R")
                st.latex(f"\\sqrt{{{val_sum_tex}}} = \\sqrt{{{max_r_sq:.2f}}} \\implies R \\geq {radius:.2f}")
                
                st.success(f"Conclusion: For any $R \\geq {radius:.2f}$, the ball $B(0, R)$ contains $C$. Thus **$C$ is Bounded**.")
            else:
                st.error("Conclusion: Could not find finite bounds for all variables. **Boundedness is Indeterminate**.")

            # 3. Compactness
            st.markdown("### 3. Compactness")
            st.write("From **Definition [5.43]**, a subset is Compact if it is Closed and Bounded.")
            
            if is_bounded:
                st.success("Since $C$ is Closed and Bounded, **$C$ is COMPACT**.")
                st.markdown("""
                **Implication:** From **Theorem [5.66]** (Extreme Value Theorem), a continuous function on a compact set attains its maximum and minimum. 
                Therefore, the optimization problem has a solution.
                """)
            else:
                st.warning("Since Boundedness is not proven, **Compactness is Undetermined**.")

        except Exception as e:
            st.error(f"Error parsing inputs: {e}")
            st.write("Ensure variables match used constraints.")