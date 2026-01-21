import streamlit as st
import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Advanced Solvers")
utils.setup_page()

# --- LOCAL UTILS (Missing in utils.py) ---
def p_multiselect(label, options, key, default=None, **kwargs):
    """
    Local wrapper for multiselect following the project's state pattern.
    """
    if default is None:
        default = []
    utils.init_state(key, default)
    def on_change():
        st.session_state[key] = st.session_state[f"w_{key}"]
    return st.multiselect(label, options, default=st.session_state[key], key=f"w_{key}", on_change=on_change, **kwargs)

# --- MAIN PAGE ---
st.markdown("<h1 class='main-header'>Advanced Mathematical Solvers</h1>", unsafe_allow_html=True)

# Mode Selection
mode = utils.p_radio(
    "Select Algorithm", 
    ["Fourier-Motzkin Elimination", "Least Squares Fitting (Normal Equations)"], 
    "adv_solver_mode"
)

st.markdown("---")

# ==========================================
# FOURIER-MOTZKIN ELIMINATION
# ==========================================
if mode == "Fourier-Motzkin Elimination":
    st.subheader("Fourier-Motzkin Elimination")
    st.caption("Projecting high-dimensional polytopes onto lower dimensions by eliminating variables.")
    
    # 
    
    st.info("""
    **Format:** This tool automatically rearranges inequalities to isolate variables.
    Enter inequalities like $x + y \le 5$ or $2*x - z >= 0$.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. System Setup")
        
        # Default example: A 3D pyramid shape
        default_ineqs = "2*x + y <= 6\nx + 2*y <= 6\nx + 2*y >= 2\nx >= 0\ny >= 0\nz - x - y <= 0\n-z + x + y <= 0"
        
        ineq_text = utils.p_text_area(
            "System of Inequalities (one per line):", 
            "fm_ineq_input", 
            default_ineqs, 
            height=250
        )
        
    with col2:
        st.subheader("2. Configuration")
        
        # Parse variables dynamically
        detected_vars = set()
        try:
            if ineq_text.strip():
                # Rough parsing to find symbols
                raw_text = ineq_text.replace('<',' ').replace('>',' ').replace('=',' ')
                for part in raw_text.split():
                    # Skip numbers
                    if part.replace('.','',1).isdigit(): continue
                    try:
                        sym = sp.sympify(part)
                        if isinstance(sym, sp.Symbol):
                            detected_vars.add(sym)
                        elif hasattr(sym, 'free_symbols'):
                            detected_vars.update(sym.free_symbols)
                    except: pass
        except Exception as e:
            st.warning(f"Could not auto-detect variables: {e}")

        sorted_vars = sorted([str(s) for s in detected_vars])
        
        if not sorted_vars:
            st.warning("No variables detected yet.")
            elimination_order = []
        else:
            st.markdown(f"**Detected variables:** `{', '.join(sorted_vars)}`")
            
            st.write("Select order of elimination (variables to remove):")
            # Use local wrapper p_multiselect
            elimination_order = p_multiselect(
                "Drag and drop or select order:",
                options=sorted_vars,
                key="fm_elim_order",
                default=sorted_vars[:-1] # Default: keep the last one
            )
            
            remaining = [v for v in sorted_vars if v not in elimination_order]
            if remaining:
                st.success(f"Final result will be inequalities in terms of: **{', '.join(remaining)}**")

    # Helper for display
    def format_leq(expr):
        """Rewrites 'expr <= 0' to 'Variables <= Constants' for cleaner display."""
        syms = expr.free_symbols
        if not syms:
            return f"{sp.latex(expr)} \le 0"
        constant_part, variable_part = expr.as_independent(*syms)
        # We have: variable + constant <= 0  ->  variable <= -constant
        return f"{sp.latex(variable_part)} \le {sp.latex(-constant_part)}"

    if st.button("Run Elimination", type="primary") and sorted_vars:
        try:
            # --- STEP 1: PARSING ---
            current_ineqs = []
            raw_lines = [line.strip() for line in ineq_text.split('\n') if line.strip()]
            
            for line in raw_lines:
                # Normalize everything to expression <= 0
                if "<=" in line:
                    parts = line.split("<=")
                    expr = sp.sympify(parts[0]) - sp.sympify(parts[1])
                elif ">=" in line:
                    parts = line.split(">=")
                    expr = sp.sympify(parts[1]) - sp.sympify(parts[0]) # Flip
                elif "<" in line:
                    parts = line.split("<")
                    expr = sp.sympify(parts[0]) - sp.sympify(parts[1])
                elif ">" in line:
                    parts = line.split(">")
                    expr = sp.sympify(parts[1]) - sp.sympify(parts[0]) # Flip
                else:
                    continue
                current_ineqs.append(expr)

            st.divider()
            st.subheader("Computation Steps")
            
            with st.expander(f"Initial System ({len(current_ineqs)} inequalities)", expanded=False):
                 for ieq in current_ineqs: 
                     st.latex(format_leq(ieq))

            # --- STEP 2: ELIMINATION LOOP ---
            step_count = 1
            possible = True
            
            for elim_var_name in elimination_order:
                var_sym = sp.symbols(elim_var_name)
                
                upper_bounds = [] 
                lower_bounds = []
                independent = []
                
                # Partition inequalities based on the variable
                for expr in current_ineqs:
                    coeff = expr.coeff(var_sym)
                    rest = expr - coeff * var_sym
                    
                    if abs(coeff) < 1e-9: # Essentially 0
                        independent.append(expr)
                    elif coeff > 0:
                        # coeff*x + rest <= 0  ->  x <= -rest/coeff
                        upper_bounds.append(sp.simplify(-rest / coeff))
                    else: 
                        # coeff*x + rest <= 0 (coeff is neg) -> x >= -rest/coeff
                        lower_bounds.append(sp.simplify(-rest / coeff))
                
                st.markdown(f"#### Step {step_count}: Eliminate ${elim_var_name}$")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption(f"Lower Bounds ($L \le {elim_var_name}$)")
                    if lower_bounds:
                        for lb in lower_bounds: st.latex(f"{sp.latex(lb)} \le {elim_var_name}")
                    else: st.write("None ($-\infty$)")
                with col_b:
                    st.caption(f"Upper Bounds (${elim_var_name} \le U$)")
                    if upper_bounds:
                        for ub in upper_bounds: st.latex(f"{elim_var_name} \le {sp.latex(ub)}")
                    else: st.write("None ($+\infty$)")

                # Generate new constraints
                new_constraints = []
                latex_copy_lines = []

                # Pair every lower bound with every upper bound
                if lower_bounds and upper_bounds:
                    st.markdown("**New Constraints (Pairing $L \le U$):**")
                    for lb in lower_bounds:
                        for ub in upper_bounds:
                            # L <= U  =>  L - U <= 0
                            full_latex = f"{sp.latex(lb)} \le {sp.latex(ub)}"
                            st.latex(full_latex)
                            latex_copy_lines.append(full_latex + ",")
                            new_constraints.append(lb - ub)
                elif independent:
                    st.write("(No pairings possible. Only independent constraints carried forward).")
                else:
                    st.write("(Variable effectively unconstrained or constraints vanished).")

                # Carry over independent
                new_constraints.extend(independent)
                
                # Cleanup and simplify
                cleaned_constraints = []
                seen_constraints = set()
                
                for c in new_constraints:
                    simp = sp.simplify(c)
                    # Check for contradiction (e.g., 5 <= 0)
                    if simp.is_number:
                        if simp > 0: 
                            st.error(f"❌ Contradiction found: {float(simp)} <= 0. System is infeasible.")
                            possible = False; break
                        else: continue # Trivial (e.g. -5 <= 0)
                    
                    # Deduplication using formatted string
                    formatted_str = format_leq(simp)
                    if formatted_str not in seen_constraints:
                        cleaned_constraints.append(simp)
                        seen_constraints.add(formatted_str)
                
                if not possible: break
                current_ineqs = cleaned_constraints
                step_count += 1
                st.divider()

            # --- STEP 3: FINAL RESULT ---
            if possible:
                st.subheader("Final Projected Region")
                if not current_ineqs:
                    st.success("The entire space (for remaining variables) is feasible.")
                else:
                    st.success(f"Projection onto space of *{', '.join(remaining)}*:")
                    
                    final_latex_lines = []
                    for c in current_ineqs:
                        latex_str = format_leq(c)
                        st.latex(latex_str)
                        final_latex_lines.append(latex_str + ",")

                    if final_latex_lines:
                        with st.expander("Show LaTeX Code"):
                            st.code("\n".join(final_latex_lines), language="latex")

        except Exception as e:
            st.error(f"Elimination Error: {e}")


# ==========================================
# LEAST SQUARES METHOD
# ==========================================
elif mode == "Least Squares Fitting (Normal Equations)":
    st.subheader("Least Squares Method")
    
    st.info("""
    **Curriculum Reference: Section 5.4 & Theorem 5.16** Finds the parameter vector $x$ that minimizes the error $|b - Ax|^2$ by solving the Normal Equations:
    """)
    st.latex(r"(A^T A)x = A^T b")
    
    # 

    # Model Selection using utils wrapper
    fit_type = utils.p_radio(
        "Choose Model Type:", 
        ["Polynomial Fit (y = a_0 + a_1 x + ...)", "Circle Fit (Exercise 5.22)"],
        "ls_model_type",
        horizontal=True
    )

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Input Data")
        
        default_data = "1, 2\n2, 3\n3, 5\n4, 7" if "Polynomial" in fit_type else "0, 2\n0, 3\n2, 0\n3, 1"
        data_input = utils.p_text_area(
            "Points (x, y) one per line:", 
            "ls_data_input", 
            default_data, 
            height=150
        )
        
        st.markdown("### 📘 Model Construction")
        if "Polynomial" in fit_type:
            degree = utils.p_slider("Polynomial Degree:", "ls_poly_degree", 1, 5, 1)
            st.markdown(f"""
            **Goal:** Fit $y = a_0 + a_1 x + \dots + a_n x^n$.
            
            We construct matrix $A$ where row $i$ is $[1, x_i, x_i^2, \dots]$ and vector $b$ is $[y_i]$.
            """)
        else:
            st.markdown(r"""
            **Goal:** Fit $(x-a)^2 + (y-b)^2 = r^2$.
            
            Linearized form (Ex 5.22): $2ax + 2by + (r^2 - a^2 - b^2) = x^2 + y^2$.
            Unknowns are $a, b$ and auxiliary $c$.
            """)

    with col2:
        st.subheader("2. Solution")
        if st.button("Compute Best Fit", type="primary"):
            try:
                # --- DATA PARSING ---
                points = []
                for line in data_input.split('\n'):
                    if ',' in line:
                        parts = line.split(',')
                        try:
                            points.append((float(parts[0].strip()), float(parts[1].strip())))
                        except: pass
                
                if not points:
                    st.error("No valid points found. Use format `x, y`.")
                    st.stop()

                x_vals = np.array([p[0] for p in points])
                y_vals = np.array([p[1] for p in points])
                num_pts = len(points)

                # --- MATRIX CONSTRUCTION ---
                if "Polynomial" in fit_type:
                    # Rows are [1, x, x^2, ...]
                    A_cols = [np.ones(num_pts)]
                    for d in range(1, degree + 1):
                        A_cols.append(x_vals ** d)
                    A_np = np.column_stack(A_cols)
                    b_np = y_vals
                    
                    param_names = [f"a_{i}" for i in range(degree + 1)]
                    
                else: # Circle Fit
                    col_1 = 2 * x_vals
                    col_2 = 2 * y_vals
                    col_3 = np.ones(num_pts)
                    
                    A_np = np.column_stack([col_1, col_2, col_3])
                    b_np = x_vals**2 + y_vals**2
                    
                    param_names = ["a (center x)", "b (center y)", "c (aux)"]

                # --- CALCULATIONS ---
                ATA = A_np.T @ A_np
                ATb = A_np.T @ b_np
                
                try:
                    x_sol = np.linalg.solve(ATA, ATb)
                except np.linalg.LinAlgError:
                    st.error("Matrix $A^T A$ is singular. Points might be collinear or insufficient.")
                    st.stop()

                # --- DISPLAY ---
                c_a, c_b = st.columns(2)
                with c_a:
                    st.markdown("**Matrix $A^T A$**")
                    st.latex(sp.latex(sp.Matrix(np.round(ATA, 2))))
                with c_b:
                    st.markdown("**Vector $A^T b$**")
                    st.latex(sp.latex(sp.Matrix(np.round(ATb, 2))))

                st.markdown("**Solution Parameters:**")
                df_params = pd.DataFrame([x_sol], columns=param_names)
                st.table(df_params)

                # --- PLOTTING ---
                fig, ax = plt.subplots(figsize=(8, 5))
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
                        st.warning("Calculated radius squared is negative (imaginary circle).")
                        r = 0
                    else:
                        r = np.sqrt(r_squared)
                        st.success(f"**Best Fit Circle:** Center $({a:.3f}, {b:.3f})$, Radius $r={r:.3f}$")
                        
                        circle = plt.Circle((a, b), r, color='blue', fill=False, label='Fitted Circle')
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