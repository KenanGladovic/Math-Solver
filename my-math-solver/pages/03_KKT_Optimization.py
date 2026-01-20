import streamlit as st
import sympy as sp
import numpy as np
import utils

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()

st.markdown("<h1 class='main-header'>KKT Computation & Proof Generator</h1>", unsafe_allow_html=True)

# 2. Curriculum Context
with st.expander("📘 Curriculum References (Chapter 9)", expanded=False):
    st.markdown("""
    * **Definition 9.24 (KKT Conditions):** The necessary conditions for $x^*$ to be an optimal solution.
    * **Definition 9.33 (Strict Feasibility):** There exists a point $z_0$ where all $g_i(z_0) < 0$.
    * **Theorem 9.34:** If the problem is convex and **strictly feasible**, KKT conditions are sufficient for optimality.
    * **Section 9.5.1 (Strategy):** We solve by testing combinations of $\lambda_i = 0$ vs $\lambda_i > 0$. 
    """)

# 3. Unified Input Section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Problem Definition")
    # Default from Jan 2025 Exam Q1 style
    obj_input = utils.p_text_input("Minimize Objective $f(x,y,...)$:", "kkt_master_obj", "(x + y)**2 - x - y")
    
    default_const = "x**2 + y**2 <= 4\n1 - x - y <= 0"
    constraints_input = utils.p_text_area("Constraints $g_i(x) \le 0$ (one per line):", "kkt_master_const", default_const, height=150)
    
    st.info("💡 **Tip:** Enter constraints **exactly** as they appear in the exam. The tool automatically converts them to $g(x) \le 0$.")

with col2:
    st.subheader("2. Analysis Mode")
    
    # Global Parse Logic
    f_expr = utils.parse_expr(obj_input)
    raw_lines = [l.strip() for l in constraints_input.split('\n') if l.strip()]
    parsed_constraints = []
    valid_input = True
    
    if f_expr:
        vars_sym = sorted(list(f_expr.free_symbols), key=lambda s: s.name)
        for c_str in raw_lines:
            try:
                if "<=" in c_str:
                    lhs, rhs = c_str.split("<=")
                    g = utils.parse_expr(lhs) - utils.parse_expr(rhs)
                    parsed_constraints.append(g)
                elif ">=" in c_str:
                    lhs, rhs = c_str.split(">=")
                    g = utils.parse_expr(rhs) - utils.parse_expr(lhs)
                    parsed_constraints.append(g)
                else:
                    st.error(f"Invalid syntax in: '{c_str}'. Use <= or >=.")
                    valid_input = False
            except:
                st.error(f"Could not parse: '{c_str}'")
                valid_input = False
    else:
        valid_input = False

    # Tabs
    tab_sym, tab_num = st.tabs(["📝 Generate Conditions (Symbolic)", "✅ Verify Candidate (Detailed Report)"])

    # ============================================================
    # TAB 1: SYMBOLIC GENERATION
    # ============================================================
    with tab_sym:
        st.write("Generates the equations from **(9.24)** in your curriculum.")
        
        if st.button("Generate KKT Conditions", type="primary"):
            if valid_input and f_expr:
                latex_report = [] # Store lines for copyable output
                
                # Helper to display and log
                def log_latex(latex_str, label=None):
                    if label:
                        st.markdown(label)
                        latex_report.append(f"% {label}")
                    st.latex(latex_str)
                    latex_report.append(f"$$ {latex_str} $$")

                # 1. Problem Statement
                st.markdown("### 1. Problem Statement")
                latex_report.append("\\section*{1. Problem Statement}")
                latex_report.append(f"Minimize: $$ f = {sp.latex(f_expr)} $$")
                st.latex(f"f = {sp.latex(f_expr)}")
                
                latex_report.append("Subject to constraints ($g_i \\le 0$):")
                st.write("Subject to constraints ($g_i \le 0$):")
                for g in parsed_constraints:
                    log_latex(f"{sp.latex(g)} \\le 0")
                st.markdown("---")

                # 2. KKT Conditions
                st.markdown("### 2. The KKT Conditions (Formula 9.24)")
                latex_report.append("\n\\section*{2. The KKT Conditions (Formula 9.24)}")
                
                lambdas = [sp.symbols(f'lambda_{i+1}') for i in range(len(parsed_constraints))]
                
                # Condition 1
                st.markdown("**1. Multiplier Signs** ($\lambda_i \ge 0$)")
                if lambdas:
                    lam_tex = ", ".join([sp.latex(l) for l in lambdas])
                    log_latex(f"{lam_tex} \\ge 0", "Multiplier Signs:")
                else:
                    st.write("No inequality constraints.")

                # Condition 2
                st.markdown("**2. Constraint Feasibility** ($g_i(v_0) \le 0$)")
                latex_report.append("\n% Constraint Feasibility")
                for i, g in enumerate(parsed_constraints):
                    log_latex(f"{sp.latex(g)} \\le 0")

                # Condition 3
                st.markdown("**3. Product Condition** ($\lambda_i g_i(v_0) = 0$)")
                latex_report.append("\n% Product Condition")
                for i, (lam, g) in enumerate(zip(lambdas, parsed_constraints)):
                    log_latex(f"\\lambda_{{{i+1}}} ({sp.latex(g)}) = 0")
                
                # Condition 4
                st.markdown("**4. Gradient Equation** ($\\nabla f(v_0) + \sum \lambda_i \\nabla g_i(v_0) = 0$)")
                latex_report.append("\n% Gradient Equation")
                
                L = f_expr
                for lam, g in zip(lambdas, parsed_constraints):
                    L += lam * g
                
                for var in vars_sym:
                    eq = sp.diff(L, var)
                    log_latex(f"{sp.latex(eq)} = 0 \\quad \\text{{(w.r.t. }} {var.name} \\text{{)}}")

                # Output Code Block
                st.markdown("### 📋 LaTeX Code")
                st.code("\n".join(latex_report), language="latex")

            else:
                st.error("Please fix input errors above first.")

    # ============================================================
    # TAB 2: NUMERICAL VERIFICATION
    # ============================================================
    with tab_num:
        st.write("Calculates required multipliers and tests the candidate point against all 4 conditions.")
        candidate_str = utils.p_text_input("Candidate Point (e.g. 2, 0):", "kkt_master_cand", "2, 0")
        
        if st.button("Verify Candidate", type="primary"):
            if valid_input and f_expr:
                try:
                    vals = [float(x.strip()) for x in candidate_str.split(',')]
                    if len(vals) != len(vars_sym):
                        st.error(f"Dimension mismatch. Expected {len(vars_sym)} variables ({vars_sym}), got {len(vals)}.")
                    else:
                        latex_report = []
                        point_map = dict(zip(vars_sym, vals))
                        point_tex = tuple(vals)
                        
                        st.markdown(f"### Analysis of Point $v_0 = {point_tex}$")
                        latex_report.append(f"\\section*{{Analysis of Candidate Point $v_0 = {point_tex}$}}")
                        
                        st.write("We will now insert this point into the KKT conditions.")
                        st.markdown("---")

                        # --- STEP 0: CALCULATE GRADIENTS ---
                        grad_f_vals = [float(sp.diff(f_expr, v).subs(point_map)) for v in vars_sym]
                        
                        grad_g_vals = []
                        g_vals = []
                        for g in parsed_constraints:
                            g_val = float(g.subs(point_map))
                            g_vals.append(g_val)
                            grad_g_vals.append([float(sp.diff(g, v).subs(point_map)) for v in vars_sym])

                        # --- STEP 1: SOLVE FOR MULTIPLIERS ---
                        # We use the Gradient Equation to find lambdas for active constraints
                        active_indices = [i for i, val in enumerate(g_vals) if abs(val) < 1e-4]
                        
                        if active_indices:
                            A_active = np.array([grad_g_vals[i] for i in active_indices]).T
                            b_vec = -np.array(grad_f_vals)
                            lam_active, residuals, _, _ = np.linalg.lstsq(A_active, b_vec, rcond=None)
                        else:
                            lam_active = []

                        lambdas_calc = [0.0] * len(parsed_constraints)
                        for i, idx in enumerate(active_indices):
                            lambdas_calc[idx] = lam_active[i]

                        # Show the linear system being solved
                        st.info("**Solving for Multipliers:** Using active constraints to solve $\\nabla f + \sum \lambda_i \\nabla g_i = 0$")
                        latex_report.append("\\subsection*{1. Calculation of Lagrange Multipliers}")
                        latex_report.append("Solving the gradient equation for active constraints:")
                        
                        eq_sys_tex = []
                        grad_f_tex = sp.latex(sp.Matrix(grad_f_vals))
                        grad_sum_tex = []
                        
                        for i, val in enumerate(lambdas_calc):
                            if abs(val) > 1e-5: # Active
                                grad_g_tex = sp.latex(sp.Matrix(grad_g_vals[i]))
                                grad_sum_tex.append(f"\\lambda_{{{i+1}}} {grad_g_tex}")
                        
                        full_eq = f"{grad_f_tex} + " + " + ".join(grad_sum_tex) + " = 0" if grad_sum_tex else f"{grad_f_tex} = 0"
                        st.latex(full_eq)
                        latex_report.append(f"$$ {full_eq} $$")

                        lam_res_str = ", ".join([f"\\lambda_{{{i+1}}} = {lam:.4f}" for i, lam in enumerate(lambdas_calc)])
                        st.write(f"Result: ${lam_res_str}$")
                        latex_report.append(f"Result: $$ {lam_res_str} $$")

                        st.markdown("---")

                        # --- STEP 2: VERIFY ALL CONDITIONS ---
                        latex_report.append("\\subsection*{2. Verification of Conditions}")
                        
                        # 1. Multiplier Signs
                        st.markdown("#### 1. Multiplier Signs Check ($\lambda_i \ge 0$)")
                        latex_report.append("\\textbf{1. Multiplier Signs} ($\\lambda_i \\ge 0$):")
                        dual_pass = True
                        for i, lam in enumerate(lambdas_calc):
                            check = "\\ge" if lam >= -1e-5 else "<"
                            msg = f"\\lambda_{{{i+1}}} = {lam:.4f} {check} 0"
                            st.latex(msg)
                            latex_report.append(f"$$ {msg} $$")
                            if lam < -1e-5: dual_pass = False

                        if dual_pass: st.success("Condition Satisfied.")
                        else: st.warning("Condition Failed.")

                        # 2. Constraint Feasibility
                        st.markdown("#### 2. Constraint Feasibility Check ($g_i(v_0) \le 0$)")
                        latex_report.append("\n\\textbf{2. Constraint Feasibility} ($g_i(v_0) \\le 0$):")
                        primal_pass = True
                        for i, val in enumerate(g_vals):
                            check = "\\le" if val <= 1e-5 else ">"
                            msg = f"g_{{{i+1}}}(v_0) = {val:.4f} {check} 0"
                            st.latex(msg)
                            latex_report.append(f"$$ {msg} $$")
                            if val > 1e-5: primal_pass = False
                        
                        if primal_pass: st.success("Condition Satisfied.")
                        else: st.warning("Condition Failed.")

                        # 3. Product Condition
                        st.markdown("#### 3. Product Condition Check ($\lambda_i g_i(v_0) = 0$)")
                        latex_report.append("\n\\textbf{3. Product Condition} ($\\lambda_i g_i(v_0) = 0$):")
                        slack_pass = True
                        for i, (lam, val) in enumerate(zip(lambdas_calc, g_vals)):
                            product = lam * val
                            check = "\\approx" if abs(product) < 1e-4 else "\\ne"
                            msg = f"\\lambda_{{{i+1}}} \\cdot g_{{{i+1}}} = {lam:.4f} \\cdot {val:.4f} = {product:.4f} {check} 0"
                            st.latex(msg)
                            latex_report.append(f"$$ {msg} $$")
                            if abs(product) > 1e-4: slack_pass = False
                        
                        if slack_pass: st.success("Condition Satisfied.")
                        else: st.warning("Condition Failed.")

                        # 4. Gradient Equation
                        st.markdown("#### 4. Gradient Equation Check ($\nabla L = 0$)")
                        latex_report.append("\n\\textbf{4. Gradient Equation} ($\\nabla L = 0$):")
                        grad_L = np.array(grad_f_vals)
                        for i, lam in enumerate(lambdas_calc):
                            grad_L += lam * np.array(grad_g_vals[i])
                        
                        residual_norm = np.linalg.norm(grad_L)
                        msg = f"|\\nabla L| = {residual_norm:.4f}"
                        st.latex(msg)
                        latex_report.append(f"$$ {msg} $$")
                        
                        if residual_norm < 1e-4:
                            st.success(f"Condition Satisfied.")
                            stat_pass = True
                        else:
                            st.error(f"Condition Failed.")
                            stat_pass = False

                        # --- STEP 3: FINAL VERDICT ---
                        st.markdown("---")
                        st.subheader("Final Verdict")
                        if dual_pass and primal_pass and slack_pass and stat_pass:
                            st.balloons()
                            res_msg = f"The point $v_0 = {point_tex}$ satisfies all KKT conditions."
                            st.success(res_msg)
                            st.markdown("Since the problem is Convex (assuming convexity of f and C) and Strictly Feasible, this is a **Global Minimum** (Theorem 9.34).")
                            latex_report.append(f"\\section*{{Conclusion}}\n{res_msg}\n\nPer Theorem 9.34, if the problem is convex and strictly feasible, this is a global minimum.")
                        else:
                            res_msg = f"The point $v_0 = {point_tex}$ fails one or more KKT conditions."
                            st.error(res_msg)
                            latex_report.append(f"\\section*{{Conclusion}}\n{res_msg}")

                        # OUTPUT CODE
                        st.markdown("### 📋 LaTeX Code")
                        st.code("\n".join(latex_report), language="latex")

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Please fix input errors above first.")