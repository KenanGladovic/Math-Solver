import streamlit as st
import sympy as sp
import numpy as np
import utils

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()

st.markdown("<h1 class='main-header'>KKT Computation</h1>", unsafe_allow_html=True)

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
    # Default from Jan 2025 Exam Q1
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
                
                st.markdown("### 1. Problem Statement")
                st.write("Minimize:")
                st.latex(f"f = {sp.latex(f_expr)}")
                st.write("Subject to constraints ($g_i \le 0$):")
                for g in parsed_constraints:
                    st.latex(f"{sp.latex(g)} \\le 0")
                st.markdown("---")

                st.markdown("### 2. The KKT Conditions (Formula 9.24)")
                
                lambdas = [sp.symbols(f'lambda_{i+1}') for i in range(len(parsed_constraints))]
                
                st.markdown("**1. Dual Feasibility** ($\lambda_i \ge 0$)")
                if lambdas:
                    lam_tex = ", ".join([sp.latex(l) for l in lambdas])
                    st.latex(f"{lam_tex} \ge 0")
                else:
                    st.write("No inequality constraints.")

                st.markdown("**2. Primal Feasibility** ($g_i(v_0) \le 0$)")
                for i, g in enumerate(parsed_constraints):
                    st.latex(f"{sp.latex(g)} \le 0")

                st.markdown("**3. Complementary Slackness** ($\lambda_i g_i(v_0) = 0$)")
                for i, (lam, g) in enumerate(zip(lambdas, parsed_constraints)):
                    st.latex(f"\\lambda_{{{i+1}}} ({sp.latex(g)}) = 0")
                
                st.markdown("**4. Stationarity** ($\\nabla f(v_0) + \sum \lambda_i \\nabla g_i(v_0) = 0$)")
                
                L = f_expr
                for lam, g in zip(lambdas, parsed_constraints):
                    L += lam * g
                
                for var in vars_sym:
                    eq = sp.diff(L, var)
                    st.latex(f"{sp.latex(eq)} = 0 \\quad \\text{{(w.r.t. }} {var.name} \\text{{)}}")

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
                        point_map = dict(zip(vars_sym, vals))
                        st.markdown(f"### Analysis of Point $v_0 = {tuple(vals)}$")
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

                        # --- STEP 1: SOLVE FOR MULTIPLIERS (LAMBDA) ---
                        # We try to satisfy Stationarity (Cond 4) first to find candidate lambdas.
                        # Equation: A^T * lambda = -grad_f
                        # We only consider 'active' constraints (g approx 0) for the solve, 
                        # but we verify against all.
                        
                        # Identify active constraints for the linear solver
                        active_indices = [i for i, val in enumerate(g_vals) if abs(val) < 1e-4]
                        
                        # Build matrix A (columns are gradients of active constraints)
                        if active_indices:
                            A_active = np.array([grad_g_vals[i] for i in active_indices]).T
                            b_vec = -np.array(grad_f_vals)
                            
                            # Solve least squares
                            lam_active, residuals, _, _ = np.linalg.lstsq(A_active, b_vec, rcond=None)
                        else:
                            lam_active = []

                        # Construct full lambda vector
                        lambdas_calc = [0.0] * len(parsed_constraints)
                        for i, idx in enumerate(active_indices):
                            lambdas_calc[idx] = lam_active[i]

                        st.info(f"**Calculated Multipliers:** We solve the Stationarity equation to find the required $\lambda$ values:\n\n" + 
                                ", ".join([f"$\lambda_{i+1} = {lam:.4f}$" for i, lam in enumerate(lambdas_calc)]))

                        st.markdown("---")

                        # --- STEP 2: VERIFY ALL CONDITIONS ---
                        
                        # 1. Dual Feasibility
                        st.markdown("#### 1. Dual Feasibility Check ($\lambda_i \ge 0$)")
                        dual_pass = True
                        for i, lam in enumerate(lambdas_calc):
                            if lam < -1e-5:
                                st.error(f"❌ $\lambda_{i+1} = {lam:.4f} < 0$. Condition Failed.")
                                dual_pass = False
                            else:
                                st.write(f"✅ $\lambda_{i+1} = {lam:.4f} \ge 0$.")
                        
                        if dual_pass: st.success("result: Condition Satisfied.")
                        else: st.warning("result: Condition Failed.")

                        # 2. Primal Feasibility
                        st.markdown("#### 2. Primal Feasibility Check ($g_i(v_0) \le 0$)")
                        primal_pass = True
                        for i, val in enumerate(g_vals):
                            if val > 1e-5:
                                st.error(f"❌ $g_{i+1}(v_0) = {val:.4f} > 0$. Constraint Violated.")
                                primal_pass = False
                            else:
                                st.write(f"✅ $g_{i+1}(v_0) = {val:.4f} \le 0$.")
                        
                        if primal_pass: st.success("result: Condition Satisfied.")
                        else: st.warning("result: Condition Failed.")

                        # 3. Complementary Slackness
                        st.markdown("#### 3. Complementary Slackness Check ($\lambda_i g_i(v_0) = 0$)")
                        slack_pass = True
                        for i, (lam, val) in enumerate(zip(lambdas_calc, g_vals)):
                            product = lam * val
                            if abs(product) > 1e-4:
                                st.error(f"❌ $\lambda_{i+1} \cdot g_{i+1} = {lam:.4f} \cdot {val:.4f} = {product:.4f} \\ne 0$. Failed.")
                                slack_pass = False
                            else:
                                st.write(f"✅ $\lambda_{i+1} \cdot g_{i+1} \\approx 0$.")
                        
                        if slack_pass: st.success("result: Condition Satisfied.")
                        else: st.warning("result: Condition Failed.")

                        # 4. Stationarity
                        st.markdown("#### 4. Stationarity Check ($\nabla L = 0$)")
                        # Recalculate gradient sum
                        grad_L = np.array(grad_f_vals)
                        for i, lam in enumerate(lambdas_calc):
                            grad_L += lam * np.array(grad_g_vals[i])
                        
                        residual_norm = np.linalg.norm(grad_L)
                        
                        st.write(f"Sum of gradients: $\\nabla f + \sum \lambda_i \\nabla g_i = {np.round(grad_L, 4)}$")
                        if residual_norm < 1e-4:
                            st.success(f"result: Condition Satisfied (Residual: {residual_norm:.1e}).")
                            stat_pass = True
                        else:
                            st.error(f"result: Condition Failed. The gradients do not cancel out (Residual: {residual_norm:.4f}).")
                            stat_pass = False

                        # --- STEP 3: FINAL VERDICT ---
                        st.markdown("---")
                        st.subheader("Final Verdict")
                        if dual_pass and primal_pass and slack_pass and stat_pass:
                            st.balloons()
                            st.success(f"**OPTIMAL.** The point {tuple(vals)} satisfies all KKT conditions.")
                            st.markdown("Since the problem is Convex (check this!) and Strictly Feasible, this is a **Global Minimum**.")
                        else:
                            st.error(f"**NOT OPTIMAL.** The point {tuple(vals)} fails one or more KKT conditions.")

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Please fix input errors above first.")