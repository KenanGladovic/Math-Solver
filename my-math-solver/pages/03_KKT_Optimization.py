import streamlit as st
import sympy as sp
import numpy as np
import utils

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()

st.markdown("<h1 class='main-header'>KKT Computation</h1>", unsafe_allow_html=True)

# 2. Curriculum Context (Strictly adhering to Chapter 9)
with st.expander("📘 Curriculum References (Chapter 9)", expanded=False):
    st.markdown("""
    * **Definition 9.24 (KKT Conditions):** The necessary conditions for $x^*$ to be an optimal solution.
    * **Definition 9.33 (Strict Feasibility):** There exists a point $z_0$ where all $g_i(z_0) < 0$.
    * **Theorem 9.34:** If the problem is convex and **strictly feasible**, KKT conditions are sufficient for optimality.
    * **Section 9.5.1 (Strategy):** We solve by testing combinations of $\lambda_i = 0$ vs $\lambda_i > 0$. 
        * If $\lambda_i > 0$, then $g_i(x) = 0$.
    """)

# 3. Unified Input Section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Problem Definition")
    # Default from Jan 2025 Exam Q1
    obj_input = utils.p_text_input("Minimize Objective $f(x,y,...)$:", "kkt_master_obj", "(x + y)**2 - x - y")
    
    default_const = "x**2 + y**2 <= 4\n1 - x - y <= 0"
    constraints_input = utils.p_text_area("Constraints $g_i(x) \le 0$ (one per line):", "kkt_master_const", default_const, height=150)
    
    st.info("💡 **Tip:** Enter constraints **exactly** as they appear in the exam (e.g. `x + y >= 1`). The tool automatically flips them to the standard mathematical form ($g(x) \le 0$) for the calculation.")
    
                                                                                                                                                                             

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
                    # Convert g >= 0 to -g <= 0
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
    tab_sym, tab_num = st.tabs(["📝 Generate Conditions (Symbolic)", "✅ Verify Candidate (Numerical)"])

    # ============================================================
    # TAB 1: SYMBOLIC GENERATION
    # ============================================================
    with tab_sym:
        st.write("Generates the formal LaTeX equations for your exam paper.")
        
        if st.button("Generate KKT Conditions", type="primary"):
            if valid_input and f_expr:
                
                # --- A. Problem Preview ---
                st.markdown("### 1. Problem Statement")
                st.write("Minimize:")
                st.latex(f"f = {sp.latex(f_expr)}")
                st.write("Subject to constraints ($g_i \le 0$):")
                for g in parsed_constraints:
                    st.latex(f"{sp.latex(g)} \\le 0")
                st.markdown("---")

                # --- B. The Conditions ---
                st.markdown("### 2. The KKT Conditions (Def 9.24)")
                
                # 1. Primal
                st.markdown("**I. Primal Feasibility** ($g_i(x) \le 0$)")
                for i, g in enumerate(parsed_constraints):
                    st.latex(f"g_{{{i+1}}}: {sp.latex(g)} \le 0")
                
                # 2. Dual
                st.markdown("**II. Dual Feasibility** ($\lambda_i \ge 0$)")
                lambdas = [sp.symbols(f'lambda_{i+1}') for i in range(len(parsed_constraints))]
                if lambdas:
                    lam_tex = ", ".join([sp.latex(l) for l in lambdas])
                    st.latex(f"{lam_tex} \ge 0")
                else:
                    st.write("No inequality constraints.")

                # 3. Complementary Slackness
                st.markdown("**III. Complementary Slackness** ($\lambda_i g_i(x) = 0$)")
                for i, (lam, g) in enumerate(zip(lambdas, parsed_constraints)):
                    st.latex(f"\\lambda_{{{i+1}}} \\cdot ({sp.latex(g)}) = 0")
                
                # 4. Stationarity
                st.markdown("**IV. Stationarity** ($\nabla f + \sum \lambda_i \\nabla g_i = 0$)")
                st.write("The gradient of the Lagrangian must be zero:")
                
                L = f_expr + sum(lam * g for lam, g in zip(lambdas, parsed_constraints))
                
                for var in vars_sym:
                    derivative = sp.diff(L, var)
                    st.latex(f"\\frac{{\partial L}}{{\partial {var.name}}} = {sp.latex(derivative)} = 0")
            else:
                st.error("Please fix input errors above first.")

    # ============================================================
    # TAB 2: NUMERICAL VERIFICATION
    # ============================================================
    with tab_num:
        st.write("Checks if a specific point (e.g., from an exam question) is optimal.")
        candidate_str = utils.p_text_input("Candidate Point (e.g. 2, 0):", "kkt_master_cand", "2, 0")
        
        if st.button("Check Optimality", type="primary"):
            if valid_input and f_expr:
                try:
                    # --- A. Problem Preview ---
                    st.markdown("### 1. Problem Context")
                    st.latex(f"\\min f = {sp.latex(f_expr)}")
                    
                    # --- B. Verification Logic ---
                    vals = [float(x.strip()) for x in candidate_str.split(',')]
                    if len(vals) != len(vars_sym):
                        st.error(f"Dimension mismatch. Expected {len(vars_sym)} variables ({vars_sym}), got {len(vals)}.")
                    else:
                        point_map = dict(zip(vars_sym, vals))
                        
                        st.markdown("### 2. Analysis of Point " + str(tuple(vals)))
                        
                        # Step 1: Constraints
                        feasible = True
                        equality_indices = [] # Instead of "Active"
                        st.markdown("**Step 1: Primal Feasibility ($g_i \le 0$)**")
                        
                        for i, g in enumerate(parsed_constraints):
                            val = float(g.subs(point_map))
                            if val > 1e-5:
                                st.error(f"❌ Constraint {i+1} Violated: {sp.latex(g)} = {val:.4f} > 0")
                                feasible = False
                            elif abs(val) < 1e-5:
                                st.info(f"⚠️ Constraint {i+1} is equality: {sp.latex(g)} = 0. ($\lambda_{{{i+1}}}$ can be $>0$)")
                                equality_indices.append(i)
                            else:
                                st.success(f"✅ Constraint {i+1} is strict inequality: {sp.latex(g)} = {val:.4f} < 0")
                                st.caption(f"By Complementary Slackness, $\lambda_{{{i+1}}}$ must be **0**.")
                        
                        if not feasible:
                            st.error("Conclusion: Point is **NOT Primal Feasible**. Cannot be optimal.")
                        else:
                            # Step 2: Stationarity
                            st.markdown("**Step 2: Stationarity & Dual Feasibility**")
                            st.caption("We solve for the multipliers $\lambda$ in the stationarity equation.")
                            
                            grad_f = [float(sp.diff(f_expr, v).subs(point_map)) for v in vars_sym]
                            
                            # We only solve for lambdas where g_i = 0. The others are FORCED to 0.
                            grad_gs_eq = []
                            for idx in equality_indices:
                                g = parsed_constraints[idx]
                                grad_g = [float(sp.diff(g, v).subs(point_map)) for v in vars_sym]
                                grad_gs_eq.append(grad_g)
                            
                            # Linear System: A.T * lambda = -grad_f
                            if not equality_indices:
                                norm_grad = np.linalg.norm(grad_f)
                                if norm_grad < 1e-5:
                                    st.success("✅ **Stationarity Holds** (Unconstrained local min, all $\lambda=0$).")
                                    st.balloons()
                                else:
                                    st.error(f"❌ **Stationarity Fails**. Gradient is not zero: {grad_f}")
                            else:
                                A = np.array(grad_gs_eq).T
                                b = -np.array(grad_f)
                                
                                lambdas_sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                                reconstructed = A @ lambdas_sol
                                error = np.linalg.norm(reconstructed - b)
                                
                                if error > 1e-5:
                                    st.error(f"❌ **Stationarity Fails**. Gradients of constraints cannot cancel $\\nabla f$. Residual: {error:.4f}")
                                else:
                                    st.success("✅ **Stationarity Holds**.")
                                    
                                    # Step 3: Dual Feasibility
                                    dual_feasible = True
                                    st.write("Checking Dual Feasibility ($\lambda_i \ge 0$):")
                                    
                                    # Map solution back to full list
                                    full_lambdas = [0.0] * len(parsed_constraints)
                                    for local_idx, true_idx in enumerate(equality_indices):
                                        lam_val = lambdas_sol[local_idx]
                                        full_lambdas[true_idx] = lam_val
                                        
                                        if lam_val < -1e-5:
                                            st.error(f"❌ $\lambda_{{{true_idx+1}}} = {lam_val:.4f}$ (Violation: Must be $\ge 0$)")
                                            dual_feasible = False
                                        else:
                                            st.success(f"✅ $\lambda_{{{true_idx+1}}} = {lam_val:.4f}$")
                                    
                                    if dual_feasible:
                                        st.balloons()
                                        st.success("**VERDICT: Point satisfies ALL KKT conditions.**")
                                        st.info("If the problem is Convex and Strictly Feasible, this is a Global Minimum (Theorem 9.34).")
                                    else:
                                        st.warning("**VERDICT: Point is NOT optimal (Violates Dual Feasibility).**")

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Please fix input errors above first.")