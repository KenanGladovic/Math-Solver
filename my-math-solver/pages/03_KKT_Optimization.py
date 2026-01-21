import streamlit as st
import sympy as sp
import numpy as np
import utils  # Assumes you have your utils.py file

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()

st.markdown("<h1 class='main-header'>KKT Computation & Proof Generator (v2)</h1>", unsafe_allow_html=True)

# 2. Curriculum Context
with st.expander("📘 Curriculum References (Chapter 9)", expanded=False):
    st.markdown("""
    * **Definition 9.24 (KKT Conditions):** Necessary conditions for optimality.
    * **Equalities vs Inequalities:** * Inequalities ($g \le 0$) require $\lambda \ge 0$.
        * Equalities ($h = 0$) have multipliers $\lambda$ with **unrestricted sign**.
    * **Theorem 9.34:** If convex and strictly feasible, KKT conditions are sufficient.
    """)

# 3. Unified Input Section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Problem Definition")
    # Default tailored to IMOMaj25 Opgave 2
    obj_input = utils.p_text_input("Minimize Objective $f(x,y,...)$:", "kkt_master_obj", "x**2 + x*y + y**2 + x*z + z**2 + y*z")
    
    default_const = "x + y + 2*z = 1"
    constraints_input = utils.p_text_area("Constraints (one per line):", "kkt_master_const", default_const, height=150)
    
    st.info("💡 **Tip:** Supports both inequality (`<=`, `>=`) and equality (`=`) constraints.")

with col2:
    st.subheader("2. Analysis Mode")
    
    # --- PARSING LOGIC ---
    f_expr = utils.parse_expr(obj_input)
    raw_lines = [l.strip() for l in constraints_input.split('\n') if l.strip()]
    parsed_constraints = []
    valid_input = True
    
    if f_expr:
        vars_sym = sorted(list(f_expr.free_symbols), key=lambda s: s.name)
        
        for c_str in raw_lines:
            try:
                # Determine type of constraint
                if "<=" in c_str:
                    lhs, rhs = c_str.split("<=")
                    g = utils.parse_expr(lhs) - utils.parse_expr(rhs)
                    ctype = 'ineq' # g <= 0
                elif ">=" in c_str:
                    lhs, rhs = c_str.split(">=")
                    g = utils.parse_expr(rhs) - utils.parse_expr(lhs) # Flip to g <= 0
                    ctype = 'ineq'
                elif "=" in c_str:
                    lhs, rhs = c_str.split("=")
                    g = utils.parse_expr(lhs) - utils.parse_expr(rhs)
                    ctype = 'eq'   # g = 0
                else:
                    st.error(f"Invalid syntax: '{c_str}'. Use <=, >=, or =.")
                    valid_input = False
                    continue
                
                parsed_constraints.append({'expr': g, 'type': ctype, 'orig': c_str})
            except Exception as e:
                st.error(f"Could not parse '{c_str}': {e}")
                valid_input = False
    else:
        valid_input = False

    # Tabs
    tab_sym, tab_num = st.tabs(["📝 Generate Conditions (Symbolic)", "✅ Verify Candidate (Detailed Report)"])

    # ============================================================
    # TAB 1: SYMBOLIC GENERATION
    # ============================================================
    with tab_sym:
        if st.button("Generate KKT Conditions", type="primary"):
            if valid_input and f_expr:
                latex_report = [] 
                
                def log_latex(latex_str, label=None):
                    if label:
                        st.markdown(label)
                        latex_report.append(f"% {label}")
                    st.latex(latex_str)
                    latex_report.append(f"$$ {latex_str} $$")

                # 1. Problem Statement
                st.markdown("### 1. Problem Statement")
                latex_report.append("\\section*{1. Problem Statement}")
                st.latex(f"\\min f = {sp.latex(f_expr)}")
                
                st.write("Subject to:")
                for i, item in enumerate(parsed_constraints):
                    op = "=" if item['type'] == 'eq' else "\\le"
                    log_latex(f"g_{{{i+1}}}: {sp.latex(item['expr'])} {op} 0")
                st.markdown("---")

                # 2. KKT Conditions
                st.markdown("### 2. The KKT Conditions")
                latex_report.append("\n\\section*{2. The KKT Conditions}")
                
                lambdas = [sp.symbols(f'lambda_{i+1}') for i in range(len(parsed_constraints))]
                
                # Condition 1: Multiplier Signs
                st.markdown("**1. Multiplier Signs**")
                ineq_lambdas = [lam for lam, item in zip(lambdas, parsed_constraints) if item['type'] == 'ineq']
                eq_lambdas = [lam for lam, item in zip(lambdas, parsed_constraints) if item['type'] == 'eq']
                
                if ineq_lambdas:
                    lam_str = ", ".join([sp.latex(l) for l in ineq_lambdas])
                    log_latex(f"{lam_str} \\ge 0")
                if eq_lambdas:
                    lam_str = ", ".join([sp.latex(l) for l in eq_lambdas])
                    st.write(f"${lam_str}$ unrestricted (Equality constraints).")
                    latex_report.append(f"$$ {lam_str} \\text{{ unrestricted}} $$")
                
                # Condition 2: Feasibility
                st.markdown("**2. Primal Feasibility**")
                for i, item in enumerate(parsed_constraints):
                    op = "=" if item['type'] == 'eq' else "\\le"
                    log_latex(f"{sp.latex(item['expr'])} {op} 0")

                # Condition 3: Complementary Slackness
                st.markdown("**3. Complementary Slackness**")
                for i, (lam, item) in enumerate(zip(lambdas, parsed_constraints)):
                    # Strictly speaking, for equalities this is trivial, but standard form often lists it
                    log_latex(f"\\lambda_{{{i+1}}} ({sp.latex(item['expr'])}) = 0")

                # Condition 4: Gradient Equation
                st.markdown("**4. Gradient Equation** ($\\nabla \\mathcal{L} = 0$)")
                L = f_expr
                for lam, item in zip(lambdas, parsed_constraints):
                    L += lam * item['expr']
                
                for var in vars_sym:
                    eq = sp.diff(L, var)
                    log_latex(f"{sp.latex(eq)} = 0 \\quad (\\partial / \\partial {var.name})")

                # Output
                st.markdown("### 📋 LaTeX Code")
                st.code("\n".join(latex_report), language="latex")
            else:
                st.error("Fix input errors first.")

    # ============================================================
    # TAB 2: NUMERICAL VERIFICATION
    # ============================================================
    with tab_num:
        candidate_str = utils.p_text_input("Candidate Point (e.g. 0, 0, 0.5):", "kkt_master_cand", "0, 0, 0.5")
        
        if st.button("Verify Candidate", type="primary"):
            if valid_input and f_expr:
                try:
                    # Parse Point
                    vals = [float(x.strip()) for x in candidate_str.split(',')]
                    if len(vals) != len(vars_sym):
                        st.error(f"Expected {len(vars_sym)} coordinates, got {len(vals)}.")
                    else:
                        latex_report = []
                        point_map = dict(zip(vars_sym, vals))
                        point_tex = tuple(vals)
                        
                        st.markdown(f"### Analysis of Point $v_0 = {point_tex}$")
                        latex_report.append(f"\\section*{{Analysis of Candidate $v_0 = {point_tex}$}}")

                        # --- CALC VALUES ---
                        grad_f_vals = [float(sp.diff(f_expr, v).subs(point_map)) for v in vars_sym]
                        
                        grad_g_vals = []
                        g_vals = []
                        
                        for item in parsed_constraints:
                            g_expr = item['expr']
                            g_val = float(g_expr.subs(point_map))
                            g_vals.append(g_val)
                            grad_g_vals.append([float(sp.diff(g_expr, v).subs(point_map)) for v in vars_sym])

                        # --- STEP 1: SOLVE LAMBDAS ---
                        # Constraint is active if:
                        # 1. Type is 'eq' (ALWAYS active)
                        # 2. Type is 'ineq' AND |val| approx 0
                        active_indices = []
                        for i, item in enumerate(parsed_constraints):
                            if item['type'] == 'eq':
                                active_indices.append(i)
                            elif abs(g_vals[i]) < 1e-4:
                                active_indices.append(i)
                        
                        if active_indices:
                            A_active = np.array([grad_g_vals[i] for i in active_indices]).T
                            b_vec = -np.array(grad_f_vals)
                            # Solve linear system
                            lam_active_res, residuals, _, _ = np.linalg.lstsq(A_active, b_vec, rcond=None)
                        else:
                            lam_active_res = []

                        lambdas_calc = [0.0] * len(parsed_constraints)
                        for i, idx in enumerate(active_indices):
                            lambdas_calc[idx] = lam_active_res[i]

                        st.info(f"**Calculated Multipliers:** Found {len(active_indices)} active constraints.")
                        lam_res_str = ", ".join([f"\\lambda_{{{i+1}}} = {lam:.4f}" for i, lam in enumerate(lambdas_calc)])
                        st.latex(lam_res_str)
                        latex_report.append(f"Calculated Multipliers: $$ {lam_res_str} $$")
                        st.markdown("---")

                        # --- STEP 2: VERIFICATION ---
                        
                        # Check 1: Multiplier Signs
                        st.markdown("#### 1. Multiplier Signs")
                        dual_pass = True
                        for i, lam in enumerate(lambdas_calc):
                            ctype = parsed_constraints[i]['type']
                            if ctype == 'ineq':
                                check = "\\ge" if lam >= -1e-5 else "<"
                                msg = f"\\lambda_{{{i+1}}} = {lam:.4f} {check} 0"
                                if lam < -1e-5: dual_pass = False
                            else:
                                check = "\\in"
                                msg = f"\\lambda_{{{i+1}}} = {lam:.4f} \\in \\mathbb{{R}} \\quad (\\text{{Equality}})"
                            
                            st.latex(msg)
                            latex_report.append(f"$$ {msg} $$")
                        
                        if dual_pass: st.success("Condition Satisfied.")
                        else: st.warning("Condition Failed (Negative multiplier for inequality).")

                        # Check 2: Feasibility
                        st.markdown("#### 2. Feasibility")
                        primal_pass = True
                        for i, val in enumerate(g_vals):
                            ctype = parsed_constraints[i]['type']
                            if ctype == 'ineq':
                                check = "\\le" if val <= 1e-5 else ">"
                                msg = f"g_{{{i+1}}}(v_0) = {val:.4f} {check} 0"
                                if val > 1e-5: primal_pass = False
                            else:
                                check = "\\approx" if abs(val) < 1e-4 else "\\ne"
                                msg = f"g_{{{i+1}}}(v_0) = {val:.4f} {check} 0"
                                if abs(val) > 1e-4: primal_pass = False
                            
                            st.latex(msg)
                            latex_report.append(f"$$ {msg} $$")

                        if primal_pass: st.success("Condition Satisfied.")
                        else: st.warning("Condition Failed.")

                        # Check 3: Slackness (Product)
                        st.markdown("#### 3. Complementary Slackness")
                        slack_pass = True
                        for i, (lam, val) in enumerate(zip(lambdas_calc, g_vals)):
                            prod = lam * val
                            check = "\\approx" if abs(prod) < 1e-4 else "\\ne"
                            msg = f"\\lambda_{{{i+1}}} g_{{{i+1}}} = {prod:.4f} {check} 0"
                            st.latex(msg)
                            latex_report.append(f"$$ {msg} $$")
                            if abs(prod) > 1e-4: slack_pass = False
                        
                        if slack_pass: st.success("Condition Satisfied.")
                        else: st.warning("Condition Failed.")

                        # Check 4: Gradient (Stationarity)
                        st.markdown("#### 4. Gradient Equation")
                        grad_L = np.array(grad_f_vals)
                        for i, lam in enumerate(lambdas_calc):
                            grad_L += lam * np.array(grad_g_vals[i])
                        
                        norm_L = np.linalg.norm(grad_L)
                        msg = f"|\\nabla \\mathcal{{L}}| = {norm_L:.5f}"
                        st.latex(msg)
                        latex_report.append(f"$$ {msg} $$")
                        
                        stat_pass = (norm_L < 1e-4)
                        if stat_pass: st.success("Condition Satisfied.")
                        else: st.error("Condition Failed (Gradient not zero).")

                        # --- FINAL ---
                        st.markdown("---")
                        st.subheader("Final Verdict")
                        if dual_pass and primal_pass and slack_pass and stat_pass:
                            # NO BALLOONS as requested
                            msg = f"✅ The point $v_0 = {point_tex}$ satisfies all KKT conditions."
                            st.success(msg)
                            st.write("If the problem is Convex, this is a Global Minimum.")
                            latex_report.append(f"\\section*{{Conclusion}}\n{msg}")
                        else:
                            msg = f"❌ The point $v_0 = {point_tex}$ fails one or more conditions."
                            st.error(msg)
                            latex_report.append(f"\\section*{{Conclusion}}\n{msg}")

                        st.markdown("### 📋 LaTeX Code")
                        st.code("\n".join(latex_report), language="latex")

                except Exception as e:
                    st.error(f"Error during verification: {e}")
            else:
                st.error("Fix input errors first.")