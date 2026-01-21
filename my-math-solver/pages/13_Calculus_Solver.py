import streamlit as st
import sympy as sp
import utils

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()

st.markdown("<h1 class='main-header'>Calculus & Optimization Assistant</h1>", unsafe_allow_html=True)

# 2. Mode Selection
tool_mode = st.radio(
    "Select Mode:", 
    ["Single Variable (Diff/Limit)", "Multivariate Optimization (Global Min/Convexity)"], 
    horizontal=True
)

st.markdown("---")

# ============================================================
# MODE A: SINGLE VARIABLE (Legacy Tools)
# ============================================================
if tool_mode == "Single Variable (Diff/Limit)":
    col1, col2 = st.columns([1, 1])
    with col1:
        calc_mode = utils.p_radio("Operation", ["Derivative", "Limit"], "calc_mode_single")
        expr_input = utils.p_text_input("Expression (in terms of x):", "calc_expr_single", "x**2 * sin(x)")

    expr = utils.parse_expr(expr_input)
    x = sp.symbols('x')

    with col2:
        if calc_mode == "Derivative":
            order_str = utils.p_text_input("Order of Derivative (n):", "calc_order", "1")
            if st.button("Differentiate", type="primary"):
                if expr is None: st.error("Invalid expression.")
                else:
                    try:
                        order = int(order_str)
                        res = sp.diff(expr, x, order)
                        st.markdown("### Result")
                        st.latex(f"\\frac{{d^{order}}}{{dx^{order}}} \\left( {sp.latex(expr)} \\right) = {sp.latex(res)}")
                    except Exception as e: st.error(f"Error: {e}")

        elif calc_mode == "Limit":
            target = utils.p_text_input("Limit as x approaches:", "calc_limit_target", "0")
            if st.button("Calculate Limit", type="primary"):
                if expr is None: st.error("Invalid expression.")
                else:
                    try:
                        target_expr = utils.parse_expr(target)
                        res = sp.limit(expr, x, target_expr)
                        st.markdown("### Result")
                        st.latex(f"\\lim_{{x \\to {sp.latex(target_expr)}}} \\left( {sp.latex(expr)} \\right) = {sp.latex(res)}")
                    except Exception as e: st.error(f"Error: {e}")

# ============================================================
# MODE B: MULTIVARIATE OPTIMIZATION
# ============================================================
else:
    st.subheader("Multivariate Global Optimization")
    
    # Input
    f_input = utils.p_text_input(
        "Enter function $f(x, y, z, ...)$:", 
        "calc_f_multi", 
        "(1 - x - y - 2*z)**2 + (1 - x - 2*y - 3*z)**2 + (t - y - 2*z)**2"
    )

    f_expr = utils.parse_expr(f_input)

    if f_expr:
        vars_sym = sorted(list(f_expr.free_symbols), key=lambda s: s.name)
        st.write(f"**Detected Variables:** {', '.join([str(v) for v in vars_sym])}")
        
        st.markdown("### Global Minima & Convexity Analysis")
        
        if st.button("Analyze Function", type="primary"):
            with st.spinner("Solving system..."):
                # 1. Gradient Calculation
                grad = [sp.diff(f_expr, v) for v in vars_sym]
                
                # 2. Solve Gradient = 0
                solutions = sp.solve(grad, vars_sym, dict=True)
                
                # --- GENERATE "SVAR" (ANSWER) ---
                st.markdown("#### SVAR (Analysis):")
                
                st.write(f"We examine the function $f({', '.join([v.name for v in vars_sym])})$.")
                
                if not solutions:
                    st.warning("No critical points found (or solver failed). Cannot determine global minimum easily.")
                else:
                    sol = solutions[0] # Take the generic solution
                    
                    # Identify Free Variables (Parameters)
                    known_vars = set(sol.keys())
                    all_vars_set = set(vars_sym)
                    
                    # Symbols appearing in the solution values (RHS)
                    rhs_symbols = set()
                    for val in sol.values():
                        rhs_symbols.update(val.free_symbols)
                        
                    # Missing variables are also free
                    missing_vars = all_vars_set - known_vars
                    
                    free_vars = list(rhs_symbols.union(missing_vars))
                    free_vars = sorted(free_vars, key=lambda s: s.name)

                    # --- LOGIC OUTPUT ---
                    
                    # 1. Minimum Existence
                    st.markdown("**1. Global Minimum:**")
                    val_at_sol = f_expr.subs(sol).simplify()
                    
                    if val_at_sol == 0:
                        st.write("Since $f$ is a sum of squares, $f \ge 0$.")
                        st.write("Solving $\\nabla f = 0$ gives a value of $0$. Thus, these are **Global Minima**.")
                    else:
                        st.write(f"Solving $\\nabla f = 0$ gives critical value: {sp.latex(val_at_sol)}")

                    # 2. Solution Set Description
                    st.markdown("**2. Solution Set:**")
                    if free_vars:
                        st.write(f"The linear system corresponding to $\\nabla f = 0$ has **infinitely many solutions**.")
                        st.write(f"The solution depends on the free variable(s): ${', '.join([str(v) for v in free_vars])}$.")
                        
                        # Construct the parametric point string
                        pt_defs = []
                        for v in vars_sym:
                            if v in sol:
                                pt_defs.append(f"{v.name} = {sp.latex(sol[v])}")
                            else:
                                pt_defs.append(f"{v.name} = {v.name}") # It's free
                        
                        st.latex(f"\\text{{Solution Set: }} \\{{ ({', '.join([v.name for v in vars_sym])}) \\mid {', '.join(pt_defs)} \\}}")
                        
                        st.markdown(f"For every value of the free variables (e.g., ${free_vars[0].name}$), there is a solution.")
                    else:
                        st.write("There is exactly **one unique solution** to the system.")
                        st.latex(f"P = {tuple(sol.values())}")

                    # 3. Strict Convexity Conclusion
                    st.markdown("**3. Strict Convexity Verdict:**")
                    
                    if free_vars:
                        st.error(f"**Conclusion:** $f$ is NOT strictly convex.")
                        
                        # THEOREM BOX
                        st.info("""
                        **Reasoning (Curriculum Logic):**
                        A strictly convex function can have **at most one** global minimum (Theorem regarding Strict Convexity & Uniqueness).
                        
                        Since we found infinitely many global minima (a solution depending on free variables), 
                        $f$ **cannot** be strictly convex.
                        """)
                    else:
                        # If unique, we need to check Hessian to be sure it's strictly convex (could be just convex)
                        hessian = [[sp.diff(f_expr, v1, v2) for v1 in vars_sym] for v2 in vars_sym]
                        H = sp.Matrix(hessian)
                        
                        # Check Positive Definiteness via Leading Principal Minors
                        is_pos_def = True
                        for k in range(1, len(vars_sym) + 1):
                            if H[:k, :k].det() <= 0: is_pos_def = False
                        
                        if is_pos_def:
                            st.success("**Conclusion:** $f$ IS Strictly Convex.")
                            st.write("Reason: The Hessian is Positive Definite everywhere (Unique Global Minimum).")
                        else:
                            st.warning("**Conclusion:** $f$ is likely Convex but NOT Strictly Convex (or Saddle).")
    else:
        st.info("Enter a function to begin.")