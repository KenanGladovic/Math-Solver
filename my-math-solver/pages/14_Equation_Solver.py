import streamlit as st
import sympy as sp
import utils

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Math Solver")
utils.setup_page()

st.markdown("<h1 class='main-header'>Mathematical Solver Suite</h1>", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def format_solution(sol):
    """Formats SymPy solutions for display."""
    if isinstance(sol, list):
        return [format_solution(s) for s in sol]
    elif isinstance(sol, dict):
        return ", ".join([f"{k} = {sp.latex(v)}" for k, v in sol.items()])
    elif isinstance(sol, tuple):
        return sp.latex(sol)
    elif isinstance(sol, sp.Set):
        return sp.latex(sol)
    else:
        return sp.latex(sol)

def explain_chain_rule(expr, var):
    """Generates step-by-step chain rule explanation."""
    steps = []
    if expr.is_Function:
        if len(expr.args) == 1 and not expr.args[0].is_Symbol:
            inner = expr.args[0]
            outer_sym = sp.Symbol('u')
            outer = expr.func(outer_sym)
            du_dx = sp.diff(inner, var)
            dy_du = sp.diff(outer, outer_sym)
            steps.append(f"**Step 1 (Identify)**: Inner $u = {sp.latex(inner)}$, Outer $y = {sp.latex(outer)}$")
            steps.append(f"**Step 2 (Diff)**: $u' = {sp.latex(du_dx)}$, $y' = {sp.latex(dy_du)}$")
            steps.append(f"**Step 3 (Combine)**: $y' \cdot u' = {sp.latex(dy_du.subs(outer_sym, inner))} \cdot {sp.latex(du_dx)}$")
            return steps
    elif expr.is_Pow:
        base, exp = expr.args
        if not base.is_Symbol and base.has(var):
            inner = base
            outer_sym = sp.Symbol('u')
            outer = outer_sym ** exp
            du_dx = sp.diff(inner, var)
            dy_du = sp.diff(outer, outer_sym)
            steps.append(f"**Step 1 (Identify)**: Inner $u = {sp.latex(inner)}$, Power form $u^{{{sp.latex(exp)}}}$")
            steps.append(f"**Step 2 (Diff)**: $u' = {sp.latex(du_dx)}$, Outer deriv = ${sp.latex(dy_du)}$")
            steps.append(f"**Step 3 (Combine)**: ${sp.latex(dy_du.subs(outer_sym, inner))} \cdot {sp.latex(du_dx)}$")
            return steps
    return ["Standard differentiation applied."]

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📐 System & Equation Solver", 
    "📉 Inequalities & Regions", 
    "∫ Calculus & Chain Rule",
    "✅ Solution Verifier"
])

# --- TAB 1: ALGEBRAIC & SYSTEM SOLVER ---
with tab1:
    st.markdown("### Solve Equations & Linear Systems")
    st.caption("Solves polynomial equations, linear systems, and algebraic relations.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        # Default: Maj 2025 Opgave 2 (Stationary Points Condition)
        eq_input = utils.p_text_area(
            "Enter Equations (one per line). Use `==` for equality.", 
            "alg_input", 
            "2*x + y + z == 0\nx + 2*y + z == 0\nx + y + 2*z == 0", 
            height=150
        )
    with col2:
        var_input = utils.p_text_input(
            "Solve for variables (comma separated):", 
            "alg_vars", 
            "x, y, z"
        )
        st.info("💡 **Tip:** Default example is from **Maj 2025 Opgave 2** (Stationary Points).")

    if st.button("Solve Equations", type="primary", key="btn_solve_eq"):
        st.markdown("---")
        try:
            var_strs = [v.strip() for v in var_input.split(",") if v.strip()]
            vars_sym = [sp.symbols(v) for v in var_strs]
            
            raw_lines = [line.strip() for line in eq_input.split('\n') if line.strip()]
            equations = []
            
            for line in raw_lines:
                if "==" in line:
                    lhs_str, rhs_str = line.split("==")
                    lhs, rhs = utils.parse_expr(lhs_str), utils.parse_expr(rhs_str)
                    equations.append(sp.Eq(lhs, rhs))
                else:
                    equations.append(sp.Eq(utils.parse_expr(line), 0))

            solution = sp.solve(equations, vars_sym)
            
            st.subheader("Results")
            if not solution:
                st.warning("No solutions found.")
            else:
                formatted = format_solution(solution)
                if isinstance(formatted, list):
                    for fs in formatted:
                        st.latex(fs)
                else:
                    st.latex(formatted)
        except Exception as e:
            st.error(f"Solver Error: {e}")

# --- TAB 2: INEQUALITIES ---
with tab2:
    st.markdown("### Find Region Boundaries")
    st.caption("Solves for the **boundaries** of an inequality. E.g., if you have $x^2 + y^2 \le 1$, solving for $y$ gives $y = \pm \sqrt{1-x^2}$.")
    
    col_i1, col_i2 = st.columns([2, 1])
    with col_i1:
        ineq_input = utils.p_text_input("Enter Inequality", "ineq_input", "x**2 + y**2 <= 1")
    with col_i2:
        ineq_var = utils.p_text_input("Solve for variable", "ineq_var", "y")
    
    if st.button("Find Boundaries", type="primary", key="btn_solve_ineq"):
        st.markdown("---")
        try:
            # 1. Parse Expression
            expr = utils.parse_expr(ineq_input)
            target_var = sp.symbols(ineq_var.strip())
            
            # 2. Extract Boundary Equation
            # If it's a relational object (like x < y), get LHS - RHS
            if isinstance(expr, (sp.LessThan, sp.StrictLessThan, sp.GreaterThan, sp.StrictGreaterThan, sp.Le, sp.Lt, sp.Ge, sp.Gt)):
                boundary_eq = sp.Eq(expr.lhs, expr.rhs)
                st.info(f"Inequality detected. Solving boundary equation: ${sp.latex(boundary_eq)}$")
            elif isinstance(expr, sp.Eq):
                boundary_eq = expr
            else:
                # If just an expression, assume = 0
                boundary_eq = sp.Eq(expr, 0)

            # 3. Solve the Equality (Robust)
            solutions = sp.solve(boundary_eq, target_var)
            
            st.subheader("Boundary Solutions")
            if not solutions:
                st.warning("No boundaries found (the region might be empty or all-encompassing, or too complex).")
            else:
                for sol in solutions:
                    st.latex(f"{ineq_var} = {sp.latex(sol)}")
                
                st.markdown("---")
                st.success("✅ These equations define the edge of your feasible region.")
            
        except Exception as e:
            st.error(f"Could not solve for boundaries. \nError: {e}")

# --- TAB 3: CALCULUS ---
with tab3:
    st.markdown("### Differentiation & Step-by-Step Chain Rule")
    
    c_col1, c_col2 = st.columns([2, 1])
    with c_col1:
        expr_str = utils.p_text_input("Enter Function", "calc_expr", "exp((x+1)**2)")
    with c_col2:
        var_str = utils.p_text_input("With respect to", "calc_var", "x")

    if st.button("Compute Derivative", type="primary", key="btn_calc"):
        st.markdown("---")
        try:
            expression = utils.parse_expr(expr_str)
            variable = sp.symbols(var_str)
            
            st.latex(f"\\frac{{d}}{{d{var_str}}} \\left[ {sp.latex(expression)} \\right]")
            
            # Check for Chain Rule
            is_composite = False
            if expression.is_Function or expression.is_Pow:
                if not (len(expression.args) == 1 and expression.args[0] == variable):
                        is_composite = True

            if is_composite:
                st.subheader("Step-by-Step (Chain Rule)")
                steps = explain_chain_rule(expression, variable)
                for step in steps:
                    st.markdown(step)
            
            final_derivative = sp.diff(expression, variable)
            st.subheader("Final Answer")
            st.latex(f"= {sp.latex(sp.simplify(final_derivative))}")
                
        except Exception as e:
            st.error(f"Differentiation Error: {e}")

# --- TAB 4: VERIFIER ---
with tab4:
    st.markdown("### Verify a Solution")
    st.caption("Check if a specific point satisfies an equation or inequality.")
    
    check_eq = utils.p_text_input("Equation / Inequality to check", "check_eq", "x**2 + y**2 <= 4")
    check_point = utils.p_text_input("Point to test (e.g. `x=1, y=1`)", "check_vals", "x=1, y=1")
    
    if st.button("Check Validity", key="btn_verify"):
        try:
            # Parse point string into dict
            val_strs = check_point.split(",")
            subs_dict = {}
            for v in val_strs:
                key, val = v.split("=")
                subs_dict[sp.symbols(key.strip())] = float(val) if '.' in val else int(val)
            
            # Check logic
            if "<" in check_eq or ">" in check_eq:
                # Inequality check
                is_valid = sp.sympify(check_eq).subs(subs_dict)
            else:
                # Equality check
                if "==" in check_eq:
                    l, r = check_eq.split("==")
                    expr = utils.parse_expr(l) - utils.parse_expr(r)
                else:
                    expr = utils.parse_expr(check_eq)
                
                res = expr.subs(subs_dict)
                is_valid = (abs(float(res)) < 1e-9) 
                
            if is_valid:
                st.success(f"✅ The point {check_point} satisfies: ${check_eq}$")
            else:
                st.error(f"❌ The point {check_point} does NOT satisfy: ${check_eq}$")
        except Exception as e:
            st.error(f"Error checking point: {e}")