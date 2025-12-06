import streamlit as st
import sympy as sp
import utils
st.set_page_config(layout="wide")
utils.setup_page()
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
    ineq_text = utils.p_text_area("System of Inequalities:", "fm_ineqs", default_ineqs, height=200)

with col2:
    st.subheader("2. Solver Settings")
    
    # Auto-detect variables from input
    dummy_expr = utils.parse_expr("0") 
    detected_vars = set()
    for line in ineq_text.split('\n'):
        if line.strip():
            for part in line.replace('<',' ').replace('>',' ').replace('=',' ').split():
                try:
                    sym = utils.parse_expr(part)
                    if sym: detected_vars.update(sym.free_symbols)
                except: pass
    
    sorted_vars = sorted([s.name for s in detected_vars])
    
    if not sorted_vars:
        st.warning("No variables detected yet.")
        target_var = None
    else:
        st.markdown(f"**Detected Variables:** `{', '.join(sorted_vars)}`")
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
                inequalities.append(utils.parse_expr(lhs) - utils.parse_expr(rhs))
            elif ">=" in line:
                lhs, rhs = line.split(">=")
                inequalities.append(utils.parse_expr(rhs) - utils.parse_expr(lhs))
            elif "<" in line:
                lhs, rhs = line.split("<")
                inequalities.append(utils.parse_expr(lhs) - utils.parse_expr(rhs))
            elif ">" in line:
                lhs, rhs = line.split(">")
                inequalities.append(utils.parse_expr(rhs) - utils.parse_expr(lhs))

        # 2. Determine Elimination Order
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
                    coeff = expr.coeff(elim_var)
                    rest = expr - coeff * elim_var
                    
                    if coeff == 0:
                        others.append(expr)
                    elif coeff > 0:
                        upper_bounds.append(-rest / coeff)
                    elif coeff < 0:
                        lower_bounds.append(-rest / coeff)
                
                st.markdown(f"* **Lower Bounds ($L \\le {elim_var_name}$):** {len(lower_bounds)}")
                if lower_bounds: st.latex(", ".join([sp.latex(b) for b in lower_bounds]) + f" \le {elim_var_name}")
                
                st.markdown(f"* **Upper Bounds (${elim_var_name} \\le U$):** {len(upper_bounds)}")
                if upper_bounds: st.latex(f"{elim_var_name} \le " + ", ".join([sp.latex(b) for b in upper_bounds]))
                
                new_constraints = []
                for lb in lower_bounds:
                    for ub in upper_bounds:
                        new_constraints.append(lb - ub)
                
                new_constraints.extend(others)
                
                simplified_constraints = []
                for c in new_constraints:
                    simp = sp.simplify(c)
                    if simp == True or (simp.is_number and simp <= 0): continue
                    if simp == False or (simp.is_number and simp > 0):
                        st.error(f"**Contradiction Found!** Derived constraint ${sp.latex(simp)} \le 0$ is impossible.")
                        st.stop()
                    simplified_constraints.append(simp)
                
                current_ineqs = list(set(simplified_constraints))
                st.write(f"**Resulting constraints:** {len(current_ineqs)}")

        # 3. Final Results
        st.markdown("---")
        st.subheader(f"3. Final Bounds for ${target_var}$")
        st.caption(f"This corresponds to the inequalities derived in **(4.14)** before back-substitution.")
        
        final_lower = []
        final_upper = []
        t_sym = sp.symbols(target_var)
        
        for expr in current_ineqs:
            coeff = expr.coeff(t_sym)
            rest = expr - coeff * t_sym
            
            if coeff == 0:
                if rest > 0: st.error("System is Infeasible!")
            elif coeff > 0:
                final_upper.append(-rest/coeff)
            elif coeff < 0:
                final_lower.append(-rest/coeff)
        
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Lower Bounds ($L \le z$)**")
            if final_lower:
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
                try:
                    nums = [float(val) for val in final_upper if val.is_number]
                    if nums: st.info(f"Tightest numerical upper bound: **{min(nums)}**")
                except: pass
                for val in final_upper: st.latex(f"{target_var} \le {sp.latex(val)}")
            else:
                st.write("None ($+\infty$)")

    except Exception as e:
        st.error(f"Error: {e}")