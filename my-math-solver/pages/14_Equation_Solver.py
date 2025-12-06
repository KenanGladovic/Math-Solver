import streamlit as st
import sympy as sp
import utils
st.set_page_config(layout="wide")
utils.setup_page()
st.markdown("<h1 class='main-header'>Equation Solver</h1>", unsafe_allow_html=True)

eq_input = utils.p_text_input("Enter Equation (use `==` for equality, e.g. `x**2 - 4 == 0`):", "eq_input", "x**2 - 5*x + 6")
st.caption("Note: If you don't type `==`, it assumes `= 0`.")

solve_for = utils.p_text_input("Solve for variable:", "eq_var", "x")

if st.button("Solve", type="primary"):
    try:
        var_sym = sp.symbols(solve_for)
        if "==" in eq_input:
            lhs_str, rhs_str = eq_input.split("==")
            lhs = utils.parse_expr(lhs_str)
            rhs = utils.parse_expr(rhs_str)
            eq = sp.Eq(lhs, rhs)
        else:
            eq = utils.parse_expr(eq_input) 
        
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