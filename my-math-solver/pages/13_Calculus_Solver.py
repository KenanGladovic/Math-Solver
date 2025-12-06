import streamlit as st
import sympy as sp
import utils
st.set_page_config(layout="wide")
utils.setup_page()
st.markdown("<h1 class='main-header'>Calculus Assistant</h1>", unsafe_allow_html=True)

calc_mode = utils.p_radio("Operation", ["Derivative", "Limit"], "calc_mode", horizontal=True)
expr_input = utils.p_text_input("Expression (in terms of x):", "calc_expr", "x**2 * sin(x)")

expr = utils.parse_expr(expr_input)
x = sp.symbols('x')

if calc_mode == "Derivative":
    order_str = utils.p_text_input("Order of Derivative (Enter an integer):", "calc_order", "1")
    if st.button("Differentiate", type="primary"):
        if expr is None: st.error("Invalid expression.")
        else:
            try:
                order = int(order_str)
                res = sp.diff(expr, x, order)
                st.markdown("### Result")
                st.latex(f"\\frac{{d^{order}}}{{dx^{order}}} ({sp.latex(expr)}) = {sp.latex(res)}")
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
                st.latex(f"\\lim_{{x \\to {sp.latex(target_expr)}}} ({sp.latex(expr)}) = {sp.latex(res)}")
            except Exception as e: st.error(f"Error: {e}")