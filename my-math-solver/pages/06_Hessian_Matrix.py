import streamlit as st
import sympy as sp
import utils
st.set_page_config(layout="wide")
utils.setup_page()
st.markdown("<h1 class='main-header'>Hessian Analysis</h1>", unsafe_allow_html=True)
st.markdown("Analyze a function $f(x, y, ...)$ by calculating its Gradient and Hessian matrix.")

col1, col2 = st.columns([1, 1])
with col1:
    func_str = st.text_input("Function f(x, y, ...):", "x**3 + x*y + y**3")
    f = utils.parse_expr(func_str)
    
if f is not None:
    vars_sym = sorted(list(f.free_symbols), key=lambda s: s.name)
    
    with col2:
        st.write(f"Detected Variables: {', '.join([str(v) for v in vars_sym])}")
    
    if st.button("Calculate Hessian", type="primary"):
        st.markdown("---")
        
        # 1. Gradient
        grad = [sp.diff(f, v) for v in vars_sym]
        grad_matrix = sp.Matrix(grad)
        
        st.subheader("1. Gradient ($\\nabla f$)")
        st.caption("**Definition 7.6** (Gradient Vector)")
        st.latex(sp.latex(grad_matrix))
        
        # 2. Hessian
        hessian = sp.hessian(f, vars_sym)
        st.subheader("2. Hessian Matrix ($H_f$)")
        st.caption("**Definition 8.2** (Hessian Matrix)")
        st.latex(sp.latex(hessian))
        
        st.info("""
        **Curriculum Reference: Theorem 8.12 (Classification)**
        To classify a critical point $P$ where $\\nabla f(P) = 0$:
        * **Local Minimum:** $H_f(P)$ is Positive Definite.
        * **Local Maximum:** $H_f(P)$ is Negative Definite.
        * **Saddle Point:** $H_f(P)$ is Indefinite.
        * **Inconclusive:** $H_f(P)$ is Semi-Definite (Zero eigenvalue).
        
        **Curriculum Reference: Theorem 8.23 (Convexity)**
        * $f$ is **Convex** if $H_f$ is Positive Semi-Definite everywhere.
        * $f$ is **Strictly Convex** if $H_f$ is Positive Definite everywhere.
        """)
        
        # 3. Critical Points
        st.subheader("3. Critical Points ($\\nabla f = 0$)")
        st.caption("**Definition 7.17** (Critical Point)")
        try:
            crit_pts = sp.solve(grad, vars_sym, dict=True)
            if not crit_pts:
                st.write("No critical points found analytically.")
            else:
                for i, pt in enumerate(crit_pts):
                    pt_str = ", ".join([f"{v}: {pt[v]}" for v in vars_sym])
                    st.write(f"**Point {i+1}:** $({pt_str})$")
                    
                    # Show Hessian at this point
                    H_at_pt = hessian.subs(pt)
                    st.latex(f"H(P_{{{i+1}}}) = {sp.latex(H_at_pt)}")
        except Exception as e:
            st.error(f"Solver error: {e}")