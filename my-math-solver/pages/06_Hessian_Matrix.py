import streamlit as st
import sympy as sp
import utils

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()

st.markdown("<h1 class='main-header'>Hessian Analysis</h1>", unsafe_allow_html=True)
st.write("Analyze $f(x, y, ...)$ and classify critical points using **Symmetric Reduction** (Diagonalization).")

# --- HELPER: SYMMETRIC REDUCTION ---
def get_diagonal_entries_with_steps(matrix):
    """
    Performs schematic symmetric reduction to find diagonal entries.
    Returns: (Diagonal Elements List, List of LaTeX steps)
    """
    M = matrix.copy()
    n = M.rows
    step_descriptions = []
    
    for i in range(n):
        # 1. Pivot Strategy
        if M[i, i] == 0:
            swap_idx = -1
            for k in range(i + 1, n):
                if M[k, k] != 0:
                    swap_idx = k
                    break
            
            if swap_idx != -1:
                M.row_swap(i, swap_idx)
                M.col_swap(i, swap_idx)
                step_descriptions.append(f"R_{i+1} \\leftrightarrow R_{swap_idx+1}, \\; C_{i+1} \\leftrightarrow C_{swap_idx+1}")
            else:
                # If all diagonals are 0, try to create one using addition
                found_mix = False
                for k in range(i + 1, n):
                    if M[i, k] != 0:
                        M.row_op(i, lambda r, j: r + M.row(k)[j])
                        M.col_op(i, lambda c, j: c + M.col(k)[j])
                        step_descriptions.append(f"R_{i+1} \\leftarrow R_{i+1} + R_{k+1}, \\; C_{i+1} \\leftarrow C_{i+1} + C_{k+1}")
                        found_mix = True
                        break
                if not found_mix and M[i, i] == 0:
                    continue 

        pivot = M[i, i]
        if pivot == 0: continue

        # 2. Elimination
        for j in range(i + 1, n):
            if M[j, i] != 0:
                factor = M[j, i] / pivot
                # Store rational/simplified factor for display
                fac_disp = sp.latex(sp.simplify(factor))
                
                M.row_op(j, lambda r, k: r - factor * M.row(i)[k])
                M.col_op(j, lambda c, k: c - factor * M.col(i)[k])
                
                step_descriptions.append(f"R_{j+1} \\leftarrow R_{j+1} - ({fac_disp})R_{i+1}")

    diags = [M[i, i] for i in range(n)]
    return diags, step_descriptions

# --- MAIN UI ---
col1, col2 = st.columns([1, 1])
with col1:
    func_str = utils.p_text_input("Function f(x, y, ...):", "hess_func", "x**3 + x*y + y**3")
    f = utils.parse_expr(func_str)
    
if f is not None:
    vars_sym = sorted(list(f.free_symbols), key=lambda s: s.name)
    
    with col2:
        st.info(f"**Variables:** ${', '.join([sp.latex(v) for v in vars_sym])}$")
    
    if st.button("Analyze Function", type="primary"):
        st.divider()
        
        # 1. Gradient & Hessian Calculation
        c1, c2 = st.columns(2)
        
        grad = [sp.diff(f, v) for v in vars_sym]
        grad_matrix = sp.Matrix(grad)
        
        hessian = sp.hessian(f, vars_sym)
        
        with c1:
            st.subheader("1. Gradient $\\nabla f$")
            st.latex(sp.latex(grad_matrix))
            
        with c2:
            st.subheader("2. Hessian Matrix $H_f$")
            st.latex(sp.latex(hessian))
            
        st.divider()
        
        # 3. Critical Points Loop
        st.subheader("3. Critical Points & Classification")
        
        try:
            crit_pts = sp.solve(grad, vars_sym, dict=True)
            if not crit_pts:
                st.warning("No critical points found (system might be inconsistent or too complex).")
            else:
                for i, pt in enumerate(crit_pts):
                    # --- Create a "Card" for each point ---
                    with st.container():
                        st.markdown(f"### $\\bullet$ Critical Point {i+1}")
                        
                        # 1. Show Point Coordinates
                        pt_latex = ", ".join([f"{sp.latex(v)} = {sp.latex(val)}" for v, val in pt.items()])
                        st.latex(f"P_{{{i+1}}}: \\quad ({pt_latex})")
                        
                        # 2. Show Hessian at this point
                        H_num = hessian.subs(pt)
                        st.write("Hessian at point:")
                        st.latex(f"H(P_{{{i+1}}}) = {sp.latex(H_num)}")
                        
                        # 3. Analyze Definiteness (if numeric)
                        if not H_num.free_symbols:
                            diags, steps = get_diagonal_entries_with_steps(H_num)
                            D_matrix = sp.diag(*diags)
                            
                            # Layout: Matrix on Left, Steps on Right (if any)
                            r1, r2 = st.columns([1, 2])
                            
                            with r1:
                                st.write("**Symmetric Reduction:**")
                                st.latex(f"D = {sp.latex(D_matrix)}")
                            
                            with r2:
                                if steps:
                                    with st.expander("Show Reduction Steps"):
                                        for s in steps:
                                            st.latex(s)
                                else:
                                    st.caption("Matrix is already diagonal.")
                            
                            # 4. Classification Verdict
                            pos = [d > 0 for d in diags]
                            neg = [d < 0 for d in diags]
                            
                            # Latex for diagonal elements
                            diag_vals = ", ".join([str(d) for d in diags])
                            
                            if all(pos):
                                st.success(f"**Strict Local Minimum**")
                                st.write(f"Explanation: All diagonal entries $d_i > 0$. Matrix is Positive Definite.")
                            elif all(neg):
                                st.success(f"**Strict Local Maximum**")
                                st.write(f"Explanation: All diagonal entries $d_i < 0$. Matrix is Negative Definite.")
                            elif any(pos) and any(neg):
                                st.error(f"**Saddle Point**")
                                st.write(f"Explanation: Diagonal entries have mixed signs. Matrix is Indefinite.")
                            else:
                                st.warning(f"**Inconclusive (Semi-Definite)**")
                                st.write(f"Explanation: Some entries are 0. Further analysis required.")
                                
                        else:
                            st.warning("Hessian depends on unresolved parameters. Cannot simplify automatically.")
                        
                        st.divider()

        except Exception as e:
            st.error(f"Computation Error: {e}")