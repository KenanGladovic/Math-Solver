import streamlit as st
import sympy as sp
import utils

# 1. SETUP
st.set_page_config(layout="wide", page_title="Hessian Analysis", page_icon="📉")
try:
    utils.setup_page()
except AttributeError:
    pass 

# --- HELPER FUNCTIONS ---

def get_exam_latex(expr):
    """Forces parentheses for matrices (pmatrix) as used in the curriculum."""
    return sp.latex(expr, mat_delim='(')

def show_math_block_with_copy(label, lhs, expr):
    """Displays math and provides a copy button for the code."""
    rhs_tex = get_exam_latex(expr)
    full_eq = f"{lhs} = {rhs_tex}"
    
    st.markdown(f"**{label}**")
    st.latex(full_eq)
    with st.expander(f"📝 Latex Code for {label}"):
        st.code(f"$$ {full_eq} $$", language="latex")

def get_diagonal_entries(matrix):
    """
    Computes the diagonal entries (pivots) d_i via Symmetric Reduction
    (Row/Col operations) to determine definiteness.
    """
    M = matrix.copy()
    n = M.rows
    pivots = []
    
    for i in range(n):
        # 1. Pivot Strategy
        if M[i, i] == 0:
            swap_idx = -1
            # Try to find a non-zero diagonal to swap
            for k in range(i + 1, n):
                if M[k, k] != 0:
                    swap_idx = k
                    break
            
            if swap_idx != -1:
                M.row_swap(i, swap_idx)
                M.col_swap(i, swap_idx)
            else:
                # If all diagonals are 0, add row/col to create pivot
                for k in range(i + 1, n):
                    if M[i, k] != 0:
                        M.row_op(i, lambda r, j: r + M.row(k)[j])
                        M.col_op(i, lambda c, j: c + M.col(k)[j])
                        break
        
        pivot = M[i, i]
        if pivot == 0: 
            pivots.append(0)
            continue
            
        pivots.append(pivot)

        # 2. Elimination (Symmetric)
        for j in range(i + 1, n):
            if M[j, i] != 0:
                factor = M[j, i] / pivot
                M.row_op(j, lambda r, k: r - factor * M.row(i)[k])
                M.col_op(j, lambda c, k: c - factor * M.col(i)[k])
                
    return pivots

# --- MAIN PAGE UI ---

st.title("📉 Hessian Analysis")
st.caption("Analyzes critical points using Symmetric Reduction and Theorem 8.12.")

# 1. INPUT SECTION
with st.container(border=True):
    col1, col2 = st.columns([2, 1])
    with col1:
        func_str = utils.p_text_input("Function $f(x, y, ...)$:", "hess_func_curr", "x**3 - 3*x + y**2")
        f = utils.parse_expr(func_str)
    
    vars_sym = []
    if f is not None:
        vars_sym = sorted(list(f.free_symbols), key=lambda s: s.name)
        vars_str = ", ".join([v.name for v in vars_sym])
        with col2:
            st.info(f"**Variables:** $({vars_str})$")

if f is not None and len(vars_sym) > 0:
    
    # 2. GLOBAL DERIVATIVES
    grad = [sp.diff(f, v) for v in vars_sym]
    grad_matrix = sp.Matrix(grad) 
    hessian = sp.hessian(f, vars_sym)
    v_str = f"({', '.join([v.name for v in vars_sym])})"
    
    st.divider()
    st.subheader("1. Derivatives")
    
    c1, c2 = st.columns(2)
    with c1:
        # Gradient Block with Copy
        show_math_block_with_copy("Gradient Vector", f"\\nabla f{v_str}", grad_matrix)
        
    with c2:
        # Hessian Block with Copy
        show_math_block_with_copy("Hessian Matrix", f"\\nabla^2 f{v_str}", hessian)

    # 3. CRITICAL POINTS ANALYSIS
    st.divider()
    st.subheader("2. Critical Points & Classification")
    st.write("Solving $\\nabla f = 0$...")
    
    # Solve system
    try:
        crit_pts = sp.solve(grad, vars_sym, dict=True)
    except:
        crit_pts = []
        
    if not crit_pts:
        st.warning("No critical points found.")
    else:
        for i, pt in enumerate(crit_pts):
            # --- CARD FOR EACH POINT ---
            with st.container(border=True):
                # Header
                st.markdown(f"#### 📍 Critical Point $P_{{{i+1}}}$")
                
                # Setup Data
                coords = [pt[v] for v in vars_sym]
                coords_tex = f"({', '.join([sp.latex(c) for c in coords])})"
                H_num = hessian.subs(pt)
                
                # Display Coords & Matrix
                xc1, xc2 = st.columns([1, 1])
                with xc1:
                    st.write("**Coordinates:**")
                    st.latex(f"P_{{{i+1}}} = {coords_tex}")
                with xc2:
                    st.write(f"**Hessian $\\nabla^2 f(P_{{{i+1}}})$:**")
                    st.latex(get_exam_latex(H_num))
                
                st.divider()
                
                # --- CLASSIFICATION LOGIC ---
                st.markdown("**🔍 Classification (Theorem 8.12):**")
                
                if not H_num.free_symbols:
                    # 1. Get Diagonals via Symmetric Reduction
                    diags = get_diagonal_entries(H_num)
                    diags_simple = [sp.simplify(d) for d in diags]
                    
                    # 2. Check Signs
                    pos_count = sum(1 for d in diags_simple if d > 0)
                    neg_count = sum(1 for d in diags_simple if d < 0)
                    n = len(diags_simple)
                    
                    diags_tex = ", ".join([sp.latex(d) for d in diags_simple])
                    
                    # 3. Generate Verdict & Argument
                    if pos_count == n:
                        # Positive Definite
                        st.success(f"**✅ Conclusion: Strict Local Minimum**")
                        st.markdown(f"""
                        Using **Symmetric Reduction**, the diagonal entries are $d_i = {diags_tex}$.
                        
                        Since all $d_i > 0$, the matrix is **positive definite**.  
                        By **Theorem 8.12 (i)**, the point $P_{{{i+1}}}$ is a **strict local minimum**.
                        """)
                    
                    elif neg_count == n:
                        # Negative Definite
                        st.success(f"**✅ Conclusion: Strict Local Maximum**")
                        st.markdown(f"""
                        Using **Symmetric Reduction**, the diagonal entries are $d_i = {diags_tex}$.
                        
                        Since all $d_i < 0$, the matrix is **negative definite**.  
                        By **Theorem 8.12 (ii)**, the point $P_{{{i+1}}}$ is a **strict local maximum**.
                        """)
                        
                    elif pos_count > 0 and neg_count > 0:
                        # Indefinite
                        st.error(f"**⚠️ Conclusion: Saddle Point**")
                        st.markdown(f"""
                        Using **Symmetric Reduction**, the diagonal entries are $d_i = {diags_tex}$.
                        
                        Since the diagonal contains both positive and negative entries, the matrix is **indefinite**.  
                        By **Theorem 8.12 (iii)** (and Definition 8.9), the point $P_{{{i+1}}}$ is a **saddle point**.
                        """)
                        
                    else:
                        # Semi-Definite / Inconclusive
                        st.warning(f"**❓ Conclusion: Inconclusive (Semi-Definite)**")
                        st.markdown(f"""
                        The diagonal entries are $d_i = {diags_tex}$.
                        
                        Some entries are zero, making the matrix **semi-definite**.  
                        **Theorem 8.12** does not provide a conclusion for this case (further analysis required).
                        """)
                        
                else:
                    st.warning("Hessian contains parameters. Cannot classify numerically.")