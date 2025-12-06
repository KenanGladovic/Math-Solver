import streamlit as st
import sympy as sp
import numpy as np
import utils
st.set_page_config(layout="wide")
utils.setup_page()
st.markdown(r"<h1 class='main-header'>Symmetric Diagonalization B<sup>T</sup> A B = D </h1>", unsafe_allow_html=True)

st.info("""
**Curriculum Reference: Section 8.7 (Schematic Procedure)**
This tool finds an invertible matrix $B$ and a diagonal matrix $D$ such that $B^T A B = D$.
It uses the schematic procedure of transforming $\\begin{pmatrix} I \\\\ A \\end{pmatrix} \\to \\begin{pmatrix} B \\\\ D \\end{pmatrix}$.
""")

# Template Selection
template_size = utils.p_selectbox("Select Template Size:", ["2x2", "3x3", "4x4"], "sym_template")

default_text = "[[1, 2], [2, 8]]"
if template_size == "3x3":
    default_text = "[[1, 2, 3], [2, 8, 4], [3, 4, 16]]" # Example 8.30
elif template_size == "4x4":
    default_text = "[[0, 0, 1, 1], [0, 0, 2, 3], [1, 2, 1, 4], [1, 3, 4, 0]]" # Example 8.31

mat_input = utils.p_text_area("Input Symmetric Matrix A:", "sym_matrix", default_text, height=150)

# --- HELPER FUNCTIONS ---
def clean_matrix(M):
    M_clean = M.copy()
    M_clean[np.abs(M_clean) < 1e-10] = 0
    return M_clean

def format_aug_latex(aug, n):
    B_part = sp.Matrix(clean_matrix(aug[:n, :]))
    A_part = sp.Matrix(clean_matrix(aug[n:, :]))
    return r"\begin{pmatrix} " + sp.latex(B_part).replace(r"\left[", "").replace(r"\right]", "") + r" \\ \hline " + sp.latex(A_part).replace(r"\left[", "").replace(r"\right]", "") + r" \end{pmatrix}"

def diagonalize_symmetric_algo(A_in):
    A = np.array(A_in, dtype=float)
    n = A.shape[0]
    
    if not np.allclose(A, A.T):
        return None, None, [], "Error: Matrix is not symmetric!"
        
    history = []
    aug = np.vstack([np.eye(n), A.copy()])
    history.append(("**Initial Setup:** Form Augmented Matrix", format_aug_latex(aug, n)))
    
    for k in range(n):
        pivot = aug[n + k, k]
        
        # --- 1. Pivot Handling (if zero) ---
        if np.isclose(pivot, 0):
            swap_idx = -1
            for j in range(k + 1, n):
                if not np.isclose(aug[n + j, j], 0):
                    swap_idx = j
                    break
            
            if swap_idx != -1:
                desc = f"**Pivot Fix (Swap):** Swap col/row {k+1} $\\leftrightarrow$ {swap_idx+1}"
                aug[:, [k, swap_idx]] = aug[:, [swap_idx, k]]
                aug[[n + k, n + swap_idx], :] = aug[[n + swap_idx, n + k], :]
                history.append((desc, format_aug_latex(aug, n)))
                pivot = aug[n + k, k]
            else:
                add_idx = -1
                for j in range(k + 1, n):
                    if not np.isclose(aug[n + k, j], 0):
                        add_idx = j
                        break
                if add_idx != -1:
                    desc = f"**Pivot Fix (Add):** Col/Row {k+1} $\\leftarrow$ Col/Row {k+1} + Col/Row {add_idx+1}"
                    aug[:, k] += aug[:, add_idx]
                    aug[n + k, :] += aug[n + add_idx, :]
                    history.append((desc, format_aug_latex(aug, n)))
                    pivot = aug[n + k, k]

        if np.isclose(pivot, 0): continue

        # --- 2. Elimination ---
        for j in range(k + 1, n):
            target = aug[n + k, j]
            if not np.isclose(target, 0):
                m = target / pivot
                desc = f"**Eliminate $(A)_{{{k+1},{j+1}}}$:** $C_{j+1} - {m:.3g} C_{k+1}$ / $R_{j+1} - {m:.3g} R_{k+1}$"
                aug[:, j] -= m * aug[:, k]
                aug[n + j, :] -= m * aug[n + k, :]
                history.append((desc, format_aug_latex(aug, n)))

    B = clean_matrix(aug[:n, :])
    D = clean_matrix(aug[n:, :])
    return B, D, history, None

# --- MAIN EXECUTION ---
if st.button("Diagonalize", type="primary"):
    try:
        A_list = eval(mat_input)
        A_arr = np.array(A_list)
        B_res, D_res, history, err = diagonalize_symmetric_algo(A_arr)
        
        if err:
            st.error(err)
        else:
            st.subheader("1. Results")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Original Matrix ($A$)**")
                st.latex(sp.latex(sp.Matrix(A_arr)))
            with c2:
                st.markdown("**Transformation Matrix ($B$)**")
                st.latex(sp.latex(sp.Matrix(B_res)))
            with c3:
                st.markdown("**Diagonal Matrix ($D$)**")
                st.latex(sp.latex(sp.Matrix(D_res)))

            st.subheader("2. Verification")
            calculated_D = B_res.T @ A_arr @ B_res
            calculated_D = clean_matrix(calculated_D)
            
            c_ver1, c_ver2 = st.columns([2, 1])
            with c_ver1:
                st.markdown("We check if $B^T A B$ equals $D$:")
                st.latex(r"B^T A B = " + sp.latex(sp.Matrix(calculated_D)))
                if np.allclose(calculated_D, D_res):
                    st.success("✅ **Verified:** $B^T A B = D$")
                else:
                    st.error("❌ **Verification Failed:** Result does not match D.")

            with c_ver2:
                st.markdown("**Definiteness (from $D$)**")
                diag = np.diag(D_res)
                if all(d > 0 for d in diag): st.info("Positive Definite (+ + +)")
                elif all(d < 0 for d in diag): st.info("Negative Definite (- - -)")
                elif all(d >= 0 for d in diag): st.info("Positive Semi-Definite (+ 0 +)")
                elif all(d <= 0 for d in diag): st.info("Negative Semi-Definite (- 0 -)")
                else: st.info("Indefinite (+ - +)")

            st.markdown("---")
            st.subheader("3. Step-by-Step Computations")
            st.caption(f"Following **Schematic Procedure (Section 8.7)**. Top block tracks $B$, bottom block tracks $A \\to D$.")
            
            for step_num, (desc, latex) in enumerate(history):
                with st.expander(f"Step {step_num}: {desc.split('**')[1] if '**' in desc else 'Setup'}", expanded=False):
                    st.markdown(desc)
                    st.latex(latex)

    except Exception as e:
        st.error(f"Error: {e}")