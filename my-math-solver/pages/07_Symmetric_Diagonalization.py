import streamlit as st
import sympy as sp
import numpy as np
import utils

st.set_page_config(layout="wide")
utils.setup_page()

# --- CUSTOM CSS FOR PROFESSIONAL LAYOUT ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-header {
        color: #6c757d;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .step-box {
        border-left: 3px solid #2196F3;
        background-color: #f1f8ff;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 0 5px 5px 0;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(r"<h1 class='main-header'>Symmetric Diagonalization $B^T A B = D$</h1>", unsafe_allow_html=True)

st.info("""
**Curriculum Reference: Section 8.7 (Schematic Procedure)**
This tool finds an invertible matrix $B$ and a diagonal matrix $D$ such that $B^T A B = D$.
It uses the schematic procedure of transforming $\\begin{pmatrix} I \\\\ A \\end{pmatrix} \\to \\begin{pmatrix} B \\\\ D \\end{pmatrix}$.
""")

# --- PRESET DEFINITIONS ---
presets = {
    "2x2": "[[1, 2], [2, 8]]",
    "3x3": "[[1, 2, 3], [2, 8, 4], [3, 4, 16]]", # Example 8.30
    "4x4": "[[0, 0, 1, 1], [0, 0, 2, 3], [1, 2, 1, 4], [1, 3, 4, 0]]" # Example 8.31
}

# --- CALLBACK TO LOAD PRESET ---
def load_preset():
    sel = st.session_state.sym_template
    st.session_state.sym_matrix_val = presets[sel]

if "sym_matrix_val" not in st.session_state:
    st.session_state.sym_matrix_val = presets["2x2"]

# --- INPUT SECTION ---
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### ⚙️ Setup")
    st.selectbox(
        "Template Size:", 
        list(presets.keys()), 
        key="sym_template", 
        on_change=load_preset
    )

with col2:
    st.markdown("### 🔢 Matrix Input")
    mat_input = st.text_area(
        "Input Symmetric Matrix A (Python List of Lists):", 
        key="sym_matrix_val", 
        height=100
    )

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
    history.append(("<b>Initial Setup:</b> Form Augmented Matrix", format_aug_latex(aug, n)))
    
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
                desc = f"<b>Pivot Fix (Swap):</b> Swap col/row {k+1} ↔ {swap_idx+1}"
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
                    desc = f"<b>Pivot Fix (Add):</b> Col/Row {k+1} ← Col/Row {k+1} + Col/Row {add_idx+1}"
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
                # Simplified description string to avoid latex parsing errors inside HTML
                # Using standard text indexing A(row, col)
                desc = f"<b>Eliminate A({k+1}, {j+1}):</b> C{j+1} - {m:.3g} C{k+1} & R{j+1} - {m:.3g} R{k+1}"
                
                aug[:, j] -= m * aug[:, k]
                aug[n + j, :] -= m * aug[n + k, :]
                history.append((desc, format_aug_latex(aug, n)))

    B = clean_matrix(aug[:n, :])
    D = clean_matrix(aug[n:, :])
    return B, D, history, None

# --- MAIN EXECUTION ---
st.markdown("---")

if st.button("🚀 Diagonalize Matrix", type="primary", use_container_width=True):
    try:
        A_list = eval(mat_input)
        A_arr = np.array(A_list)
        B_res, D_res, history, err = diagonalize_symmetric_algo(A_arr)
        
        if err:
            st.error(err)
        else:
            # --- 1. RESULTS SECTION ---
            st.markdown("### 1. Final Results")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown("""<div class="metric-card"><div class="metric-header">Original Matrix A</div>""", unsafe_allow_html=True)
                st.latex(sp.latex(sp.Matrix(A_arr)))
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("""<div class="metric-card"><div class="metric-header">Transformation B</div>""", unsafe_allow_html=True)
                st.latex(sp.latex(sp.Matrix(B_res)))
                st.markdown("</div>", unsafe_allow_html=True)
            with c3:
                st.markdown("""<div class="metric-card"><div class="metric-header">Diagonal D</div>""", unsafe_allow_html=True)
                st.latex(sp.latex(sp.Matrix(D_res)))
                st.markdown("</div>", unsafe_allow_html=True)

            # --- 2. VERIFICATION ---
            st.markdown("### 2. Verification & Properties")
            with st.container():
                calculated_D = B_res.T @ A_arr @ B_res
                calculated_D = clean_matrix(calculated_D)
                
                v1, v2 = st.columns([2, 1])
                with v1:
                    st.write("**Verification Calculation ($B^T A B$):**")
                    st.latex(r"B^T A B = " + sp.latex(sp.Matrix(calculated_D)))
                    if np.allclose(calculated_D, D_res):
                        st.success("✅ **Mathematical Verification Passed**")
                    else:
                        st.error("❌ **Verification Failed**")
                
                with v2:
                    st.write("**Definiteness:**")
                    diag = np.diag(D_res)
                    if all(d > 1e-9 for d in diag): 
                        st.info("Positive Definite (+ + +)")
                    elif all(d < -1e-9 for d in diag): 
                        st.info("Negative Definite (- - -)")
                    elif all(d >= -1e-9 for d in diag): 
                        st.warning("Positive Semi-Definite (+ 0 +)")
                    elif all(d <= 1e-9 for d in diag): 
                        st.warning("Negative Semi-Definite (- 0 -)")
                    else: 
                        st.warning("Indefinite (+ - +)")

            # --- 3. LOG ---
            st.markdown("### 3. Step-by-Step Schematic Procedure")
            st.caption(f"Following **Schematic Procedure (Section 8.7)**.")
            
            for step_num, (desc, latex) in enumerate(history):
                # Simple clean title logic
                step_title = desc.replace("<b>", "").replace("</b>", "").split(":")[0]
                with st.expander(f"Step {step_num}: {step_title}", expanded=False):
                    st.markdown(f"<div class='step-box'>{desc}</div>", unsafe_allow_html=True)
                    st.latex(latex)

    except Exception as e:
        st.error(f"Input Error: {e}")