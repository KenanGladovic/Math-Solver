import streamlit as st
import sympy as sp
import utils

# 1. Setup
st.set_page_config(layout="wide", page_title="Matrix Operations", page_icon="🧮")
utils.setup_page()

# --- CSS FOR CLEAN LOOK ---
st.markdown("""
    <style>
        .main-header { font-size: 2.2rem; color: #2C3E50; border-bottom: 2px solid #2196F3; padding-bottom: 10px; margin-bottom: 20px; }
        .def-container { background-color: #f8f9fa; border-left: 5px solid #2196F3; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        h4 { margin-top: 0; color: #2C3E50; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Matrix Operations</h1>", unsafe_allow_html=True)

# --- OPERATION SELECTOR ---
op = st.selectbox(
    "Select Operation:", 
    ["Inverse", "Transpose", "Multiply", "Row Reduction (Steps)"],
    index=0
)

# --- CURRICULUM DEFINITIONS ---
with st.container():
    st.markdown("<div class='def-container'>", unsafe_allow_html=True)
    
    if op == "Inverse":
        st.markdown("#### 🎓 Definition: Inverse Matrix (Section 8.2)")
        st.write("An $n \\times n$ matrix $A$ is invertible if there exists a matrix $B$ (denoted $A^{-1}$) such that:")
        st.latex(r"A B = B A = I_n")
        st.write("where $I_n$ is the identity matrix.")

    elif op == "Row Reduction (Steps)":
        st.markdown("#### 🎓 Method: Gaussian Elimination (Section 8.3)")
        st.write("To find the inverse of $A$, we form the augmented matrix $[A \\mid I]$ and apply row operations:")
        st.latex(r"[A \mid I] \xrightarrow{\text{Row Ops}} [I \mid A^{-1}]")
        st.write("If the left side cannot be reduced to $I$, the matrix is singular.")

    elif op == "Transpose":
        st.markdown("#### 🎓 Definition: Transpose")
        st.write("The transpose $A^T$ is formed by swapping rows and columns.")
        st.latex(r"(A^T)_{ij} = A_{ji}")

    elif op == "Multiply":
        st.markdown("#### 🎓 Definition: Matrix Multiplication")
        st.write("The product $C = AB$ is defined if columns of $A$ equal rows of $B$.")
        st.latex(r"C_{ij} = \sum_{k} A_{ik} B_{kj}")
        

    st.markdown("</div>", unsafe_allow_html=True)

# --- ALGORITHMS ---
def get_rref_steps(M):
    rows = M.rows
    cols = M.cols
    if rows == cols:
        aug = M.row_join(sp.eye(rows))
    else:
        aug = M.copy()

    steps = []
    pivot_row = 0
    loop_cols = cols if rows != cols else cols
    
    for col in range(loop_cols):
        if pivot_row >= rows: break
        
        if aug[pivot_row, col] == 0:
            swap_idx = -1
            for r in range(pivot_row + 1, rows):
                if aug[r, col] != 0:
                    swap_idx = r
                    break
            if swap_idx != -1:
                aug.row_swap(pivot_row, swap_idx)
                steps.append((f"Swap R{pivot_row+1} <-> R{swap_idx+1}", aug.copy()))
        
        pivot = aug[pivot_row, col]
        if pivot != 0:
            if pivot != 1:
                aug.row_op(pivot_row, lambda v, j: v / pivot)
                steps.append((f"R{pivot_row+1} <- R{pivot_row+1} / {sp.latex(pivot)}", aug.copy()))
            
            for r in range(rows):
                if r != pivot_row and aug[r, col] != 0:
                    factor = aug[r, col]
                    aug.row_op(r, lambda v, j: v - factor * aug[pivot_row, j])
                    steps.append((f"R{r+1} <- R{r+1} - ({sp.latex(factor)}) * R{pivot_row+1}", aug.copy()))
            
            pivot_row += 1
            
    return steps, aug[:, cols:] if rows == cols else aug

# --- INPUT & CALCULATION ---
if op == "Multiply":
    c1, c2, c3 = st.columns([1, 0.2, 1])
    with c1: a_str = st.text_area("Matrix A", "[[1, 2], [3, 4]]", height=100)
    with c2: st.markdown("<br><h2 style='text-align: center;'>×</h2>", unsafe_allow_html=True)
    with c3: b_str = st.text_area("Matrix B", "[[1, 0], [0, 1]]", height=100)
        
    if st.button("Calculate", type="primary"):
        try:
            A = sp.Matrix(eval(a_str))
            B = sp.Matrix(eval(b_str))
            if A.cols != B.rows: st.error("Dimension Mismatch.")
            else:
                res_tex = f"{sp.latex(A)} \\cdot {sp.latex(B)} = {sp.latex(A*B)}"
                st.markdown("### Result")
                st.latex(res_tex)
                st.code(f"$$ {res_tex} $$", language="latex")
        except: st.error("Invalid Input")

else:
    c1, c2 = st.columns([1, 1.5])
    with c1:
        default_a = "[[1, 2], [3, 4]]"
        mat_input = st.text_area("Matrix A", value=default_a, height=100)
        calc_btn = st.button("Calculate", type="primary", use_container_width=True)

    with c2:
        if calc_btn:
            try:
                A = sp.Matrix(eval(mat_input))
                st.markdown("**Input Matrix:**")
                st.latex(sp.latex(A))
                
                if op == "Transpose":
                    st.markdown("**Result:**")
                    st.latex(f"A^T = {sp.latex(A.T)}")
                    
                elif op == "Inverse":
                    if A.det() == 0:
                        st.error("Matrix is Singular (Determinant is 0).")
                    else:
                        st.markdown("**Result:**")
                        st.latex(f"A^{{-1}} = {sp.latex(A.inv())}")
                        
                elif op == "Row Reduction (Steps)":
                    steps, res = get_rref_steps(A)
                    st.markdown("**Gaussian Elimination Steps:**")
                    for i, (desc, mat) in enumerate(steps):
                        st.markdown(f"**Step {i+1}:** {desc}")
                        st.latex(sp.latex(mat))
                        st.markdown("---")
                    st.success("Final Result:")
                    st.latex(sp.latex(res))

            except Exception as e: st.error(f"Input Error: {e}")