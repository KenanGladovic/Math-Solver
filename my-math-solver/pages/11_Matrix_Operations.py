import streamlit as st
import sympy as sp
import utils

st.set_page_config(layout="wide")
utils.setup_page()
st.markdown("<h1 class='main-header'>Matrix Operations</h1>", unsafe_allow_html=True)

# Operation Selector
op = utils.p_selectbox("Operation", ["Determinant", "Inverse", "Transpose", "Multiply", "Row Reduction (Steps)"], "mat_op")

# --- ROW REDUCTION STEPPER ---
def get_rref_steps(M):
    """Performs Gaussian elimination on [M | I] to find inverse steps."""
    rows = M.rows
    cols = M.cols
    # Augment with Identity for inverse tracking
    aug = M.row_join(sp.eye(rows))
    steps = []
    
    # Simple Gaussian Elimination (Forward only for simplicity in demo)
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows: break
        
        # Find pivot
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
            # Normalize pivot row
            if pivot != 1:
                aug.row_op(pivot_row, lambda v, j: v / pivot)
                steps.append((f"R{pivot_row+1} / {sp.latex(pivot)}", aug.copy()))
            
            # Eliminate others
            for r in range(rows):
                if r != pivot_row and aug[r, col] != 0:
                    factor = aug[r, col]
                    aug.row_op(r, lambda v, j: v - factor * aug[pivot_row, j])
                    steps.append((f"R{r+1} - ({sp.latex(factor)})R{pivot_row+1}", aug.copy()))
            
            pivot_row += 1
            
    return steps, aug[:, cols:]

# --- UI HANDLER ---
if op in ["Determinant", "Inverse", "Transpose", "Row Reduction (Steps)"]:
    st.subheader(f"Matrix Setup")
    default_a = "[[1, 2], [3, 4]]"
    mat_input = st.text_area("Input Matrix A:", value=default_a, height=100)
    
    if st.button("Calculate", type="primary"):
        try:
            mat_list = eval(mat_input)
            A = sp.Matrix(mat_list)
            
            st.write("Input A:")
            st.latex(sp.latex(A))
            
            if op == "Determinant":
                st.latex(f"\\det(A) = {sp.latex(A.det())}")
            
            elif op == "Transpose":
                st.latex(f"A^T = {sp.latex(A.T)}")
                
            elif op == "Inverse":
                if A.det() == 0:
                    st.error("Matrix is Singular (No Inverse)")
                else:
                    st.latex(f"A^{{-1}} = {sp.latex(A.inv())}")
            
            elif op == "Row Reduction (Steps)":
                if A.rows != A.cols:
                    st.warning("Standard inverse steps require a square matrix. Showing RREF on input.")
                    
                st.subheader("Gaussian Elimination Steps [A | I] -> [I | A^-1]")
                steps, res = get_rref_steps(A)
                
                for desc, mat in steps:
                    c1, c2 = st.columns([1, 3])
                    c1.write(f"**{desc}**")
                    c2.latex(sp.latex(mat))
                
                st.success("Resulting Inverse (Right side of bar):")
                st.latex(sp.latex(res))

        except Exception as e:
            st.error(f"Error: {e}")

elif op == "Multiply":
    c1, c2 = st.columns(2)
    with c1:
        a_str = st.text_area("Matrix A", "[[1,2],[3,4]]")
    with c2:
        b_str = st.text_area("Matrix B", "[[1,0],[0,1]]")
        
    if st.button("Multiply"):
        A = sp.Matrix(eval(a_str))
        B = sp.Matrix(eval(b_str))
        st.latex(f"{sp.latex(A)} \\cdot {sp.latex(B)} = {sp.latex(A*B)}")