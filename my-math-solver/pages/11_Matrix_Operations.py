import streamlit as st
import sympy as sp
import utils
st.set_page_config(layout="wide")
utils.setup_page()
st.markdown("<h1 class='main-header'>Matrix Operations</h1>", unsafe_allow_html=True)

# Operation Selector
op = utils.p_selectbox("Operation", ["Determinant", "Inverse", "Transpose", "Matrix Multiplication"], "mat_op")

# --- 1. UNARY OPERATIONS (Determinant, Inverse, Transpose) ---
if op in ["Determinant", "Inverse", "Transpose"]:
    st.subheader(f"Matrix A Setup for {op}")
    
    # Callback to update default text based on template selection
    # In multipage, we handle this slightly differently with key management, 
    # but to match original logic we check session state manually.
    template = st.selectbox(
        "Select Matrix Template Size:", 
        ["2x2", "3x3", "4x4"], 
        key="mat_template_sel"
    )
    
    default_a = "[[2, 1], [1, 2]]"
    if "3x3" in template: default_a = "[[1, 0, -1], [-2, 2, -1], [1, -1, 1]]"
    elif template == "4x4": default_a = "[[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]"

    st.write("Input Matrix (Python list of lists format)")
    mat_input = st.text_area("Matrix A:", value=default_a, height=150)
    
    if st.button("Calculate", type="primary"):
        try:
            mat_list = eval(mat_input, {'__builtins__': None}, locals())
            A = sp.Matrix(mat_list)
            
            st.subheader("Input")
            st.latex(f"A = {sp.latex(A)}")
            
            st.subheader("Result")
            if op == "Determinant":
                det = A.det()
                st.latex(f"\\det(A) = {sp.latex(det)}")
            elif op == "Inverse":
                if A.rows != A.cols:
                    st.error("Matrix must be square to calculate the inverse.")
                elif A.det() == 0:
                    st.error("Matrix is singular (Determinant is 0), cannot invert.")
                else:
                    inv = A.inv()
                    st.latex(f"A^{{-1}} = {sp.latex(inv)}")
            elif op == "Transpose":
                trans = A.transpose()
                st.latex(f"A^{{T}} = {sp.latex(trans)}")
                    
        except Exception as e:
            st.error(f"Error parsing matrix: {e}")

# --- 2. BINARY OPERATIONS (Matrix Multiplication) ---
elif op == "Matrix Multiplication":
    st.subheader("Matrix A (m x n) $\\cdot$ Matrix B (n x r)")
    st.caption("Reference: **Section 3.3** (Matrix Multiplication)")
    
    col1, col2 = st.columns(2)
    with col1:
        mat_a_str = utils.p_text_area("Matrix A:", "mat_mul_a", "[[1, 2], [3, 4]]", height=150)
    with col2:
        mat_b_str = utils.p_text_area("Matrix B:", "mat_mul_b", "[[1, 0], [0, 1]]", height=150)
        
    if st.button("Multiply A $\\cdot$ B", type="primary"):
        try:
            A = sp.Matrix(eval(mat_a_str, {'__builtins__': None}, locals()))
            B = sp.Matrix(eval(mat_b_str, {'__builtins__': None}, locals()))
            
            if A.cols != B.rows:
                st.error(f"Dimension Mismatch: A has {A.cols} cols, B has {B.rows} rows.")
                st.stop()
                
            res = A * B
            st.subheader("Result")
            st.latex(f"{sp.latex(A)} \\cdot {sp.latex(B)} = {sp.latex(res)}")
            
        except Exception as e:
            st.error(f"Error: {e}")