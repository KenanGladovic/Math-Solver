import streamlit as st
import sympy as sp
import utils

# 1. SETUP
st.set_page_config(layout="wide", page_title="Parametric Matrix Analysis", page_icon="🎛️")
try:
    utils.setup_page()
except AttributeError:
    pass

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .result-card {
            background-color: #f8f9fa;
            border-left: 5px solid #2196F3;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        .step-box {
            font-family: 'Courier New', monospace;
            background-color: #f1f1f1;
            padding: 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎛️ Parametric Matrix Analysis")
st.markdown("""
**Curriculum Focus:** Analysis of matrices with parameters ($a, t, k$) using row operations.
* **Invertibility:** Gaussian Elimination $\\rightarrow$ Row Echelon Form (Check Pivots).
* **Definiteness:** Symmetric Reduction $\\rightarrow$ Diagonal Matrix $D$ (Check Signs).
""")

# --- HELPER FUNCTIONS ---

def get_exam_latex(expr):
    """Formats matrix with parentheses (pmatrix) per curriculum."""
    return sp.latex(expr, mat_delim='(')

def gaussian_elimination_pivots(matrix):
    """
    Reduces matrix to Row Echelon Form (REF) using fraction-free Gaussian elimination
    to keep parameters readable. Returns the REF matrix and the pivots.
    """
    # echelon_form() in SymPy does fraction-free row reduction
    ref_matrix = matrix.echelon_form()
    # Extract diagonal entries (pivots)
    pivots = [ref_matrix[i, i] for i in range(min(ref_matrix.rows, ref_matrix.cols))]
    return ref_matrix, pivots

def symmetric_reduction_diagonal(matrix):
    """
    Performs Schematic Symmetric Reduction (P^T A P) to find diagonal entries.
    Returns the list of diagonal entries.
    """
    M = matrix.copy()
    n = M.rows
    diags = []
    
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
            else:
                # Add row/col if no pivot found
                for k in range(i + 1, n):
                    if M[i, k] != 0:
                        M.row_op(i, lambda r, j: r + M.row(k)[j])
                        M.col_op(i, lambda c, j: c + M.col(k)[j])
                        break
        
        pivot = M[i, i]
        diags.append(pivot)
        if pivot == 0: continue

        # 2. Elimination
        for j in range(i + 1, n):
            if M[j, i] != 0:
                factor = M[j, i] / pivot
                M.row_op(j, lambda r, k: r - factor * M.row(i)[k])
                M.col_op(j, lambda c, k: c - factor * M.col(i)[k])
                
    return diags

# --- MAIN UI ---

# Input Mode
mode = st.radio("Select Analysis Mode:", 
    ["1. Matrix Properties (Invertibility & Definiteness)", 
     "2. Quadratic Form → Symmetric Matrix"], 
    horizontal=True
)

st.divider()

if mode == "1. Matrix Properties (Invertibility & Definiteness)":
    col_input, col_info = st.columns([1, 1])
    
    with col_input:
        st.subheader("Input Matrix")
        # Exam presets
        preset = st.selectbox("Load Exam Template:", ["Custom", "Maj 2024 (Definiteness)", "Jan 2025 (System)"])
        if preset == "Maj 2024 (Definiteness)":
            def_mat = "[[1, 1, 1], [1, 2, 1], [1, 1, a]]"
        elif preset == "Jan 2025 (System)":
            def_mat = "[[2, 1], [1, 2]]"
        else:
            def_mat = "[[1, a], [a, 1]]"

        mat_str = utils.p_text_area("Matrix A (Python list of lists):", "param_mat", def_mat)

    if st.button("Analyze Symbolic Matrix", type="primary"):
        try:
            # 1. Parse Matrix
            # Define common exam parameters as symbols
            a, b, c, t, k, x, y, z = sp.symbols('a b c t k x y z', real=True)
            parse_locals = {'a':a, 'b':b, 'c':c, 't':t, 'k':k, 'x':x, 'y':y, 'z':z}
            
            mat_list = eval(mat_str, {"__builtins__": None}, parse_locals)
            A = sp.Matrix(mat_list)
            
            # Show Input A
            with col_info:
                st.subheader("Parsed Matrix A")
                st.latex(f"A = {get_exam_latex(A)}")

            st.divider()
            
            # Create two main analysis columns
            c_inv, c_def = st.columns(2)

            # --- A. INVERTIBILITY (GAUSSIAN ELIMINATION) ---
            with c_inv:
                with st.container(border=True):
                    st.subheader("A. Invertibility")
                    st.caption("Method: Gaussian Elimination (Row Echelon Form)")
                    
                    # Compute REF
                    ref, pivots = gaussian_elimination_pivots(A)
                    
                    st.write("**Row Echelon Form (REF):**")
                    st.latex(f"{get_exam_latex(ref)}")
                    
                    st.write("**Pivots (Diagonal entries of REF):**")
                    pivots_tex = ", ".join([sp.latex(p) for p in pivots])
                    st.latex(f"p_1, \\dots, p_n = {pivots_tex}")
                    
                    # Logic: Invertible if all pivots != 0
                    st.markdown("#### Conclusion:")
                    zero_pivots = [p for p in pivots if p == 0]
                    
                    if zero_pivots:
                        st.error("Matrix is **Singular** (Not Invertible).")
                        st.write("Reason: One or more pivots are strictly zero.")
                    else:
                        # Symbolic check
                        conds = []
                        for p in pivots:
                            if not p.is_number:
                                conds.append(f"{sp.latex(p)} \\neq 0")
                        
                        if not conds:
                            if all(p != 0 for p in pivots):
                                st.success("Matrix is **Always Invertible**.")
                        else:
                            st.warning("Matrix is **Invertible** if:")
                            st.latex(" \\quad \\text{and} \\quad ".join(conds))

            # --- B. DEFINITENESS (SYMMETRIC REDUCTION) ---
            with c_def:
                with st.container(border=True):
                    st.subheader("B. Definiteness")
                    st.caption("Method: Symmetric Reduction (Diagonalization)")
                    
                    if not A.is_symmetric():
                        st.error("Matrix is NOT symmetric.")
                        st.write("Definiteness analysis requires a symmetric matrix.")
                    else:
                        # Compute Symmetric Reduction
                        diags = symmetric_reduction_diagonal(A)
                        simple_diags = [sp.simplify(d) for d in diags]
                        
                        st.write("**Diagonal Entries ($d_i$):**")
                        # Display D matrix
                        D_mat = sp.diag(*simple_diags)
                        st.latex(f"D = {get_exam_latex(D_mat)}")
                        
                        st.markdown("#### Conclusion (Thm 8.12):")
                        
                        # Logic for verdict
                        # Note: We can't always evaluate symbolic definiteness fully, 
                        # so we display the conditions.
                        
                        st.markdown("**Conditions for Positive Definite:**")
                        pd_conds = [f"{sp.latex(d)} > 0" for d in simple_diags]
                        st.latex(" \\quad \\text{and} \\quad ".join(pd_conds))
                        
                        st.markdown("**Conditions for Positive Semi-Definite:**")
                        psd_conds = [f"{sp.latex(d)} \\ge 0" for d in simple_diags]
                        st.latex(" \\quad \\text{and} \\quad ".join(psd_conds))

                        with st.expander("Show Argument Logic"):
                            st.write("From Symmetric Reduction, we have found $D$.")
                            st.write("- If all $d_i > 0 \\implies$ Positive Definite.")
                            st.write("- If all $d_i < 0 \\implies$ Negative Definite.")
                            st.write("- If mixed signs $\\implies$ Indefinite.")

        except Exception as e:
            st.error(f"Analysis Error: {e}")
            st.write("Check your matrix syntax.")

elif mode == "2. Quadratic Form → Symmetric Matrix":
    st.subheader("Polynomial to Symmetric Matrix")
    st.info("Converts $f(x,y,z)$ into matrix $A$ such that $f(v) = v^T A v$.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        poly_str = utils.p_text_input("Polynomial f(x, y...):", "param_poly", "x**2 + x*y + y**2 + x*z + z**2 + y*z")
    
    if st.button("Extract Matrix A", type="primary"):
        try:
            # 1. Parse Expression
            expr = utils.parse_expr(poly_str)
            vars_sym = sorted(list(expr.free_symbols), key=lambda s: s.name)
            
            with col2:
                st.write("**Variables:**")
                st.latex(f"({', '.join([sp.latex(v) for v in vars_sym])})")
            
            # 2. Compute Matrix (Hessian / 2)
            # Curriculum method: The matrix entry a_ij is coefficient of x_i x_j (divided by 2 if i!=j)
            # This is mathematically equivalent to 1/2 Hessian.
            H = sp.hessian(expr, vars_sym)
            A = sp.Rational(1, 2) * H
            
            # 3. Display
            st.divider()
            
            c_res1, c_res2 = st.columns(2)
            
            with c_res1:
                st.subheader("Symmetric Matrix A")
                st.latex(f"A = {get_exam_latex(A)}")
                with st.expander("Copy LaTeX"):
                    st.code(f"A = {get_exam_latex(A)}", language="latex")
            
            with c_res2:
                st.subheader("Verification")
                v = sp.Matrix(vars_sym)
                check = sp.expand((v.T * A * v)[0])
                st.write("Calculating $v^T A v$:")
                st.latex(sp.latex(check))
                
                if sp.simplify(check - expr) == 0:
                    st.success("✅ Verified")
                else:
                    st.error("❌ Verification Failed (Is it linear/constant?)")

        except Exception as e:
            st.error(f"Error: {e}")