import streamlit as st
import sympy as sp
import utils

st.set_page_config(layout="wide")
utils.setup_page()

st.markdown("<h1 class='main-header'>Parametric Matrix Analysis</h1>", unsafe_allow_html=True)
st.markdown("""
**Exam Focus:** Problems involving parameters ($a, t, k$) instead of just numbers.
* **Invertibility:** Find $t$ such that $\det(A) \\neq 0$ (Matrix is invertible).
* **Definiteness:** Find $a$ such that $A$ is Positive Definite (using symbolic reduction).
* **Quadratic Forms:** Convert a polynomial $f(x,y,z)$ into a symmetric matrix $A$.
""")

# Input Mode
mode = utils.p_radio("Analysis Mode", 
    ["1. Matrix Properties (Invertibility/Definiteness)", 
     "2. Quadratic Form -> Matrix"], 
    "param_mode", horizontal=True
)

if mode == "1. Matrix Properties (Invertibility/Definiteness)":
    st.subheader("1. Input Matrix with Symbols")
    
    # Presets
    preset = st.selectbox("Load Exam Example:", ["Custom", "Maj 2024 Q2 (Definiteness)", "Jan 2025 Q2 (Invertibility)"])
    if preset == "Maj 2024 Q2 (Definiteness)":
        def_mat = "[[1, 1, 1], [1, 2, 1], [1, 1, a]]"
    elif preset == "Jan 2025 Q2 (Invertibility)":
        def_mat = "[[2, 1], [1, 2]]" # The exam had a system, this is the A matrix part roughly
    else:
        def_mat = "[[1, a], [a, 1]]"

    mat_str = utils.p_text_area("Matrix A (use 'a', 't', 'k' etc):", "param_mat", def_mat)
    
    if st.button("Analyze Symbolic Matrix", type="primary"):
        try:
            # Parse
            a, b, c, t, k, x, y, z = sp.symbols('a b c t k x y z')
            parse_locals = {'a':a, 'b':b, 'c':c, 't':t, 'k':k, 'x':x, 'y':y, 'z':z}
            
            mat_list = eval(mat_str, {"__builtins__": None}, parse_locals)
            A = sp.Matrix(mat_list)
            
            # Display A
            st.latex(f"A = {sp.latex(A)}")
            
            col1, col2 = st.columns(2)
            
            # --- INVERTIBILITY ---
            with col1:
                st.subheader("A. Invertibility")
                det = A.det()
                st.latex(f"\\det(A) = {sp.latex(det)}")
                
                if det == 0:
                    st.error("Matrix is Singular for ALL values (Determinant is 0).")
                elif det.is_number:
                    if det != 0: st.success("Matrix is Always Invertible (Determinant is constant non-zero).")
                else:
                    st.write("Matrix is **Singular** (Not Invertible) when:")
                    sols = sp.solve(det, dict=True)
                    if sols:
                        for s in sols:
                            st.latex(f"{', '.join([f'{k} = {sp.latex(v)}' for k,v in s.items()])}")
                    else:
                        st.write("No real solutions found.")

            # --- DEFINITENESS (Symbolic Symmetric Reduction) ---
            with col2:
                st.subheader("B. Definiteness (Symmetric Reduction)")
                if not A.is_symmetric():
                    st.warning("Matrix is NOT symmetric. Definiteness analysis usually applies to symmetric matrices.")
                
                # Symbolic Reduction Algorithm
                n = A.rows
                D = A.copy()
                steps = []
                
                # We perform raw row operations to get diagonals. 
                # Note: Full symbolic pivoting is hard, we assume standard order works or warn.
                try:
                    for i in range(n):
                        pivot = D[i,i]
                        if pivot == 0:
                            steps.append(f"Pivot at {i+1},{i+1} is 0. Analysis difficult without specific value.")
                            continue
                            
                        for j in range(i+1, n):
                            if D[j,i] != 0:
                                factor = D[j,i]/pivot
                                # Row op
                                D.row_op(j, lambda r,idx: r - factor*D.row(i)[idx])
                                # Col op
                                D.col_op(j, lambda c,idx: c - factor*D.col(i)[idx])
                    
                    diags = [D[i,i] for i in range(n)]
                    
                    st.write("After Symmetric Reduction ($B^T A B = D$), the diagonal entries are:")
                    for idx, d in enumerate(diags):
                        st.latex(f"d_{{{idx+1}}} = {sp.latex(sp.simplify(d))}")
                        
                    st.info("""
                    **Conclusion Guide:**
                    * **Pos. Definite:** All $d_i > 0$
                    * **Pos. Semi-Definite:** All $d_i \ge 0$
                    * **Indefinite:** Mixed signs
                    """)
                    
                except Exception as e:
                    st.error(f"Symbolic reduction failed: {e}")

        except Exception as e:
            st.error(f"Error parsing matrix: {e}")

elif mode == "2. Quadratic Form -> Matrix":
    st.subheader("2. Polynomial to Symmetric Matrix")
    st.info("Convert $f(x,y,z)$ into $A$ such that $f(v) = v^T A v$. (Maj 2025 Q2)")
    
    poly_str = utils.p_text_input("Polynomial f(x, y...):", "param_poly", "x**2 + x*y + y**2 + x*z + z**2 + y*z")
    
    if st.button("Extract Matrix", type="primary"):
        try:
            expr = utils.parse_expr(poly_str)
            vars_sym = sorted(list(expr.free_symbols), key=lambda s: s.name)
            
            st.write(f"Variables found: {vars_sym}")
            
            # The Matrix A is 1/2 * Hessian (for quadratic forms)
            # This works perfectly because Hessian[i,j] = d^2f / dx_i dx_j
            # For x_i^2 term: deriv is 2, Hessian is 2. 1/2 * 2 = 1. Correct.
            # For x_i*x_j term: deriv is 1, Hessian is 1. 1/2 * 1 = 0.5. Correct (split between A_ij and A_ji).
            
            H = sp.hessian(expr, vars_sym)
            A = sp.Rational(1, 2) * H
            
            st.subheader("Resulting Symmetric Matrix A")
            st.latex(f"A = {sp.latex(A)}")
            
            # Verify
            v = sp.Matrix(vars_sym)
            check = sp.expand((v.T * A * v)[0])
            st.write("Verification ($v^T A v$):")
            st.latex(sp.latex(check))
            
            if sp.simplify(check - expr) == 0:
                st.success("Verified! The matrix exactly represents the polynomial.")
            else:
                st.error("Verification failed. Is the function purely quadratic?")
                
        except Exception as e:
            st.error(f"Error: {e}")