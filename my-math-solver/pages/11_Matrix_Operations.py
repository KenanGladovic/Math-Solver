import streamlit as st
import sympy as sp
from utils import setup_page, p_text_area, p_text_input, p_selectbox

# 1. Page Configuration
setup_page()

st.markdown('<div class="main-header">Matrix Operations & Linear Systems</div>', unsafe_allow_html=True)

# Create Main Tabs
tab_systems, tab_ops = st.tabs(["📝 Linear Systems Solver (Av=b)", "🧮 Matrix Operations (Step-by-Step)"])

# ==============================================================================
# TAB 1: LINEAR SYSTEMS SOLVER (From Script 1)
# ==============================================================================
with tab_systems:
    st.markdown("### 1. Define System")
    st.markdown("Enter equations (one per line). The tool will extract $A$ and $\mathbf{b}$ and solve $A\mathbf{v} = \mathbf{b}$ (or find the Least Squares solution).")
    
    # Standard Exam Example
    default_eqs = "x + y + z = 1\nx + 2*y + 3*z = 1\ny + 2*z = 1"
    
    col1, col2 = st.columns([3, 1])
    with col1:
        eq_input = p_text_area("Equations:", "mo_eq_input", default_eqs, height=150)
    with col2:
        # User defines which letters are the variables to solve for
        vars_input = p_text_input("Unknowns (comma sep):", "mo_vars", "x, y, z")

    if eq_input:
        try:
            # 1. Parse Variables
            var_strs = [v.strip() for v in vars_input.split(',') if v.strip()]
            variables = [sp.Symbol(v) for v in var_strs]
            
            # 2. Parse Equations
            raw_lines = [line.strip() for line in eq_input.split('\n') if line.strip()]
            eqs = []
            
            for line in raw_lines:
                if "=" in line:
                    lhs, rhs = line.split("=")
                    eq_expr = sp.sympify(lhs) - sp.sympify(rhs)
                else:
                    eq_expr = sp.sympify(line)
                eqs.append(eq_expr)
            
            # 3. Extract A and b
            A, b = sp.linear_eq_to_matrix(eqs, variables)
            
            st.markdown("---")
            st.markdown("### 2. Matrix Form")
            
            c1, c2 = st.columns(2)
            with c1:
                st.latex(f"A = {sp.latex(A)}")
            with c2:
                st.latex(f"\\mathbf{{b}} = {sp.latex(b)}")
                
            # 4. Consistency & Rank Analysis
            st.markdown("### 3. Analysis & Solution")
            
            try:
                rank_A = A.rank()
                augmented = A.col_insert(A.shape[1], b)
                rank_Ab = augmented.rank()
                n = len(variables)
                
                st.write(f"**Rank(A):** {rank_A}, **Rank(A|b):** {rank_Ab}, **Unknowns:** {n}")
                
                if rank_A == rank_Ab:
                    if rank_A == n:
                        st.success("✅ Unique Solution (Consistent)")
                        sol = A.LUsolve(b)
                        st.latex(f"\\mathbf{{v}} = {sp.latex(sol)}")
                    else:
                        st.info("⚠️ Infinitely Many Solutions (Consistent, Underdetermined)")
                        sol = sp.linsolve((A, b), variables)
                        st.latex(sp.latex(sol))
                else:
                    st.error("❌ No Solution (Inconsistent)")
                    st.markdown("**Least Squares Solution (Normal Equations)**")
                    st.markdown("Since the system is inconsistent, we solve $A^T A \\mathbf{v} = A^T \\mathbf{b}$ (Theorem 5.19).")
                    
                    ATA = sp.simplify(A.T * A)
                    ATb = sp.simplify(A.T * b)
                    
                    c3, c4 = st.columns(2)
                    with c3:
                        st.latex(r"A^T A = " + sp.latex(ATA))
                    with c4:
                        st.latex(r"A^T \mathbf{b} = " + sp.latex(ATb))
                    
                    try:
                        ls_sol = ATA.LUsolve(ATb)
                        st.latex(f"\\mathbf{{v}}_{{LS}} = {sp.latex(ls_sol)}")
                    except:
                        st.write("Could not solve Normal Equations symbolically.")

            except Exception as e:
                st.warning(f"Could not compute numerical rank (likely due to symbolic parameters). Solution attempt:")
                try:
                    sol = sp.linsolve((A, b), variables)
                    st.latex(sp.latex(sol))
                except:
                    st.error("Solver failed.")

        except Exception as e:
            st.error(f"Error parsing system: {e}")


# ==============================================================================
# TAB 2: MATRIX OPERATIONS & STEPS (From Script 2)
# ==============================================================================
with tab_ops:
    # --- OPERATION SELECTOR ---
    op = p_selectbox(
        "Select Operation:", 
        ["Inverse", "Transpose", "Multiply", "Row Reduction (Steps)"],
        key="mo_op_select",
        default_idx=0
    )

    # --- CURRICULUM DEFINITIONS ---
    with st.container():
        st.markdown("""
        <style>
            .def-container { background-color: #f0f2f6; border-left: 5px solid #2196F3; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            h4 { margin-top: 0; color: #2C3E50; }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='def-container'>", unsafe_allow_html=True)
        
        if op == "Inverse":
            st.markdown("#### 🎓 Definition: Inverse Matrix (Section 8.2)")
            st.write("An $n \\times n$ matrix $A$ is invertible if there exists a matrix $B$ (denoted $A^{-1}$) such that:")
            st.latex(r"A B = B A = I_n")
            st.write("where $I_n$ is the identity matrix.")

        elif op == "Row Reduction (Steps)":
            st.markdown("#### 🎓 Method: Gaussian Elimination (Section 8.3)")
            st.write("To find the inverse of $A$ or solve systems, we apply row operations:")
            st.latex(r"[A \mid I] \xrightarrow{\text{Row Ops}} [I \mid A^{-1}]")

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
        with c1: 
            a_str = p_text_area("Matrix A", "mo_mat_a_mult", "[[1, 2], [3, 4]]", height=100)
        with c2: 
            st.markdown("<br><h2 style='text-align: center;'>×</h2>", unsafe_allow_html=True)
        with c3: 
            b_str = p_text_area("Matrix B", "mo_mat_b_mult", "[[1, 0], [0, 1]]", height=100)
            
        if st.button("Calculate Product", type="primary"):
            try:
                A = sp.Matrix(sp.sympify(a_str))
                B = sp.Matrix(sp.sympify(b_str))
                if A.cols != B.rows: 
                    st.error(f"Dimension Mismatch: {A.shape} vs {B.shape}")
                else:
                    res = A * B
                    res_tex = f"{sp.latex(A)} \\cdot {sp.latex(B)} = {sp.latex(res)}"
                    st.markdown("### Result")
                    st.latex(res_tex)
            except Exception as e: 
                st.error(f"Invalid Input: {e}")

    else:
        # Standard Single Matrix Input
        c1, c2 = st.columns([1, 1.5])
        with c1:
            default_a = "[[1, 2], [3, 4]]"
            mat_input = p_text_area("Matrix A", "mo_mat_single", default_a, height=100)
            calc_btn = st.button("Calculate", type="primary", use_container_width=True)

        with c2:
            if calc_btn and mat_input:
                try:
                    A = sp.Matrix(sp.sympify(mat_input))
                    st.markdown("**Input Matrix:**")
                    st.latex(sp.latex(A))
                    
                    if op == "Transpose":
                        st.markdown("**Result:**")
                        st.latex(f"A^T = {sp.latex(A.T)}")
                        
                    elif op == "Inverse":
                        if A.rows != A.cols:
                            st.error("Inverse only exists for square matrices.")
                        elif A.det() == 0:
                            st.error("Matrix is Singular (Determinant is 0).")
                        else:
                            st.markdown("**Result:**")
                            st.latex(f"A^{{-1}} = {sp.latex(A.inv())}")
                            
                    elif op == "Row Reduction (Steps)":
                        try:
                            steps, res = get_rref_steps(A)
                            st.markdown("**Gaussian Elimination Steps:**")
                            for i, (desc, mat) in enumerate(steps):
                                st.markdown(f"**Step {i+1}:** {desc}")
                                st.latex(sp.latex(mat))
                                st.markdown("---")
                            st.success("Final Result:")
                            st.latex(sp.latex(res))
                        except Exception as e:
                            st.error(f"Calculation error: {e}")

                except Exception as e: 
                    st.error(f"Input Error: {e}")