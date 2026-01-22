import streamlit as st
import sympy as sp
import utils

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="Gaussian Elimination", page_icon="🧮")
utils.setup_page()

st.markdown("<h1 class='main-header'>Gaussian Elimination (Chapter 2)</h1>", unsafe_allow_html=True)
st.caption("Step-by-step row reduction to solve $Ax = b$ or find $A^{-1}$.")

# --- INPUT SECTION ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. System Setup")
    mode = utils.p_radio("Mode:", ["Solve System (Ax = b)", "Find Inverse (A⁻¹)"], "gauss_mode")
    
    # Default values based on Mode
    if mode == "Solve System (Ax = b)":
        def_A = "1, 2, 1\n2, 5, 2\n1, 3, 4" # Invertible 3x3
        def_b = "4\n10\n9"
        
        st.markdown("**Matrix A**")
        a_str = utils.p_text_area("", "gauss_a", def_A, height=120, label_visibility="collapsed")
        
        st.markdown("**Vector b**")
        b_str = utils.p_text_area("", "gauss_b", def_b, height=120, label_visibility="collapsed")
        
    else: # Inverse
        def_A = "1, 2, 3\n0, 1, 4\n5, 6, 0"
        st.markdown("**Matrix A**")
        a_str = utils.p_text_area("", "gauss_a_inv", def_A, height=150, label_visibility="collapsed")
        b_str = None # Not needed for inverse

with col2:
    st.subheader("2. Result Preview")
    
    if st.button("Start Elimination", type="primary"):
        st.session_state["gauss_run"] = True
    
    if st.session_state.get("gauss_run"):
        try:
            # Parse Input
            rows_A = [row.split(',') for row in a_str.split('\n') if row.strip()]
            A = sp.Matrix([[utils.parse_expr(c.strip()) for c in r] for r in rows_A])
            
            if mode == "Solve System (Ax = b)":
                rows_b = [row.split(',') for row in b_str.split('\n') if row.strip()]
                b = sp.Matrix([[utils.parse_expr(c.strip()) for c in r] for r in rows_b])
                
                # Check dimensions
                if A.rows != b.rows:
                    st.error(f"Dimension Mismatch: A is {A.shape}, b is {b.shape}")
                    st.stop()
                
                # Create Augmented Matrix [A | b]
                M = A.row_join(b)
                st.markdown("**Augmented Matrix $[A \\mid b]$:**")
                st.latex(sp.latex(M))
                
            else: # Inverse
                if A.rows != A.cols:
                    st.error("Inverse only exists for square matrices.")
                    st.stop()
                
                # Create Augmented Matrix [A | I]
                I = sp.eye(A.rows)
                M = A.row_join(I)
                st.markdown("**Augmented Matrix $[A \\mid I]$:**")
                st.latex(sp.latex(M))

        except Exception as e:
            st.error(f"Parsing Error: {e}")
            st.stop()

# --- ALGORITHM & VISUALIZATION ---

if st.session_state.get("gauss_run"):
    st.divider()
    st.subheader("3. Step-by-Step Reduction")
    
    # We perform Gaussian Elimination manually to capture steps
    # Note: SymPy's rref() does it all at once. We need a custom loop.
    
    current_mat = M.copy()
    rows, cols = current_mat.shape
    pivot_row = 0
    
    # Store steps for display
    steps_log = []
    
    # We only eliminate up to the columns of A (not the augmented part)
    target_cols = A.cols 
    
    try:
        for col in range(target_cols):
            if pivot_row >= rows: break
            
            # 1. FIND PIVOT
            # Look for non-zero in current column at or below pivot_row
            pivot_val = current_mat[pivot_row, col]
            swap_target = -1
            
            if pivot_val == 0:
                for r in range(pivot_row + 1, rows):
                    if current_mat[r, col] != 0:
                        swap_target = r
                        break
                
                if swap_target != -1:
                    # Operation: SWAP
                    current_mat.row_swap(pivot_row, swap_target)
                    steps_log.append({
                        "op": f"R_{{{pivot_row+1}}} \\leftrightarrow R_{{{swap_target+1}}}",
                        "desc": f"Swap row {pivot_row+1} and {swap_target+1} to get a non-zero pivot.",
                        "mat": current_mat.copy()
                    })
                    pivot_val = current_mat[pivot_row, col] # Update pivot val
                else:
                    # Column is all zeros, move to next column
                    continue

            # 2. NORMALIZE PIVOT (Make it 1)
            # Chapter 2 usually prefers making the pivot 1 immediately
            if pivot_val != 1 and pivot_val != 0:
                current_mat.row_op(pivot_row, lambda v, j: v / pivot_val)
                steps_log.append({
                    "op": f"R_{{{pivot_row+1}}} \\leftarrow \\frac{{1}}{{{sp.latex(pivot_val)}}} R_{{{pivot_row+1}}}",
                    "desc": f"Scale row {pivot_row+1} to make the pivot 1.",
                    "mat": current_mat.copy()
                })
            
            # 3. ELIMINATE BELOW (Row Echelon Form)
            for r in range(pivot_row + 1, rows):
                factor = current_mat[r, col]
                if factor != 0:
                    current_mat.row_op(r, lambda v, j: v - factor * current_mat[pivot_row, j])
                    steps_log.append({
                        "op": f"R_{{{r+1}}} \\leftarrow R_{{{r+1}}} - ({sp.latex(factor)}) R_{{{pivot_row+1}}}",
                        "desc": f"Eliminate entry below pivot in row {r+1}.",
                        "mat": current_mat.copy()
                    })

            pivot_row += 1

        # 4. BACK SUBSTITUTION (Reduced Row Echelon Form)
        # Iterate backwards from the last pivot
        # We find the actual pivots again to be safe
        pivots = []
        for r in range(rows):
            for c in range(target_cols):
                if current_mat[r, c] != 0:
                    pivots.append((r, c))
                    break
        
        for r_pivot, c_pivot in reversed(pivots):
            for r_above in range(r_pivot):
                factor = current_mat[r_above, c_pivot]
                if factor != 0:
                    current_mat.row_op(r_above, lambda v, j: v - factor * current_mat[r_pivot, j])
                    steps_log.append({
                        "op": f"R_{{{r_above+1}}} \\leftarrow R_{{{r_above+1}}} - ({sp.latex(factor)}) R_{{{r_pivot+1}}}",
                        "desc": f"Eliminate entry above pivot in row {r_above+1} (Back Substitution).",
                        "mat": current_mat.copy()
                    })
        
        # --- RENDER STEPS ---
        for i, step in enumerate(steps_log):
            with st.container():
                c_desc, c_math = st.columns([2, 3])
                with c_desc:
                    st.markdown(f"**Step {i+1}:** {step['desc']}")
                    st.latex(step['op'])
                with c_math:
                    st.latex(sp.latex(step['mat']))
                st.divider()

        # --- FINAL INTERPRETATION ---
        st.subheader("4. Final Solution")
        
        final_mat = steps_log[-1]['mat'] if steps_log else M
        
        if mode == "Solve System (Ax = b)":
            # Extract last column
            sol_vec = final_mat[:, -1]
            # Check consistency (Zero row on left, non-zero on right)
            rank_A = final_mat[:, :-1].rank()
            rank_Ab = final_mat.rank()
            
            if rank_A < rank_Ab:
                st.error("System is **Inconsistent** (No Solution). 0 = Non-Zero.")
            elif rank_A < A.cols:
                st.warning("System has **Infinitely Many Solutions** (Free Variables).")
                st.latex(sp.latex(final_mat))
            else:
                st.success("Unique Solution Found:")
                # We can construct the vector explicitly
                vars_list = sp.symbols(f"x_1:{A.cols+1}")
                st.latex(f"x = {sp.latex(sol_vec)}")
                
        else: # Inverse
            # Left side should be Identity
            left_block = final_mat[:, :A.cols]
            right_block = final_mat[:, A.cols:]
            
            if left_block == sp.eye(A.rows):
                st.success("Matrix is Invertible. The Inverse is:")
                st.latex(f"A^{{-1}} = {sp.latex(right_block)}")
            else:
                st.error("Matrix is **Singular** (Not Invertible). Reduced form does not yield Identity.")

    except Exception as e:
        st.error(f"Calculation Error during steps: {e}")