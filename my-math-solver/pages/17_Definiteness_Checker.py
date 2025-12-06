import streamlit as st
import numpy as np
import utils

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()
st.markdown("<h1 class='main-header'>Matrix Definiteness</h1>", unsafe_allow_html=True)

# 2. Curriculum Reference
with st.expander("📘 Curriculum References (Section 8.6 & 8.4)", expanded=True):
    st.markdown(r"""
    **How to decide definiteness (Section 8.6):**
    1.  **Theorem 8.29:** Transform symmetric matrix $A$ to diagonal $D$ ($B^T A B = D$).
    2.  **Exercise 8.28:** We classify $D$ based on its diagonal entries $d_{ii}$:
        * **Positive Definite:** All $d_{ii} > 0$.
        * **Negative Definite:** All $d_{ii} < 0$.
        * **Positive Semi-Definite:** All $d_{ii} \ge 0$.
        * **Negative Semi-Definite:** All $d_{ii} \le 0$.
        * **Indefinite:** Entries have mixed signs (some positive, some negative).
    
    **Connection to Optimization (Theorem 8.12):**
    * **Positive Definite** $\rightarrow$ Local Minimum 
    * **Negative Definite** $\rightarrow$ Local Maximum 
    * **Indefinite** $\rightarrow$ Saddle Point 
    """)

# --- HELPER ALGORITHMS (From Section 8.7) ---
def clean_matrix(M):
    M_clean = M.copy()
    M_clean[np.abs(M_clean) < 1e-10] = 0
    return M_clean

def get_diagonal_D(A_in):
    """
    Performs the symmetric reduction B.T @ A @ B = D 
    and returns the diagonal matrix D.
    """
    A = np.array(A_in, dtype=float)
    n = A.shape[0]
    
    # Augmented matrix is [I | A] ... we only need to track A to get D
    aug = np.vstack([np.eye(n), A.copy()])
    
    for k in range(n):
        pivot = aug[n + k, k]
        
        # 1. Pivot Handling (Swap/Add if 0)
        if np.isclose(pivot, 0):
            swap_idx = -1
            for j in range(k + 1, n):
                if not np.isclose(aug[n + j, j], 0):
                    swap_idx = j
                    break
            
            if swap_idx != -1:
                # Swap Col/Row k <-> swap_idx
                aug[:, [k, swap_idx]] = aug[:, [swap_idx, k]]
                aug[[n + k, n + swap_idx], :] = aug[[n + swap_idx, n + k], :]
                pivot = aug[n + k, k]
            else:
                add_idx = -1
                for j in range(k + 1, n):
                    if not np.isclose(aug[n + k, j], 0):
                        add_idx = j
                        break
                if add_idx != -1:
                    # Add Col/Row add_idx to k
                    aug[:, k] += aug[:, add_idx]
                    aug[n + k, :] += aug[n + add_idx, :]
                    pivot = aug[n + k, k]
        
        if np.isclose(pivot, 0): continue

        # 2. Elimination
        for j in range(k + 1, n):
            target = aug[n + k, j]
            if not np.isclose(target, 0):
                m = target / pivot
                aug[:, j] -= m * aug[:, k]
                aug[n + j, :] -= m * aug[n + k, :]

    D = clean_matrix(aug[n:, :])
    return D

# --- UI SECTION ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Matrix")
    default_mat = "[[2, -1], [-1, 2]]" 
    mat_str = utils.p_text_area("Symmetric Matrix A:", "def_mat_curr", default_mat, height=150)

with col2:
    if st.button("Analyze Definiteness", type="primary"):
        try:
            # Parse
            mat_list = eval(mat_str)
            A = np.array(mat_list, dtype=float)
            
            # Check Symmetry
            if not np.allclose(A, A.T):
                st.error("Matrix is NOT symmetric. The theorems in Chapter 8 apply to symmetric matrices.")
            else:
                # Run Curriculum Method
                D = get_diagonal_D(A)
                diag_entries = np.diag(D)
                
                st.subheader("2. Result ($B^T A B = D$)")
                st.write("We reduce $A$ to the diagonal matrix $D$:")
                st.write(D)
                
                st.write("**Diagonal Entries ($d_{ii}$):**")
                st.code(str(diag_entries))
                
                # Check Signs (Exercise 8.28)
                has_pos = np.any(diag_entries > 1e-10)
                has_neg = np.any(diag_entries < -1e-10)
                has_zero = np.any(np.isclose(diag_entries, 0, atol=1e-10))
                
                st.subheader("3. Conclusion")
                
                if has_pos and not has_neg and not has_zero:
                    st.success("Result: **Positive Definite (PD)**")
                    st.caption("Reason: All entries in $D$ are strictly positive ($>0$).")
                    # --- UPDATED LINE ---
                    st.info("Optimization: Critical point is a **Local Minimum** (Theorem 8.12(i)).")
                    
                elif has_neg and not has_pos and not has_zero:
                    st.error("Result: **Negative Definite (ND)**")
                    st.caption("Reason: All entries in $D$ are strictly negative ($<0$).")
                    st.info("Optimization: Critical point is a **Local Maximum** (Theorem 8.12(ii)).")
                    
                elif has_pos and not has_neg and has_zero:
                    st.warning("Result: **Positive Semi-Definite (PSD)**")
                    st.caption("Reason: Entries are $\ge 0$ (some are zero).")
                    
                elif has_neg and not has_pos and has_zero:
                    st.warning("Result: **Negative Semi-Definite (NSD)**")
                    st.caption("Reason: Entries are $\le 0$ (some are zero).")
                    
                elif has_pos and has_neg:
                    st.warning("Result: **Indefinite**")
                    st.caption("Reason: $D$ contains both positive and negative numbers.")
                    st.info("Optimization: Critical point is a **Saddle Point** (Theorem 8.12(iii)).")
                    
                else:
                    st.write("Result: Zero Matrix")

        except Exception as e:
            st.error(f"Error: {e}")
