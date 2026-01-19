import streamlit as st
import sympy as sp
import numpy as np
import utils

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()

st.markdown("<h1 class='main-header'>Newton's Method (Optimization)</h1>", unsafe_allow_html=True)

# 2. Theory Block (Curriculum References)
with st.expander("📘 Theory & Curriculum Reference (Section 8.3)", expanded=True):
    st.markdown(r"""
    **Goal:** Find a critical point ($x^*$) where $\nabla f(x^*) = 0$.
    
    **The Method (Section 8.3):**
    Newton's method approximates the function $f$ as a quadratic function (using the Taylor expansion) and jumps directly to its minimum.
    
    **The Iteration Step:**
    $$
    v_{k+1} = v_k - [H_f(v_k)]^{-1} \nabla f(v_k)
    $$
    * $v_k$: Current point (vector).
    * $\nabla f(v_k)$: **Gradient** at current point (Definition 7.6).
    * $H_f(v_k)$: **Hessian Matrix** at current point (Definition 8.2).
    
    **Note:** This method converges **quadratically** (very fast) if started close to a minimum, but fails if $H_f$ is singular (determinant = 0).
    """)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Setup")
    # Default: A function where Newton works well
    func_str = utils.p_text_input("Function f(x, y):", "newton_func", "x**4 + y**4 - 4*x*y")
    start_str = utils.p_text_input("Start Point (x, y):", "newton_start", "2.0, 3.0")
    iterations = st.slider("Max Iterations:", 1, 10, 3)

with col2:
    if st.button("Run Newton Iteration", type="primary"):
        try:
            # --- A. MATH SETUP ---
            f = utils.parse_expr(func_str)
            if f is None:
                st.error("Invalid function. Please check your syntax.")
                st.stop()
                
            vars_sym = sorted(list(f.free_symbols), key=lambda s: s.name)
            if len(vars_sym) == 0:
                st.error("Function must have variables (e.g. x, y).")
                st.stop()
            
            # Symbolic Derivatives
            grad_sym = sp.Matrix([sp.diff(f, v) for v in vars_sym])
            hessian_sym = sp.hessian(f, vars_sym)
            
            # Numeric Conversion (for speed & clean formatting)
            f_num = sp.lambdify(vars_sym, f, 'numpy')
            grad_num_func = sp.lambdify(vars_sym, grad_sym, 'numpy')
            hess_num_func = sp.lambdify(vars_sym, hessian_sym, 'numpy')
            
            # Initial Values
            try:
                curr_vals = np.array([float(v.strip()) for v in start_str.split(',')])
            except:
                st.error(f"Start point format error. Expected {len(vars_sym)} numbers separated by comma.")
                st.stop()

            if len(curr_vals) != len(vars_sym):
                st.error(f"Dimension mismatch! Function has {len(vars_sym)} variables ({vars_sym}), but you entered {len(curr_vals)} coordinates.")
                st.stop()
            
            st.subheader("2. Iteration Process")
            
            # --- B. ITERATION LOOP ---
            for k in range(iterations):
                with st.container():
                    st.markdown(f"#### Iteration {k+1}")
                    
                    # 1. Evaluate at current point
                    # Note: lambdify returns numpy arrays, we flatten them to simple lists for display
                    g_val = np.array(grad_num_func(*curr_vals)).flatten()
                    H_val = np.array(hess_num_func(*curr_vals))
                    
                    # 2. Check Singularity (Invertibility)
                    det_H = np.linalg.det(H_val)
                    if abs(det_H) < 1e-9:
                        st.error(f"**STOP:** Hessian is Singular (Determinant $\\approx 0$) at {tuple(np.round(curr_vals, 4))}.")
                        st.write("Cannot calculate inverse $H^{-1}$. Newton's method fails here.")
                        st.stop()
                    
                    H_inv = np.linalg.inv(H_val)
                    
                    # 3. Newton Step: v_new = v_old - H_inv * grad
                    step = - H_inv @ g_val
                    new_vals = curr_vals + step
                    
                    # --- DISPLAY BLOCK ---
                    c_left, c_right = st.columns(2)
                    
                    with c_left:
                        st.write(f"**Current Point ($v_{k}$):**")
                        st.latex(f"{sp.latex(sp.Matrix(np.round(curr_vals, 4)).T)}")
                        
                        st.write("**Gradient ($\\nabla f$):**")
                        st.latex(sp.latex(sp.Matrix(np.round(g_val, 4)).T))
                        
                    with c_right:
                        st.write("**Inverse Hessian ($H^{-1}$):**")
                        st.latex(sp.latex(sp.Matrix(np.round(H_inv, 4))))
                        
                        st.write("**Update ($v_{k+1}$):**")
                        # Show the math: Old + Step = New
                        st.latex(f"v_{{{k+1}}} = {sp.latex(sp.Matrix(np.round(new_vals, 4)).T)}")

                    # Update for next loop
                    curr_vals = new_vals
                    st.divider()

            # --- C. FINAL RESULT ---
            # Format nicely as standard python floats (removes np.float64 wrapper)
            final_tuple = tuple([float(x) for x in np.round(curr_vals, 5)])
            
            st.success(f"**Final Result:** $v_{{{iterations}}} \\approx {final_tuple}$")
            
            # Optional: Check if gradient is actually close to 0
            final_grad = np.linalg.norm(grad_num_func(*curr_vals))
            if final_grad < 1e-3:
                st.caption(f"✅ Converged! Gradient norm is tiny ({final_grad:.1e}). This is likely a critical point.")
            else:
                st.warning(f"⚠️ Has not fully converged yet. Gradient norm is {final_grad:.4f}. Try more iterations.")

        except Exception as e:
            st.error(f"Computation Error: {e}")