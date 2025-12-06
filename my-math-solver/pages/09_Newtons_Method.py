import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import utils

# 1. Ensure Wide Layout
st.set_page_config(layout="wide")

utils.setup_page()
st.markdown("<h1 class='main-header'>Newton's Method Solver</h1>", unsafe_allow_html=True)

problem_type = utils.p_radio(
    "Choose Application:", 
    ["1D Root Finding (Solve f(x) = 0)", "Multivariable Optimization (Minimize f(v))"],
    "newton_mode",
    horizontal=True
)

st.markdown("---")

# ==========================================
# MODE A: 1D ROOT FINDING (Section 6.3.5)
# ==========================================
if problem_type == "1D Root Finding (Solve f(x) = 0)":
    st.info("Iteratively finds a root where $f(x) = 0$ using tangent lines. (**Section 6.3.5**)")
    st.latex(r"x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Input")
        f_str = utils.p_text_input("Function f(x):", "newton_root_func", "x**2 - 2")
        df_str = st.text_input("Derivative f'(x) (Optional):", "")
        
        x0 = utils.p_number_input("Start Value ($x_0$):", "newton_root_start", 1.0)
        iterations = utils.p_slider("Iterations:", "newton_iter_root", 1, 10, 5)
        
        st.markdown("### 📊 Plot Settings")
        c1, c2 = st.columns(2)
        x_min_u = c1.number_input("x min", value=-2.0)
        x_max_u = c2.number_input("x max", value=3.0)
        
        # --- NEW Y-AXIS CONTROLS ---
        c3, c4 = st.columns(2)
        y_min_u = c3.number_input("y min", value=-5.0)
        y_max_u = c4.number_input("y max", value=10.0)

    with col2:
        if st.button("Run Newton-Raphson (1D)", type="primary"):
            try:
                x = sp.symbols('x')
                f_expr = utils.parse_expr(f_str)
                
                if df_str.strip():
                    df_expr = utils.parse_expr(df_str)
                else:
                    df_expr = sp.diff(f_expr, x)
                    st.caption(f"Computed Derivative automatically: ${sp.latex(df_expr)}$")
                
                f_num = sp.lambdify(x, f_expr, 'numpy')
                df_num = sp.lambdify(x, df_expr, 'numpy')
                
                x_curr = x0
                history = [] 
                history.append((x_curr, f_num(x_curr)))
                
                st.subheader("Iteration Log")
                
                for i in range(iterations):
                    val = float(f_num(x_curr))
                    deriv = float(df_num(x_curr))
                    
                    if abs(deriv) < 1e-8:
                        st.error(f"Derivative near zero at x={x_curr:.4f}. Stopping.")
                        break
                        
                    x_next = x_curr - val / deriv
                    history.append((x_next, f_num(x_next)))
                    st.write(f"**Iter {i+1}:** $x = {x_next:.6f}$")
                    x_curr = x_next
                    
                st.success(f"Approximated Root: **{x_curr:.6f}**")
                
                # --- PLOTTING ---
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Create X values for the curve
                # We extend slightly beyond your chosen min/max to ensure the line looks continuous
                x_vals = np.linspace(x_min_u - 0.5, x_max_u + 0.5, 400)
                y_vals = f_num(x_vals)
                
                ax.plot(x_vals, y_vals, 'b-', label=f'$f(x)$', linewidth=2)
                ax.axhline(0, color='black', linewidth=1)
                
                hx = [p[0] for p in history]
                hy = [p[1] for p in history]
                
                ax.scatter(hx, hy, color='red', s=50, zorder=5, label='Newton Steps')
                
                for j in range(len(history)-1):
                    x_start, y_start = history[j]
                    x_end, y_end = history[j+1]
                    # Tangent line visual
                    ax.plot([x_start, x_end], [y_start, 0], 'r--', alpha=0.4)
                    # Vertical drop visual
                    ax.plot([x_end, x_end], [0, f_num(x_end)], 'k:', alpha=0.3)

                ax.set_title(f"Root Finding for ${sp.latex(f_expr)}$")
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                
                # --- APPLY USER SETTINGS ---
                ax.set_xlim(x_min_u, x_max_u)
                ax.set_ylim(y_min_u, y_max_u)
                
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# MODE B: MULTIVARIABLE OPTIMIZATION
# ==========================================
else:
    st.info("Iteratively finds a critical point (min/max/saddle) using the Hessian. (**Section 8.3**)")
    st.latex(r"v_{k+1} = v_k - [\nabla^2 F(v_k)]^{-1} \nabla F(v_k)")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("1. Problem Setup")
        func_str = utils.p_text_input("Objective Function $f(x, y)$:", "newton_func", "x**2 + 3*x*log(y) - y**3")
        start_point = utils.p_text_input("Start Point $v_0$ (comma separated):", "newton_start", "1.0, 1.0")
        iterations = utils.p_slider("Iterations:", "newton_iter", 1, 20, 5)

    with col2:
        st.subheader("2. Visualization")
        show_plot = st.checkbox("Show Contour Plot", value=True)
        zoom_level = utils.p_slider("Zoom:", "newton_zoom", 1.0, 10.0, 4.0)

    if st.button("Run Newton (Optimization)", type="primary"):
        try:
            f = utils.parse_expr(func_str)
            if f is None:
                st.error("Could not parse function.")
                st.stop()
                
            vars_sym = sorted(list(f.free_symbols), key=lambda s: s.name)
            
            grad = sp.Matrix([sp.diff(f, v) for v in vars_sym])
            hessian = sp.hessian(f, vars_sym)
            
            f_num = sp.lambdify(vars_sym, f, 'numpy')
            grad_num = sp.lambdify(vars_sym, grad, 'numpy')
            hess_num = sp.lambdify(vars_sym, hessian, 'numpy')

            current_vals = np.array([float(x.strip()) for x in start_point.split(',')])

            history = [current_vals.copy()]
            st.subheader("Iteration Log")
            
            for k in range(iterations):
                val_grad = np.array(grad_num(*current_vals)).flatten().astype(float)
                val_hess = np.array(hess_num(*current_vals)).astype(float)
                
                try:
                    H_inv = np.linalg.inv(val_hess)
                except np.linalg.LinAlgError:
                    st.error(f"**Iteration {k}:** Hessian is singular. Stopping.")
                    break
                
                step = H_inv @ val_grad
                current_vals = current_vals - step
                history.append(current_vals.copy())
                
                st.write(f"**Iter {k+1}:** $v = {np.round(current_vals, 4)}$")

            if show_plot and len(vars_sym) == 2:
                history_arr = np.array(history)
                x_center, y_center = np.mean(history_arr[:, 0]), np.mean(history_arr[:, 1])
                span = max(np.ptp(history_arr[:, 0]), np.ptp(history_arr[:, 1]))
                if span == 0: span = 1.0
                margin = span * 0.5 + zoom_level
                
                x_vis = np.linspace(x_center - margin, x_center + margin, 100)
                y_vis = np.linspace(y_center - margin, y_center + margin, 100)
                X, Y = np.meshgrid(x_vis, y_vis)
                Z = f_num(X, Y)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                cp = ax.contour(X, Y, Z, levels=15, cmap='Greys', alpha=0.5)
                ax.plot(history_arr[:, 0], history_arr[:, 1], 'k--', linewidth=1.5, alpha=0.8)
                ax.scatter(history_arr[0, 0], history_arr[0, 1], color='green', s=80, label='Start', zorder=5)
                ax.scatter(history_arr[-1, 0], history_arr[-1, 1], color='red', s=120, marker='*', label='End', zorder=5)
                ax.set_title(f"Minimization Path")
                ax.legend()
                ax.grid(False)
                st.pyplot(fig)

            st.success(f"**Result:** Critical point at $v \\approx {np.round(current_vals, 5)}$")

        except Exception as e:
            st.error(f"An error occurred: {e}")