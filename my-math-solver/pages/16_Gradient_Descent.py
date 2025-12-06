import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import utils

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()
st.markdown("<h1 class='main-header'>Gradient Descent Visualizer</h1>", unsafe_allow_html=True)

# 2. Theory Block
with st.expander("📘 Theory & Curriculum References", expanded=True):
    st.markdown(r"""
    **Reference: Section 7.1 & Lemma 7.19**
    
    Gradient Descent is an iterative method to find a local minimum of a differentiable function $f$.
    
    **The Algorithm:**
    Start at a point $v_0$. Update the position iteratively:
    $$
    v_{k+1} = v_k - \lambda \nabla f(v_k)
    $$
    * $\nabla f(v_k)$ is the **Gradient** (points uphill).
    * $-\nabla f(v_k)$ points **downhill** (steepest descent).
    * $\lambda$ is the **Learning Rate** (step size). 
    
    **Lemma 7.19** guarantees that for a small enough $\lambda$, we move to a lower value: $f(v_{k+1}) < f(v_k)$.
    """)

# 3. Inputs
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Problem Setup")
    
    # Presets from Curriculum
    preset = st.selectbox(
        "Choose Example:", 
        ["Custom", "Simple Bowl (Ex 7.20)", "The Waffle (Ex 7.23)", "Elongated Valley"]
    )
    
    if preset == "Simple Bowl (Ex 7.20)":
        def_func = "x**2 + y**2"
        def_start = "2.0, 1.5"
        def_lr = 0.1
    elif preset == "The Waffle (Ex 7.23)":
        # Strictly convex but wavy
        def_func = "x**2 + y**2 - cos(x) - sin(y)"
        def_start = "1.5, 1.5"
        def_lr = 0.2
    elif preset == "Elongated Valley":
        # Harder for GD
        def_func = "x**2 + 10*y**2"
        def_start = "2.0, 1.0"
        def_lr = 0.05
    else:
        def_func = "x**2 + y**2"
        def_start = "2.0, 1.5"
        def_lr = 0.1

    func_str = utils.p_text_input("Minimize f(x, y):", "gd_func_new", def_func)
    start_str = utils.p_text_input("Start Point (x, y):", "gd_start_new", def_start)
    
    st.subheader("2. Hyperparameters")
    c1, c2 = st.columns(2)
    lr = c1.number_input("Learning Rate ($\lambda$):", value=def_lr, step=0.01, format="%.3f")
    steps = c2.slider("Iterations:", 1, 100, 20)

with col2:
    if st.button("Run Descent", type="primary"):
        try:
            # --- A. MATH SETUP ---
            f_expr = utils.parse_expr(func_str)
            vars_sym = sorted(list(f_expr.free_symbols), key=lambda s: s.name)
            
            if len(vars_sym) != 2:
                st.error("Visualization requires exactly 2 variables (e.g. x, y).")
                st.stop()
            
            # Gradients
            grad = [sp.diff(f_expr, v) for v in vars_sym]
            
            # Fast numerical functions
            f_num = sp.lambdify(vars_sym, f_expr, 'numpy')
            grad_num = sp.lambdify(vars_sym, grad, 'numpy')
            
            # --- B. ITERATION LOOP ---
            curr = np.array([float(s) for s in start_str.split(',')])
            path = [curr.copy()]
            z_path = [float(f_num(*curr))]
            
            for i in range(steps):
                g_val = np.array(grad_num(*curr)).flatten().astype(float)
                curr = curr - lr * g_val # The Update Rule
                
                path.append(curr.copy())
                z_path.append(float(f_num(*curr)))
            
            path = np.array(path)
            
            # --- C. RESULTS ---
            st.subheader("3. Results")
            st.success(f"Ended at: $({curr[0]:.4f}, {curr[1]:.4f})$")
            st.write(f"Final Value: $f(v) = {z_path[-1]:.5f}$")
            
            # --- D. 3D VISUALIZATION (Plotly) ---
            # Create a grid around the path
            margin = 1.0
            x_min, x_max = np.min(path[:,0]) - margin, np.max(path[:,0]) + margin
            y_min, y_max = np.min(path[:,1]) - margin, np.max(path[:,1]) + margin
            
            grid_res = 50
            x_vis = np.linspace(x_min, x_max, grid_res)
            y_vis = np.linspace(y_min, y_max, grid_res)
            X, Y = np.meshgrid(x_vis, y_vis)
            Z = f_num(X, Y)
            
            # 3D Surface
            fig3d = go.Figure()
            
            # The Landscape
            fig3d.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, name='Function'))
            
            # The Ball (Path)
            fig3d.add_trace(go.Scatter3d(
                x=path[:,0], y=path[:,1], z=z_path,
                mode='lines+markers',
                marker=dict(size=5, color='red'),
                line=dict(color='red', width=4),
                name='Descent Path'
            ))
            
            fig3d.update_layout(
                title='3D Descent Path (Drag to Rotate)',
                scene=dict(
                    xaxis_title='x',
                    yaxis_title='y',
                    zaxis_title='f(x,y)'
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                height=500
            )
            st.plotly_chart(fig3d, use_container_width=True)
            
            # --- E. 2D CONTOUR PLOT (Matplotlib) ---
            st.subheader("2D Contour View")
            fig2d, ax = plt.subplots(figsize=(8, 5))
            
            # Contour map
            cp = ax.contourf(X, Y, Z, levels=25, cmap='viridis', alpha=0.6)
            fig2d.colorbar(cp)
            
            # Path arrows
            ax.plot(path[:,0], path[:,1], 'r-', linewidth=1.5, alpha=0.8)
            ax.scatter(path[0,0], path[0,1], c='green', s=100, label='Start', zorder=5)
            ax.scatter(path[-1,0], path[-1,1], c='black', marker='x', s=100, label='End', zorder=5)
            
            # Quiver plot (little arrows showing gradient direction)
            # We subsample the grid to avoid clutter
            skip = 5
            q_X, q_Y = X[::skip, ::skip], Y[::skip, ::skip]
            # Negative gradient points downhill
            grads = np.array([grad_num(x, y) for x, y in zip(np.ravel(q_X), np.ravel(q_Y))])
            U = -grads[:,0].reshape(q_X.shape)
            V = -grads[:,1].reshape(q_Y.shape)
            ax.quiver(q_X, q_Y, U, V, color='white', alpha=0.3)

            ax.set_title("Contour Map with Steepest Descent Directions")
            ax.legend()
            st.pyplot(fig2d)
            
            # --- F. CONVERGENCE GRAPH ---
            st.subheader("Convergence (Value over Time)")
            fig_conv, ax_conv = plt.subplots(figsize=(8, 3))
            ax_conv.plot(z_path, 'b.-')
            ax_conv.set_xlabel("Iteration")
            ax_conv.set_ylabel("f(x,y)")
            ax_conv.set_title("Does it go down?")
            ax_conv.grid(True)
            st.pyplot(fig_conv)

        except Exception as e:
            st.error(f"Computation Error: {e}")