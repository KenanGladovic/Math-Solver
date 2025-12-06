import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import utils

# 1. Ensure Wide Layout
st.set_page_config(layout="wide")

utils.setup_page()
st.markdown("<h1 class='main-header'>Math Plotter & Sketcher</h1>", unsafe_allow_html=True)
st.info("Visualizes 1D functions, 2D contours, and 3D surfaces (Interactive).")

# --- 2D PLOTTING HELPER FUNCTIONS ---
def plot_objective_contours(f_func, X, Y, ax, f_expr):
    """Draws contour lines for the 2D objective function."""
    Z = f_func(X, Y)
    contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_title(f"Contour Plot of ${sp.latex(f_expr)}$ with Feasible Region")

def plot_constraints_region(const_str, x_sym, y_sym, X, Y, x_min, x_max, y_min, y_max, ax):
    """Parses constraints, computes feasible mask, and shades the region."""
    constraints = []
    lines = [l.strip() for l in const_str.split('\n') if l.strip()]
    
    for l in lines:
        if "<=" in l:
            lhs, rhs = l.split("<=")
            expr = utils.parse_expr(lhs) - utils.parse_expr(rhs)
            if expr is not None: constraints.append(sp.lambdify((x_sym, y_sym), expr, 'numpy'))
        elif ">=" in l:
            lhs, rhs = l.split(">=")
            expr = utils.parse_expr(rhs) - utils.parse_expr(lhs)
            if expr is not None: constraints.append(sp.lambdify((x_sym, y_sym), expr, 'numpy'))
    
    feasible_mask = np.ones_like(X, dtype=bool)
    if constraints:
        for g_func in constraints:
            val = g_func(X, Y)
            if np.isscalar(val):
                if val > 0: feasible_mask[:] = False
            else:
                feasible_mask &= (val <= 0)

        ax.imshow(feasible_mask, extent=[x_min, x_max, y_min, y_max], origin='lower', 
                  alpha=0.3, cmap='Greens', aspect='auto')
        
        for g_func in constraints:
            g_val = g_func(X, Y)
            if not np.isscalar(g_val):
                ax.contour(X, Y, g_val, levels=[0], colors='red', linewidths=2, linestyles='dashed')

def generate_2d_plot(func_str, const_str, x_min, x_max, y_min, y_max):
    x_sym, y_sym = sp.symbols('x y')
    f = utils.parse_expr(func_str)
    
    if f is None:
        st.error("Invalid objective function.")
        return
        
    f_func = sp.lambdify((x_sym, y_sym), f, 'numpy')
    
    res = 100
    x_vals = np.linspace(x_min, x_max, res)
    y_vals = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plot_objective_contours(f_func, X, Y, ax, f)
    plot_constraints_region(const_str, x_sym, y_sym, X, Y, x_min, x_max, y_min, y_max, ax)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    st.pyplot(fig)
    st.caption("**Legend:** Green shaded area = Feasible Region ($C$). Red dashed lines = Constraint Boundaries.")

# --- MAIN UI ---
plot_type = utils.p_radio(
    "Plot Type:", 
    ["1D Function (f(x))", "2D Function & Constraints", "3D Surface (Interactive)"], 
    "plot_type_main",
    horizontal=True
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Setup")
    
    if plot_type == "1D Function (f(x))":
        func_default = "x**2 - 2*x + 1"
        const_default = ""
    elif plot_type == "2D Function & Constraints":
        # Example 4.33 from the curriculum
        func_default = "-x - y"
        const_default = "2*x + y <= 1\nx + 2*y <= 1\nx >= 0\ny >= 0"
    else: # 3D Surface
        func_default = "(x + y)**2 - x - y"
        const_default = "" 
        
    func_str = st.text_input("Objective Function $f(x, y, t, ...)$:", value=func_default)
    
    if plot_type == "2D Function & Constraints":
        st.write("Constraints (for 2D shading):")
        const_str = st.text_area("Inequalities (one per line):", const_default)
    else:
        const_str = ""
    
    st.subheader("Ranges")
    c1, c2 = st.columns(2)
    x_min = c1.number_input("x min", value=-3.0)
    x_max = c2.number_input("x max", value=3.0)
    
    # --- CHANGED: Always show Y-Axis controls now ---
    y_min = c1.number_input("y min", value=-3.0)
    y_max = c2.number_input("y max", value=3.0)

with col2:
    if st.button("Generate Plot", type="primary"):
        try:
            if plot_type == "1D Function (f(x))":
                f = utils.parse_expr(func_str)
                x_sym = sp.symbols('x')
                if x_sym not in f.free_symbols and len(f.free_symbols) > 0:
                    st.error("Function must be in terms of 'x' for 1D plotting.")
                else:
                    f_func = sp.lambdify(x_sym, f, 'numpy')
                    x_vals = np.linspace(x_min, x_max, 500)
                    y_vals = f_func(x_vals)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(x_vals, y_vals, label=f"f(x) = ${sp.latex(f)}$")
                    ax.set_xlabel('x')
                    ax.set_ylabel('f(x)')
                    ax.set_title(f"Plot of $f(x) = {sp.latex(f)}$")
                    ax.axhline(0, color='gray', linewidth=0.5)
                    ax.axvline(0, color='gray', linewidth=0.5)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    
                    # --- CHANGED: Apply Y Limits here ---
                    ax.set_ylim(y_min, y_max)
                    # ------------------------------------
                    
                    ax.legend()
                    st.pyplot(fig)

            elif plot_type == "2D Function & Constraints":
                generate_2d_plot(func_str, const_str, x_min, x_max, y_min, y_max)

            elif plot_type == "3D Surface (Interactive)":
                x_sym, y_sym = sp.symbols('x y')
                f = utils.parse_expr(func_str)
                if f is None:
                    st.error("Invalid objective function.")
                    st.stop()
                f_func = sp.lambdify((x_sym, y_sym), f, 'numpy')
                
                res = 100 
                x_vals = np.linspace(x_min, x_max, res)
                y_vals = np.linspace(y_min, y_max, res)
                X, Y = np.meshgrid(x_vals, y_vals)
                Z = f_func(X, Y)
                
                st.subheader("Interactive 3D Surface Plot")
                
                fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

                fig.update_layout(
                    scene=dict(
                        xaxis_title='X Axis',
                        yaxis_title='Y Axis',
                        zaxis_title='f(x,y)',
                        aspectratio=dict(x=1, y=1, z=0.7), 
                        camera_eye=dict(x=1.2, y=1.2, z=0.6)
                    ),
                    margin=dict(l=0, r=0, b=0, t=0), 
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Plotting Error: {e}")