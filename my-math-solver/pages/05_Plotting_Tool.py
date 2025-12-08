import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import utils

# --- CONFIGURATION ---
st.set_page_config(page_title="Math Visualizer", layout="wide", page_icon="📈")
utils.setup_page()

st.markdown("<h1 class='main-header'>Math Visualization Tool</h1>", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def plot_1d_analysis(f_expr, x_range, y_range=None, show_crit=True):
    """Plots a 1D function with optional critical points."""
    x_sym = sp.Symbol('x')
    f_func = sp.lambdify(x_sym, f_expr, 'numpy')
    
    # Plot Data
    x_vals = np.linspace(x_range[0], x_range[1], 500)
    y_vals = f_func(x_vals)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_vals, y_vals, label=f"$f(x) = {sp.latex(f_expr)}$", linewidth=2, color="#2196F3")
    
    # Derivative for critical points
    if show_crit:
        try:
            f_prime = sp.diff(f_expr, x_sym)
            crit_points = sp.solve(f_prime, x_sym)
            real_crits = [float(p.evalf()) for p in crit_points if p.is_real and x_range[0] <= p <= x_range[1]]
            
            for cp in real_crits:
                y_cp = f_func(cp)
                if y_range is None or (y_range[0] <= y_cp <= y_range[1]):
                    ax.scatter([cp], [y_cp], color='red', zorder=5, s=80)
                    ax.annotate(f"Crit: ({cp:.2f}, {y_cp:.2f})", (cp, y_cp), xytext=(0, 10), textcoords='offset points')
        except:
            pass

    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)
        
    ax.legend()
    ax.set_title("1D Function Analysis")
    return fig

def plot_2d_optimization(f_expr, constraints_list, x_range, y_range, show_contours=False):
    """Plots 2D Feasible Region + Optional Contours."""
    x_sym, y_sym = sp.symbols('x y')
    f_func = sp.lambdify((x_sym, y_sym), f_expr, 'numpy')

    # Grid Setup
    res = 200
    x = np.linspace(x_range[0], x_range[1], res)
    y = np.linspace(y_range[0], y_range[1], res)
    X, Y = np.meshgrid(x, y)
    Z = f_func(X, Y)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate Feasible Mask
    feasible_mask = np.ones_like(X, dtype=bool)
    
    for const_str in constraints_list:
        try:
            if "<=" in const_str:
                lhs, rhs = const_str.split("<=")
                g_expr = utils.parse_expr(lhs) - utils.parse_expr(rhs)
                g_func = sp.lambdify((x_sym, y_sym), g_expr, 'numpy')
                val = g_func(X, Y)
                
                if np.isscalar(val):
                    if val > 0: feasible_mask[:] = False
                else:
                    feasible_mask &= (val <= 1e-5)
                
                ax.contour(X, Y, val, levels=[0], colors='red', linewidths=2.5, linestyles='--')

            elif ">=" in const_str:
                lhs, rhs = const_str.split(">=")
                g_expr = utils.parse_expr(rhs) - utils.parse_expr(lhs)
                g_func = sp.lambdify((x_sym, y_sym), g_expr, 'numpy')
                val = g_func(X, Y)

                if np.isscalar(val):
                    if val > 0: feasible_mask[:] = False
                else:
                    feasible_mask &= (val <= 1e-5)

                ax.contour(X, Y, val, levels=[0], colors='red', linewidths=2.5, linestyles='--')
        except: pass

    # Plot Visuals
    ax.imshow(feasible_mask, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
              origin='lower', cmap='Greens', alpha=0.4, aspect='auto')

    if show_contours:
        contours = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
        ax.clabel(contours, inline=True, fontsize=8)

    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_title(f"Objective: ${sp.latex(f_expr)}$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    return fig

def plot_3d_surface(f_expr, x_range, y_range):
    """Standard 3D Surface Plot (f(x,y))"""
    x_sym, y_sym = sp.symbols('x y')
    f_func = sp.lambdify((x_sym, y_sym), f_expr, 'numpy')

    x = np.linspace(x_range[0], x_range[1], 50)
    y = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = f_func(X, Y)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        title=f"Surface: f(x,y)",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )
    return fig

def plot_3d_geometric_analysis_with_function(constraints_list, f_expr_3d, x_range, y_range, z_range):
    """
    Plots 3D Feasible Region C (like a pyramid) AND an optional function surface f(x,y,z)=k.
    Uses Plotly Isosurface.
    """
    x_sym, y_sym, z_sym = sp.symbols('x y z')
    
    # Grid Setup (Higher Res for sharper look)
    res = 40 
    x = np.linspace(x_range[0], x_range[1], res)
    y = np.linspace(y_range[0], y_range[1], res)
    z = np.linspace(z_range[0], z_range[1], res)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # 1. Create Constraint Volume
    # We want max(g1, g2, ...) <= 0.
    max_g_val = np.full_like(X, -1.0e9) 
    has_constraints = False
    
    for const_str in constraints_list:
        try:
            if "<=" in const_str:
                lhs, rhs = const_str.split("<=")
                g_expr = utils.parse_expr(lhs) - utils.parse_expr(rhs)
            elif ">=" in const_str:
                lhs, rhs = const_str.split(">=")
                g_expr = utils.parse_expr(rhs) - utils.parse_expr(lhs)
            else:
                continue
            
            has_constraints = True
            g_func = sp.lambdify((x_sym, y_sym, z_sym), g_expr, 'numpy')
            val = g_func(X, Y, Z)
            
            # Combine: Intersection of constraints corresponds to max(g_i)
            max_g_val = np.maximum(max_g_val, val)
        except: pass

    fig = go.Figure()

    # 2. Plot Feasible Region (g <= 0)
    if has_constraints:
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=max_g_val.flatten(),
            isomin=-1000, # Fill everything below 0
            isomax=0,     # Boundary at 0
            surface_count=2,
            colorscale='Greens',
            opacity=0.3, 
            name='Feasible Region C',
            showscale=False,
            caps=dict(x_show=False, y_show=False)
        ))

    # 3. Plot Objective Function Surface (f(x,y,z) = 0)
    if f_expr_3d:
        try:
            f_func = sp.lambdify((x_sym, y_sym, z_sym), f_expr_3d, 'numpy')
            f_vals = f_func(X, Y, Z)
            
            # Plot isosurface where f(x,y,z) is close to 0
            fig.add_trace(go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=f_vals.flatten(),
                isomin=-0.5, # Small range around 0 to simulate a surface
                isomax=0.5,
                surface_count=3,
                colorscale='Oranges',
                opacity=0.4,
                name='Objective f=0'
            ))
        except:
            pass

    fig.update_layout(
        title="3D Geometric Analysis",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(range=x_range, showgrid=True, gridcolor='gray'),
            yaxis=dict(range=y_range, showgrid=True, gridcolor='gray'),
            zaxis=dict(range=z_range, showgrid=True, gridcolor='gray'),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700
    )
    return fig

# --- TABS ---
tabs = st.tabs(["📈 1D Functions", "🎨 2D Optimization", "🧊 3D Surfaces (f)", "📦 3D Constraints (C)"])

# ==============================================================================
# TAB 1: 1D ANALYSIS
# ==============================================================================
with tabs[0]:
    st.subheader("1D Analysis (Derivatives & Roots)")
    c1, c2 = st.columns([1, 2])
    with c1:
        f_1d = st.text_input("f(x):", "x**3 - 3*x + 1", key="f1")
        cx1, cx2 = st.columns(2)
        x_min = cx1.number_input("X Min", -3.0, key="x1min")
        x_max = cx2.number_input("X Max", 3.0, key="x1max")
        
        use_ylim = st.checkbox("Set Manual Y-Limits", value=False)
        cy1, cy2 = st.columns(2)
        y_min_1d = cy1.number_input("Y Min", -10.0, disabled=not use_ylim)
        y_max_1d = cy2.number_input("Y Max", 10.0, disabled=not use_ylim)
        
        # New Option: Hide Critical Points
        show_crit = st.checkbox("Show Critical Points", value=True)
        
    with c2:
        if f_1d:
            expr = utils.parse_expr(f_1d)
            if expr:
                y_range = [y_min_1d, y_max_1d] if use_ylim else None
                st.pyplot(plot_1d_analysis(expr, [x_min, x_max], y_range, show_crit))
                
                x = sp.Symbol('x')
                st.markdown("### Math Details")
                st.latex(f"f'(x) = {sp.latex(sp.diff(expr, x))}")
                st.latex(f"f''(x) = {sp.latex(sp.diff(expr, x, 2))}")

# ==============================================================================
# TAB 2: 2D OPTIMIZATION
# ==============================================================================
with tabs[1]:
    st.subheader("2D Optimization & Constraints")
    col_input, col_plot = st.columns([1, 1.5])
    
    with col_input:
        scenario = st.selectbox("Load Exam Problem:", 
            ["Custom", 
             "Jan 2025: Op 1 (Circle + Line)", 
             "Jan 2024: Op 1 (Intersection of Circles)", 
             "Jan 2023: Op 1 (Polygon)"])
        
        if scenario == "Jan 2025: Op 1 (Circle + Line)":
            def_f = "(x+y)**2 - x - y"
            def_c = "x**2 + y**2 <= 4\nx + y >= 1"
            def_lims = [-3.0, 3.0, -3.0, 3.0]
        elif scenario == "Jan 2024: Op 1 (Intersection of Circles)":
            def_f = "x**2 - y"
            def_c = "x**2 + y**2 <= 1\n(x+1)**2 + (y-1)**2 <= 1"
            def_lims = [-2.0, 1.0, -1.0, 2.0]
        elif scenario == "Jan 2023: Op 1 (Polygon)":
            def_f = "3*x + 2*y"
            def_c = "x + y <= 10\n2*x + y <= 15\nx >= 0\ny >= 0"
            def_lims = [-1.0, 12.0, -1.0, 12.0]
        else:
            def_f = "-x - y"
            def_c = "2*x + y <= 1\nx + 2*y <= 1\nx >= 0\ny >= 0"
            def_lims = [-0.5, 2.0, -0.5, 2.0]

        with st.form("plot2d"):
            f_in = st.text_input("Objective f(x,y):", def_f)
            c_in = st.text_area("Constraints (one per line):", def_c, height=120)
            
            sc1, sc2 = st.columns(2)
            xm = sc1.number_input("X Min", value=def_lims[0])
            xM = sc2.number_input("X Max", value=def_lims[1])
            ym = sc1.number_input("Y Min", value=def_lims[2])
            yM = sc2.number_input("Y Max", value=def_lims[3])
            
            show_cnt = st.checkbox("Show Objective Contours", value=False)
            submitted = st.form_submit_button("Plot Graph")

    with col_plot:
        if submitted or f_in:
            f_expr = utils.parse_expr(f_in)
            if f_expr:
                constraints = [c.strip() for c in c_in.split('\n') if c.strip()]
                fig = plot_2d_optimization(f_expr, constraints, [xm, xM], [ym, yM], show_cnt)
                st.pyplot(fig)
                
                st.markdown("### 🧮 Gradient & Hessian")
                st.latex(f"\\nabla f = {sp.latex([sp.diff(f_expr, 'x'), sp.diff(f_expr, 'y')])}")
                H = [[sp.diff(f_expr, 'x', 2), sp.diff(f_expr, 'x', 'y')],
                     [sp.diff(f_expr, 'y', 'x'), sp.diff(f_expr, 'y', 2)]]
                st.latex(r"\nabla^2 f = " + sp.latex(sp.Matrix(H)))

# ==============================================================================
# TAB 3: 3D SURFACES
# ==============================================================================
with tabs[2]:
    st.subheader("3D Function Visualization")
    st.info("Visualizes the graph of a function $z = f(x,y)$. Use this to see if a function is convex (bowl-shaped) or has saddle points.")
    
    c1, c2 = st.columns([1, 3])
    with c1:
        f_3d = st.text_input("f(x,y):", "x**2 - y**2", key="f3_func")
        rng_3d = st.slider("Range", 1, 10, 5, key="rng_3d_func")
    
    with c2:
        expr = utils.parse_expr(f_3d)
        if expr:
            st.plotly_chart(plot_3d_surface(expr, [-rng_3d, rng_3d], [-rng_3d, rng_3d]), use_container_width=True)

# ==============================================================================
# TAB 4: 3D CONSTRAINTS (New)
# ==============================================================================
with tabs[3]:
    st.subheader("3D Geometric Analysis")
    st.info("Visualizes 3D sets defined by inequalities (like the Pyramid in Jan 22).")
    
    # --- Callback to handle preset loading ---
    def load_preset():
        selection = st.session_state.preset_selector
        if selection == "IMO Jan 22 Q1 (Pyramid + Sphere)":
            st.session_state.c3d_text = "z - x <= 0\nz + x <= 0\nz - y <= 0\nz + y <= 0\n-1 - z <= 0"
            st.session_state.f_opt_text = "(x**2 + y**2 + (z-1)**2)" # The sphere
        else:
            # Default/Custom
            st.session_state.c3d_text = "z - x <= 0\nz + x <= 0\nz - y <= 0\nz + y <= 0\n-1 - z <= 0"
            st.session_state.f_opt_text = ""

    # Initialize Session State if not present
    if "c3d_text" not in st.session_state:
        st.session_state.c3d_text = "z - x <= 0\nz + x <= 0\nz - y <= 0\nz + y <= 0\n-1 - z <= 0"
    if "f_opt_text" not in st.session_state:
        st.session_state.f_opt_text = ""

    c1, c2 = st.columns([1, 2])
    
    with c1:
        # 1. Preset Selector with Callback
        st.selectbox(
            "Load Example:", 
            ["Custom", "IMO Jan 22 Q1 (Pyramid + Sphere)"], 
            key="preset_selector",
            on_change=load_preset
        )
        
        # 2. Text Areas linked to Session State
        c_3d = st.text_area(
            "Constraints (g(x,y,z) <= 0):", 
            key="c3d_text", 
            height=150
        )
        
        f_opt = st.text_input(
            "Optional Function f(x,y,z) (Plot Level Set = 0):", 
            key="f_opt_text"
        )
        
        r3 = st.slider("Axis Range (+/-)", 1.0, 5.0, 2.0, key="r3d")
        
    with c2:
        if c_3d:
            const_list_3d = [l for l in c_3d.split('\n') if l.strip()]
            
            f_expr_opt = None
            if f_opt:
                f_expr_opt = utils.parse_expr(f_opt)
                
            st.plotly_chart(plot_3d_geometric_analysis_with_function(const_list_3d, f_expr_opt, [-r3, r3], [-r3, r3], [-r3, r3]), use_container_width=True)