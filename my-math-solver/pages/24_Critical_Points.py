import streamlit as st
import sympy as sp
import plotly.graph_objects as go
import numpy as np
from utils import setup_page, parse_expr, p_text_input

# 1. Page Configuration
setup_page()

st.markdown('<div class="main-header">Critical Points & Hessian Classifier</div>', unsafe_allow_html=True)

# 2. Input Section (Shared State)
st.markdown("### 1. Define Function")
default_func = "x**2+y**2+z**3+x*y+x+y-z" # Slightly more interesting Hessian

func_str = p_text_input(
    label="Enter f(x, y, ...):",
    key="cp_func_input",
    default=default_func
)

expr = parse_expr(func_str)

if not expr:
    st.info("Awaiting valid input...")
    st.stop()

# Auto-detect variables
variables = sorted(list(expr.free_symbols), key=lambda s: s.name)

if not variables:
    st.warning("Expression must contain variables.")
    st.stop()

# --- PRE-COMPUTATION (Shared Data) ---
# We compute derivatives and solutions once so they are available to all tabs
partials = [sp.diff(expr, v) for v in variables]
try:
    # Attempt to find critical points
    solutions = sp.solve(partials, variables, dict=True)
except Exception as e:
    solutions = []
    st.error(f"Could not solve system analytically: {e}")

# --- TABS CONFIGURATION ---
tab1, tab2 = st.tabs(["🔍 Find Critical Points", "🎓 Hessian Classification"])

# ==============================================================================
# TAB 1: FIND CRITICAL POINTS (Gradient Analysis)
# ==============================================================================
with tab1:
    st.markdown("### Step 1: Compute Gradient $\\nabla f$")
    
    # Display Partials
    grad_latex = []
    for var, diff in zip(variables, partials):
        grad_latex.append(f"f_{{{var.name}}} &= {sp.latex(diff)}")
    st.latex(r"\begin{aligned} " + r" \\ ".join(grad_latex) + r" \end{aligned}")

    st.markdown("### Step 2: Solve $\\nabla f = 0$")
    system_latex = [f"{sp.latex(eq)} = 0" for eq in partials]
    st.latex(r"\begin{cases} " + r" \\ ".join(system_latex) + r" \end{cases}")

    st.markdown("### Step 3: Critical Points")
    if not solutions:
        st.markdown('<div class="error-box">No critical points found.</div>', unsafe_allow_html=True)
    else:
        # Summary Box
        point_strs = []
        for sol in solutions:
            coords = [sol.get(v, sp.S.Zero) for v in variables] # Handle cases where var might cancel out
            tuple_str = r"\left(" + ",".join([sp.latex(c) for c in coords]) + r"\right)"
            point_strs.append(tuple_str)
        
        st.latex(r"\boxed{ " + ", \\quad ".join(point_strs) + r" }")

    # Visualization (2D Only)
    if len(variables) == 2 and solutions:
        st.markdown("---")
        st.markdown("**Surface Plot**")
        try:
            f_func = sp.lambdify(variables, expr, 'numpy')
            
            # Determine plot center
            cx, cy = 0, 0
            for sol in solutions:
                try:
                    # Use first real solution as center
                    vals = [complex(sol[v]) for v in variables]
                    if all(np.isreal(v) for v in vals):
                        cx, cy = vals[0].real, vals[1].real
                        break
                except: continue

            rng = 5
            x_range = np.linspace(cx - rng, cx + rng, 50)
            y_range = np.linspace(cy - rng, cy + rng, 50)
            X, Y = np.meshgrid(x_range, y_range)
            Z = f_func(X, Y)

            fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])

            # Plot Critical Points
            for sol in solutions:
                try:
                    vals = [complex(sol[v]) for v in variables]
                    if all(np.isreal(v) for v in vals):
                        real_vals = [v.real for v in vals]
                        z_val = complex(expr.subs(zip(variables, real_vals))).real
                        fig.add_trace(go.Scatter3d(
                            x=[real_vals[0]], y=[real_vals[1]], z=[z_val],
                            mode='markers', marker=dict(size=8, color='red'),
                            name='Crit. Point'
                        ))
                except: pass

            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plotting error: {e}")

# ==============================================================================
# TAB 2: HESSIAN CLASSIFICATION (Chapter 8)
# ==============================================================================
with tab2:
    if not solutions:
        st.info("No critical points to classify.")
    else:
        st.markdown("### Step 1: The Hessian Matrix")
        st.markdown("""
        The Hessian $\\nabla^2 f$ contains the second-order partial derivatives (Definition 8.2). 
        It is used to classify critical points (Theorem 8.12).
        """)

        # 1. Compute Hessian Matrix Symbolically
        # hessian is a Matrix object
        hessian_mat = sp.hessian(expr, variables)
        
        st.latex(r"\nabla^2 f = " + sp.latex(hessian_mat))

        st.markdown("### Step 2: Classification (Theorem 8.12)")
        
        for i, sol in enumerate(solutions, 1):
            # Evaluate Hessian at this point
            H_at_point = hessian_mat.subs(sol)
            
            # Prepare Latex for the point
            coords = [sol.get(v, sp.S.Zero) for v in variables]
            pt_latex = r"\left(" + ",".join([sp.latex(c) for c in coords]) + r"\right)"

            st.markdown(f"**Point {i}:** ${pt_latex}$")
            
            # Check for imaginary coordinates
            if any(c.has(sp.I) for c in coords):
                st.warning("Complex critical point - Classification not applicable.")
                continue

            # Display Evaluated Hessian
            st.latex(f"H(P_{i}) = {sp.latex(H_at_point)}")

            # Determine Definiteness
            # Theorem 8.12: 
            # Pos Def -> Min
            # Neg Def -> Max
            # Indefinite -> Saddle
            try:
                if H_at_point.is_positive_definite:
                    classification = "Local Minimum"
                    reason = "Positive Definite ($v^T H v > 0$)"
                    color = "success-box"
                elif H_at_point.is_negative_definite:
                    classification = "Local Maximum"
                    reason = "Negative Definite ($v^T H v < 0$)"
                    color = "success-box"
                elif H_at_point.is_indefinite:
                    classification = "Saddle Point"
                    reason = "Indefinite (mixed positive/negative)"
                    color = "result-card" # Neutral
                else:
                    # Semi-definite or singular cases (Inconclusive in strict terms of Thm 8.12)
                    classification = "Inconclusive (Degenerate)"
                    reason = "Determinant is zero or Semi-Definite"
                    color = "error-box"
                
                # Render Classification
                st.markdown(f"""
                <div class="{color}">
                    <b>Classification:</b> {classification}<br>
                    <small>Reason: Matrix is {reason}</small>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.warning(f"Could not classify numerically: {e}")
            
            st.markdown("---")