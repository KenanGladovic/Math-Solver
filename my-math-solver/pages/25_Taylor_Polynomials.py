import streamlit as st
import sympy as sp
import plotly.graph_objects as go
import numpy as np
from utils import setup_page, parse_expr, p_text_input, p_number_input

# 1. Page Configuration
setup_page()

st.markdown('<div class="main-header">Multivariate Taylor Polynomials</div>', unsafe_allow_html=True)

# FIXED: Added 'r' before the string to treat backslashes literally
st.markdown(r"""
Compute the $k$-th degree Taylor polynomial approximation of a function $f$ around a point $\mathbf{a}$.
For optimization, the **2nd Degree (Quadratic)** approximation is most critical:

$$ 
P_2(\mathbf{x}) = f(\mathbf{a}) + \nabla f(\mathbf{a})^T (\mathbf{x} - \mathbf{a}) + \frac{1}{2} (\mathbf{x} - \mathbf{a})^T \nabla^2 f(\mathbf{a}) (\mathbf{x} - \mathbf{a}) 
$$
""")

# 2. Input Section
col1, col2 = st.columns([2, 1])

with col1:
    # Default from IMO Jun 2022 Problem 2 style
    default_func = "x**2 * y + 3 * y**2" 
    func_str = p_text_input("Function f(x, y, ...):", "tp_func", default_func)

with col2:
    degree = int(p_number_input("Degree (k):", "tp_degree", 2, min_value=1, max_value=4, step=1))
    
default_point = "1, 2"
point_str = p_text_input("Expansion point a (comma separated):", "tp_point", default_point)

# 3. Processing
expr = parse_expr(func_str)

if expr:
    # Auto-detect variables
    variables = sorted(list(expr.free_symbols), key=lambda s: s.name)
    
    # Parse point string
    try:
        point_vals = [float(x.strip()) for x in point_str.split(',')]
        if len(point_vals) != len(variables):
            st.error(f"Error: Expected {len(variables)} coordinates for variables {variables}, got {len(point_vals)}.")
            st.stop()
        
        # Create substitution dictionary: {x: 1, y: 2}
        sub_dict = dict(zip(variables, point_vals))
        point_map = {v: val for v, val in zip(variables, point_vals)}
        
    except ValueError:
        st.warning("Please enter numeric coordinates for the expansion point.")
        st.stop()

    st.markdown("---")
    
    # 4. Calculation Logic
    poly = 0
    
    # A. 0th Order Term: f(a)
    f_val = expr.subs(point_map)
    poly += f_val
    
    st.markdown(r"### 1. Function Value at $\mathbf{a}$")
    st.latex(f"f({point_str}) = {sp.latex(f_val)}")
    
    # B. 1st Order Terms: Gradient * (x-a)
    st.markdown(r"### 2. Gradient Term (Degree 1)")
    
    grad = []
    first_order_terms = 0
    
    for i, var in enumerate(variables):
        diff = sp.diff(expr, var)
        val = diff.subs(point_map)
        grad.append(val)
        
        term = val * (var - point_vals[i])
        first_order_terms += term
    
    poly += first_order_terms
    
    # Display Gradient Vector
    # We use double backslashes for newlines inside the matrix logic
    grad_latex = r"\begin{bmatrix} " + r" \\ ".join([sp.latex(g) for g in grad]) + r" \end{bmatrix}"
    st.latex(r"\nabla f(\mathbf{a}) = " + grad_latex)
    
    # C. 2nd Order Terms (if k >= 2)
    if degree >= 2:
        st.markdown(r"### 3. Hessian Term (Degree 2)")
        
        hessian = sp.zeros(len(variables), len(variables))
        second_order_terms = 0
        
        for i, v1 in enumerate(variables):
            for j, v2 in enumerate(variables):
                # Calculate second derivative
                diff2 = sp.diff(expr, v1, v2)
                val2 = diff2.subs(point_map)
                hessian[i, j] = val2
                
                # Add to polynomial: (1/2) * f_xy * (x-a)(y-b)
                term = val2 * (v1 - point_vals[i]) * (v2 - point_vals[j])
                second_order_terms += term
        
        poly += (sp.S(1)/2) * second_order_terms
        
        # Display Hessian Matrix
        st.latex(r"\nabla^2 f(\mathbf{a}) = " + sp.latex(hessian))

    # D. Higher Orders
    if degree > 2:
        st.info("Calculating higher order terms...")
        # Simple loop for higher order terms (Symbolic approach)
        # Note: This does not display the tensor notation, just adds the terms to the polynomial
        current_poly_terms = poly
        
        # Iteratively differentiate the expression up to 'degree' times
        # This is a conceptual implementation. For strict Taylor expansion of multivariate:
        # P_k = P_{k-1} + (1/k!) * (d^k f)
        # Due to complexity of "combinations with replacement" loops in pure streamlit script,
        # we stick to the library function 'series' if available, or manual expansion:
        
        # Using sp.series is 1D. Multivariate series is complex. 
        # We manually add 3rd order terms if requested (k=3)
        if degree >= 3:
             third_order = 0
             for i, v1 in enumerate(variables):
                 for j, v2 in enumerate(variables):
                     for k, v3 in enumerate(variables):
                         diff3 = sp.diff(expr, v1, v2, v3)
                         val3 = diff3.subs(point_map)
                         term = val3 * (v1 - point_vals[i]) * (v2 - point_vals[j]) * (v3 - point_vals[k])
                         third_order += term
             poly += (sp.S(1)/6) * third_order # 1/3!

        if degree >= 4:
            fourth_order = 0
            for i, v1 in enumerate(variables):
                 for j, v2 in enumerate(variables):
                     for k, v3 in enumerate(variables):
                         for l, v4 in enumerate(variables):
                            diff4 = sp.diff(expr, v1, v2, v3, v4)
                            val4 = diff4.subs(point_map)
                            term = val4 * (v1 - point_vals[i]) * (v2 - point_vals[j]) * (v3 - point_vals[k]) * (v4 - point_vals[l])
                            fourth_order += term
            poly += (sp.S(1)/24) * fourth_order # 1/4!

    # 5. Final Result
    st.markdown(f"### Final Polynomial $P_{{{degree}}}(\mathbf{{x}})$")
    
    # Simplify the polynomial for display
    poly_simplified = sp.expand(poly)
    st.latex(f"P_{{{degree}}} = {sp.latex(poly_simplified)}")
    
    # 6. Visualization (2D Only)
    if len(variables) == 2:
        st.markdown("---")
        st.markdown("### Visualization (Function vs. Approximation)")
        
        try:
            # Create grid centered at point a
            cx, cy = point_vals[0], point_vals[1]
            rng = 2.0 
            
            x_range = np.linspace(cx - rng, cx + rng, 40)
            y_range = np.linspace(cy - rng, cy + rng, 40)
            X, Y = np.meshgrid(x_range, y_range)
            
            # Lambdify both original and approximation
            f_func = sp.lambdify(variables, expr, 'numpy')
            p_func = sp.lambdify(variables, poly, 'numpy')
            
            Z_orig = f_func(X, Y)
            Z_poly = p_func(X, Y)
            
            fig = go.Figure()
            
            # Original Function
            fig.add_trace(go.Surface(
                z=Z_orig, x=X, y=Y, 
                colorscale='Blues', opacity=0.7, 
                name='Original f(x,y)', showscale=False
            ))
            
            # Taylor Polynomial
            fig.add_trace(go.Surface(
                z=Z_poly, x=X, y=Y, 
                colorscale='Reds', opacity=0.7, 
                name=f'Taylor P{degree}', showscale=False
            ))
            
            # Mark the expansion point
            z_pt = float(f_val)
            fig.add_trace(go.Scatter3d(
                x=[cx], y=[cy], z=[z_pt],
                mode='markers', marker=dict(size=6, color='black'),
                name='Expansion Point'
            ))

            fig.update_layout(
                title=f"Approximation near ({cx}, {cy})",
                scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Blue: Original Function | Red: Taylor Approximation")
            
        except Exception as e:
            st.warning(f"Visualization error: {e}")

else:
    st.info("Awaiting valid function input...")