import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import utils

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()
st.markdown("<h1 class='main-header'>Lagrange Interpolation</h1>", unsafe_allow_html=True)

# 2. Curriculum Explanation
with st.expander("📘 Theory & Curriculum References", expanded=True):
    st.markdown("""
    **Reference: Section 2.5.1 - The Magic of Lagrange Polynomials** 
    
    Given $n$ points $(x_1, y_1), \dots, (x_n, y_n)$, we can construct a unique polynomial of degree $n-1$ that passes through them.
    
    The formula uses **Lagrange Basis Polynomials** $L_i(x)$ (Eq 2.17):
    $$
    L_i(x) = \\frac{\prod_{j \\neq i} (x - x_j)}{\prod_{j \\neq i} (x_i - x_j)}
    $$
    
    The final polynomial is the weighted sum (Remark 2.32):
    $$
    f(x) = y_1 L_1(x) + y_2 L_2(x) + \dots + y_n L_n(x)
    $$
    """)

col1, col2 = st.columns([1, 1])

# 3. Input Section
with col1:
    st.subheader("1. Input Data")
    # Default is Example 2.34 from the text 
    default_points = "1, 2\n2, 3\n3, 5" 
    points_str = utils.p_text_area("Points (x, y):", "lag_points", default_points, height=150)
    
    st.info("""
    **Example 2.34:**
    Input: `(1,2), (2,3), (3,5)`
    Result should be: $f(x) = x^2 - 2x + 3$ (or similar form).
    """)


# 4. Calculation & Visualization
with col2:
    if st.button("Interpolate & Explain", type="primary"):
        try:
            # --- A. PARSE INPUT ---
            x_vals = []
            y_vals = []
            for line in points_str.split('\n'):
                if ',' in line:
                    parts = line.split(',')
                    x_vals.append(float(parts[0].strip()))
                    y_vals.append(float(parts[1].strip()))
            
            n = len(x_vals)
            if n < 2:
                st.error("Need at least 2 points to interpolate.")
                st.stop()

            # --- B. COMPUTE BASIS POLYNOMIALS (Li) ---
            x = sp.symbols('x')
            full_poly = 0
            
            st.subheader("2. Step-by-Step Construction")
            st.write(f"Constructing {n} Basis Polynomials $L_i(x)$:")
            
            for i in range(n):
                # 1. Numerator: product of (x - x_j)
                # 2. Denominator: product of (x_i - x_j)
                num = 1
                den = 1
                numerator_latex_terms = []
                
                for j in range(n):
                    if i != j:
                        num *= (x - x_vals[j])
                        den *= (x_vals[i] - x_vals[j])
                        numerator_latex_terms.append(f"(x - {x_vals[j]})")
                
                # Create the term L_i
                L_i = num / den
                
                # Add to total sum: y_i * L_i
                term = y_vals[i] * L_i
                full_poly += term
                
                # Display this step (Like Example 2.34)
                num_str = "".join(numerator_latex_terms)
                st.markdown(f"**Point {i+1}** $(x_{i+1}={x_vals[i]}, y_{i+1}={y_vals[i]})$:")
                st.latex(f"L_{{{i+1}}}(x) = \\frac{{{num_str}}}{{{den:.1f}}}")
                st.write(f"Contribution: ${y_vals[i]} \\cdot L_{{{i+1}}}(x)$")
                st.markdown("---")

            # --- C. FINAL RESULT ---
            simplified_poly = sp.simplify(full_poly)
            
            st.subheader("3. Final Polynomial")
            st.markdown("Summing all contributions and simplifying:")
            st.latex(f"P(x) = {sp.latex(simplified_poly)}")
            
            # --- D. PLOTTING ---
            f_num = sp.lambdify(x, simplified_poly, 'numpy')
            
            # Range: slightly wider than data
            x_min, x_max = min(x_vals) - 1, max(x_vals) + 1
            xs = np.linspace(x_min, x_max, 200)
            ys = f_num(xs)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            # Plot the curve
            ax.plot(xs, ys, 'b-', label=f"Interpolation (Deg {n-1})", linewidth=2)
            # Plot the original points
            ax.scatter(x_vals, y_vals, color='red', s=100, label="Data Points", zorder=5)
            
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title("Lagrange Interpolation")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            
            st.pyplot(fig)
            

        except Exception as e:
            st.error(f"Error: {e}")