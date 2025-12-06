import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp  # <--- THIS WAS MISSING
import utils
st.set_page_config(layout="wide")
utils.setup_page()
st.markdown("<h1 class='main-header'>Least Squares Method</h1>", unsafe_allow_html=True)

st.info("""
**Curriculum Reference: Section 5.4 & Theorem 5.16**
This tool finds the parameter vector $x$ that minimizes the error $|b - Ax|^2$.
The optimal solution is found by solving the **Normal Equations**:
""")
st.latex(r"(A^T A)x = A^T b")

fit_type = utils.p_radio(
    "Choose Model Type:", 
    ["Polynomial Fit (y = a_0 + a_1 x + ...)", "Circle Fit (Exercise 5.22)"],
    "ls_type",
    horizontal=True
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Data")
    default_data = "1, 2\n2, 3\n3, 5\n4, 7" if "Polynomial" in fit_type else "0, 2\n0, 3\n2, 0\n3, 1"
    data_input = utils.p_text_area("Points (x, y) one per line:", "ls_data", default_data, height=150)
    
    st.markdown("### 📘 How A and b are built")
    if "Polynomial" in fit_type:
        degree = utils.p_slider("Polynomial Degree:", "ls_degree", 1, 5, 1)
        st.markdown(f"""
        **Goal:** Fit $y = a_0 + a_1 x + \dots + a_n x^n$.
        
        We treat the coefficients $a_0, \dots, a_n$ as the unknowns. For each point $(x_i, y_i)$, we write one linear equation:
        $$a_0 \cdot 1 + a_1 \cdot x_i + \dots + a_n \cdot x_i^n = y_i$$
        
        **Matrix Form ($Ax=b$):**
        * **Matrix A:** The $i$-th row contains the powers of $x_i$: $[1, x_i, x_i^2, \dots]$
        * **Vector b:** The $i$-th entry is the observed value $y_i$.
        * **Unknowns x:** The vector $[a_0, a_1, \dots]^T$.
        """)
    else:
        st.markdown(r"""
        **Goal:** Fit $(x-a)^2 + (y-b)^2 = r^2$.
        
        This is non-linear in $a, b, r$. We use the trick from **Exercise 5.22**:
        Expand: $x^2 - 2ax + a^2 + y^2 - 2by + b^2 = r^2$
        Rearrange: $2ax + 2by + (r^2 - a^2 - b^2) = x^2 + y^2$
        
        Let $c = r^2 - a^2 - b^2$. Now it is linear in unknowns $a, b, c$:
        $$a(2x_i) + b(2y_i) + c(1) = x_i^2 + y_i^2$$
        
        **Matrix Form ($Ax=b$):**
        * **Matrix A:** Row $i$ is $[2x_i, 2y_i, 1]$.
        * **Vector b:** Entry $i$ is $x_i^2 + y_i^2$.
        * **Unknowns x:** $[a, b, c]^T$.
        """)

with col2:
    if st.button("Compute Best Fit", type="primary"):
        try:
            points = []
            for line in data_input.split('\n'):
                if ',' in line:
                    parts = line.split(',')
                    points.append((float(parts[0].strip()), float(parts[1].strip())))
            
            if not points:
                st.error("No valid points found.")
                st.stop()

            x_vals = np.array([p[0] for p in points])
            y_vals = np.array([p[1] for p in points])
            num_pts = len(points)

            if "Polynomial" in fit_type:
                A_cols = [np.ones(num_pts)]
                for d in range(1, degree + 1):
                    A_cols.append(x_vals ** d)
                A_np = np.column_stack(A_cols)
                b_np = y_vals
                param_names = [f"a_{i}" for i in range(degree + 1)]
                
            else: # Circle
                col_1 = 2 * x_vals
                col_2 = 2 * y_vals
                col_3 = np.ones(num_pts)
                A_np = np.column_stack([col_1, col_2, col_3])
                b_np = x_vals**2 + y_vals**2
                param_names = ["a (center x)", "b (center y)", "c (aux)"]

            ATA = A_np.T @ A_np
            ATb = A_np.T @ b_np
            
            try:
                x_sol = np.linalg.solve(ATA, ATb)
            except np.linalg.LinAlgError:
                st.error("Matrix $A^T A$ is singular. Points might be collinear or insufficient.")
                st.stop()

            st.subheader("2. The Normal Equations")
            c_a, c_b = st.columns(2)
            with c_a:
                st.markdown("**Matrix $A^T A$**")
                st.latex(sp.latex(sp.Matrix(np.round(ATA, 2))))
            with c_b:
                st.markdown("**Vector $A^T b$**")
                st.latex(sp.latex(sp.Matrix(np.round(ATb, 2))))

            st.subheader("3. Solution Parameters")
            df_params = pd.DataFrame([x_sol], columns=param_names)
            st.table(df_params)

            # --- PLOTTING ---
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(x_vals, y_vals, color='red', label='Data Points', zorder=5)
            
            margin = 1.0
            x_range = np.linspace(min(x_vals)-margin, max(x_vals)+margin, 200)

            if "Polynomial" in fit_type:
                poly_str = f"{x_sol[0]:.3f}"
                y_plot = np.full_like(x_range, x_sol[0])
                for i in range(1, len(x_sol)):
                    poly_str += f" + {x_sol[i]:.3f}x^{i}"
                    y_plot += x_sol[i] * (x_range ** i)
                
                st.success(f"**Best Fit Polynomial:** $y = {poly_str}$")
                ax.plot(x_range, y_plot, 'b-', label=f'Poly Fit (Deg {degree})')
                
            else: # Circle
                a, b, c = x_sol
                r_squared = c + a**2 + b**2
                if r_squared < 0:
                    st.warning("Calculated radius squared is negative. No real circle fits these points well.")
                    r = 0
                else:
                    r = np.sqrt(r_squared)
                    st.success(f"**Best Fit Circle:** Center $({a:.3f}, {b:.3f})$, Radius $r={r:.3f}$")
                    
                    circle = plt.Circle((a, b), r, color='blue', fill=False, label='Least Squares Circle')
                    ax.add_patch(circle)
                    ax.plot(a, b, 'b+', markersize=10, label='Center')
                    ax.set_aspect('equal')
                    ax.set_xlim(a - r - 1, a + r + 1)
                    ax.set_ylim(b - r - 1, b + r + 1)

            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend()
            ax.set_title("Least Squares Fit")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Computation Error: {e}")