import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()
st.markdown("<h1 class='main-header'>Perceptron Learning Algorithm</h1>", unsafe_allow_html=True)

st.info("""
**Curriculum Reference: Section 5.3.2**
This algorithm finds a linear boundary (hyperplane) that separates data points into two classes (+1 and -1).
It works by iteratively updating a normal vector $\\alpha$ until all points satisfy $\\alpha \\cdot \\hat{v}_i > 0$.
**Note:** If data is not linearly separable, this algorithm will loop forever. Use the 'Max Iterations' slider to force a stop.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Input Data")
    
    # Presets
    preset = utils.p_selectbox(
        "Load Example:", 
        ["Custom", "Example 5.12 (Simple)", "Exercise 5.9 (Impossible/XOR)", "Exercise 5.13"], 
        "perc_preset"
    )
    
    # Define the data for each preset
    if preset == "Example 5.12 (Simple)":
        new_data = "0, 0, 1\n1, 1, 1\n1, -1, -1"
    elif preset == "Exercise 5.9 (Impossible/XOR)":
        new_data = "-1, 1, -1\n1, -1, -1\n-1, -1, 1\n1, 1, 1"
    elif preset == "Exercise 5.13":
        new_data = "0, 0, -1\n0, 1, 1\n1, 1, 1\n1, 0, 1"
    else:
        new_data = "1, 2, 1\n2, 1, -1"

    # --- STATE SYNC LOGIC ---
    if "perc_last_preset" not in st.session_state:
        st.session_state["perc_last_preset"] = preset

    if st.session_state["perc_last_preset"] != preset:
        st.session_state["perc_data"] = new_data
        st.session_state["w_perc_data"] = new_data
        st.session_state["perc_last_preset"] = preset
        st.rerun()
    # ----------------------------------

    data_input = utils.p_text_area("Points $(x, y, label)$:", "perc_data", new_data, height=150)
    st.caption("Format: `x, y, label` per line. Label must be `1` or `-1`.")
    
    # Safety Slider
    max_iter = st.slider("Max Iterations (Safety Stop):", 100, 20000, 1000, help="Prevents infinite loops if data is not separable.")

with col2:
    st.subheader("2. Theory & Transformation")
    st.markdown(r"""
    **The "Augmented" Trick (Eq 5.7):**
    To handle the offset $c$ in $ax+by+c=0$, we transform 2D points into 3D vectors.
    For a point $(x, y)$ with label $\ell$:
    $$
    \hat{v} = (\ell x, \ell y, \ell)
    $$
    We then look for a vector $\alpha = (a, b, c)$ such that $\alpha \cdot \hat{v} > 0$ for all points.
    """)

if st.button("Run Perceptron", type="primary"):
    try:
        raw_points = []
        vectors = []
        
        for line in data_input.split('\n'):
            if not line.strip(): continue
            parts = [float(p.strip()) for p in line.split(',')]
            if len(parts) != 3:
                st.error(f"Invalid format in line: {line}")
                st.stop()
            
            x, y, label = parts
            raw_points.append((x, y, label))
            # The augmented vector (Eq 5.7)
            v_hat = np.array([label * x, label * y, label])
            vectors.append(v_hat)
        
        # Perceptron Algorithm
        alpha = np.zeros(3) 
        history = []
        converged = False
        
        st.markdown("---")
        st.subheader("3. Execution Trace")
        
        df_vecs = pd.DataFrame(vectors, columns=["lx", "ly", "l (bias)"])
        st.write("**Augmented Vectors ($\\hat{v}_i$):**")
        st.dataframe(df_vecs.T)
        
        log_container = st.expander("Show Iteration Log", expanded=False)
        with log_container:
            progress_bar = st.progress(0)
            
            for k in range(max_iter):
                misclassified = None
                mis_idx = -1
                
                for idx, v in enumerate(vectors):
                    if np.dot(alpha, v) <= 0:
                        misclassified = v
                        mis_idx = idx
                        break
                
                if misclassified is None:
                    converged = True
                    break
                    
                old_alpha = alpha.copy()
                alpha = alpha + misclassified
                history.append(alpha.copy())
                
                # Only log first 20 and last few to save memory/rendering time
                if k < 20 or k % 100 == 0:
                    st.write(f"**Step {k+1}:** Misclassified point {mis_idx+1} ({raw_points[mis_idx]}).")
                    st.latex(r"\alpha \leftarrow " + f"{np.round(old_alpha, 2)} + {np.round(misclassified, 2)} = {np.round(alpha, 2)}")
                
                if k % 100 == 0:
                    progress_bar.progress(min(k / max_iter, 1.0))
            
            progress_bar.empty()

        if converged:
            st.success(f"**Converged in {len(history)} steps!**") 
            a, b, c = alpha
            st.markdown(f"**Final Normal Vector:** $\\alpha = ({a:.2f}, {b:.2f}, {c:.2f})$")
            st.markdown(f"**Separating Line Equation:** ${a:.2f}x + {b:.2f}y + {c:.2f} = 0$")
            
            # --- Plotting ---
            fig, ax = plt.subplots(figsize=(8, 6))
            xs = [p[0] for p in raw_points]
            ys = [p[1] for p in raw_points]
            labels = [p[2] for p in raw_points]
            colors = ['blue' if l == 1 else 'red' for l in labels]
            
            # 1. Plot Points
            ax.scatter(xs, ys, c=colors, s=100, edgecolors='k', zorder=5)
            
            # 2. Determine Plot Limits
            x_min, x_max = min(xs)-1, max(xs)+1
            y_min, y_max = min(ys)-1, max(ys)+1
            x_grid = np.linspace(x_min, x_max, 100)
            
            # 3. Plot Line: ax + by + c = 0  =>  y = (-c - ax) / b
            if abs(b) > 1e-5: 
                y_grid = (-c - a * x_grid) / b
                ax.plot(x_grid, y_grid, 'g-', linewidth=2, label='Separating Line')
                
                # Optional: Shade the "Positive" side
                # We need to know which side is positive. 
                # If b > 0, upper side is positive. If b < 0, lower side is positive.
                fill_y = y_max if b > 0 else y_min
                ax.fill_between(x_grid, y_grid, fill_y, color='green', alpha=0.1)
            else:
                # Vertical line case
                if abs(a) > 1e-5:
                    vert_x = -c / a
                    ax.axvline(vert_x, color='g', linewidth=2, label='Separating Line')
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.grid(True, linestyle=':')
            st.pyplot(fig)
            
        else:
            st.error(f"**Algorithm Stopped:** Did not converge within {max_iter} iterations.")
            st.warning("Conclusion: The data is likely **NOT** linearly separable (e.g. XOR problem).")
            
            # Plot points anyway to show why
            fig, ax = plt.subplots(figsize=(8, 6))
            xs = [p[0] for p in raw_points]
            ys = [p[1] for p in raw_points]
            labels = [p[2] for p in raw_points]
            colors = ['blue' if l == 1 else 'red' for l in labels]
            ax.scatter(xs, ys, c=colors, s=100, edgecolors='k', zorder=5)
            ax.set_title("Data Points (Non-Separable)")
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error: {e}")