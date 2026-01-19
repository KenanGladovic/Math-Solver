import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles, make_moons, make_blobs
import utils

# --- CONFIGURATION ---
st.set_page_config(page_title="SVM Solver", layout="wide", page_icon="⚔️")
utils.setup_page()

st.markdown("<h1 class='main-header'>Support Vector Machines & Kernels</h1>", unsafe_allow_html=True)

# --- HELPER FUNCTION (UPDATED) ---
def plot_svm_boundary(X, y, clf, title, support_vectors=None, transformer=None):
    """
    Plots decision boundary, margins, data points, and support vectors.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Dynamic grid sizing
    if len(X) > 0:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    else:
        x_min, x_max, y_min, y_max = -2, 2, -2, 2
        
    h = (x_max - x_min) / 200
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Prepare Grid Points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # --- FIX: Apply transformation if model expects higher dimensions ---
    if transformer is not None:
        grid_points = transformer(grid_points)

    # Decision Function
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(grid_points)
    else:
        Z = clf.predict_proba(grid_points)[:, 1]
        
    Z = Z.reshape(xx.shape)

    # Plot Boundary & Margins
    ax.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
               levels=[-1, 0, 1], linewidths=1.5)
    
    # Background color
    ax.pcolormesh(xx, yy, Z > 0, cmap=plt.cm.coolwarm, alpha=0.1, shading='auto')

    # Plot Points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm_r, s=80, edgecolors='k')
    
    # Highlight Support Vectors
    if support_vectors is not None:
        # support_vectors might be 3D, we only plot the first 2 dimensions (x, y)
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=200,
                   linewidth=1.5, facecolors='none', edgecolors='k', label='Support Vectors')

    ax.set_title(title)
    ax.set_xlabel("$x_1$ (x)")
    ax.set_ylabel("$x_2$ (y)")
    return fig

# --- TABS ---
tabs = st.tabs(["🛠️ Solver & Analysis", "📚 Explorer (Prof's Examples)", "📖 Lecture Notes"])

# ==============================================================================
# TAB 1: SOLVER & ANALYSIS
# ==============================================================================
with tabs[0]:
    st.subheader("1. Problem Setup")
    st.info("Enter points to analyze separability, generate proofs, and solve.")

    col_input, col_analysis = st.columns([1, 1.5])

    with col_input:
        # Integrated Exam Presets
        example_choice = st.selectbox("Load Exam Example:", 
                                      ["Custom", 
                                       "Jan 2025 Opgave 3 (Part a)", 
                                       "Jan 2024 Part A (Non-Separable)", 
                                       "Jan 2024 Part C (Separable)"])
        
        if example_choice == "Jan 2024 Part A (Non-Separable)":
            default_txt = "0, 1, 1\n1, 0, -1\n1, 2, -1\n2, 1, 1"
            def_sep = "Linear (Hyperplane)"
        elif example_choice == "Jan 2024 Part C (Separable)":
            default_txt = "0, 1, 1\n1, 0, 1\n1, 2, -1\n2, 1, -1"
            def_sep = "Linear (Hyperplane)"
        elif example_choice == "Jan 2025 Opgave 3 (Part a)":
            default_txt = "1, 1, -1\n1, 3, 1\n2, 1, 1"
            def_sep = "Linear (Hyperplane)"
        else:
            default_txt = "1, 1, 1\n2, 2, -1"
            def_sep = "Linear (Hyperplane)"

        with st.form("solver_form"):
            data_text = st.text_area("Data Points (x, y, label):", value=default_txt, height=150,
                                     help="Enter one point per line: x, y, label")
            
            separator_type = st.radio("Separator Type:", 
                                      ["Linear (Hyperplane)", "Circular (x² + y²)"],
                                      index=0 if def_sep.startswith("Lin") else 1)
            
            submitted = st.form_submit_button("Analyze & Solve")

    if submitted:
        try:
            raw_data = []
            for line in data_text.strip().split('\n'):
                if not line.strip(): continue
                parts = [float(x.strip()) for x in line.split(',')]
                if len(parts) != 3: raise ValueError
                raw_data.append(parts)
            data_arr = np.array(raw_data)
            X_custom = data_arr[:, :2]
            y_custom = data_arr[:, 2]
        except:
            st.error("Invalid format! Use: x, y, label")
            st.stop()

        # --- PRE-CHECK SEPARABILITY (With Safety Limit) ---
        HARD_MARGIN_C = 1e6 
        is_separable = False
        clf_check = None
        # Define the transformer for 3D mapping if needed
        transformer_func = None
        
        if separator_type == "Circular (x² + y²)":
            # Map (x, y) -> (x, y, x^2 + y^2)
            transformer_func = lambda d: np.column_stack((d[:, 0], d[:, 1], d[:, 0]**2 + d[:, 1]**2))
            X_transformed = transformer_func(X_custom)
            
            clf_check = svm.SVC(kernel='linear', C=HARD_MARGIN_C, max_iter=20000)
            clf_check.fit(X_transformed, y_custom)
            if clf_check.score(X_transformed, y_custom) >= 1.0:
                is_separable = True
        else:
            clf_check = svm.SVC(kernel='linear', C=HARD_MARGIN_C, max_iter=20000)
            clf_check.fit(X_custom, y_custom)
            if clf_check.score(X_custom, y_custom) >= 1.0:
                is_separable = True

        with col_analysis:
            # --- SEPARABILITY STATUS ---
            if is_separable:
                st.success("✅ **Data is Separable**")
            else:
                st.error("❌ **Data is NOT Separable**")
            
            st.markdown("---")
            st.subheader("2. Theoretical Analysis")

            if separator_type == "Linear (Hyperplane)":
                st.markdown("#### A. Inequality Setup")
                st.write("Condition: $y_i(ax + by + c) > 0$")
                
                latex_eqs = []
                opt_latex = []
                constraints_for_kkt = []
                fm_data = [] 
                
                for i, (point, label) in enumerate(zip(X_custom, y_custom)):
                    x_val, y_val = point
                    
                    # 1. Standard Inequality
                    a_coeff = f"{label*x_val:.0f}a" if x_val != 0 else ""
                    b_coeff = f"{label*y_val:+.0f}b" if y_val != 0 else ""
                    c_coeff = f"{label:+.0f}c"
                    eq_str = f"{a_coeff} {b_coeff} {c_coeff}".strip().replace("+ -", "- ").replace("1a", "a").replace("1b", "b").replace("1c", "c").lstrip("+")
                    latex_eqs.append(f"{eq_str} > 0 \\quad ({i+1})")
                    
                    # 2. Optimization Form
                    opt_latex.append(f"{eq_str} \\ge 1")
                    constraints_for_kkt.append(eq_str)
                    
                    # 3. Fourier-Motzkin Setup
                    rhs_a = f"{-x_val:.0f}a" if x_val != 0 else ""
                    rhs_b = f"{-y_val:+.0f}b" if y_val != 0 else ""
                    rhs_str = f"{rhs_a} {rhs_b}".strip().replace("+ -", "- ").replace("1a", "a").replace("1b", "b").lstrip("+")
                    if not rhs_str: rhs_str = "0"
                    
                    if label == 1:
                        fm_data.append({'rhs': rhs_str, 'type': 'lower', 'id': i+1})
                    else:
                        fm_data.append({'rhs': rhs_str, 'type': 'upper', 'id': i+1})

                st.latex(r" \begin{cases} " + r" \\ ".join(latex_eqs) + r" \end{cases}")

                # --- BRANCHING LOGIC ---
                if not is_separable:
                    st.markdown("#### B. Proof of Non-Separability (Fourier-Motzkin)")
                    st.info("Since the data is not separable, we prove it by deriving a contradiction.")
                    
                    st.write("**Step 1: Isolate $c$**")
                    lower_bounds = [f"c > {d['rhs']} \\quad ({d['id']})" for d in fm_data if d['type'] == 'lower']
                    upper_bounds = [f"c < {d['rhs']} \\quad ({d['id']})" for d in fm_data if d['type'] == 'upper']
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("Lower Bounds ($c > \dots$):")
                        if lower_bounds: st.latex(r" \\ ".join(lower_bounds))
                        else: st.write("None")
                    with c2:
                        st.write("Upper Bounds ($c < \dots$):")
                        if upper_bounds: st.latex(r" \\ ".join(upper_bounds))
                        else: st.write("None")
                        
                    st.write("**Step 2: Pairwise Elimination**")
                    for l in [d for d in fm_data if d['type'] == 'lower']:
                        for u in [d for d in fm_data if d['type'] == 'upper']:
                            st.markdown(f"**Combine ({l['id']}) and ({u['id']}):**")
                            st.latex(f"{l['rhs']} < c < {u['rhs']} \\implies {l['rhs']} < {u['rhs']}")
                    
                    st.warning("🔎 Look for contradictions like $0 < -2$ or $b < -a$ AND $b > -a$.")

                else:
                    st.markdown("#### B. KKT Conditions (Definition 9.24)")
                    with st.expander("Show KKT Conditions", expanded=True):
                        st.markdown("**1. Optimization Problem:**")
                        st.latex(r"\min a^2 + b^2 \quad \text{s.t.} \quad " + r" \\ ".join(opt_latex))
                        
                        st.markdown("**2. The Conditions:**")
                        kkt_lines = []
                        lambda_list = [f"\\lambda_{{{i+1}}}" for i in range(len(X_custom))]
                        kkt_lines.append(", ".join(lambda_list) + " \\ge 0")
                        
                        for eq in constraints_for_kkt:
                            kkt_lines.append(f"1 - ({eq}) \\le 0")
                        for i, eq in enumerate(constraints_for_kkt):
                            kkt_lines.append(f"\\lambda_{{{i+1}}} (1 - ({eq})) = 0")
                        
                        grad_a_terms = ["2a"]
                        grad_b_terms = ["2b"]
                        grad_c_terms = []
                        for i, (point, label) in enumerate(zip(X_custom, y_custom)):
                            term_a = f"{(-label * point[0]):+.0f}\\lambda_{{{i+1}}}"
                            term_b = f"{(-label * point[1]):+.0f}\\lambda_{{{i+1}}}"
                            term_c = f"{(-label):+.0f}\\lambda_{{{i+1}}}"
                            grad_a_terms.append(term_a)
                            grad_b_terms.append(term_b)
                            grad_c_terms.append(term_c)
                        
                        kkt_lines.append("".join(grad_a_terms).replace("+ -", "- ") + " = 0")
                        kkt_lines.append("".join(grad_b_terms).replace("+ -", "- ") + " = 0")
                        if grad_c_terms:
                            kkt_lines.append("".join(grad_c_terms).lstrip("+").replace("+ -", "- ") + " = 0")

                        st.latex(r" \\ ".join(kkt_lines))

            # --- NUMERICAL VISUALIZATION ---
            st.markdown("---")
            st.subheader("3. Visualization")
            
            if separator_type == "Circular (x² + y²)":
                if is_separable:
                    w = clf_check.coef_[0]
                    b = clf_check.intercept_[0]
                    if abs(w[2]) > 1e-5:
                        xc = -w[0] / (2 * w[2])
                        yc = -w[1] / (2 * w[2])
                        R = np.sqrt(xc**2 + yc**2 - (b / w[2]))
                        st.latex(r"(x - " + f"{xc:.2f})^2 + (y - {yc:.2f})^2 = {R:.2f}^2")
                        st.success(f"Center: ({xc:.2f}, {yc:.2f}), Radius: {R:.2f}")
                
                # Use updated plot function with transformer
                st.pyplot(plot_svm_boundary(X_custom, y_custom, clf_check, "Optimal Separator", 
                                          clf_check.support_vectors_, transformer=transformer_func))
                
            else:
                if is_separable:
                    w = clf_check.coef_[0]
                    b = clf_check.intercept_[0]
                    st.latex(f"{w[0]:.2f}x + {w[1]:.2f}y + ({b:.2f}) = 0")
                    st.pyplot(plot_svm_boundary(X_custom, y_custom, clf_check, "Optimal Separator", 
                                              clf_check.support_vectors_))
                else:
                    st.warning("No optimal separating hyperplane exists (Data Overlap).")
                    fig, ax = plt.subplots(figsize=(5,5))
                    ax.scatter(X_custom[:, 0], X_custom[:, 1], c=y_custom, cmap=plt.cm.coolwarm_r, s=100, edgecolors='k')
                    ax.set_title("Data Points (Non-Separable)")
                    st.pyplot(fig)

# ==============================================================================
# TAB 2: INTERACTIVE EXPLORER
# ==============================================================================
with tabs[1]:
    st.subheader("🎛️ Interactive SVM Explorer")
    st.markdown("Use this to replicate the professor's `SVM.ipynb` examples or explore new datasets.")

    col_ctrl, col_vis = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### 1. Dataset Selection")
        dataset_type = st.selectbox("Data Source", 
                                    ["Circles (Gaussian Ex.)", 
                                     "Professor's Poly Ex. (5 pts)", 
                                     "Moons", 
                                     "Blobs (Linear)"])
        
        if dataset_type in ["Circles (Gaussian Ex.)", "Moons", "Blobs (Linear)"]:
            noise = st.slider("Noise Level", 0.0, 0.5, 0.1)
            n_samples = st.slider("N Samples", 50, 500, 200)
        else:
            st.info("Using fixed points from notebook.")

        st.markdown("#### 2. SVM Parameters")
        kernel_type = st.selectbox("Kernel", ["Linear", "Polynomial", "RBF (Gaussian)"])
        c_param = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
        
        params = {"C": c_param}

        if kernel_type == "Polynomial":
            d = st.slider("Degree ($d$)", 1, 10, 4, help="Professor's example uses d=4")
            coef0 = st.slider("Coef0 ($r$)", 0.0, 10.0, 1.0, help="Professor's example uses r=1")
            gamma_poly = st.slider("Gamma ($\\gamma$)", 0.1, 5.0, 1.0)
            params.update({"kernel": "poly", "degree": d, "coef0": coef0, "gamma": gamma_poly})
            st.latex(r"K(u,v) = (\gamma u^\top v + r)^d")
            
        elif kernel_type == "RBF (Gaussian)":
            g = st.slider("Gamma ($\\gamma$)", 0.1, 10.0, 2.0, help="Professor's example uses gamma=2")
            params.update({"kernel": "rbf", "gamma": g})
            st.latex(r"K(u,v) = e^{-\gamma |u-v|^2}")
            
        else:
            params.update({"kernel": "linear"})
            st.latex(r"K(u,v) = u^\top v")

    with col_vis:
        if dataset_type == "Circles (Gaussian Ex.)":
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.4, random_state=42)
            y = np.where(y==0, -1, 1)
        elif dataset_type == "Professor's Poly Ex. (5 pts)":
            X = np.array([[0,0], [1,1], [-1, 1], [-1, -1], [1, -1]])
            y = np.array([-1, 1, 1, 1, 1])
        elif dataset_type == "Moons":
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
            y = np.where(y==0, -1, 1)
        else: 
            X, y = make_blobs(n_samples=n_samples, centers=2, random_state=42, cluster_std=1.0 + (noise * 5))
            y = np.where(y==0, -1, 1)

        clf = svm.SVC(**params)
        clf.fit(X, y)
        st.pyplot(plot_svm_boundary(X, y, clf, f"SVM: {kernel_type} Kernel (C={c_param})"))

# ==============================================================================
# TAB 3: LECTURE NOTES
# ==============================================================================
with tabs[2]:
    st.markdown("## 📖 Professor's Notes (Summarized)")
    st.write("Key concepts and definitions extracted from the course notebook.")
    
    st.markdown(r"""
    ### 1. The Optimization Problem
    We want to find the hyperplane $H = \{v \mid \alpha^T v + \beta = 0\}$ that maximizes distance to points.
    
    **Primal Problem (Convex):**
    $$
    \begin{aligned}
    &\text{min } |\alpha|^2 \\
    &\text{s.t. } y_i(\alpha^T v_i + \beta) \geq 1
    \end{aligned}
    $$
    
    **Dual Problem:**
    $$
    \begin{aligned}
    &\text{max } \sum \lambda_i - \frac{1}{2} \Lambda^T D \Lambda \\
    &\text{s.t. } \Lambda \geq 0 \text{ and } \sum y_i \lambda_i = 0
    \end{aligned}
    $$
    where $D_{ij} = y_i y_j v_i \cdot v_j$.
    
    ### 2. Kernels
    To separate non-linear data, we map points to a higher dimension $\varphi(v)$. The kernel calculates the dot product in that space:
    $$K(u, v) = \varphi(u) \cdot \varphi(v)$$
    """)