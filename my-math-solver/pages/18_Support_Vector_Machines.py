import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles
import utils

# --- CONFIGURATION ---
st.set_page_config(page_title="SVM Solver", layout="wide", page_icon="⚔️")
utils.setup_page()

st.markdown("<h1 class='main-header'>Support Vector Machines & Kernels</h1>", unsafe_allow_html=True)

# --- HELPER FUNCTION ---
def plot_svm_boundary(X, y, clf, title, support_vectors=None):
    """
    Plots decision boundary, margins, data points, and support vectors.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Dynamic grid sizing
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max - x_min) / 200
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Decision Function
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        
    Z = Z.reshape(xx.shape)

    # Plot Boundary & Margins
    # Z=0 is boundary. Z=1, -1 are margins.
    ax.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
               levels=[-1, 0, 1], linewidths=1.5)
    
    # Background color
    ax.pcolormesh(xx, yy, Z > 0, cmap=plt.cm.coolwarm, alpha=0.1, shading='auto')

    # Plot Points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm_r, s=80, edgecolors='k')
    
    # Highlight Support Vectors
    if support_vectors is not None:
        ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=200,
                   linewidth=1.5, facecolors='none', edgecolors='k', label='Support Vectors')

    ax.set_title(title)
    ax.set_xlabel("$x_1$ (x)")
    ax.set_ylabel("$x_2$ (y)")
    return fig

# --- TABS ---
tabs = st.tabs(["🛠️ Custom Solver & Analysis", "🎓 Exam 2025 Case", "🎓 Exam 2024 Case", "📚 Explorer", "📖 Lecture Notes"])

# ==============================================================================
# TAB 1: CUSTOM PROBLEM SOLVER (ENHANCED)
# ==============================================================================
with tabs[0]:
    st.subheader("1. Problem Setup")
    st.info("Enter points to generate inequalities, set up Fourier-Motzkin, and find the optimal hyperplane.")

    col_input, col_analysis = st.columns([1, 1.5])

    with col_input:
        example_choice = st.selectbox("Load Example Data:", 
                                      ["Custom", "Jan 2024 Part A (Non-Separable)", "Jan 2024 Part C (Separable)", "Jan 2025 Opgave 3"])
        
        if example_choice == "Jan 2024 Part A (Non-Separable)":
            default_txt = "0, 1, 1\n1, 0, -1\n1, 2, -1\n2, 1, 1"
            def_sep = "Linear (Hyperplane)"
        elif example_choice == "Jan 2024 Part C (Separable)":
            default_txt = "0, 1, 1\n1, 0, 1\n1, 2, -1\n2, 1, -1"
            def_sep = "Linear (Hyperplane)"
        elif example_choice == "Jan 2025 Opgave 3":
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

        with col_analysis:
            # --- THEORETICAL ANALYSIS SECTION ---
            st.markdown("### 2. Theoretical Analysis")
            
            if separator_type == "Linear (Hyperplane)":
                st.markdown("#### A. Inequality Derivation")
                st.markdown("For a line $L: ax + by + c = 0$ to strictly separate points, we need $y_i(ax_i + by_i + c) > 0$.")
                
                latex_eqs = []
                fm_eqs = []
                for i, (point, label) in enumerate(zip(X_custom, y_custom)):
                    x_val, y_val = point
                    
                    # Simplify coefficients
                    a_coeff = f"{label*x_val:.0f}a" if x_val != 0 else ""
                    b_coeff = f"{label*y_val:+.0f}b" if y_val != 0 else ""
                    c_coeff = f"{label:+.0f}c"
                    
                    # Formatting
                    eq_str = f"{a_coeff} {b_coeff} {c_coeff}".strip().replace("+ -", "- ").replace("1a", "a").replace("1b", "b").replace("1c", "c").lstrip("+")
                    
                    latex_eqs.append(f"{eq_str} > 0")
                    fm_eqs.append(f"{eq_str} > 0")

                st.latex(r" \begin{cases} " + r" \\ ".join(latex_eqs) + r" \end{cases}")
                
                st.markdown("#### B. Fourier-Motzkin Setup")
                st.write("To check for solvability (or prove non-existence), isolate $c$:")
                
                fm_latex = []
                for i, (point, label) in enumerate(zip(X_custom, y_custom)):
                    x_val, y_val = point
                    rhs = f"{-1*x_val:.0f}a {-1*y_val:+.0f}b".replace("+ -", "- ").replace("1a", "a").replace("1b", "b").lstrip("+")
                    if not rhs: rhs = "0"
                    
                    if label > 0:
                        fm_latex.append(f"c > {rhs}")
                    else:
                        fm_latex.append(f"c < {rhs}")
                        
                st.latex(r" \begin{cases} " + r" \\ ".join(fm_latex) + r" \end{cases}")
                
                st.info("""
                **Tip for Fourier-Motzkin:**
                Combine inequalities by taking pairs where:
                $$
                \\text{Lower Bound} < c < \\text{Upper Bound}
                $$
                If any pair results in a contradiction (e.g., $0 < -2$), then **no solution exists**.
                """)

                st.markdown("#### C. Optimization Problem Formulation")
                st.write("To find the *best* line, we minimize $|\mathbf{w}|^2$ ($a^2+b^2$) subject to constraints scaled to $\ge 1$.")
                
                opt_latex = []
                for eq in latex_eqs:
                    opt_latex.append(eq.replace(">", r"\ge 1"))
                
                st.latex(r"\text{min } a^2 + b^2 \quad \text{s.t.} \quad \begin{cases} " + r" \\ ".join(opt_latex) + r" \end{cases}")

            st.markdown("---")
            st.markdown("### 3. Numerical Solution")

            # Force strict separation (Hard Margin)
            HARD_MARGIN_C = 1e10

            if separator_type == "Circular (x² + y²)":
                X_transformed = np.column_stack((X_custom[:, 0], X_custom[:, 1], X_custom[:, 0]**2 + X_custom[:, 1]**2))
                clf = svm.SVC(kernel='linear', C=HARD_MARGIN_C)
                clf.fit(X_transformed, y_custom)
                
                # Check for separability
                acc = clf.score(X_transformed, y_custom)
                if acc < 1.0:
                    st.error(f"⚠️ Data is **NOT** separable by a circle! (Accuracy: {acc*100:.0f}%)")
                else:
                    st.success("✅ Data is separable by a circle.")

                w = clf.coef_[0]
                b = clf.intercept_[0]
                
                if abs(w[2]) > 1e-5:
                    xc = -w[0] / (2 * w[2])
                    yc = -w[1] / (2 * w[2])
                    R_sq = xc**2 + yc**2 - (b / w[2])
                    if R_sq > 0:
                        R = np.sqrt(R_sq)
                        st.latex(r"(x - " + f"{xc:.2f})^2 + (y - {yc:.2f})^2 = {R:.2f}^2")
                        st.write(f"**Center:** `({xc:.2f}, {yc:.2f})`, **Radius:** `{R:.2f}`")
                
                # Custom Plot for Circle
                fig, ax = plt.subplots()
                ax.scatter(X_custom[:, 0], X_custom[:, 1], c=y_custom, cmap=plt.cm.coolwarm_r, s=100, edgecolors='k')
                x_min, x_max = X_custom[:, 0].min()-1, X_custom[:, 0].max()+1
                y_min, y_max = X_custom[:, 1].min()-1, X_custom[:, 1].max()+1
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
                Z = w[0]*xx + w[1]*yy + w[2]*(xx**2 + yy**2) + b
                ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
                ax.set_aspect('equal')
                st.pyplot(fig)

            else:
                clf = svm.SVC(kernel='linear', C=HARD_MARGIN_C)
                clf.fit(X_custom, y_custom)
                
                acc = clf.score(X_custom, y_custom)
                if acc < 1.0:
                    st.error(f"⚠️ Data is **NOT** linearly separable! (Accuracy: {acc*100:.0f}%)")
                    st.warning("The Fourier-Motzkin system derived above has **no solution**.")
                else:
                    st.success("✅ Data is linearly separable.")

                w = clf.coef_[0]
                b = clf.intercept_[0]
                
                st.latex(f"{w[0]:.2f}x + {w[1]:.2f}y + ({b:.2f}) = 0")
                
                if abs(w[1]) > 1e-5:
                    m = -w[0] / w[1]
                    c = -b / w[1]
                    st.caption(f"y = {m:.2f}x + {c:.2f}")
                
                st.pyplot(plot_svm_boundary(X_custom, y_custom, clf, "Solution", clf.support_vectors_))

# ==============================================================================
# TAB 2: EXAM 2025 (OPGAVE 3)
# ==============================================================================
with tabs[1]:
    st.markdown("## 🎓 Exam Jan 2025: Opgave 3")
    col_prob, col_sol = st.columns([1, 1.5])
    
    with col_prob:
        st.markdown("**1. Data & Inequalities (Part a)**")
        st.latex(r"P_1(1,1): -1, \quad P_2(1,3): +1, \quad P_3(2,1): +1")
        st.write("Condition: $y_i(ax+by+c) > 0$")
        st.latex(r"""
        \begin{aligned}
        (-1)(a+b+c) > 0 &\Rightarrow a+b+c < 0 \\
        (1)(a+3b+c) > 0 &\Rightarrow a+3b+c > 0 \\
        (1)(2a+b+c) > 0 &\Rightarrow 2a+b+c > 0
        \end{aligned}
        """)
        
        st.markdown("**2. Optimization (Part c)**")
        st.markdown("Minimize $a^2 + b^2$ subject to constraints scaled to $\ge 1$.")

    with col_sol:
        st.markdown("**3. Verification (Part d)**")
        st.write("Verify line: **$2x + y - 4 = 0$** ($a=2, b=1, c=-4$)")
        
        X_exam = np.array([[1, 1], [1, 3], [2, 1]])
        x_vals = np.linspace(0, 3, 100)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(x_vals, -2*x_vals + 4, 'k-', label="2x+y-4=0")
        ax.plot(x_vals, -2*x_vals + 5, 'r--', alpha=0.3, label="Margin +1")
        ax.plot(x_vals, -2*x_vals + 3, 'b--', alpha=0.3, label="Margin -1")
        ax.scatter(1, 1, c='blue', s=100, label="-1 P1")
        ax.scatter([1, 2], [3, 1], c='red', s=100, label="+1 P2, P3")
        ax.legend()
        st.pyplot(fig)
        st.success("Points lie exactly on margins. $2(1)+1(1)-4 = -1$ and $2(1)+1(3)-4 = 1$.")

# ==============================================================================
# TAB 3: EXAM 2024 (OPGAVE 3)
# ==============================================================================
with tabs[2]:
    st.markdown("## 🎓 Exam Jan 2024: Opgave 3")
    
    col_a, col_b = st.columns([1, 1.5])
    
    with col_a:
        st.markdown("### Part (a): Points & Inequalities")
        st.markdown("Points: $(0,1)[+1], (1,0)[-1], (1,2)[-1], (2,1)[+1]$")
        
        st.write("Show that if line $L$ separates points, then:")
        st.latex(r"""
        \begin{aligned}
        1(b+c) > 0 &\Rightarrow b+c > 0 \quad (1) \\
        -1(a+c) > 0 &\Rightarrow a+c < 0 \quad (2) \\
        -1(a+2b+c) > 0 &\Rightarrow a+2b+c < 0 \quad (3) \\
        1(2a+b+c) > 0 &\Rightarrow 2a+b+c > 0 \quad (4)
        \end{aligned}
        """)
        
        st.markdown("### Part (b): Fourier-Motzkin")
        st.markdown("**Goal:** Show no solution exists.")
        st.markdown("**Step 1: Isolate c**")
        st.latex(r"""
        \begin{aligned}
        c &> -b \quad (1) \\
        c &< -a \quad (2) \\
        c &< -a - 2b \quad (3) \\
        c &> -2a - b \quad (4)
        \end{aligned}
        """)
        
        st.markdown("**Step 2: Combine Bounds**")
        st.write("We need $Lower < c < Upper$. Let's test pairs.")
        st.write("Take (1) and (3):")
        st.latex(r"-b < c < -a - 2b \implies -b < -a - 2b \implies \mathbf{b < -a}")
        
        st.write("Take (4) and (2):")
        st.latex(r"-2a - b < c < -a \implies -2a - b < -a \implies -b < a \implies \mathbf{b > -a}")
        
        st.error("Contradiction: We cannot have $b < -a$ AND $b > -a$. Thus, no solution exists.")

    with col_b:
        st.markdown("### Part (c): New Labels (Separable)")
        st.markdown("New Labels: $(0,1)[+1], (1,0)[+1], (1,2)[-1], (2,1)[-1]$")
        
        # Verify Solution for Part D
        st.write("Part (d) asks to verify: $a=-1, b=-1, c=2$")
        st.latex(r"L: -x -y + 2 = 0 \Rightarrow y = -x + 2")
        
        X_24 = np.array([[0,1], [1,0], [1,2], [2,1]])
        y_24 = np.array([1, 1, -1, -1])
        
        fig24, ax24 = plt.subplots(figsize=(5, 4))
        x_vals = np.linspace(-1, 3, 100)
        ax24.plot(x_vals, -x_vals + 2, 'k-', label="-x -y + 2 = 0")
        
        # Plot Margins: -x-y+2 = 1 => y = -x + 1  AND -x-y+2 = -1 => y = -x + 3
        ax24.plot(x_vals, -x_vals + 1, 'r--', alpha=0.3, label="Margin +1")
        ax24.plot(x_vals, -x_vals + 3, 'b--', alpha=0.3, label="Margin -1")
        
        ax24.scatter(X_24[:2,0], X_24[:2,1], c='red', s=100, label="+1")
        ax24.scatter(X_24[2:,0], X_24[2:,1], c='blue', s=100, label="-1")
        ax24.legend()
        ax24.grid(True, alpha=0.3)
        st.pyplot(fig24)
        
        st.success("Verification: Points lie on margins or correct side. Line separates optimally.")

# ==============================================================================
# TAB 3: INTERACTIVE EXPLORER (Interactive Playground)
# ==============================================================================
with tabs[3]:
    st.subheader("🎛️ Interactive SVM Explorer")
    st.markdown("Generate synthetic datasets and tune parameters to see how the SVM decision boundary changes.")

    col_ctrl, col_vis = st.columns([1, 2])

    with col_ctrl:
        st.markdown("#### 1. Dataset Generation")
        dataset_type = st.selectbox("Shape", ["Circles", "Moons", "Blobs (Linear)"])
        noise = st.slider("Noise Level", 0.0, 0.5, 0.1)
        n_samples = st.slider("N Samples", 50, 500, 100)

        st.markdown("#### 2. SVM Parameters")
        kernel_type = st.selectbox("Kernel", ["Linear", "Polynomial", "RBF (Gaussian)"])
        
        # C Parameter (Regularization) - Useful for seeing Soft Margin effects
        c_param = st.slider("C (Regularization)", 0.1, 10.0, 1.0, help="Low C = Soft Margin (smoother), High C = Hard Margin (strict).")
        
        params = {"C": c_param}

        if kernel_type == "Polynomial":
            d = st.slider("Degree", 1, 10, 3)
            coef0 = st.slider("Coef0", 0.0, 10.0, 1.0)
            params.update({"kernel": "poly", "degree": d, "coef0": coef0})
            st.latex(r"K(u,v) = (\gamma u^\top v + r)^d")
            
        elif kernel_type == "RBF (Gaussian)":
            g = st.slider("Gamma", 0.1, 10.0, 1.0)
            params.update({"kernel": "rbf", "gamma": g})
            st.latex(r"K(u,v) = e^{-\gamma |u-v|^2}")
            
        else:
            params.update({"kernel": "linear"})
            st.latex(r"K(u,v) = u^\top v")

    with col_vis:
        # Generate Data on the fly
        # Ensure imports are available
        from sklearn.datasets import make_circles, make_moons, make_blobs

        if dataset_type == "Circles":
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
            y = np.where(y==0, -1, 1)
        elif dataset_type == "Moons":
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
            y = np.where(y==0, -1, 1)
        else: # Blobs (Linear)
            X, y = make_blobs(n_samples=n_samples, centers=2, random_state=42, cluster_std=1.0 + (noise * 5))
            y = np.where(y==0, -1, 1)

        # Train & Plot
        clf = svm.SVC(**params)
        clf.fit(X, y)
        
        st.pyplot(plot_svm_boundary(X, y, clf, f"SVM: {kernel_type} Kernel (C={c_param})"))

# ==============================================================================
# TAB 5: LECTURE NOTES (RESTORED)
# ==============================================================================
with tabs[4]:
    st.markdown("## 📖 Professor's Notes (SVM.ipynb)")
    
    st.markdown(r"""
    ### Support vector machines
    Suppose that $v_1, \dots, v_m\in \mathbb{R}^n$ are given along with labels $y_1, \dots, y_m\in \{-1, 1\}$. 
    Recall that the hyperplane given by $\alpha\in \mathbb{R}^n\setminus\{0\}$ and $\beta\in \mathbb{R}$ is

    $$
    H = \left\{v\in \mathbb{R}^n \middle| \alpha^T v + \beta = 0\right\}.
    $$

    **Example**
    
    If $n=2$, then $H$ is just a line in the plane given by $\alpha = (a, b)\in \mathbb{R}^2\setminus\{(0,0)\}$ and $\beta = c$ i.e.,

    $$
    H = \left\{(x, y)\in \mathbb{R}^2 \middle| a x + b y + c = 0\right\}
    $$

    **Separation**
    
    $H$ is said to separate $v_1, \dots, v_m$ (strictly) if
    $$
    y_i (\alpha^T v_i + \beta) >0 \tag{*}
    $$
    for $i = 1, \dots, m$.

    **Distance from point to hyperplane**
    
    The distance from $H = \left\{v\in \mathbb{R}^n \middle| \alpha^T v + \beta = 0 \right\}$ to $v\in \mathbb{R}^n$ is given by the formula

    $$
    \frac{|\alpha^T v + \beta|}{|\alpha|}. \tag{**}
    $$

    **Formulation of the problem**
    
    We wish to find the "best" separating hyperplane. What does this mean? Under the assumption that $\alpha$ and $\beta$ give a hyperplane $H$ that separates the points, we want it to have maximal distance to the points. This distance is defined as the distance to the point $u$ closest to $H$.

    **The scaling trick**
    
    Suppose that $\alpha$ and $\beta$ satisfy ($*$). Then the distance of the hyperplane $H$ to the points is given by
    $$
    \frac{|\alpha^T u + \beta|}{|\alpha|},
    $$
    where $u$ is the point closest to $H$. This distance and the hyperplane do *not* change if we scale $\alpha$ and $\beta$ by $1/|\alpha^T u + \beta|$. Therefore we may assume that
    $$
    |\alpha^T v_i + \beta| \geq 1
    $$
    for every $i =1, \dots, n$.

    **The convex optimization problem**
    
    Finding the best hyperplane therefore is therefore the solution to the maximization problem

    $$
    \begin{align*}
        &\text{max}\,\, 1/|\alpha|\\
        &\text{with constraints}\\
        &y_i(\alpha^\top v_i + \beta) \geq 1, \quad\text{for } i = 1, \dots, m
    \end{align*}
    $$

    for $i = 1, \dots, m$. This is the same as solving the (convex) minimization problem

    $$
    \begin{align*}
        &\text{min}\,\, |\alpha|^2\\
        &\text{with constraints}\\
        &y_i(\alpha^\top v_i + \beta) \geq 1, \quad\text{for } i = 1, \dots, m
    \end{align*}\tag{!}
    $$

    **The dual optimization problem**
    
    The dual optimization problem to (!) is a maximization problem in the Lagrange multipliers $\Lambda = (\lambda_1, \dots, \lambda_m)$ associated with the contraints in (!)

    $$
    \begin{align*}
        &\text{max}\,\, \lambda_1 + \cdots + \lambda_m - \frac{1}{2} \Lambda^T D \Lambda\\
        &\text{with constraints}\\
        &\Lambda \geq 0\\
        &(y_1, \dots, y_m) \Lambda = 0,
    \end{align*}\tag{D}
    $$
    where $D$ is the symmetric $m\times m$ matrix given by
    $$
    D_{ij} = y_i y_j v_i \cdot v_j.
    $$

    If $\Lambda = (\lambda_1, \dots, \lambda_m)$ is an optimal solution to (D), then

    $$
    \alpha = \lambda_1 y_1 v_1 + \cdots + \lambda_m y_m v_m
    $$

    is optimal in (!) with $\beta$ available for $\lambda_i > 0$ through the binding KKT condition $y_i(\alpha^\top v_i + \beta) = 1$ in $(!)$.

    **Separation hack**
    
    We can sometimes use a (feature) map

    $$
    \varphi: \mathbb{R}^n \rightarrow \mathbb{R}^N
    $$

    to get more room and separate $\varphi(v_1), \dots, \varphi(v_m)$ with similar labels in $\mathbb{R}^N$ instead of separating $v_1, \dots, v_m$ in $\mathbb{R}^n$.

    Suppose that $\varphi(v_1), \dots, \varphi(v_m)$ are separated by the hyperplane in $\mathbb{R}^N$ given by $\alpha\in \mathbb{R}^N$ and $\beta\in \mathbb{R}$. Then we get for
    $$
    \mu(v) = \alpha^T \varphi(v) + \beta
    $$
    that
    $$
    \begin{align*}
    \mu(v_i) > 0\quad&\text{for } y_i = 1\\
    \mu(v_i) < 0\quad&\text{for } y_1 = -1.
    \end{align*}
    $$

    **The kernel trick**
    
    This is where the dual optimization problem (D) really shines! Suppose we separate using a function $\varphi: \mathbb{R}^n\rightarrow \mathbb{R}^N$. A kernel function associated with $\varphi$ is defined as a function $K: \mathbb{R}^n\times \mathbb{R}^n \rightarrow \mathbb{R}$, such that
    $$
    K(u, v) = \varphi(u)\cdot \varphi(v).
    $$
    With this in mind, a miracle happens! Using $\varphi$, the dual optimization problem for the optimization problem (!) in $\mathbb{R}^N$ only depends on $K(v_i, v_j)$. The dot product in $\mathbb{R}^N$ with potentially very big (or even infinite) $N$ can be handled in $\mathbb{R}^n$ by the kernel function $K$.

    ### We only use the kernel

    In calling SVM from for example sklearn in python, one only specifies the kernel through the following three options:
    """)

    # --- NOTES EXAMPLES RENDERED AS PLOTS ---
    st.markdown("#### 1. Linear Kernel")
    st.latex(r"K(u, v) = u\cdot v.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.code("""
points = [[1,1], [2,2]]
labels = [-1, 1]
clf = svm.SVC(kernel='linear')
clf.fit(X, y)
        """)
    with c2:
        points = [[1,1], [2,2]]
        labels = [-1, 1]
        X = np.array(points)
        y = np.array(labels)
        clf = svm.SVC(kernel='linear')
        clf.fit(X, y)
        st.pyplot(plot_svm_boundary(X, y, clf, 'Linear Kernel Example'))

    st.markdown("#### 2. Polynomial Kernel")
    st.latex(r"K(u, v) = (\gamma u\cdot v + r)^d")
    
    c1, c2 = st.columns(2)
    with c1:
        st.code("""
points = [[0,0], [1,1], [-1, 1], [-1, -1], [1, -1]]
labels = [-1, 1, 1, 1, 1]
clf = svm.SVC(kernel='poly', degree=4, coef0=1, gamma=1)
        """)
    with c2:
        points = [[0,0], [1,1], [-1, 1], [-1, -1], [1, -1]]
        labels = [-1, 1, 1, 1, 1]
        X = np.array(points)
        y = np.array(labels)
        clf = svm.SVC(kernel='poly', degree=4, coef0=1, gamma=1)
        clf.fit(X, y)
        st.pyplot(plot_svm_boundary(X, y, clf, 'Polynomial Kernel Example'))

    st.markdown("#### 3. Gaussian (RBF) Kernel")
    st.latex(r"K(u, v) = e^{- \gamma |u -v|^2}.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.code("""
X, y = make_circles(n_samples=200, noise=0.1, factor=0.4)
clf = svm.SVC(kernel='rbf', gamma=2)
        """)
    with c2:
        X, y = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=42)
        y = np.where(y==0, -1, 1)
        clf = svm.SVC(kernel='rbf', gamma=2)
        clf.fit(X, y)
        st.pyplot(plot_svm_boundary(X, y, clf, 'Gaussian (RBF) Kernel Example'))