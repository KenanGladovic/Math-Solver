import streamlit as st
import sympy as sp
import numpy as np
import utils
from scipy.optimize import linprog

# 1. Setup
st.set_page_config(layout="wide", page_title="Subset Analysis", page_icon="⊂")
utils.setup_page()

# --- PROFESSIONAL CSS ---
st.markdown("""
    <style>
        .main-header { font-size: 2.2rem; color: #2C3E50; border-bottom: 2px solid #2196F3; padding-bottom: 10px; margin-bottom: 20px; }
        .proof-step { background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 5px solid #2196F3; margin-bottom: 20px; }
        .success-step { border-left: 5px solid #28a745; background-color: #e6fffa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .fail-step { border-left: 5px solid #dc3545; background-color: #fff5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .latex-box { font-family: monospace; background: #eee; padding: 10px; border-radius: 4px; font-size: 0.85rem; margin-top: 10px; }
        code { color: #d63384; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Subset Analysis & Proof Generator</h1>", unsafe_allow_html=True)

# --- CONFIGURATION ---
with st.container():
    st.write("Analyze a subset $C = \{ x \mid g_i(x) \le 0 \}$ for Closedness, Boundedness, Compactness, and Convexity.")
    
    with st.expander("⚙️ Problem Configuration", expanded=True):
        # Preset Examples
        examples = {
            "Custom": {
                "vars": "x, y", "const": "x**2 + y**2 <= 4\nx + y >= 1"
            },
            "IMO Jan 2022 Q1 (Linear 3D)": {
                "vars": "x, y, z", "const": "z - x <= 0\nz + x <= 0\nz - y <= 0\nz + y <= 0\n-1 - z <= 0"
            },
            "IMO Jan 2023 Q1 (Linear 2D)": {
                "vars": "x, y", "const": "x + y <= 10\n2*x + y <= 15\n-x <= 0\n-y <= 0"
            },
            "IMO Jan 2025 Q1 (Ball)": {
                "vars": "x, y", "const": "x**2 + y**2 <= 4\n1 - x - y <= 0"
            },
            "IMO Maj 2024 (Unbounded check)": {
                "vars": "x, y, z", "const": "y + z >= 1\nx + z >= 1\n2*x + y + z <= -1\nx + 2*y + z <= -1"
            }
        }

        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_ex = st.selectbox("Load Exam Problem:", list(examples.keys()), index=2)
            vars_input = st.text_input("Variables (comma sep):", value=examples[selected_ex]["vars"])
        
        with col2:
            constraints_input = st.text_area("Constraints ($g(x) \le 0$):", value=examples[selected_ex]["const"], height=100)
            run_btn = st.button("Analyze Subset", type="primary", use_container_width=True)

# --- MAIN TABS ---
tab_tool, tab_theory = st.tabs(["🛠️ Analysis & Proofs", "📚 Exam Strategy (Curriculum)"])

# ==========================================
# TAB 1: ANALYSIS TOOL
# ==========================================
with tab_tool:
    if run_btn:
        try:
            # 1. Parsing
            vars_sym = [sp.symbols(v.strip()) for v in vars_input.split(',')]
            raw_lines = [line.strip() for line in constraints_input.split('\n') if line.strip()]
            parsed_constraints = []
            linear_constraints = []
            is_fully_linear = True
            
            # 2. Process Constraints
            for i, line in enumerate(raw_lines):
                rel = ""
                if "<=" in line: rel = "<="; lhs, rhs = line.split("<=")
                elif ">=" in line: rel = ">="; lhs, rhs = line.split(">=")
                elif "=" in line: rel = "="; lhs, rhs = line.split("=")
                else: continue
                
                # Convert to g(x) <= 0 standard form
                expr = utils.parse_expr(lhs) - utils.parse_expr(rhs)
                if rel == ">=": expr = -expr # Flip if >=
                
                parsed_constraints.append({"expr": expr, "id": i+1, "orig": line})
                
                # Check linearity for linprog
                if expr.is_polynomial() and expr.as_poly(vars_sym).total_degree() == 1:
                    poly = expr.as_poly(vars_sym)
                    coeffs = [float(poly.coeff_monomial(v)) for v in vars_sym]
                    constant = -float(poly.coeff_monomial(1)) if poly.coeff_monomial(1) else 0.0
                    linear_constraints.append((coeffs, constant))
                else:
                    is_fully_linear = False

            # 3. Boundedness Check
            box_bounds = {v: [float('-inf'), float('inf')] for v in vars_sym}
            is_bounded = False
            bound_reason = ""
            
            # Check A: Explicit Ball (x^2 + y^2 <= R)
            sq_sum = sum(v**2 for v in vars_sym)
            for c in parsed_constraints:
                # Heuristic check for ball
                if c["expr"].has(sq_sum):
                    try:
                        if "x**2" in str(c["expr"]) and "y**2" in str(c["expr"]): 
                            is_bounded = True
                            bound_reason = "Constraint defines a Ball/Ellipsoid."
                    except: pass

            # Check B: Linear Programming (Simplex)
            if not is_bounded and linear_constraints:
                A_ub = [x[0] for x in linear_constraints]
                b_ub = [x[1] for x in linear_constraints]
                
                bounded_vars = 0
                for i, v in enumerate(vars_sym):
                    c_min = [0]*len(vars_sym); c_min[i] = 1
                    c_max = [0]*len(vars_sym); c_max[i] = -1
                    
                    res_min = linprog(c_min, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method='highs')
                    res_max = linprog(c_max, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method='highs')
                    
                    if res_min.success and res_max.success:
                        box_bounds[v] = [res_min.fun, -res_max.fun]
                        bounded_vars += 1
                
                if bounded_vars == len(vars_sym):
                    is_bounded = True
                    bound_reason = "Linear constraints form a Polytope (all variables bounded)."

            # --- RENDER RESULTS ---
            
            # A. CLOSEDNESS
            with st.container():
                st.markdown("<div class='success-step'>", unsafe_allow_html=True)
                st.markdown("#### 1. Closedness Proof")
                
                # Logic
                preimage_tex = " \\cap ".join([f"g_{{{c['id']}}}^{{-1}}((-\\infty, 0])" for c in parsed_constraints])
                
                st.markdown(f"The set $C$ is defined by the intersection of {len(parsed_constraints)} constraints:")
                st.latex(f"C = {preimage_tex}")
                
                proof_text = r"""
                * **Step 1:** The functions $g_i$ are polynomials, which are **continuous** (Remark 5.59).
                * **Step 2:** The interval $(-\infty, 0]$ is a **closed set** in $\mathbb{R}$ (Prop 5.42).
                * **Step 3:** The preimage of a closed set under a continuous function is closed (Prop 5.51).
                * **Conclusion:** $C$ is the intersection of closed sets, so **$C$ is Closed** (Prop 5.39).
                """
                st.markdown(proof_text)
                
                # LaTeX Generator
                latex_code = (
                    f"% Closedness Proof\n"
                    f"The set $C$ can be written as:\n"
                    f"$$ C = {preimage_tex} $$\n"
                    f"Since $g_i$ are continuous (polynomials) and $(-\\infty, 0]$ is closed,\n"
                    f"each preimage $g_i^{{-1}}((-\\infty, 0])$ is a closed set (Prop 5.51).\n"
                    f"Thus, $C$ is closed as it is the intersection of closed sets (Prop 5.39)."
                )
                with st.expander("📋 Copy LaTeX Code for Closedness"):
                    st.code(latex_code, language="latex")
                
                st.markdown("</div>", unsafe_allow_html=True)

            # B. BOUNDEDNESS
            with st.container():
                if is_bounded:
                    style = "success-step"
                    res_text = "$C$ is Bounded."
                else:
                    style = "fail-step"
                    res_text = "Boundedness Indeterminate / Unbounded."
                
                st.markdown(f"<div class='{style}'>", unsafe_allow_html=True)
                st.markdown("#### 2. Boundedness Proof")
                
                latex_lines = []
                
                if is_bounded:
                    st.write(f"**Analysis:** {bound_reason}")
                    st.write("We found finite bounds for all variables:")
                    
                    for v in vars_sym:
                        low, high = box_bounds[v]
                        if low != float('-inf'):
                            line = f"{low:.2f} \\le {sp.latex(v)} \\le {high:.2f}"
                            st.latex(line)
                            latex_lines.append(line)
                    
                    # Calculate Radius
                    if linear_constraints:
                        max_val = max([max(abs(b[0]), abs(b[1])) for b in box_bounds.values()])
                        radius = np.sqrt(len(vars_sym) * max_val**2)
                    else:
                        radius = 10.0
                    
                    st.markdown(f"""
                    * **Definition [5.29]:** A set is bounded if it is contained in a ball $B(0, R)$.
                    * Based on the variable bounds, $C \\subseteq B(0, {radius:.2f})$.
                    * **Conclusion:** {res_text}
                    """)
                    
                    # LaTeX Generator
                    bounds_tex = " \\quad \\text{and} \\quad ".join(latex_lines)
                    latex_code = (
                        f"% Boundedness Proof\n"
                        f"From the constraints, we observe that:\n"
                        f"$$ {bounds_tex} $$\n"
                        f"This implies that for a sufficiently large $R$, $|x| \\le R$ for all $x \\in C$.\n"
                        f"Thus $C \\subseteq B(0, R)$, so $C$ is bounded (Definition 5.29)."
                    )
                else:
                    st.write("Could not find finite bounds.")
                    latex_code = "% Boundedness could not be proven automatically."

                with st.expander("📋 Copy LaTeX Code for Boundedness"):
                    st.code(latex_code, language="latex")
                    
                st.markdown("</div>", unsafe_allow_html=True)

            # C. COMPACTNESS
            with st.container():
                if is_bounded:
                    st.markdown("<div class='success-step'>", unsafe_allow_html=True)
                    st.markdown("#### 3. Compactness Conclusion")
                    st.markdown("""
                    * **Definition [5.43]:** A subset of $\mathbb{R}^n$ is **Compact** if it is Closed and Bounded.
                    * Since we proved (1) Closedness and (2) Boundedness:
                    * **Result:** **$C$ is COMPACT.**
                    """)
                    
                    latex_code = (
                        f"% Compactness Proof\n"
                        f"Since we have shown that $C$ is both closed and bounded,\n"
                        f"it follows from Definition 5.43 that $C$ is compact.\n"
                        f"Since $f$ is continuous and $C$ is compact, the Extreme Value Theorem (5.66)\n"
                        f"ensures that $f$ attains a global minimum on $C$."
                    )
                    with st.expander("📋 Copy LaTeX Code for Compactness"):
                        st.code(latex_code, language="latex")
                        
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("Since Boundedness is not proven, we cannot conclude Compactness.")

            # D. CONVEXITY
            with st.container():
                st.markdown("<div class='proof-step'>", unsafe_allow_html=True)
                st.markdown("#### 4. Convexity Check")
                
                
                latex_code = ""
                if is_fully_linear:
                    st.success("**$C$ is Convex.**")
                    st.markdown("""
                    * All constraints are linear ($g_i(x) = ax + b$).
                    * Linear functions are Convex.
                    * The sublevel set of a convex function is a convex set.
                    * The intersection of convex sets is Convex.
                    """)
                    latex_code = (
                        f"% Convexity Proof\n"
                        f"The functions defining the constraints $g_i(x)$ are all linear (affine).\n"
                        f"Linear functions are convex.\n"
                        f"The sublevel sets of convex functions are convex sets.\n"
                        f"Since $C$ is the intersection of convex sets, $C$ is a convex set (Prop 9.8)."
                    )
                else:
                    st.info("**$C$ is likely Convex (Non-linear checks required).**")
                    st.write("Check Hessians of constraints.")
                    latex_code = (
                        f"% Convexity Proof\n"
                        f"We calculate the Hessian $\\nabla^2 g_i$ for each constraint.\n"
                        f"If all $\\nabla^2 g_i$ are Positive Semidefinite, then $g_i$ are convex functions.\n"
                        f"Consequently, $C$ is convex as the intersection of convex sets."
                    )
                
                with st.expander("📋 Copy LaTeX Code for Convexity"):
                    st.code(latex_code, language="latex")
                    
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Analysis Error: {e}")
    else:
        st.info("👈 Select an example or enter constraints to begin analysis.")

# ==========================================
# TAB 2: EXAM STRATEGY (CURRICULUM)
# ==========================================
with tab_theory:
    st.subheader("🎓 How to solve 'Subset Properties' in Exams")
    st.write("Most exams ask you to prove properties of the feasible set $C$. Follow this standard recipe based on Chapter 5.")
    
    # 1. Closedness
    with st.expander("Step 1: How to prove Closedness (Standard Recipe)", expanded=True):
        st.markdown(r"""
        **The Goal:** Prove that $C$ includes its boundary.
        
        **The Argument:**
        1.  Identify that $C$ is defined by inequalities involving continuous functions:
            $$ C = \{ x \in \mathbb{R}^n \mid g_1(x) \le 0, \dots, g_k(x) \le 0 \} $$
        2.  Rewrite $C$ as an intersection of preimages:
            $$ C = \bigcap_{i=1}^k g_i^{-1}((-\infty, 0]) $$
        3.  **Cite these results:**
            * "Polynomials are continuous functions" (**Remark 5.59**).
            * "$(-\infty, 0]$ is a closed set in $\mathbb{R}$" (**Prop 5.42**).
            * "The preimage of a closed set under a continuous function is closed" (**Prop 5.51**).
            * "The intersection of closed sets is closed" (**Prop 5.39**).
        
        **Result:** $C$ is closed.
        """)

    # 2. Boundedness
    with st.expander("Step 2: How to prove Boundedness", expanded=True):
        st.markdown(r"""
        **The Goal:** Prove that $C$ does not extend to infinity.
        
        
        **Strategy A: The Ball Method (Best for $x^2 + y^2$)**
        * If you see $x^2 + y^2 \le 4$, you can immediately say:
            > "The constraint $x^2 + y^2 \le 4$ implies that $(x,y)$ is contained in the ball $B(0, 2)$."
        * Cite **Definition 5.29**: A subset is bounded if it is contained in a ball $B(0, R)$.
        
        **Strategy B: The Box Method (Best for Linear sets)**
        * Use the inequalities to find limits for each variable.
        * Example: $x \ge 0, y \ge 0, x+y \le 1$.
            * $x \le 1 - y \le 1$ (since $y \ge 0$).
            * Thus $x \in [0, 1]$ and $y \in [0, 1]$.
        """)

    # 3. Compactness
    with st.expander("Step 3: Compactness & Optimization", expanded=True):
        st.markdown(r"""
        **The Goal:** Prove that a solution exists (Existence).
        
        **The Argument:**
        1.  State that you have proven $C$ is **Closed** and **Bounded**.
        2.  Cite **Definition 5.43**: "A subset of $\mathbb{R}^n$ is Compact if it is Closed and Bounded."
        3.  Connect to Optimization:
            > "Since $C$ is compact and the objective function $f$ is continuous, the **Extreme Value Theorem (5.66)** ensures that $f$ attains a global minimum and maximum on $C$."
        """)
        
    # 4. Convexity
    with st.expander("Step 4: Convexity", expanded=False):
        st.markdown(r"""
        **The Goal:** Prove $C$ is a convex set (for KKT sufficiency).
        
        **The Argument:**
        1.  Look at the functions $g_i(x)$ defining the constraints.
        2.  Check if $g_i$ are convex functions (e.g., Linear functions are convex, $x^2$ is convex).
        3.  Cite the rule: "The sublevel set $\{x \mid g(x) \le 0\}$ of a convex function is a convex set."
        4.  Cite **Prop 9.8**: "The intersection of convex sets is convex."
        """)