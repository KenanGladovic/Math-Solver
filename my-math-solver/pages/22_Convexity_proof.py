import streamlit as st
import sympy as sp
import utils

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()

st.markdown("<h1 class='main-header'>Strict Convexity Proof Solver</h1>", unsafe_allow_html=True)

# 2. Shared Input Section (Function)
st.subheader("1. Define Function")
col_main, _ = st.columns([1, 1])
with col_main:
    # Global function input used by both tabs
    f_input = utils.p_text_input("Enter function $f(x, ...)$:", "cvx_proof_f", "-sqrt(x)")
    f_expr = utils.parse_expr(f_input)
    st.caption("Note: For square roots, use `sqrt(x)` or `x**(1/2)`.")

if f_expr:
    vars_sym = sorted(list(f_expr.free_symbols), key=lambda s: s.name)
else:
    vars_sym = []

# 3. Tabs for Different Proof Methods
tab_def, tab_deriv = st.tabs(["Definition 6.1 (Algebraic)", "Derivative Tests (Cor 6.52 & Thm 8.23)"])

# ============================================================
# TAB 1: DEFINITION 6.1 (ALGEBRAIC PROOF)
# ============================================================
with tab_def:
    with st.expander("📘 Definition 6.1 (Strict Convexity)", expanded=False):
        st.markdown("""
        To show a function is strictly convex using **Definition [6.1]**, we must show:
        $$f((1-t)u + tv) < (1-t)f(u) + tf(v)$$
        for every $t$ with $0 < t < 1$ and distinct points $u, v$.
        
        """)

    st.subheader("2. Choose Points u, v")
    if f_expr:
        c_u, c_v = st.columns(2)
        u_vals = {}
        v_vals = {}
        
        with c_u:
            st.markdown("**Point $u$**")
            for var in vars_sym:
                u_vals[var] = st.number_input(f"$u_{{{var.name}}}$", value=0.0, step=1.0, key=f"u_{var.name}")
        
        with c_v:
            st.markdown("**Point $v$**")
            for var in vars_sym:
                v_vals[var] = st.number_input(f"$v_{{{var.name}}}$", value=4.0, step=1.0, key=f"v_{var.name}")

        st.markdown("---")
        if st.button("📝 Formulate Proof (Def 6.1)", type="primary"):
            
            # Check if u == v
            if all(u_vals[v] == v_vals[v] for v in vars_sym):
                st.error("Error: Points $u$ and $v$ must be distinct ($u \\neq v$) for Definition [6.1].")
            else:
                try:
                    t = sp.symbols('t', real=True)
                    latex_source = [] 
                    
                    # --- CALCULATION ENGINE ---
                    # 1. Substitute numbers
                    f_u = f_expr.subs(u_vals)
                    f_v = f_expr.subs(v_vals)
                    
                    # 2. Construct Mixed Point
                    mixed_subs = {}
                    for var in vars_sym:
                        val_mix = (1 - t) * u_vals[var] + t * v_vals[var]
                        mixed_subs[var] = sp.simplify(val_mix)
                    
                    # 3. Expressions
                    lhs_expr = f_expr.subs(mixed_subs)
                    lhs_simp = sp.simplify(lhs_expr) # f((1-t)u+tv)
                    
                    rhs_expr = (1 - t) * f_u + t * f_v
                    rhs_simp = sp.simplify(rhs_expr) # (1-t)f(u) + tf(v)
                    
                    # 4. Analysis Term (RHS - LHS)
                    gap = rhs_simp - lhs_simp
                    gap_factor = sp.factor(gap)
                    
                    # --- LATEX GENERATION ---
                    latex_source.append(r"\subsection*{Proof using Definition [6.1]}")
                    latex_source.append(r"We check the inequality $$f((1-t)u +tv) < (1-t)f(u)+tf(v)$$")
                    
                    u_str = ", ".join([f"{k}={v}" for k, v in u_vals.items()])
                    v_str = ", ".join([f"{k}={v}" for k, v in v_vals.items()])
                    latex_source.append(f"Where we set ${u_str}$, ${v_str}$, and $t \\in (0,1)$.")
                    
                    latex_source.append(f"$$ {sp.latex(lhs_simp)} < {sp.latex(rhs_simp)} \\implies $$")
                    
                    latex_source.append(r"We move all terms to the right hand side:")
                    latex_source.append(f"$$ 0 < {sp.latex(rhs_simp)} - ({sp.latex(lhs_simp)}) \\implies $$")
                    latex_source.append(f"$$ 0 < {sp.latex(gap_factor)} $$")
                    
                    check_val = float(gap_factor.subs(t, 0.5))
                    
                    if gap_factor == 0:
                        latex_source.append(r"Since the expressions are equal, strict convexity \textbf{fails} (it might be linear).")
                    elif check_val > 1e-9:
                        latex_source.append(r"For $t \in (0,1)$, we know $t(1-t) > 0$. The remaining factors are positive.")
                        latex_source.append(f"Therefore, $f(x)={f_input}$ satisfies strict convexity at these points.")
                    else:
                        latex_source.append(r"The inequality evaluates to something negative. Thus, $f$ is \textbf{not} convex.")

                    st.subheader("Generated Proof")
                    for line in latex_source:
                        if line.startswith("$$"): st.latex(line.replace("$$", ""))
                        elif line.startswith("\\"): st.markdown(line)
                        else: st.markdown(line)
                        
                    st.markdown("---")
                    st.text_area("LaTeX Source", value="\n".join(latex_source), height=200)

                except Exception as e:
                    st.error(f"Algebra Error: {e}")

# ============================================================
# TAB 2: DERIVATIVE TESTS (COR 6.52 & THM 8.23)
# ============================================================
with tab_deriv:
    with st.expander("📘 Curriculum Reference: Derivative Tests", expanded=True):
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### **1. Single Variable (Corollary 6.52)**")
            st.markdown("""
            Let $I \\subseteq \mathbb{R}$ be an open interval.
            * **Method:** Calculate $f''(x)$.
            * **Strict Convexity:** If $f''(x) > 0$ for all $x \\in I$, then $f$ is strictly convex.
            * **Convexity:** If $f''(x) \\ge 0$ for all $x \\in I$, then $f$ is convex.
            """)
            
        with c2:
            st.markdown("#### **2. Multi-Variable (Theorem 8.23)**")
            st.markdown("""
            Let $f: U \\to \mathbb{R}$ be twice differentiable, where $U \\subseteq \mathbb{R}^n$ is a **convex open subset**.
            * **Convexity:** $f$ is convex $\\iff \\nabla^2 f(x)$ is **Positive Semidefinite** for every $x \\in U$.
            * **Strict Convexity:** If $\\nabla^2 f(x)$ is **Positive Definite** for every $x \\in U$, then $f$ is strictly convex.
            """)
            st.caption("Note: Positive Definite $\\implies$ all leading principal minors $D_i > 0$.")

    st.subheader("2. Derivative Analysis")
    
    if st.button("🚀 Analyze Derivatives", type="primary"):
        if f_expr:
            try:
                latex_report = []
                latex_report.append(r"\section*{Convexity Analysis}")
                
                # 1. Gradient
                grad = [sp.diff(f_expr, v) for v in vars_sym]
                grad_tex = "\\\\ ".join([sp.latex(g) for g in grad])
                
                st.markdown("**Step 1: First Derivative (Gradient)**")
                grad_display = f"\\nabla f = \\begin{{pmatrix}} {grad_tex} \\end{{pmatrix}}"
                st.latex(grad_display)
                latex_report.append(f"$$ {grad_display} $$")

                # 2. Hessian
                st.markdown("**Step 2: Second Derivative (Hessian)**")
                hessian = sp.hessian(f_expr, vars_sym)
                hessian_tex = sp.latex(hessian)
                
                hess_display = f"\\nabla^2 f = {hessian_tex}"
                st.latex(hess_display)
                latex_report.append(f"$$ {hess_display} $$")

                # 3. Analysis Logic
                st.markdown("**Step 3: Sign/Definiteness Check**")
                latex_report.append(r"\subsection*{Conclusion}")
                
                if len(vars_sym) == 1:
                    # --- COROLLARY 6.52 LOGIC ---
                    second_deriv = hessian[0, 0]
                    latex_report.append(r"\textbf{Applying Corollary 6.52 (Single Variable):}")
                    latex_report.append(f"We examine the sign of $f''(x) = {sp.latex(second_deriv)}$.")
                    
                    st.info(f"Check if $f''(x) = {sp.latex(second_deriv)}$ is strictly positive for all $x$ in your domain.")
                    
                    if second_deriv.is_constant() and second_deriv > 0:
                        msg = "Since $f''(x) > 0$ is a positive constant, $f$ is **Strictly Convex**."
                        st.success(msg)
                        latex_report.append(msg)
                    else:
                        latex_report.append(r"If $f''(x) > 0$ for all valid $x$, then $f$ is strictly convex.")

                else:
                    # --- THEOREM 8.23 LOGIC ---
                    latex_report.append(r"\textbf{Applying Theorem 8.23 (Multivariate):}")
                    latex_report.append(r"We check if $\nabla^2 f$ is Positive Definite on the convex open set $U$.")
                    
                    dets = []
                    st.markdown("Checking Principal Minors (Sylvester's Criterion):")
                    for i in range(1, len(vars_sym) + 1):
                        sub_matrix = hessian[:i, :i]
                        det = sub_matrix.det()
                        dets.append(det)
                        st.latex(f"D_{i} = {sp.latex(det)}")
                        latex_report.append(f"$$ D_{i} = {sp.latex(det)} $$")
                    
                    st.caption("Theorem 8.23: If all leading principal minors $D_i > 0$ for all $x \in U$, $f$ is Strictly Convex.")

                # Output LaTeX
                st.markdown("---")
                st.text_area("LaTeX Source Code", value="\n".join(latex_report), height=250)

            except Exception as e:
                st.error(f"Derivative Error: {e}")