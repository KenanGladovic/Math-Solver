import streamlit as st
import sympy as sp
import utils
st.set_page_config(layout="wide")
utils.setup_page()
st.markdown("<h1 class='main-header'>Subset Analysis & Proof Generator</h1>", unsafe_allow_html=True)
st.info("Analyze properties of a subset $C$ (Convexity, Closedness, Boundedness, Compactness) with curriculum proofs.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Define Subset C")
    vars_input = st.text_input("Variables:", "x, y")
    constraints_input = st.text_area(
        "Inequalities (e.g. x**2 + y**2 <= 4):", 
        "x**2 + y**2 <= 4\nx + y >= 1"
    )

with col2:
    if st.button("Analyze & Generate Proof", type="primary"):
        try:
            vars_sym = [sp.symbols(v.strip()) for v in vars_input.split(',')]
            
            # Parse Constraints
            raw_lines = [line.strip() for line in constraints_input.split('\n') if line.strip()]
            parsed_constraints = []
            
            for line in raw_lines:
                if "<=" in line:
                    lhs, rhs = line.split("<=")
                    expr = utils.parse_expr(lhs) - utils.parse_expr(rhs)
                    parsed_constraints.append((expr, "<="))
                elif ">=" in line:
                    lhs, rhs = line.split(">=")
                    expr = utils.parse_expr(rhs) - utils.parse_expr(lhs)
                    parsed_constraints.append((expr, "<="))
                elif "=" in line:
                    lhs, rhs = line.split("=")
                    expr = utils.parse_expr(lhs) - utils.parse_expr(rhs)
                    parsed_constraints.append((expr, "="))
                elif "<" in line:
                    lhs, rhs = line.split("<")
                    expr = utils.parse_expr(lhs) - utils.parse_expr(rhs)
                    parsed_constraints.append((expr, "<"))
                elif ">" in line:
                    lhs, rhs = line.split(">")
                    expr = utils.parse_expr(rhs) - utils.parse_expr(lhs)
                    parsed_constraints.append((expr, "<"))

            st.subheader("Formal Proof")
            
            # 1. CONVEXITY
            st.markdown("### 1. Convexity Analysis")
            st.markdown("<div class='proof-step'><b>Strategy:</b> Check Hessian of constraints (Theorem 8.23) and intersection properties (Exercise 4.23).</div>", unsafe_allow_html=True)
            
            all_convex = True
            for g, rel in parsed_constraints:
                hessian = sp.hessian(g, vars_sym)
                try:
                    if hessian.is_zero_matrix:
                        st.write(f"Constraint ${sp.latex(g)} {rel} 0$ is linear.")
                        st.caption("Linear functions define convex sets (Hyperplanes/Half-spaces).")
                    else:
                        evals = hessian.eigenvals()
                        is_psd = True
                        for ev in evals:
                            if (ev.is_real and ev < 0) or (ev.is_number and ev < 0):
                                is_psd = False
                        
                        if is_psd:
                            st.write(f"Function $g(x) = {sp.latex(g)}$ has a **Positive Semi-Definite Hessian**.")
                            st.latex(f"H_g = {sp.latex(hessian)}")
                            st.caption("By **Theorem 8.23**, $g$ is convex. By **Lemma 4.27**, the sublevel set is convex.")
                        else:
                            st.write(f"Constraint ${sp.latex(g)} {rel} 0$: Convexity check indeterminate (Hessian not clearly PSD).")
                            all_convex = False
                except:
                    st.write(f"Constraint ${sp.latex(g)} {rel} 0$: Could not automatically verify convexity.")
                    all_convex = False

            if all_convex:
                st.success("**Conclusion:** $C$ is the intersection of convex sets. By **Exercise 4.23**, $C$ is **CONVEX**.")
            else:
                st.warning("**Conclusion:** Could not prove convexity for all constraints automatically.")

            # 2. CLOSEDNESS
            st.markdown("### 2. Closedness Analysis")
            st.markdown("<div class='proof-step'><b>Strategy:</b> Check inequality types using **Proposition 5.51** (Continuous Preimages).</div>", unsafe_allow_html=True)
            
            is_closed = True
            for g, rel in parsed_constraints:
                if rel in ["<=", ">=", "="]:
                    st.write(f"Constraint ${sp.latex(g)} {rel} 0$: Defines a closed set.")
                    st.caption("Preimage of a closed interval under a continuous function is closed (**Prop 5.51**).")
                else:
                    st.error(f"Constraint ${sp.latex(g)} {rel} 0$: Strict inequality usually defines an OPEN set.")
                    is_closed = False
            
            if is_closed:
                st.success("**Conclusion:** $C$ is the intersection of closed sets. By **Proposition 5.39**, $C$ is **CLOSED**.")
            else:
                st.error("**Conclusion:** $C$ is **NOT CLOSED** (contains strict inequalities).")

            # 3. BOUNDEDNESS
            st.markdown("### 3. Boundedness Analysis")
            st.markdown("<div class='proof-step'><b>Strategy:</b> Check for ball constraints $|x|^2 \le R^2$ (**Definition 5.29**).</div>", unsafe_allow_html=True)
            
            is_bounded = False
            squared_sum = sum(v**2 for v in vars_sym)
            
            for g, rel in parsed_constraints:
                diff = sp.simplify(g - squared_sum)
                if diff.is_constant() and rel == "<=":
                    R_squared = -diff
                    if R_squared > 0:
                        st.write(f"Constraint ${sp.latex(g)} \le 0$ implies $|x|^2 \le {R_squared}$.")
                        st.caption(f"This subset is contained in a ball $B(0, \sqrt{{{R_squared}}})$.")
                        is_bounded = True
            
            if is_bounded:
                st.success("**Conclusion:** $C$ is contained in a finite ball. $C$ is **BOUNDED**.")
            else:
                st.warning("**Conclusion:** Could not explicitly find a bounding constraint (like $x^2+y^2 \le R$). Boundedness is Undetermined.")

            # 4. COMPACTNESS
            st.markdown("### 4. Compactness Analysis")
            st.markdown("<div class='proof-step'><b>Strategy:</b> Combine Closed and Bounded (**Definition 5.43**).</div>", unsafe_allow_html=True)
            
            if is_closed and is_bounded:
                st.success("Since $C$ is both **CLOSED** and **BOUNDED**, by **Definition 5.43**, $C$ is **COMPACT**.")
            elif not is_closed:
                st.error("Since $C$ is **NOT CLOSED**, it is **NOT COMPACT**.")
            else:
                st.warning("Compactness is Indeterminate (Boundedness could not be proven automatically).")

        except Exception as e:
            st.error(f"Error during analysis: {e}")