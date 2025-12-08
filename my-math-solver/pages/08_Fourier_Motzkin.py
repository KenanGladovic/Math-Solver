import streamlit as st
import sympy as sp
import utils

st.set_page_config(layout="wide")
utils.setup_page()

st.markdown("<h1 class='main-header'>Fourier-Motzkin Elimination</h1>", unsafe_allow_html=True)

# --- Curriculum Context ---
with st.expander("📘 Theory & Curriculum Reference (Section 4.5)", expanded=False):
    st.markdown("""
    **Concept:** Fourier-Motzkin elimination is a method for solving systems of linear inequalities by projecting the feasible region onto a lower-dimensional space.
    
    **Optimization (Curriculum p. 109):** To minimize/maximize a function $f(x,y)$, we introduce a new variable $z$ and the constraint $z = f(x,y)$. 
    We then eliminate all variables except $z$ to find its bounds: $z_{min} \le z \le z_{max}$.
    """)

# --- PRESET DEFINITIONS ---
presets = {
    "Custom": {
        "mode": "System Analysis",
        "const": "",
        "obj": ""
    },
    "Curriculum 4.5 (System)": {
        "mode": "System Analysis",
        "const": "2*x + y <= 6\nx + 2*y <= 6\n-x - 2*y <= -2\n-x <= 0\n-y <= 0\nz - x - y <= 0\n-z + x + y <= 0",
        "obj": ""
    },
    "IMO Jan 23 Q1 (Optimization)": {
        "mode": "Optimization",
        "const": "x + y <= 10\n2*x + y <= 15\n-x <= 0\n-y <= 0",
        "obj": "3*x + 2*y"
    },
    "IMO Jan 25 Q3 (Feasibility)": {
        "mode": "System Analysis",
        "const": "a + b + c <= -0.001\n-(a + 3*b + c) <= -0.001\n-(2*a + b + c) <= -0.001",
        "obj": ""
    }
}

# --- CALLBACK TO LOAD PRESETS ---
def load_preset():
    sel = st.session_state.preset_selector
    data = presets[sel]
    st.session_state.fm_mode = data["mode"]
    st.session_state.fm_const = data["const"]
    st.session_state.fm_obj = data["obj"]
    # Update widget keys
    st.session_state["w_fm_const"] = data["const"]
    st.session_state["w_fm_obj"] = data["obj"]

# Initialize State
if "fm_mode" not in st.session_state: st.session_state.fm_mode = "System Analysis"
if "fm_const" not in st.session_state: st.session_state.fm_const = ""
if "fm_obj" not in st.session_state: st.session_state.fm_obj = ""

# --- 1. SETUP & INPUTS ---
col_setup, col_ex = st.columns([2, 1])

with col_ex:
    st.subheader("Presets")
    st.selectbox(
        "Load Example:", 
        list(presets.keys()), 
        key="preset_selector", 
        on_change=load_preset
    )
    
    if st.session_state.fm_const.startswith("2*x"):
        st.info("**Curriculum Ex 4.11:** Solves for $z$ bounds.")
    elif "3*x" in st.session_state.fm_obj:
        st.info("**IMO Jan 23:** Maximizing linear function.")

with col_setup:
    st.subheader("1. Problem Configuration")
    mode = st.radio(
        "Problem Type:", 
        ["System Analysis", "Optimization"], 
        index=0 if st.session_state.fm_mode == "System Analysis" else 1,
        horizontal=True
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        constraints_txt = utils.p_text_area(
            "Constraints (one per line):", 
            "fm_const", 
            st.session_state.fm_const, 
            height=200
        )
    
    with c2:
        if mode == "Optimization":
            obj_func = utils.p_text_input("Objective Function $f(x, ...)$:", "fm_obj", st.session_state.fm_obj)
            opt_dir = st.radio("Goal:", ["Maximize", "Minimize"], horizontal=True)
            st.info("Introduces $z = f(x)$ and eliminates others.")
            target_var_name = "z"
        else:
            st.write("**Target Variable:**")
            st.caption("Variable to solve for (keep). Auto-detected on run.")
            target_var_name = "Manual Select"

# --- 2. SOLVER LOGIC ---

if st.button("Run Fourier-Motzkin", type="primary"):
    try:
        # A. Parse Constraints
        raw_lines = [line.strip() for line in constraints_txt.split('\n') if line.strip()]
        inequalities = []
        all_symbols = set()
        
        for line in raw_lines:
            if "<=" in line: l, r = line.split("<=")
            elif ">=" in line: r, l = line.split(">=") # Flip
            elif "<" in line: l, r = line.split("<")
            elif ">" in line: r, l = line.split(">") # Flip
            else: continue
            
            expr = utils.parse_expr(l) - utils.parse_expr(r)
            inequalities.append(expr)
            all_symbols.update(expr.free_symbols)

        # B. Optimization Setup
        if mode == "Optimization":
            if not obj_func:
                st.error("Please enter an objective function.")
                st.stop()
            f_expr = utils.parse_expr(obj_func)
            z = sp.symbols('z')
            if z in all_symbols: z = sp.symbols('z_opt')
                
            # z = f(x) <=> z - f(x) <= 0 AND f(x) - z <= 0
            inequalities.append(z - f_expr)
            inequalities.append(f_expr - z)
            all_symbols.add(z)
            all_symbols.update(f_expr.free_symbols)
            target_var_name = z.name
            
        # C. Variable Ordering
        sorted_vars = sorted([s.name for s in all_symbols])
        if mode == "System Analysis":
            if 'z' in sorted_vars: target_var_name = 'z'
            elif 'a' in sorted_vars: target_var_name = 'a'
            elif sorted_vars: target_var_name = sorted_vars[-1]
        
        elimination_order = [v for v in sorted_vars if v != target_var_name]
        
        if not elimination_order:
            st.warning("Nothing to eliminate.")
        
        st.markdown("---")
        st.subheader("Step-by-Step Elimination")
        
        current_ineqs = inequalities
        target_sym = sp.symbols(target_var_name)

        # --- ELIMINATION LOOP ---
        for step_idx, elim_var_name in enumerate(elimination_order):
            elim_var = sp.symbols(elim_var_name)
            
            with st.expander(f"Step {step_idx + 1}: Eliminate ${elim_var_name}$", expanded=False):
                lower, upper, others = [], [], []
                
                for expr in current_ineqs:
                    c = expr.coeff(elim_var)
                    rest = expr - c * elim_var
                    if c == 0: others.append(expr)
                    elif c > 0: upper.append(-rest/c)
                    elif c < 0: lower.append(-rest/c)
                
                # Info
                c_info, c_math = st.columns([1, 2])
                with c_info:
                    st.markdown(f"- **{len(lower)}** Lower bounds")
                    st.markdown(f"- **{len(upper)}** Upper bounds")
                    st.markdown(f"- **{len(others)}** Passthrough")
                
                with c_math:
                    if lower: st.latex(f"\\max\\left\\{{ {', '.join([sp.latex(b) for b in lower])} \\right\\}} \\le {elim_var_name}")
                    if upper: st.latex(f"{elim_var_name} \\le \\min\\left\\{{ {', '.join([sp.latex(b) for b in upper])} \\right\\}}")

                # New constraints (L <= U)
                new_cons = []
                for l in lower:
                    for u in upper:
                        new_cons.append(l - u)
                new_cons.extend(others)
                
                # Simplify
                simplified = []
                for c in new_cons:
                    s = sp.simplify(c)
                    if s == True or (s.is_number and s <= 0): continue
                    if s == False or (s.is_number and s > 0):
                        st.error(f"**Contradiction:** ${sp.latex(s)} \\le 0$ is impossible.")
                        st.stop()
                    simplified.append(s)
                
                current_ineqs = list(set(simplified))
                st.caption(f"Generated {len(simplified)} constraints for next step.")

        # --- FINAL RESULTS ---
        st.markdown("---")
        
        final_lower = []
        final_upper = []
        
        for expr in current_ineqs:
            c = expr.coeff(target_sym)
            rest = expr - c * target_sym
            if c == 0:
                if rest > 0: st.error("Infeasible final constraint.")
            elif c > 0: final_upper.append(-rest/c)
            elif c < 0: final_lower.append(-rest/c)
            
        st.subheader(f"3. Final Solution for ${target_var_name}$")
        
        # --- PREFERRED LATEX FORMAT ---
        latex_lines = []
        
        # 1. Upper Bounds (Right Side) -> & && w &\leq& RHS \\[6pt]
        for u in final_upper:
            latex_lines.append(f"& && {target_var_name} &\\leq& {sp.latex(u)} \\\\[6pt]")
            
        # 2. Lower Bounds (Left Side) -> & LHS &\leq& w && \\[6pt]
        for l in final_lower:
            latex_lines.append(f"&{sp.latex(l)} &\\leq& {target_var_name} && \\\\[6pt]")
            
        aligned_block = "\\begin{alignedat}{5}\n" + "\n".join(latex_lines) + "\n\\end{alignedat}"
        
        # 3. Summary (Max/Min)
        try:
            nums_l = [float(v) for v in final_lower if v.is_number]
            nums_u = [float(v) for v in final_upper if v.is_number]
            val_max = max(nums_l) if nums_l else None
            val_min = min(nums_u) if nums_u else None
            
            l_str = f"\\max\\left( {', '.join([sp.latex(v) for v in final_lower])} \\right)" if final_lower else "-\\infty"
            r_str = f"\\min\\left( {', '.join([sp.latex(v) for v in final_upper])} \\right)" if final_upper else "\\infty"
            
            if val_max is not None: l_str += f" = {val_max:g}"
            if val_min is not None: r_str = f"{val_min:g} = " + r_str
            
            summary_eq = f"{l_str} \\le {target_var_name} \\le {r_str}"
        except:
            summary_eq = f"L \\le {target_var_name} \\le U"

        # DISPLAY
        col_res, col_copy = st.columns([2, 1])
        
        with col_res:
            st.markdown("#### Rendered Output")
            st.latex(aligned_block)
            st.markdown("**Summary:**")
            st.latex(summary_eq)
            
            if mode == "Optimization":
                if opt_dir == "Maximize":
                    if val_min is not None: st.success(f"**Optimal Max:** {val_min:g}")
                    else: st.warning("Unbounded (No upper limit).")
                else:
                    if val_max is not None: st.success(f"**Optimal Min:** {val_max:g}")
                    else: st.warning("Unbounded (No lower limit).")

        with col_copy:
            st.markdown("#### LaTeX Code")
            full_code = f"$$\n{aligned_block}\n$$\n\n$$\n{summary_eq}\n$$"
            st.code(full_code, language="latex")

    except Exception as e:
        st.error(f"An error occurred: {e}")