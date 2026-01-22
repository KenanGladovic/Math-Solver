import streamlit as st
import sympy as sp
import utils

st.set_page_config(layout="wide", page_title="Fourier-Motzkin Solver")
utils.setup_page()

st.markdown("<h1 class='main-header'>Fourier-Motzkin Elimination</h1>", unsafe_allow_html=True)
st.markdown("")

# --- HELPER: Parse Points for IMO Jan 25 Style Problems ---
def generate_separation_constraints(points_str):
    """
    Parses "x,y,label" lines and returns inequalities for ax + by + c.
    Label +1 => ax + by + c >= 1   => -ax -by -c <= -1
    Label -1 => ax + by + c <= -1  =>  ax + by + c <= -1
    """
    lines = points_str.strip().split('\n')
    ineqs = []
    for line in lines:
        parts = line.split(',')
        if len(parts) != 3: continue
        try:
            x, y, label = float(parts[0]), float(parts[1]), int(parts[2])
            if label == 1:
                ineqs.append(f"{-x}*a + {-y}*b + -1*c <= -1")
            elif label == -1:
                ineqs.append(f"{x}*a + {y}*b + 1*c <= -1")
        except: pass
    return "\n".join(ineqs)

# --- PRESET DEFINITIONS ---
presets = {
    "IMO Jan 25 Q3 (Point Separation)": {
        "mode": "System Analysis",
        "const": "1*a + 1*b + 1*c <= -1\n-1*a - 3*b - 1*c <= -1\n-2*a - 1*b - 1*c <= -1",
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
    "Empty": { "mode": "System Analysis", "const": "", "obj": "" }
}

# --- STATE MANAGEMENT ---
def load_preset():
    sel = st.session_state.preset_selector
    data = presets[sel]
    st.session_state.fm_mode = data["mode"]
    st.session_state.fm_const = data["const"]
    st.session_state.fm_obj = data["obj"]
    st.session_state["w_fm_const"] = data["const"]
    st.session_state["w_fm_obj"] = data["obj"]

if "fm_mode" not in st.session_state: st.session_state.fm_mode = "System Analysis"
if "fm_const" not in st.session_state: st.session_state.fm_const = ""
if "fm_obj" not in st.session_state: st.session_state.fm_obj = ""

# --- UI LAYOUT ---
col_setup, col_helper = st.columns([2, 1])

with col_helper:
    st.subheader("Presets & Helpers")
    st.selectbox("Load Example:", list(presets.keys()), key="preset_selector", on_change=load_preset)
    
    with st.expander("🛠️ Hyperplane Helper (For Opg 3)", expanded=True):
        st.caption("Enter points `x, y, label` to generate separation constraints.")
        pts_input = st.text_area("Points:", "1, 1, -1\n1, 3, 1\n2, 1, 1", height=100)
        
        if st.button("Generate Inequalities"):
            res = generate_separation_constraints(pts_input)
            st.session_state.fm_const = res
            st.session_state["w_fm_const"] = res
            st.rerun()

with col_setup:
    st.subheader("1. Configuration")
    mode = utils.p_radio("Problem Type:", ["System Analysis", "Optimization"], "fm_mode", horizontal=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Constraints**")
        constraints_txt = utils.p_text_area("Input", "fm_const", st.session_state.fm_const, height=200, label_visibility="collapsed")
    with c2:
        if mode == "Optimization":
            st.markdown("**Objective**")
            obj_func = utils.p_text_input("Max/Min Function:", "fm_obj", st.session_state.fm_obj)
            opt_dir = st.radio("Goal:", ["Maximize", "Minimize"], horizontal=True)
            target_var_name = "z"
            st.info(f"Setting $z = {obj_func}$ and finding bounds.")
        else:
            st.markdown("**Target Variable**")
            st.info("The variable that remains at the end.")
            target_var_name = "Last Variable"

# --- CORE ALGORITHM ---

if st.button("Run Fourier-Motzkin Elimination", type="primary"):
    st.markdown("---")
    try:
        # 1. PARSE & INIT
        raw_lines = [line.strip() for line in constraints_txt.split('\n') if line.strip()]
        
        # We store inequalities as objects with a 'history' tag
        # structure: {'expr': sympy_expr (<= 0), 'source': str_label}
        system = []
        all_symbols = set()
        
        for idx, line in enumerate(raw_lines):
            if "<=" in line: l, r = line.split("<=")
            elif ">=" in line: r, l = line.split(">=") 
            elif "<" in line: l, r = line.split("<")
            elif ">" in line: r, l = line.split(">")
            else: continue
            
            expr = sp.simplify(utils.parse_expr(l) - utils.parse_expr(r))
            system.append({'expr': expr, 'id': f"I_{idx+1}"})
            all_symbols.update(expr.free_symbols)

        # 2. OPTIMIZATION SETUP
        if mode == "Optimization":
            if not obj_func: st.stop()
            f_expr = utils.parse_expr(obj_func)
            z = sp.symbols('z')
            if z in all_symbols: z = sp.symbols('z_opt')
            
            # z = f(x) => z - f(x) <= 0 AND f(x) - z <= 0
            system.append({'expr': z - f_expr, 'id': "Obj_A"})
            system.append({'expr': f_expr - z, 'id': "Obj_B"})
            all_symbols.add(z)
            all_symbols.update(f_expr.free_symbols)
            target_var_name = z.name

        # 3. ORDERING
        sorted_vars = sorted([s.name for s in all_symbols])
        if mode == "System Analysis":
            # Prefer solving for 'a' (IMO style) or 'z'
            if 'a' in sorted_vars: target_var_name = 'a'
            elif 'z' in sorted_vars: target_var_name = 'z'
            elif sorted_vars: target_var_name = sorted_vars[-1]
        
        elimination_order = [v for v in sorted_vars if v != target_var_name]
        
        st.subheader("2. Step-by-Step Elimination")
        st.write(f"**Goal:** Eliminate ${', '.join(elimination_order)}$ to find bounds for **${target_var_name}$**.")
        
        is_infeasible = False
        step_counter = 1

        # --- LOOP ---
        for elim_var_name in elimination_order:
            elim_var = sp.symbols(elim_var_name)
            
            with st.expander(f"Step {step_counter}: Eliminate ${elim_var_name}$", expanded=True):
                # A. Display Current System
                st.markdown(f"**Current System (before eliminating {elim_var_name}):**")
                for item in system:
                    st.latex(f"({item['id']}) \quad {sp.latex(item['expr'])} \le 0")
                
                # B. Isolate Variable
                lower_set = [] # (expr, id) where L <= x
                upper_set = [] # (expr, id) where x <= U
                independent = []
                
                st.markdown(f"**Isolation of ${elim_var_name}$:**")
                
                for item in system:
                    expr = item['expr']
                    c = expr.coeff(elim_var)
                    rest = expr - c * elim_var
                    
                    if abs(c) < 1e-9:
                        independent.append(item)
                    elif c > 0:
                        # c*x + rest <= 0  ->  x <= -rest/c
                        bound = sp.simplify(-rest/c)
                        upper_set.append({'bound': bound, 'origin': item['id']})
                        st.latex(f"({item['id']}) \implies {elim_var_name} \le {sp.latex(bound)}")
                    elif c < 0:
                        # c*x + rest <= 0 -> x >= -rest/c
                        bound = sp.simplify(-rest/c)
                        lower_set.append({'bound': bound, 'origin': item['id']})
                        st.latex(f"({item['id']}) \implies {sp.latex(bound)} \le {elim_var_name}")

                # C. Combine (L <= U)
                new_system = []
                new_idx = 1
                
                if lower_set and upper_set:
                    st.markdown("**New Inequalities (combining $L \le U$):**")
                    for l_item in lower_set:
                        for u_item in upper_set:
                            # L <= U  =>  L - U <= 0
                            new_expr = sp.simplify(l_item['bound'] - u_item['bound'])
                            new_id = f"II_{new_idx}" if step_counter == 1 else f"III_{new_idx}"
                            
                            origin_str = f"From ({l_item['origin']}) and ({u_item['origin']})"
                            st.markdown(f"* {origin_str}:")
                            st.latex(f"{sp.latex(l_item['bound'])} \le {sp.latex(u_item['bound'])} \iff {sp.latex(new_expr)} \le 0")
                            
                            new_system.append({'expr': new_expr, 'id': new_id})
                            new_idx += 1
                elif not independent:
                    st.info(f"Variable ${elim_var_name}$ is unconstrained (removed without generating new constraints).")

                # Carry over independent
                for item in independent:
                    new_system.append(item)

                # D. Check Feasibility
                clean_system = []
                seen_exprs = set()
                
                for item in new_system:
                    s = item['expr']
                    if s == True or (s.is_number and s <= 0): continue
                    if s == False or (s.is_number and s > 0):
                        st.error(f"❌ **Contradiction:** {float(s)} <= 0")
                        is_infeasible = True
                        break
                    
                    # Deduplicate
                    s_str = str(s)
                    if s_str not in seen_exprs:
                        clean_system.append(item)
                        seen_exprs.add(s_str)
                
                if is_infeasible:
                    st.error("System is **INFEASIBLE**.")
                    break
                
                system = clean_system
                step_counter += 1

        # --- FINAL OUTPUT ---
        if not is_infeasible:
            st.markdown("---")
            st.subheader(f"3. Final Result for ${target_var_name}$")
            
            final_lower = []
            final_upper = []
            target_sym = sp.symbols(target_var_name)
            
            for item in system:
                expr = item['expr']
                c = expr.coeff(target_sym)
                rest = expr - c * target_sym
                
                if abs(c) < 1e-9:
                    if rest > 0: is_infeasible = True
                elif c > 0: final_upper.append(sp.simplify(-rest/c))
                elif c < 0: final_lower.append(sp.simplify(-rest/c))
            
            if is_infeasible:
                 st.error("System is Infeasible (Constant constraint violation).")
            else:
                # LaTeX Formatting
                lines = []
                for u in final_upper:
                    lines.append(f"& && {target_var_name} &\\le& {sp.latex(u)} \\\\[6pt]")
                for l in final_lower:
                    lines.append(f"&{sp.latex(l)} &\\le& {target_var_name} && \\\\[6pt]")
                
                if not lines:
                    st.success("Variable is unconstrained.")
                else:
                    block = "\\begin{alignedat}{5}\n" + "\n".join(lines) + "\n\\end{alignedat}"
                    st.latex(block)
                    
                    # Summary
                    l_str = f"\\max({', '.join([sp.latex(v) for v in final_lower])})" if final_lower else "-\\infty"
                    r_str = f"\\min({', '.join([sp.latex(v) for v in final_upper])})" if final_upper else "\\infty"
                    st.latex(f"{l_str} \le {target_var_name} \le {r_str}")
                    
                    if mode == "System Analysis":
                        st.success("✅ System is Feasible.")

    except Exception as e:
        st.error(f"Error: {e}")