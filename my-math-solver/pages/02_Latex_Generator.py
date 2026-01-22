import streamlit as st
import sympy as sp
import pandas as pd
import utils

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide", page_title="LaTeX Exam Generator", page_icon="📝")
utils.setup_page()

# --- PROFESSIONAL CSS ---
st.markdown("""
    <style>
        .main-header { font-size: 2.2rem; color: #2C3E50; margin-bottom: 0.5rem; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px; }
        .stTabs [aria-selected="true"] { background-color: #e3f2fd; border-bottom: 2px solid #2196F3; }
        code { color: #d63384; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>LaTeX Exam Generator</h1>", unsafe_allow_html=True)
st.caption("Generate LaTeX code for your exam.")

# --- NAVIGATION ---
tabs = st.tabs([
    "Expression Converter", 
    "Matrices & Vectors", 
    "Optimization & Systems", 
    "Calculus & Sums",       
    "Convexity (Grad/Hess)", 
    "Fourier-Motzkin",
    "Sets & Topology",
    "Tables",
    "Symbols"
])

# ==========================================
# TAB 1: EXPRESSION CONVERTER
# ==========================================
with tabs[0]:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Input")
        math_input = utils.p_text_area(
            "Python Expression:", 
            "latex_gen_expr", 
            "1/2 * (x - 1)**2 + sqrt(y)"
        )
        st.caption("Common syntax: `x**2` for $x^2$, `sqrt(x)` for $\\sqrt{x}$, `sum(i, 1, n)`.")

    with col2:
        st.subheader("Output")
        expr = utils.parse_expr(math_input)
        if expr:
            latex_code = sp.latex(expr)
            final_output = f"$$ {latex_code} $$"
            st.markdown(f"**Preview:** {final_output}")
            st.code(final_output, language="latex")
        else:
            st.info("Waiting for valid input...")

# ==========================================
# TAB 2: MATRICES & VECTORS
# ==========================================
with tabs[1]:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Configuration")
        mat_type = st.selectbox(
            "Template:", 
            ["Matrix (2x2)", "Matrix (3x3)", "Vector (2D)", "Vector (3D)", "Custom"],
            key="mat_type_select"
        )
        
        # Smart Defaults
        if mat_type == "Custom": default_mat = "1, 2\n3, 4"
        elif "2x2" in mat_type: default_mat = "1, t\nt, 2"
        elif "3x3" in mat_type: default_mat = "2, -1, 0\n-1, 2, -1\n0, -1, 2"
        elif "2D" in mat_type: default_mat = "x\ny"
        else: default_mat = "x\ny\nz"

        if "last_mat_type" not in st.session_state: st.session_state["last_mat_type"] = mat_type
        if st.session_state["last_mat_type"] != mat_type:
            st.session_state["latex_mat_input"] = default_mat
            st.session_state["w_latex_mat_input"] = default_mat
            st.session_state["last_mat_type"] = mat_type
            st.rerun()

        raw_mat = utils.p_text_area("Matrix Data (comma separated columns):", "latex_mat_input", default_mat, height=150)
        env_type = st.radio("Brackets:", ["pmatrix ( )", "bmatrix [ ]", "vmatrix | |"], horizontal=True)
        env = env_type.split()[0]

    with col2:
        st.subheader("Generated LaTeX")
        if st.button("Generate Matrix", type="primary"):
            try:
                rows = []
                for line in raw_mat.split('\n'):
                    if line.strip():
                        cells = []
                        for item in line.split(','):
                            clean = item.strip()
                            parsed = utils.parse_expr(clean)
                            cells.append(sp.latex(parsed) if parsed else clean)
                        rows.append(" & ".join(cells))
                
                body = " \\\\\n".join(rows)
                code = f"$$ \\begin{{{env}}}\n{body}\n\\end{{{env}}} $$"
                
                st.markdown("**Preview:**")
                st.latex(code.replace("$$", ""))
                st.markdown("**Code:**")
                st.code(code, language="latex")
            except Exception as e:
                st.error(f"Parsing Error: {e}")

# ==========================================
# TAB 3: OPTIMIZATION & SYSTEMS
# ==========================================
with tabs[2]:
    type_select = st.radio("Format:", ["Optimization Problem", "System of Equations / KKT"], horizontal=True)
    
    col1, col2 = st.columns([1, 1])
    
    if type_select == "Optimization Problem":
        with col1:
            st.subheader("Problem Setup")
            op_dir = st.selectbox("Direction:", ["Minimize", "Maximize"])
            func = utils.p_text_input("Objective Function:", "opt_func", "(x+y)^2")
            consts = utils.p_text_area("Constraints (one per line):", "opt_const", "x^2 + y^2 <= 1\nx >= 0")
        with col2:
            st.subheader("Output")
            f_expr = utils.parse_expr(func)
            f_tex = sp.latex(f_expr) if f_expr else func
            
            c_lines = []
            for c in consts.split('\n'):
                if c.strip():
                    c_tex = c.strip().replace("<=", "\\le").replace(">=", "\\ge").replace("lambda", "\\lambda")
                    c_lines.append(c_tex)
            
            c_block = " \\\\\n".join([f"& {c}" for c in c_lines])
            
            final_tex = f"$$ \\begin{{aligned}}\n\\text{{{op_dir} }} & {f_tex} \\\\\n\\text{{subject to }} {c_block}\n\\end{{aligned}} $$"
            st.latex(final_tex.replace("$$", ""))
            st.code(final_tex, language="latex")
            
    else: 
        with col1:
            st.subheader("Equations")
            eqs = utils.p_text_area("Equations (one per line):", "sys_eqs", "\\nabla L = 0\n\\lambda_i >= 0\n\\lambda_i g_i(x) = 0")
            style = st.selectbox("Style:", ["Aligned", "Cases (curly brace)"])
        with col2:
            st.subheader("Output")
            lines = [l.strip().replace(">=", "\\ge").replace("<=", "\\le") for l in eqs.split('\n') if l.strip()]
            tex_lines = []
            for l in lines:
                if "=" in l and style == "Aligned":
                    lhs, rhs = l.split("=", 1)
                    tex_lines.append(f"{lhs.strip()} &= {rhs.strip()}")
                else:
                    tex_lines.append(l)
            
            env = "aligned" if style == "Aligned" else "cases"
            body = " \\\\\n".join(tex_lines)
            final_tex = f"$$ \\begin{{{env}}}\n{body}\n\\end{{{env}}} $$"
            st.latex(final_tex.replace("$$", ""))
            st.code(final_tex, language="latex")

# ==========================================
# TAB 4: CALCULUS & SUMS
# ==========================================
with tabs[3]:
    st.markdown("### Calculus & Series")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        op_type = st.selectbox("Operator:", ["Summation (∑)", "Integral (∫)", "Limit (lim)", "Derivative (d/dx)"])
        func_str = utils.p_text_input("Expression:", "calc_gen_func", "f(x_i)")
        
        c_p1, c_p2, c_p3 = st.columns(3)
        
        if "Derivative" in op_type:
            var_str = c_p1.text_input("Variable:", "x")
            order = c_p2.number_input("Order:", 1, 5, 1)
            is_partial = c_p3.checkbox("Partial (∂)", value=False)
            lim_from, lim_to = None, None
        else:
            var_str = c_p1.text_input("Index/Var:", "i" if "Sum" in op_type else "x")
            lim_from = c_p2.text_input("From / Point:", "1" if "Sum" in op_type else "0")
            lim_to = c_p3.text_input("To (Optional):", "n" if "Sum" in op_type else "\\infty")

    with col2:
        st.subheader("Output")
        if st.button("Generate Calculus LaTeX", type="primary"):
            try:
                # Parse expression safely
                parsed = utils.parse_expr(func_str)
                f_tex = sp.latex(parsed) if parsed else func_str
                
                if "Sum" in op_type:
                    inner_tex = f"\\sum_{{{var_str}={lim_from}}}^{{{lim_to}}} {f_tex}"
                elif "Integral" in op_type:
                    bounds = f"_{{{lim_from}}}^{{{lim_to}}}" if lim_to else f"_{{{lim_from}}}"
                    inner_tex = f"\\int{bounds} {f_tex} \\, d{var_str}"
                elif "Limit" in op_type:
                    inner_tex = f"\\lim_{{{var_str} \\to {lim_from}}} {f_tex}"
                elif "Derivative" in op_type:
                    d_sym = "\\partial" if is_partial else "d"
                    d_ord = f"^{order}" if order > 1 else ""
                    inner_tex = f"\\frac{{{d_sym}{d_ord}}}{{{d_sym}{var_str}{d_ord}}} \\left( {f_tex} \\right)"
                
                final_tex = f"$$ {inner_tex} $$"
                st.markdown(f"**Preview:** {final_tex}")
                st.code(final_tex, language="latex")
                
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# TAB 5: CONVEXITY (GRADIENT & HESSIAN)
# ==========================================
with tabs[4]:
    st.markdown("### Convexity Tools")
    st.write("Calculates $\\nabla f$ (Gradient) and $\\nabla^2 f$ (Hessian Matrix) instantly.")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        func_str_c = utils.p_text_input("Function $f(x,y,...)$:", "calc_f", "x**2 + y**2 + 2*x*y")
        vars_str_c = utils.p_text_input("Variables (comma sep):", "calc_vars", "x, y")
        
    with col2:
        if st.button("Calculate Derivatives", type="primary"):
            try:
                f_expr = utils.parse_expr(func_str_c)
                v_list = [utils.parse_expr(v.strip()) for v in vars_str_c.split(',')]
                
                if f_expr and v_list:
                    # Gradient
                    grad = [sp.diff(f_expr, v) for v in v_list]
                    grad_tex = "\\\\ ".join([sp.latex(g) for g in grad])
                    grad_out = f"\\nabla f = \\begin{{pmatrix}} {grad_tex} \\end{{pmatrix}}"
                    
                    # Hessian
                    hess_rows = []
                    for v1 in v_list:
                        row = [sp.latex(sp.diff(f_expr, v1, v2)) for v2 in v_list]
                        hess_rows.append(" & ".join(row))
                    hess_out = f"\\nabla^2 f = \\begin{{pmatrix}} {' \\\\\\\\ '.join(hess_rows)} \\end{{pmatrix}}"
                    
                    st.markdown("**Gradient:**")
                    st.latex(grad_out)
                    st.code(f"$$ {grad_out} $$", language="latex")
                    
                    st.markdown("**Hessian:**")
                    st.latex(hess_out)
                    st.code(f"$$ {hess_out} $$", language="latex")
                else:
                    st.error("Invalid input.")
            except Exception as e:
                st.error(f"Calculation Error: {e}")

# ==========================================
# TAB 6: FOURIER-MOTZKIN
# ==========================================
with tabs[5]:
    st.markdown("### Fourier-Motzkin Formatting")
    st.write("Generates aligned inequality steps and max/min summary.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fm_var = utils.p_text_input("Variable:", "fm_var", "w")
        st.caption("Lower Bounds ($L \le w$)")
        fm_lower = utils.p_text_area("Lower Bounds (comma sep):", "fm_lower", "11/5, 5/3, 2, 0")
        st.caption("Upper Bounds ($w \le U$)")
        fm_upper = utils.p_text_area("Upper Bounds (comma sep):", "fm_upper", "27/5, 15/2, 6")
        
    with col2:
        if st.button("Generate FM LaTeX", type="primary"):
            # 1. Parse Inputs
            def fmt_val(v):
                v = v.strip()
                if "/" in v and "\\" not in v:
                    n, d = v.split("/")
                    return f"\\frac{{{n}}}{{{d}}}"
                return v

            lows = [fmt_val(x) for x in fm_lower.split(",") if x.strip()]
            ups = [fmt_val(x) for x in fm_upper.split(",") if x.strip()]
            
            # 2. Build Block 1: Aligned Inequalities
            rows = []
            
            # Uppers: & && w &\leq& {U} \\[6pt]
            for u in ups:
                rows.append(f"& && {fm_var} &\\leq& {u} \\\\[6pt]")
                
            # Lowers: &{L} &\leq& w && \\[6pt]
            for l in lows:
                rows.append(f"&{l} &\\leq& {fm_var} && \\\\[6pt]")
                
            block1 = "\\begin{alignedat}{5}\n" + "\n".join(rows) + "\n\\end{alignedat}"
            
            # 3. Build Block 2: Max/Min Summary
            l_str = ",\\; ".join(lows) if lows else "-\\infty"
            u_str = ",\\; ".join(ups) if ups else "\\infty"
            
            summary = (
                "\\begin{alignedat}{5}\n"
                f"&\\max({l_str})\\ &\\leq\\ &{fm_var}\\ &\\\\ &\\\\\n"
                f"&\\ &\\ &{fm_var}\\ &\\leq\\ &\\min({u_str})\n"
                "\\end{alignedat}"
            )
            
            st.subheader("Output")
            
            # PREVIEW (Render separately to avoid double delimiter issues)
            st.markdown("**Preview:**")
            st.latex(block1)
            st.latex(summary)
            
            # CODE (Unified block with delimiters for Copy-Paste)
            final_code = f"\\[\n{block1}\n\\]\n\\[\n{summary}\n\\]"
            st.markdown("**LaTeX Code:**")
            st.code(final_code, language="latex")

# ==========================================
# TAB 7: SETS & TOPOLOGY
# ==========================================
with tabs[6]:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Set Definition")
        
        st.write("Insert Topology Symbols:")
        c_syms = st.columns(4)
        if c_syms[0].button("ℝⁿ"): st.session_state["w_set_dom"] = "\\mathbb{R}^n"
        if c_syms[1].button("∂C"): st.session_state["w_set_dom"] = "\\partial C"
        if c_syms[2].button("C°"): st.session_state["w_set_dom"] = "C^{\\circ}"
        if c_syms[3].button("∅"): st.session_state["w_set_dom"] = "\\emptyset"
        
        domain = utils.p_text_input("Domain / Set:", "set_dom", "\\mathbb{R}^2")
        cond = utils.p_text_input("Condition:", "set_cond", "x^2 + y^2 \\le 1")
        fmt = st.radio("Notation:", ["Set Builder { | }", "Quantifier (∀)", "Quantifier (∃)"], horizontal=True)
        var = utils.p_text_input("Variable:", "set_var", "x")

    with col2:
        st.subheader("Output")
        if fmt == "Set Builder { | }":
            res = f"\\{{ {var} \\in {domain} \\mid {cond} \\}}"
        elif fmt == "Quantifier (∀)":
            res = f"\\forall {var} \\in {domain}: {cond}"
        else:
            res = f"\\exists {var} \\in {domain}: {cond}"
            
        st.markdown("**Preview:**")
        st.latex(res)
        st.code(f"$$ {res} $$", language="latex")

# ==========================================
# TAB 8: TABLES (Professional & Course Tailored)
# ==========================================
with tabs[7]:
    st.markdown("### 📊 Exam Table Generator")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Configuration")
        
        # Template Selection tailored to course content
        template = st.selectbox(
            "Load Exam Template:", 
            ["Empty", "Simplex Tableau (LP)", "Truth Table (Logic)", "Newton/Gradient Iteration"],
            key="tbl_template"
        )
        
        # Logic to set default data based on template
        if template == "Simplex Tableau (LP)":
            def_csv = "Basis, x_1, x_2, s_1, RHS\ns_1, 1, 2, 1, 10\nZ, -3, -5, 0, 0"
            def_align = "|l|ccc|r|" # Standard Simplex format: Left basis, middle vars, right RHS
        elif template == "Truth Table (Logic)":
            def_csv = "P, Q, P \\Rightarrow Q\nT, T, T\nT, F, F\nF, T, T\nF, F, T"
            def_align = "|c|c|c|"
        elif template == "Newton/Gradient Iteration":
            def_csv = "k, x_k, f(x_k), \\nabla f(x_k)\n0, 1.0, 0.5, -2\n1, 0.8, 0.1, 0.1"
            def_align = "|c|c|c|c|"
        else:
            def_csv = "Header 1, Header 2\nVal 1, Val 2"
            def_align = "|c|c|"

        # State management to prevent wiping data on reload
        if "last_tpl" not in st.session_state: st.session_state["last_tpl"] = template
        if st.session_state["last_tpl"] != template:
            st.session_state["table_csv"] = def_csv
            st.session_state["w_table_csv"] = def_csv
            st.session_state["table_align"] = def_align
            st.session_state["w_table_align"] = def_align
            st.session_state["last_tpl"] = template
            st.rerun()

        # Input Area
        csv_data = utils.p_text_area("Table Data (CSV format):", "table_csv", def_csv, height=150)
        
        c_opt1, c_opt2 = st.columns(2)
        auto_math = c_opt1.checkbox("Auto-Math ($...$)", value=True, help="Automatically wraps cell content in $ signs.")
        header_row = c_opt2.checkbox("First row is Header", value=True)
        
        # Advanced Alignment Editing
        align_str = utils.p_text_input("LaTeX Alignment String:", "table_align", def_align)
        st.caption("Tip: Use `|` for vertical lines. Ex: `|l|ccc|r|` puts lines on outsides and splits the first/last columns.")

    with col2:
        st.subheader("2. Preview & Output")
        
        rows = [r.split(',') for r in csv_data.split('\n') if r.strip()]
        
        if rows:
            try:
                # Cleaning data
                clean_data = []
                for row in rows:
                    clean_row = []
                    for cell in row:
                        txt = cell.strip()
                        # Auto-Math Logic: Wrap if requested and not already wrapped
                        if auto_math and txt and not txt.startswith("$"):
                            # Don't wrap standard text headers if user wants to keep them text, 
                            # but usually in math exams headers are math symbols too.
                            clean_row.append(f"${txt}$")
                        else:
                            clean_row.append(txt)
                    clean_data.append(clean_row)

                # 1. Visual Preview (Pandas strips $ for readability if we want, but let's show raw)
                # We create a display version without the $ for the UI table
                display_data = [[c.replace('$', '') for c in r] for r in clean_data]
                
                if header_row and len(display_data) > 0:
                    st.table(pd.DataFrame(display_data[1:], columns=display_data[0]))
                else:
                    st.table(pd.DataFrame(display_data))

                # 2. LaTeX Generation
                tex_rows = []
                for i, r in enumerate(clean_data):
                    line = " & ".join(r) + " \\\\"
                    # Double horizontal line for header if it's Simplex or standard
                    if i == 0 and header_row: 
                        line += " \\hline"
                    tex_rows.append(line)
                
                body = "\n".join(tex_rows)
                
                final_tbl = (
                    f"\\begin{{table}}[h!]\n"
                    f"  \\centering\n"
                    f"  \\begin{{tabular}}{{{align_str}}}\n"
                    f"    \\hline\n"
                    f"    {body}\n"
                    f"    \\hline\n"
                    f"  \\end{{tabular}}\n"
                    f"\\end{{table}}"
                )
                
                st.code(final_tbl, language="latex")
                
            except Exception as e:
                st.error(f"Processing Error: {e}")
        else:
            st.info("Enter data to generate table.")

# ==========================================
# TAB 9: LATEX SYMBOLS CHEAT SHEET
# ==========================================
with tabs[8]: 
    st.markdown("### 📖 Course Reference: LaTeX Symbols")
    st.caption("Symbols specifically found in 'Introduction to Mathematics and Optimization' exams.")

    # Dictionary of symbols tailored to your curriculum
    symbol_cats = {
        "Sets & Numbers": [
            ("Natural Numbers", "\\mathbb{N}", "Set {1, 2, 3...}"),
            ("Integers", "\\mathbb{Z}", "Set {..., -1, 0, 1, ...}"),
            ("Rationals", "\\mathbb{Q}", "Fractions"),
            ("Real Numbers", "\\mathbb{R}", "Continuum"),
            ("Subset", "\\subseteq", "Subset or equal"),
            ("Strict Subset", "\\subsetneq", "A is strictly inside B"),
            ("Empty Set", "\\emptyset", "Set with no elements"),
            ("Set Difference", "\\setminus", "A minus B (A \\ B)"),
            ("Intersection", "\\cap", "A intersect B"),
            ("Union", "\\cup", "A union B"),
            ("Symm. Diff.", "\\Delta", "Symmetric Difference"),
        ],
        "Logic & Quantifiers": [
            ("Implication", "\\implies", "If P, then Q"),
            ("Bi-implication", "\\iff", "P if and only if Q"),
            ("For All", "\\forall", "For all x..."),
            ("Exists", "\\exists", "There exists an x..."),
            ("Logical And", "\\wedge", "P and Q (Use \\wedge)"),
            ("Logical Or", "\\vee", "P or Q (Use \\vee)"),
            ("Negation", "\\neg", "Not P"),
        ],
        "Topology": [
            ("Boundary", "\\partial C", "Boundary of set C"),
            ("Interior", "C^\\circ", "Interior of set C"),
            ("Closure", "\\overline{C}", "Closure of C"),
            ("Open Ball", "B(u, r)", "Ball center u radius r"),
        ],
        "Linear Algebra": [
            ("Transpose", "A^\\top", "Matrix Transpose (Use \\top)"),
            ("Inverse", "A^{-1}", "Matrix Inverse"),
            ("Dot Product", "u \\cdot v", "Dot product"),
            ("Norm", "|u|", "Vector Norm (Single bars)"),
            ("Orthogonal", "\\perp", "Perpendicular"),
            ("Vector", "\\begin{pmatrix} x \\\\ y \\end{pmatrix}", "Column Vector"),
        ],
        "Calculus & Optimization": [
            ("Gradient", "\\nabla f(x)", "Gradient vector"),
            ("Hessian", "\\nabla^2 f(x)", "Hessian matrix"),
            ("Partial Deriv", "\\frac{\\partial f}{\\partial x}", "Partial derivative"),
            ("Lagrange Mult", "\\lambda", "Lambda"),
            ("Sum", "\\sum_{i=1}^{n}", "Summation"),
        ]
    }

    # Layout: 2 Columns for categories
    left_col, right_col = st.columns(2)
    
    # Helper function to render a category
    def render_category(container, title, items):
        with container:
            st.subheader(title)
            for name, code, desc in items:
                c1, c2, c3 = st.columns([0.3, 0.3, 0.4])
                with c1: st.markdown(f"**{name}**")
                with c2: st.code(code, language="latex")
                with c3: st.latex(code)
            st.divider()

    # Left Column
    render_category(left_col, "Sets & Numbers", symbol_cats["Sets & Numbers"])
    render_category(left_col, "Logic & Quantifiers", symbol_cats["Logic & Quantifiers"])
    
    # Right Column
    render_category(right_col, "Linear Algebra", symbol_cats["Linear Algebra"])
    render_category(right_col, "Topology", symbol_cats["Topology"])
    render_category(right_col, "Calculus & Optimization", symbol_cats["Calculus & Optimization"])