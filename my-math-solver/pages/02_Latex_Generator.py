import streamlit as st
import sympy as sp
import pandas as pd
import utils

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()
st.markdown("<h1 class='main-header'>LaTeX Generator</h1>", unsafe_allow_html=True)

st.info("Generate professional math code for your exam paper without typing all the backslashes manually.")

# Generator Mode Selector
mode = st.radio(
    "What do you want to create?", 
    [
        "Python Math -> LaTeX", 
        "Matrix / Vector", 
        "Cases / Systems", 
        "Optimization Problem",
        "∫ Calculus / Sums",       # NEW
        "Table Generator"        # NEW
    ],
    horizontal=True
)

st.markdown("---")

# ==========================================
# MODE 1: PYTHON EXPRESSION
# ==========================================
if mode == "Python Math -> LaTeX":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Input")
        math_input = utils.p_text_area(
            "Python Expression:", 
            "latex_gen_expr", 
            "1/2 * x**2 + sqrt(y)"
        )
        st.caption("Example: `(x+y)**2` becomes $(x+y)^2$")

    with col2:
        st.subheader("Output")
        expr = utils.parse_expr(math_input)
        if expr:
            latex_code = sp.latex(expr)
            st.markdown(f"**Preview:** ${latex_code}$")
            st.code(latex_code, language="latex")
        else:
            st.warning("Waiting for valid input...")

# ==========================================
# MODE 2: MATRIX / VECTOR
# ==========================================
elif mode == "Matrix / Vector":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Matrix Setup")
        
        mat_type = st.selectbox(
            "Type:", 
            ["Matrix (2x2)", "Matrix (3x3)", "Column Vector (2D)", "Column Vector (3D)", "Custom"],
            key="mat_gen_type"
        )
        
        if mat_type == "Custom":
            new_default = "1, 2\n3, 4"
        elif "2x2" in mat_type:
            new_default = "1, t\nt, 1"
        elif "3x3" in mat_type:
            new_default = "1, 0, -1\n-2, 2, -1\n1, -1, 1"
        elif "2D" in mat_type:
            new_default = "x\ny"
        else:
            new_default = "x\ny\nz"

        if "last_mat_type" not in st.session_state:
            st.session_state["last_mat_type"] = mat_type
            
        if st.session_state["last_mat_type"] != mat_type:
            st.session_state["latex_mat_input"] = new_default
            st.session_state["w_latex_mat_input"] = new_default
            st.session_state["last_mat_type"] = mat_type
            st.rerun()

        st.write("Enter values (comma separated columns, new line for rows):")
        raw_mat = utils.p_text_area("Values:", "latex_mat_input", new_default, height=150)

    with col2:
        st.subheader("Output")
        if st.button("Generate LaTeX Matrix", type="primary"):
            try:
                matrix_rows = []
                for line in raw_mat.split('\n'):
                    if line.strip():
                        row_cells = []
                        for item in line.split(','):
                            clean_item = item.strip()
                            parsed = utils.parse_expr(clean_item)
                            row_cells.append(sp.latex(parsed) if parsed else clean_item)
                        matrix_rows.append(" & ".join(row_cells))
                
                body = " \\\\\n".join(matrix_rows)
                env = "pmatrix"
                latex_out = f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}"
                
                st.markdown("**Preview:**")
                st.latex(latex_out)
                st.markdown("**Code:**")
                st.code(latex_out, language="latex")
                
            except Exception as e:
                st.error(f"Error parsing matrix: {e}")

# ==========================================
# MODE 3: CASES / SYSTEMS
# ==========================================
elif mode == "Cases / Systems":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Equations")
        default_sys = "2*x + y = 1\nx + 2*y = t"
        eq_input = utils.p_text_area("One equation per line:", "latex_sys_input", default_sys)
        style = st.radio("Style:", ["Aligned System (align*)", "Cases (curly brace)"])

    with col2:
        st.subheader("Output")
        lines = [l.strip() for l in eq_input.split('\n') if l.strip()]
        latex_lines = []
        for l in lines:
            if "=" in l:
                lhs, rhs = l.split("=")
                l_tex = sp.latex(utils.parse_expr(lhs))
                r_tex = sp.latex(utils.parse_expr(rhs))
                latex_lines.append(f"{l_tex} &= {r_tex}")
            else:
                latex_lines.append(sp.latex(utils.parse_expr(l)))
        
        if style == "Aligned System (align*)":
            final_tex = "\\begin{align*}\n" + " \\\\\n".join(latex_lines) + "\n\\end{align*}"
        else:
            final_tex = "\\begin{cases}\n" + " \\\\\n".join(latex_lines) + "\n\\end{cases}"
            
        st.latex(final_tex)
        st.code(final_tex, language="latex")

# ==========================================
# MODE 4: OPTIMIZATION PROBLEM
# ==========================================
elif mode == "Optimization Problem":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Problem Setup")
        op_type = st.selectbox("Type", ["Minimize", "Maximize"])
        func = utils.p_text_input("Objective Function:", "latex_opt_func", "(x+y)**2 - x")
        consts = utils.p_text_area("Constraints:", "latex_opt_const", "x**2 + y**2 <= 4\nx + y >= 1")

    with col2:
        st.subheader("Output")
        f_tex = sp.latex(utils.parse_expr(func))
        const_lines = []
        for c in consts.split('\n'):
            if c.strip():
                c_tex = c.replace("<=", "\\le").replace(">=", "\\ge").replace("**", "^").replace("*", "")
                const_lines.append(c_tex)
        
        c_block = " \\\\\n".join([f"& {c}" for c in const_lines])
        final_tex = (
            f"\\begin{{aligned}}\n"
            f"\\text{{{op_type} }} & {f_tex} \\\\\n"
            f"\\text{{subject to }} {c_block}\n"
            f"\\end{{aligned}}"
        )
        st.latex(final_tex)
        st.code(final_tex, language="latex")

# ==========================================
# MODE 5: CALCULUS / SUMS (NEW)
# ==========================================
elif mode == "∫ Calculus / Sums":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Operator Setup")
        op_type = st.selectbox("Operator:", ["Summation (∑)", "Integral (∫)", "Limit (lim)", "Derivative (d/dx)"])
        
        func_str = utils.p_text_input("Expression (e.g. x**2):", "calc_gen_func", "x**i")
        
        if op_type == "Derivative (d/dx)":
            var_str = utils.p_text_input("Variable:", "calc_gen_var", "x")
            order = st.number_input("Order:", 1, 5, 1)
            lim_from, lim_to = None, None
        else:
            c1, c2, c3 = st.columns(3)
            var_str = c1.text_input("Variable:", "i" if "Sum" in op_type else "x")
            lim_from = c2.text_input("From / Point:", "1" if "Sum" in op_type else "0")
            lim_to = c3.text_input("To (Optional):", "n" if "Sum" in op_type else "inf")

    with col2:
        st.subheader("Output")
        if st.button("Generate Math", type="primary"):
            try:
                f_tex = sp.latex(utils.parse_expr(func_str))
                
                if op_type == "Summation (∑)":
                    final_tex = f"\\sum_{{{var_str}={lim_from}}}^{{{lim_to}}} {f_tex}"
                
                elif op_type == "Integral (∫)":
                    bounds = f"_{{{lim_from}}}^{{{lim_to}}}" if lim_to else f"_{{{lim_from}}}"
                    if not lim_from and not lim_to: bounds = ""
                    final_tex = f"\\int{bounds} {f_tex} \\, d{var_str}"
                    
                elif op_type == "Limit (lim)":
                    final_tex = f"\\lim_{{{var_str} \\to {lim_from}}} {f_tex}"
                    
                elif op_type == "Derivative (d/dx)":
                    d_part = f"^{order}" if order > 1 else ""
                    final_tex = f"\\frac{{d{d_part}}}{{d{var_str}{d_part}}} \\left( {f_tex} \\right)"
                
                st.markdown(f"**Preview:** $${final_tex}$$")
                st.code(final_tex, language="latex")
                
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# MODE 6: TABLE GENERATOR
# ==========================================
elif mode == "Table Generator":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Input Data")
        
        st.info("""
        **Format Instructions (CSV):**
        * **Columns:** Separate values with a **comma** (`,`).
        * **Rows:** Press **Enter** for a new line.
        * **Math:** You can type normal math like `$x^2$` inside the cells.
        """)
        
        default_data = "Iteration, $x_n$, $f(x_n)$\n1, 0.5, 0.25\n2, 0.2, 0.04\n3, 0.1, 0.01"
        raw_data = utils.p_text_area("Table Data:", "latex_table_input", default_data, height=200)
        
        st.subheader("2. Settings")
        add_header = st.checkbox("First row is a Header (bold + line)", value=True)
        center_align = st.checkbox("Center Align All Columns", value=True)

    with col2:
        st.subheader("3. Visual Preview")
        
        # Parse data immediately for the preview
        rows = [r.split(',') for r in raw_data.split('\n') if r.strip()]
        
        if rows:
            try:
                # Clean up whitespace
                clean_rows = [[cell.strip() for cell in row] for row in rows]
                
                # Create a DataFrame for a nice "Spreadsheet" look
                if add_header:
                    header = clean_rows[0]
                    body = clean_rows[1:]
                    df = pd.DataFrame(body, columns=header)
                else:
                    df = pd.DataFrame(clean_rows)
                
                # Show the Rendered Table
                st.table(df)
                
            except Exception as e:
                st.warning(f"Preview unavailable: {e}")
        
        st.markdown("---")
        
        if st.button("Generate LaTeX Code", type="primary"):
            if not rows:
                st.error("Please enter some data first.")
            else:
                # Calculate columns based on the widest row
                num_cols = max(len(r) for r in rows)
                
                # Create alignment string (e.g., "|c|c|c|")
                col_char = "c" if center_align else "l"
                align_str = "|" + (col_char + "|") * num_cols
                
                latex_rows = []
                for i, row in enumerate(rows):
                    # Pad row if it's shorter than the max columns
                    clean_cells = [x.strip() for x in row]
                    while len(clean_cells) < num_cols:
                        clean_cells.append("")
                    
                    # Join with &
                    line_str = " & ".join(clean_cells) + " \\\\"
                    
                    # Add horizontal line after header
                    if i == 0 and add_header:
                        line_str += " \\hline"
                        
                    latex_rows.append(line_str)
                
                # Assemble full code
                body_code = "\n".join(latex_rows)
                final_tex = (
                    f"\\begin{{table}}[h!]\n"
                    f"\\centering\n"
                    f"\\begin{{tabular}}{{{align_str}}}\n"
                    f"\\hline\n"
                    f"{body_code}\n"
                    f"\\hline\n"
                    f"\\end{{tabular}}\n"
                    f"\\end{{table}}"
                )
                
                st.subheader("4. LaTeX Output")
                st.code(final_tex, language="latex")