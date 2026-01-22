import streamlit as st
import sympy as sp
import re
from utils import setup_page, parse_expr, p_text_input, p_selectbox

def app():
    setup_page()

    st.markdown('<h1 class="main-header">Algebraic Equation Manipulator</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="proof-step">
        <strong>How to use:</strong> Enter an expression (e.g., <code>x**2 - 4</code>) 
        or a relation using <code>=, &lt;, &gt;, &le;, &ge;</code> (e.g., <code>-x &le; 10</code>). 
        Select an operation to manipulate it.
    </div>
    """, unsafe_allow_html=True)

    # --- INPUT SECTION ---
    col1, col2 = st.columns([2, 1])

    with col1:
        raw_input = p_text_input(
            label="Enter Expression or Relation",
            key="alg_input",
            default="-2*x + 4 <= 10" # Changed default to demonstrate inequality
        )

    # --- PARSING LOGIC ---
    # We need to detect standard math relations. 
    # Order matters: check 2-char operators first to avoid partial matches (e.g. matching '<' inside '<=')
    # We treat "=" and "==" as Equality.
    rel_ops = ["<=", ">=", "!=", "==", "=", "<", ">"]
    
    expr_obj = None
    is_relational = False
    
    # 1. Identify if there is a relational operator in the string
    found_op = None
    split_char = None
    
    for op in rel_ops:
        if op in raw_input:
            found_op = op
            split_char = op
            break
            
    # 2. Parse based on findings
    if found_op:
        parts = raw_input.split(split_char, 1) # Split only on the first occurrence
        if len(parts) == 2:
            lhs_str, rhs_str = parts[0], parts[1]
            lhs = parse_expr(lhs_str)
            rhs = parse_expr(rhs_str)
            
            if lhs is not None and rhs is not None:
                is_relational = True
                # Create the specific SymPy object
                if found_op == "<=":
                    expr_obj = sp.Le(lhs, rhs)
                elif found_op == ">=":
                    expr_obj = sp.Ge(lhs, rhs)
                elif found_op == "<":
                    expr_obj = sp.Lt(lhs, rhs)
                elif found_op == ">":
                    expr_obj = sp.Gt(lhs, rhs)
                elif found_op == "!=":
                    expr_obj = sp.Ne(lhs, rhs)
                else: # "=" or "=="
                    expr_obj = sp.Eq(lhs, rhs)

    # Fallback: If no operator found, parse as standard expression
    if expr_obj is None:
        expr_obj = parse_expr(raw_input)

    # --- PREVIEW SECTION ---
    with col2:
        st.caption("LaTeX Preview")
        if expr_obj is not None:
            st.latex(sp.latex(expr_obj))
        else:
            st.error("Invalid Syntax")
            st.stop()

    st.markdown("---")

    # --- OPERATION SELECTION ---
    ops = ["Simplify", "Expand", "Factor", "Collect Terms"]
    
    if is_relational:
        # Relational specific operations
        ops = ["Solve for Variable", "Multiply by -1 & Flip", "Flip Sides"] + ops
    else:
        # Expression specific operations
        ops = ["Find Roots (Solve = 0)", "Differentiate", "Integrate"] + ops

    selected_op = p_selectbox("Select Operation", ops, "alg_op")

    # --- EXECUTION ENGINE ---
    result = None
    steps_text = ""

    try:
        if selected_op == "Simplify":
            if is_relational:
                # Reconstruct the relation with simplified sides
                result = expr_obj.func(sp.simplify(expr_obj.lhs), sp.simplify(expr_obj.rhs))
            else:
                result = sp.simplify(expr_obj)
            steps_text = "Applied simplification rules."

        elif selected_op == "Expand":
            if is_relational:
                result = expr_obj.func(sp.expand(expr_obj.lhs), sp.expand(expr_obj.rhs))
            else:
                result = sp.expand(expr_obj)
            steps_text = "Expanded polynomial products."

        elif selected_op == "Factor":
            if is_relational:
                # Move everything to LHS: LHS - RHS Rel 0
                temp_expr = expr_obj.lhs - expr_obj.rhs
                factored = sp.factor(temp_expr)
                # We interpret "Factor" on a relation as factoring the difference compared to 0
                result = expr_obj.func(factored, 0)
                steps_text = "Moved terms to LHS and factored."
            else:
                result = sp.factor(expr_obj)
                steps_text = "Factored the expression."

        elif selected_op == "Collect Terms":
            free_syms = list(expr_obj.free_symbols)
            if free_syms:
                var_to_collect = p_selectbox("Collect with respect to:", [str(s) for s in free_syms], "alg_collect_var")
                sym_collect = sp.Symbol(var_to_collect)
                if is_relational:
                    result = expr_obj.func(sp.collect(expr_obj.lhs, sym_collect), sp.collect(expr_obj.rhs, sym_collect))
                else:
                    result = sp.collect(expr_obj, sym_collect)
            else:
                st.warning("No variables found to collect.")

        elif selected_op == "Flip Sides":
            if is_relational:
                # For equations, easy swap. For inequalities, logic is needed: A < B -> B > A
                # SymPy usually handles this via .reversed, but let's be explicit
                if isinstance(expr_obj, sp.Eq) or isinstance(expr_obj, sp.Ne):
                    result = expr_obj.func(expr_obj.rhs, expr_obj.lhs)
                elif isinstance(expr_obj, sp.Lt): # <
                    result = sp.Gt(expr_obj.rhs, expr_obj.lhs)
                elif isinstance(expr_obj, sp.Le): # <=
                    result = sp.Ge(expr_obj.rhs, expr_obj.lhs)
                elif isinstance(expr_obj, sp.Gt): # >
                    result = sp.Lt(expr_obj.rhs, expr_obj.lhs)
                elif isinstance(expr_obj, sp.Ge): # >=
                    result = sp.Le(expr_obj.rhs, expr_obj.lhs)
                
                steps_text = "Swapped LHS and RHS (reversed inequality direction)."

        # --- NEW FUNCTIONALITY: MULTIPLY BY -1 ---
        elif selected_op == "Multiply by -1 & Flip":
            if is_relational:
                # 1. Multiply sides by -1
                new_lhs = sp.Mul(-1, expr_obj.lhs, evaluate=True)
                new_rhs = sp.Mul(-1, expr_obj.rhs, evaluate=True)
                
                # 2. Determine new operator (Flip direction)
                if isinstance(expr_obj, sp.Eq):     # = stays =
                    result = sp.Eq(new_lhs, new_rhs)
                elif isinstance(expr_obj, sp.Ne):   # != stays !=
                    result = sp.Ne(new_lhs, new_rhs)
                elif isinstance(expr_obj, sp.Lt):   # < becomes >
                    result = sp.Gt(new_lhs, new_rhs)
                elif isinstance(expr_obj, sp.Le):   # <= becomes >=
                    result = sp.Ge(new_lhs, new_rhs)
                elif isinstance(expr_obj, sp.Gt):   # > becomes <
                    result = sp.Lt(new_lhs, new_rhs)
                elif isinstance(expr_obj, sp.Ge):   # >= becomes <=
                    result = sp.Le(new_lhs, new_rhs)
                
                steps_text = "Multiplied by -1 and flipped inequality sign."

        elif selected_op == "Solve for Variable" or selected_op == "Find Roots (Solve = 0)":
            free_syms = list(expr_obj.free_symbols)
            if not free_syms:
                st.warning("No variables to solve for.")
            else:
                var_str = p_selectbox("Solve for:", [str(s) for s in free_syms], "alg_solve_var")
                solve_var = sp.Symbol(var_str)
                
                # SymPy solve_univariate_inequality or standard solve
                # If it's an inequality, we use reduce_inequalities or solveset
                if is_relational and not isinstance(expr_obj, sp.Eq):
                    try:
                        result_set = sp.reduce_inequalities(expr_obj, solve_var)
                        result = result_set
                        steps_text = f"Solved inequality for {var_str}"
                    except:
                        # Fallback if too complex
                        result_set = sp.solve(expr_obj, solve_var)
                        result = result_set
                else:
                    result_set = sp.solve(expr_obj, solve_var)
                    if not result_set:
                        steps_text = "No solution found."
                        result = "No Solution"
                    else:
                        steps_text = f"Solved for {var_str}"
                        result = result_set

        elif selected_op == "Differentiate":
            free_syms = list(expr_obj.free_symbols)
            if free_syms:
                var_str = p_selectbox("Differentiate w.r.t:", [str(s) for s in free_syms], "alg_diff_var")
                result = sp.diff(expr_obj, sp.Symbol(var_str))
                steps_text = f"Calculated derivative with respect to {var_str}"

        elif selected_op == "Integrate":
            free_syms = list(expr_obj.free_symbols)
            if free_syms:
                var_str = p_selectbox("Integrate w.r.t:", [str(s) for s in free_syms], "alg_int_var")
                result = sp.integrate(expr_obj, sp.Symbol(var_str))
                steps_text = f"Calculated indefinite integral with respect to {var_str}"

    except Exception as e:
        st.markdown(f'<div class="error-box">Error during calculation: {str(e)}</div>', unsafe_allow_html=True)
        result = None

    # --- OUTPUT DISPLAY ---
    if result is not None:
        st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"**Operation:** {selected_op}")
        if steps_text:
            st.info(steps_text)
        
        st.markdown("### Result:")
        
        # Display logic
        if isinstance(result, list):
            for i, sol in enumerate(result):
                st.latex(f"x_{i+1} = {sp.latex(sol)}")
        elif isinstance(result, str):
            st.write(result)
        else:
            st.latex(sp.latex(result))
            
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()