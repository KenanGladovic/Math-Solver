import streamlit as st
import sympy as sp
from utils import setup_page, p_text_area, p_selectbox, parse_expr

def run_fourier_motzkin():
    setup_page()
    
    st.markdown("<h1 style='text-align: center;'>Fourier-Motzkin Elimination Tool</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    <i>Based on Section 4.5: Pedagogical Elimination of Inequalities</i>
    </div>
    """, unsafe_allow_html=True)

    # --- Input Section ---
    st.markdown("### 1. Input System of Inequalities")
    st.markdown("Enter inequalities separated by newlines. The tool accepts `<=` and `>=`.")
    
    # Default matching the user's request (IMO Maj 2025 style)
    default_input = "y <= 1\nx <= 2\ny >= -3/2\n2*x - y >= -6"
    
    ineq_input = p_text_area(
        label="System of Inequalities",
        key="fm_inequalities",
        default=default_input,
        height=150
    )

    if not ineq_input.strip():
        st.info("Please enter a system of inequalities to begin.")
        return

    # --- Parsing ---
    raw_lines = [line.strip() for line in ineq_input.split('\n') if line.strip()]
    inequalities = []
    variables = set()
    
    for line in raw_lines:
        try:
            if "<=" in line:
                lhs_str, rhs_str = line.split("<=")
                op = "<="
            elif ">=" in line:
                lhs_str, rhs_str = line.split(">=")
                op = ">="
            else:
                st.error(f"Could not parse operator in line: {line}. Use '<=' or '>='.")
                return

            lhs = parse_expr(lhs_str)
            rhs = parse_expr(rhs_str)
            
            if lhs is None or rhs is None:
                st.error(f"Could not parse mathematical expression in: {line}")
                return

            # Normalize to lhs - rhs <= 0 for internal processing
            if op == ">=":
                # A >= B  --> B <= A --> B - A <= 0 --> -A + B <= 0
                expr = -lhs + rhs
            else:
                # A <= B --> A - B <= 0
                expr = lhs - rhs
            
            inequalities.append(expr)
            variables.update(expr.free_symbols)
            
        except Exception as e:
            st.error(f"Error parsing line '{line}': {e}")
            return

    if not variables:
        st.warning("No variables found in the system.")
        return

    # --- Variable Selection ---
    sorted_vars = sorted(list(variables), key=lambda v: v.name)
    var_options = [str(v) for v in sorted_vars]
    
    st.markdown("### 2. Elimination Step")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        target_var_name = p_selectbox(
            label="Select variable to eliminate",
            options=var_options,
            key="fm_target_var",
            default_idx=0
        )
    target_var = sp.Symbol(target_var_name)

    # --- Algorithm Implementation ---
    lower_bounds = [] # expr <= target
    upper_bounds = [] # target <= expr
    others = []       # Inequalities without target
    
    for expr in inequalities:
        # expr is: poly <= 0
        # poly = coeff * target + rest
        poly = expr.as_ordered_terms()
        coeff = expr.coeff(target_var)
        rest = expr - coeff * target_var
        
        if coeff == 0:
            others.append(expr)
        elif coeff > 0:
            # coeff * target + rest <= 0 -> target <= -rest/coeff
            upper_bounds.append(-rest / coeff)
        elif coeff < 0:
            # coeff * target + rest <= 0 -> coeff * target <= -rest 
            # -> target >= -rest/coeff (sign flip) -> -rest/coeff <= target
            lower_bounds.append(-rest / coeff)

    # --- LaTeX Helper with Alignment ---
    def aligned_latex(lines):
        # Wraps lines in align* environment
        content = "\\\\\n".join(lines)
        return f"\\begin{{align*}}\n{content}\n\\end{{align*}}"

    # --- Step 1: Isolation ---
    st.markdown("#### Step 1: Isolation")
    st.write(f"We isolate **${target_var_name}$** to find lower ($L_i$) and upper ($U_j$) bounds.")
    
    isolation_lines = []
    
    if upper_bounds:
        for ub in upper_bounds:
            # target <= ub
            isolation_lines.append(f"{target_var_name} &\\le {sp.latex(ub)}")
            
    if lower_bounds:
        for lb in lower_bounds:
            # lb <= target
            isolation_lines.append(f"{sp.latex(lb)} &\\le {target_var_name}")
            
    if others:
        for o in others:
            # o <= 0
            # To make it look nicer, we try to move constant to RHS
            # expr = terms + const <= 0 -> terms <= -const
            const = o.as_coeff_Add()[0]
            terms = o - const
            isolation_lines.append(f"{sp.latex(terms)} &\\le {sp.latex(-const)}")

    st.latex(aligned_latex(isolation_lines))

    # --- Step 2: Max-Min ---
    st.markdown("#### Step 2: The Max-Min Inequality")
    
    max_part = ",".join([sp.latex(lb) for lb in lower_bounds]) if lower_bounds else "-\\infty"
    min_part = ",".join([sp.latex(ub) for ub in upper_bounds]) if upper_bounds else "\\infty"
    
    max_str = f"\\max\\left( {max_part} \\right)" if len(lower_bounds) > 1 else max_part
    min_str = f"\\min\\left( {min_part} \\right)" if len(upper_bounds) > 1 else min_part

    st.latex(f"{max_str} \\le {target_var_name} \\le {min_str}")

    # --- Step 3: Pairwise Elimination ---
    st.markdown("#### Step 3: Pairwise Elimination")
    st.write(f"We form the new system by requiring $L_i \\le U_j$ for all pairs.")
    
    new_system_exprs = []
    pairwise_lines = []
    
    # 1. Pairs (Lower <= Upper)
    if lower_bounds and upper_bounds:
        for lb in lower_bounds:
            for ub in upper_bounds:
                # lb <= ub
                pairwise_lines.append(f"{sp.latex(lb)} &\\le {sp.latex(ub)}")
                new_system_exprs.append(lb - ub) # lb - ub <= 0
    elif lower_bounds:
        st.info(f"Variable {target_var_name} is unbounded from above. No pairs formed.")
    elif upper_bounds:
        st.info(f"Variable {target_var_name} is unbounded from below. No pairs formed.")
        
    # 2. Others (carry over)
    for o in others:
        const = o.as_coeff_Add()[0]
        terms = o - const
        pairwise_lines.append(f"{sp.latex(terms)} &\\le {sp.latex(-const)}")
        new_system_exprs.append(o)
        
    if pairwise_lines:
        st.latex(aligned_latex(pairwise_lines))
    else:
        st.write("No constraints remain.")

    # --- Step 4: Simplified System (The Final Block) ---
    st.markdown("#### Step 4: Final Simplified System")
    st.write("Rearranging to standard form:")
    
    final_lines = []
    for expr in new_system_exprs:
        simp = sp.simplify(expr)
        # Format: LHS <= RHS
        # Separate constant
        const = simp.as_coeff_Add()[0]
        terms = simp - const
        # terms <= -const
        final_lines.append(f"{sp.latex(terms)} &\\le {sp.latex(-const)}")
        
    if not final_lines:
        st.write("No constraints remain.")
    else:
        st.latex(aligned_latex(final_lines))

    # --- Source Code ---
    st.markdown("### LaTeX Source Code")
    full_latex = f"""% Isolation
{aligned_latex(isolation_lines)}

% Max-Min
\\[ {max_str} \\le {target_var_name} \\le {min_str} \\]

% Resulting System
{aligned_latex(final_lines)}
"""
    st.code(full_latex, language="latex")

if __name__ == "__main__":
    run_fourier_motzkin()