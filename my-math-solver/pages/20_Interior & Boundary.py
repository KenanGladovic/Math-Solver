import streamlit as st
import sympy as sp
import utils

def main():
    # 1. styling setup
    utils.setup_page()

    st.markdown("<h1 style='text-align: center;'>Subset: Interior & Boundary</h1>", unsafe_allow_html=True)
    st.info("Input the inequalities defining the set $C$. The tool will compute the Interior ($C^{\\circ}$) and Boundary ($\partial C$) and provide the LaTeX code.")

    # 2. User Input using Custom Wrapper
    default_input = "x <= 3\ny <= 3\nx + y >= 4"
    
    ineq_input = utils.p_text_area(
        label="Define Set C (one inequality per line):",
        key="subset_inequalities",
        default=default_input,
        height=150
    )

    if ineq_input:
        # 3. Processing Logic
        lines = [line.strip() for line in ineq_input.split('\n') if line.strip()]
        
        c_conditions = []        # For defining C
        interior_conditions = [] # For defining Co
        boundary_conditions = [] # For defining dC
        
        valid_parse = True
        
        for line in lines:
            # Use utils.parse_expr for safety
            expr = utils.parse_expr(line)
            
            # Validation
            if expr is None or not isinstance(expr, (sp.LessThan, sp.GreaterThan, sp.StrictLessThan, sp.StrictGreaterThan, sp.Eq, sp.Ne)):
                st.error(f"Could not parse inequality: '{line}'. Please use standard format (e.g., x <= 3).")
                valid_parse = False
                break
            
            # Extract components
            lhs, rhs = expr.lhs, expr.rhs
            lhs_latex = sp.latex(lhs)
            rhs_latex = sp.latex(rhs)
            
            # Logic: 
            # Interior uses strict inequalities (<, >).
            # Boundary uses equalities (=).
            
            if isinstance(expr, sp.LessThan): # <=
                c_op = r"\le"
                int_op = "<"
                bound_op = "="
            elif isinstance(expr, sp.GreaterThan): # >=
                c_op = r"\ge"
                int_op = ">"
                bound_op = "="
            elif isinstance(expr, sp.StrictLessThan): # <
                c_op = "<"
                int_op = "<" 
                bound_op = "="
            elif isinstance(expr, sp.StrictGreaterThan): # >
                c_op = ">"
                int_op = ">"
                bound_op = "="
            elif isinstance(expr, sp.Eq): # =
                c_op = "="
                int_op = r"\emptyset" 
                bound_op = "="
            else:
                c_op = "?"
                int_op = "?"
                bound_op = "?"

            c_conditions.append(f"{lhs_latex} {c_op} {rhs_latex}")
            interior_conditions.append(f"{lhs_latex} {int_op} {rhs_latex}")
            boundary_conditions.append(f"{lhs_latex} {bound_op} {rhs_latex}")

        if valid_parse:
            # 4. Construct LaTeX Strings
            # Interior uses comma separation
            # Boundary uses Logical OR (\lor)
            
            c_str = ", ".join(c_conditions)
            co_str = ", ".join(interior_conditions)
            dc_str = r" \lor ".join(boundary_conditions)

            # Define the sets in LaTeX
            # Using ^{\circ} as requested
            latex_c = f"C = \\{{(x, y) \\in \\mathbb{{R}}^2 \\mid {c_str}\\}}"
            latex_co = f"C^{{\\circ}} = \\{{(x, y) \\in \\mathbb{{R}}^2 \\mid {co_str}\\}}"
            # Boundary definition references 'in C'
            latex_dc = f"\\partial C = \\{{(x, y) \\in C \\mid {dc_str}\\}}"
            
            # Combined answer string "og" (and)
            combined_latex = f"{latex_co} \\quad \\text{{og}} \\quad {latex_dc}"

            # 5. Display Results
            
            st.subheader("Rendered Result")
            st.latex(latex_c)
            st.latex(latex_co)
            st.latex(latex_dc)
            
            st.divider()
            
            st.subheader("LaTeX Source Code")
            
            # Tabs for organization
            tab1, tab2, tab3, tab4 = st.tabs(["Combined Answer", "Interior Only", "Boundary Only", "Set C Only"])
            
            with tab1:
                st.markdown("**Combined Answer:**")
                st.code(combined_latex, language="latex")
            
            with tab2:
                st.markdown("**Interior ($C^{\circ}$):**")
                st.code(latex_co, language="latex")
                
            with tab3:
                st.markdown("**Boundary ($\partial C$):**")
                st.code(latex_dc, language="latex")
                
            with tab4:
                st.markdown("**Set Definition ($C$):**")
                st.code(latex_c, language="latex")

if __name__ == "__main__":
    main()