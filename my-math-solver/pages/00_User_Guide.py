import streamlit as st
import pandas as pd
import utils

# 1. Setup
st.set_page_config(layout="wide")
utils.setup_page()
st.markdown("<h1 class='main-header'>📚 User Guide & Syntax Help</h1>", unsafe_allow_html=True)

st.info("This tool runs entirely on your computer using Python. This guide explains how to format your math inputs correctly.")


# --- TABS FOR ORGANIZATION ---
tab1, tab2, tab3 = st.tabs(["🔣 Syntax Cheat Sheet", "⚠️ Common Errors", "🛠️ Tool Overviews"])

# ==================================================
# TAB 1: SYNTAX (Updated for Exams)
# ==================================================
with tab1:
    st.subheader("1. The Golden Rules")
    
    st.markdown("""
    * **Multiplication:** You must always use `*`. Writing `2x` or `xy` will cause an error. Use `2*x` or `x*y`.
    * **Powers:** Use `**`. Writing `x^2` will cause an error (in Python `^` means "XOR").
    """)

    st.subheader("2. Symbols & Notation")
    st.markdown("Use this table to translate exam questions into the tool:")

    # Comprehensive Data Table based on Jan 22-25 Exams
    data = {
        "Math Concept": [
            "Powers / Exponents", 
            "Exponential Function ($e^x$)", 
            "Natural Logarithm ($\\ln x$)",
            "Logarithm Base 10 ($\\log_{10} x$)",
            "Logarithm Base 2 ($\\log_2 x$)",
            "Trigonometry", 
            "Square Root",
            "Inequalities",
            "Equations",
            "Fractions"
        ],
        "Exam Notation": [
            "$x^2, (x+y)^2$", 
            "$e^{x-y}$", 
            "$\\ln(x)$ or $\\log(x)$", 
            "$\\log_{10}(x)$", 
            "$\\log_2(x)$", 
            "$\\cos(t), \\sin(t)$", 
            "$\\sqrt{x^2+1}$", 
            "$x+y \\le 4$", 
            "$x^2 - 5x = 0$", 
            "$\\frac{1}{2}x$"
        ],
        "Type this in Tool": [
            "`x**2`, `(x+y)**2`", 
            "`exp(x-y)`", 
            "`log(x)`", 
            "`log(x, 10)`", 
            "`log(x, 2)`", 
            "`cos(t)`, `sin(t)`", 
            "`sqrt(x**2+1)`", 
            "`x+y <= 4`", 
            "`x**2 - 5*x == 0`", 
            "`1/2 * x` or `0.5*x`"
        ]
    }
    df = pd.DataFrame(data)
    st.table(df)

    # --- SPECIAL SECTION: LOGS & EXP ---
    st.markdown("### 🚨 Important: Logarithms & 'e'")
    col_a, col_b = st.columns(2)
    with col_a:
        st.warning("**The symbol `e`**")
        st.markdown("""
        Do **NOT** type `e**x`.
        * **Why?** The computer might think 'e' is a variable (like x or y).
        * **Correct:** Use `exp(x)` for $e^x$.
        * **Example:** `exp(2*x)` is $e^{2x}$.
        """)
    with col_b:
        st.warning("**The function `log`**")
        st.markdown("""
        In this tool, `log(x)` is the **Natural Log** ($\ln$).
        * **Why?** This is standard in Python/Calculus.
        * **Base 10:** Type `log(x, 10)`.
        * **Base 2:** Type `log(x, 2)`.
        """)

# ==================================================
# TAB 2: COMMON ERRORS
# ==================================================
with tab2:
    st.subheader("Troubleshooting Guide")
    
    with st.expander("1. 'SyntaxError' or 'Parse Error'"):
        st.write("""
        **Meaning:** The program couldn't read your math.
        **Fix:**
        1. Did you forget a multiplication sign? (`2x` → `2*x`)
        2. Did you use `^` instead of `**`?
        3. Do you have mismatched parentheses? `(x + y`
        """)

    with st.expander("2. 'Singular Matrix' (Linear Algebra/Newton)"):
        st.write("""
        **Meaning:** The problem you entered has no unique solution (or infinite solutions).
        **Fix:**
        * In **Newton's Method**: Try a different "Start Point". You might have started on a flat spot where the derivative is 0.
        * In **Least Squares**: You might not have enough data points, or your points are all on a straight line.
        """)
        
    with st.expander("3. Plot is blank"):
        st.write("""
        **Meaning:** The function might be undefined in the range you selected.
        **Fix:**
        * Check your x min/max ranges.
        * Example: Plotting `log(x)` starting at `x = -5` will fail because log is undefined for negative numbers. Change start to `0.1`.
        """)

# ==================================================
# TAB 3: TOOL OVERVIEWS
# ==================================================
with tab3:
    st.subheader("Exam Strategy Map")
    
    st.markdown("""
    ### 🧠 Analysis Tools
    * **KKT Tool:** It generates the LaTeX conditions for you and verifies if points like $(2,0)$ are optimal.
    * **Subset Analysis:** Use when asked if a set $C$ is "Compact", "Closed", or "Bounded".
    
    ### 📉 Calculus & Solvers
    * **Newton's Method:** Use if asked to find roots or optimize without a formula.
    * **Fourier-Motzkin:** Use for systems of inequalities.
    * **Equation Solver:** Use to check your algebra when solving $\\nabla f = 0$.
    
    ### 📐 Linear Algebra 
    * **Matrix Operations:** Use for inverses, determinants, and multiplication.
    * **Symmetric Diagonalization:** Use for $B^T A B = D$ problems (Chapter 8).
    * **Least Squares:** Use for fitting lines/circles to points.
    """)