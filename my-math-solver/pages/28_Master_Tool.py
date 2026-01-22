import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from utils import setup_page, parse_expr, p_text_input, p_text_area, p_selectbox

def app():
    setup_page()
    st.markdown('<h1 class="main-header">Ω Master Optimization Engine</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="proof-step">
        <strong>System Capabilities:</strong> This engine handles both <em>Unconstrained</em> and <em>Constrained</em> optimization. 
        It automatically computes Gradients, Hessians, proves Convexity, derives KKT conditions, and visualizes the solution space.
    </div>
    """, unsafe_allow_html=True)

    # --- INPUT SECTION ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Problem Definition")
        # Objective Function
        obj_str = p_text_input("Objective Function f(x, y, ...)", "mo_obj", "x**2 + y**2", help="Example: x**2 + y**2 - 4*x")
        
        # Constraints Input
        constraints_str = p_text_area(
            "Constraints (one per line)", 
            "mo_constr", 
            "x + y <= 10\nx >= 0\ny >= 0", 
            height=150,
            help="Supports <=, >=, ="
        )

        st.markdown("---")
        solve_btn = st.button("Analyze & Solve", type="primary")

    # --- LOGIC ENGINE ---
    if solve_btn or st.session_state.get("mo_obj"):
        try:
            # 1. Parsing & Setup
            f_expr = parse_expr(obj_str)
            if f_expr is None:
                st.error("Could not parse objective function.")
                st.stop()
                
            # Detect variables automatically
            vars_found = sorted(list(f_expr.free_symbols), key=lambda s: s.name)
            
            # Parse Constraints
            constraints = []
            raw_lines = [line.strip() for line in constraints_str.split('\n') if line.strip()]
            
            # Parse into standard forms: g(x) <= 0 or h(x) = 0
            std_constraints = [] 
            
            for line in raw_lines:
                if "<=" in line:
                    lhs, rhs = line.split("<=")
                    expr = parse_expr(lhs) - parse_expr(rhs)
                    std_constraints.append({'type': 'ineq', 'expr': expr, 'orig': line})
                    vars_found.extend(expr.free_symbols)
                elif ">=" in line:
                    lhs, rhs = line.split(">=")
                    # Convert a >= b to b - a <= 0
                    expr = parse_expr(rhs) - parse_expr(lhs)
                    std_constraints.append({'type': 'ineq', 'expr': expr, 'orig': line})
                    vars_found.extend(expr.free_symbols)
                elif "=" in line:
                    lhs, rhs = line.split("=")
                    expr = parse_expr(lhs) - parse_expr(rhs)
                    std_constraints.append({'type': 'eq', 'expr': expr, 'orig': line})
                    vars_found.extend(expr.free_symbols)
            
            # Unify variables
            vars_found = sorted(list(set(vars_found)), key=lambda s: s.name)
            
            # --- TABS FOR ANALYSIS ---
            tab_props, tab_kkt, tab_vis, tab_num = st.tabs([
                "1. Structural Analysis (Convexity)", 
                "2. KKT & Lagrange", 
                "3. 3D Visualization",
                "4. Numerical Check"
            ])

            # =========================================================
            # TAB 1: PROPERTIES (Gradient, Hessian, Convexity)
            # =========================================================
            with tab_props:
                st.markdown("### Differential Properties")
                
                # Gradient
                grad = [sp.diff(f_expr, v) for v in vars_found]
                
                # Hessian
                hessian = [[sp.diff(f_expr, v1, v2) for v1 in vars_found] for v2 in vars_found]
                hessian_matrix = sp.Matrix(hessian)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Gradient** $\\nabla f$")
                    st.latex(sp.latex(sp.Matrix(grad)))
                with c2:
                    st.markdown("**Hessian Matrix** $H_f$")
                    st.latex(sp.latex(hessian_matrix))

                # Convexity Check (Eigenvalues)
                st.markdown("### Convexity Analysis")
                try:
                    eigenvals = hessian_matrix.eigenvals()
                    # Check if all eigenvalues are non-negative
                    is_convex = True
                    is_strict = True
                    explanation = []
                    
                    for ev, mult in eigenvals.items():
                        # Try to evaluate if constant
                        if ev.is_number:
                            if ev < 0: 
                                is_convex = False
                                explanation.append(f"Found negative eigenvalue: {ev}")
                            if ev <= 0:
                                is_strict = False
                        else:
                            explanation.append(f"Eigenvalue depends on variables: {sp.latex(ev)}. Cannot determine global convexity purely numerically.")
                            is_convex = None # Indeterminate symbolically without range
                    
                    if is_convex is True:
                        if is_strict:
                            st.success("The function is **Strictly Convex** (Hessian is Positive Definite). Global minimum guaranteed if it exists.")
                        else:
                            st.info("The function is **Convex** (Hessian is Positive Semi-Definite). Local minima are global.")
                    elif is_convex is False:
                        st.warning("The function is **Non-Convex** (Hessian is Indefinite or Negative Definite).")
                    else:
                        st.markdown('<div class="proof-step">Hessian depends on variables. Check regions defined by constraints.</div>', unsafe_allow_html=True)
                
                except Exception as e:
                    st.write("Convexity check requires simplifiable eigenvalues.", e)

            # =========================================================
            # TAB 2: KKT SYSTEM (The Exam Solver)
            # =========================================================
            with tab_kkt:
                st.markdown("### Karush-Kuhn-Tucker (KKT) Derivation")
                
                if not std_constraints:
                    st.info("No constraints detected. Solving $\\nabla f = 0$.")
                    crit_points = sp.solve(grad, vars_found, dict=True)
                    if crit_points:
                        st.write("Critical Points:", crit_points)
                    else:
                        st.write("No symbolic critical points found.")
                else:
                    # Lagrangian Construction
                    # L = f + sum(lambda * g) + sum(nu * h)
                    lambdas = sp.symbols(f'lambda_1:{len(std_constraints)+1}')
                    L = f_expr
                    
                    kkt_text = []
                    kkt_text.append(r"\textbf{1. Lagrangian Function } \mathcal{L}:")
                    
                    term_list = []
                    idx = 0
                    for c in std_constraints:
                        sym = lambdas[idx]
                        idx += 1
                        term_list.append(sym * c['expr'])
                    
                    L_expr = L + sum(term_list)
                    st.latex(r"\mathcal{L} = " + sp.latex(L_expr))
                    
                    st.markdown("---")
                    st.markdown("**2. Stationarity** ($\\nabla \mathcal{L} = 0$)")
                    stat_eqs = []
                    for v in vars_found:
                        eq = sp.diff(L_expr, v)
                        stat_eqs.append(sp.Eq(eq, 0))
                        st.latex(sp.latex(eq) + " = 0")
                        
                    st.markdown("**3. Primal Feasibility**")
                    for c in std_constraints:
                        if c['type'] == 'ineq':
                            st.latex(sp.latex(c['expr']) + r" \le 0")
                        else:
                            st.latex(sp.latex(c['expr']) + r" = 0")
                            
                    st.markdown("**4. Dual Feasibility & Complementary Slackness**")
                    idx = 0
                    for c in std_constraints:
                        sym = lambdas[idx]
                        idx += 1
                        if c['type'] == 'ineq':
                            st.latex(f"{sp.latex(sym)} \ge 0")
                            st.latex(f"{sp.latex(sym)} \\cdot ({sp.latex(c['expr'])}) = 0")
                    
                    st.markdown("""
                    <div class="result-card">
                        This system represents the necessary conditions for optimality. 
                        For convex problems (checked in Tab 1), these are also sufficient.
                    </div>
                    """, unsafe_allow_html=True)

            # =========================================================
            # TAB 3: VISUALIZATION (Plotly)
            # =========================================================
            with tab_vis:
                if len(vars_found) == 2:
                    x_sym, y_sym = vars_found[0], vars_found[1]
                    f_lamb = sp.lambdify((x_sym, y_sym), f_expr, 'numpy')
                    
                    # Create grid
                    x_range = np.linspace(-5, 5, 100)
                    y_range = np.linspace(-5, 5, 100)
                    X, Y = np.meshgrid(x_range, y_range)
                    Z = f_lamb(X, Y)
                    
                    # 3D Surface
                    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])
                    
                    # Add Constraint Planes (approximate visualization)
                    # Note: Perfect constraint clipping in 3D is complex; we show the objective surface
                    # and perhaps a contour plot below.
                    
                    fig.update_layout(title='Objective Function Landscape', autosize=True, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption("Note: Constraints are not visually clipped in 3D mode yet.")
                else:
                    st.info("Visualization is currently optimized for 2-variable problems.")

            # =========================================================
            # TAB 4: NUMERICAL CHECK (Fallback)
            # =========================================================
            with tab_num:
                st.markdown("### Numerical Solver (SciPy)")
                st.write("Attempting to find numerical minimum starting at [1, 1, ...]")
                
                from scipy.optimize import minimize
                
                if len(vars_found) > 0:
                    # Objective wrapper
                    func_n = sp.lambdify(vars_found, f_expr, 'numpy')
                    def obj_wrap(args):
                        return func_n(*args)
                    
                    # Constraint wrappers
                    cons = []
                    for c in std_constraints:
                        # Scipy constraints: 'ineq' means val >= 0. Our storage is expr <= 0 => -expr >= 0
                        c_func = sp.lambdify(vars_found, c['expr'], 'numpy')
                        
                        if c['type'] == 'ineq':
                            # We stored as expr <= 0. SciPy wants C(x) >= 0. So we pass -expr.
                            # We need a closure to capture the specific function
                            def make_cons(f):
                                return {'type': 'ineq', 'fun': lambda x: -f(*x)}
                            cons.append(make_cons(c_func))
                        else:
                            def make_eq(f):
                                return {'type': 'eq', 'fun': lambda x: f(*x)}
                            cons.append(make_eq(c_func))

                    x0 = [1.0] * len(vars_found)
                    try:
                        res = minimize(obj_wrap, x0, constraints=cons)
                        
                        if res.success:
                            st.success(f"Optimal Value Found: {res.fun:.4f}")
                            st.write("At point:")
                            st.json({str(v): float(val) for v, val in zip(vars_found, res.x)})
                        else:
                            st.error(f"Solver failed: {res.message}")
                    except Exception as e:
                        st.error(f"Numerical solver error: {e}")

        except Exception as e:
            st.error(f"An error occurred in the analysis engine: {e}")
            st.write("Tip: Ensure variables are explicit (e.g., '2*x' not '2x').")

if __name__ == "__main__":
    app()