import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- CONFIGURATION ---
def setup_page():
    st.markdown("""
    <style>
        /* --- Your Original Styles --- */
        .main-header { font-size: 2.5rem; color: #4B4B4B; text-align: center; margin-bottom: 1rem; }
        .proof-step { background-color: #f9f9f9; border-left: 4px solid #2196F3; padding: 10px; margin-bottom: 10px; }
        .result-card { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .error-box { background-color: #ffebee; border-left: 5px solid #f44336; padding: 10px; }
        .success-box { background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 10px; }

        /* --- NEW: Target ONLY the "Home" line in the sidebar --- */
        /* This targets the first item in the sidebar navigation list */
        div[data-testid="stSidebarNav"] > ul > li:first-child a {
            
            
            
            /* 2. Optional: Make it bigger or bold */
            font-size: 1.2rem !important;
            font-weight: bold !important;
            
            /* 3. Optional: Change the color */
            color: #2196F3 !important; 
        }
    </style>
    """, unsafe_allow_html=True)

# --- MATH HELPER ---
def parse_expr(input_str):
    try:
        x, y, z, t, a, b, c = sp.symbols('x y z t a b c')
        return sp.sympify(input_str, locals={'x': x, 'y': y, 'z': z, 't': t, 'a': a, 'b': b, 'c': c, 'pi': sp.pi, 'log': sp.log, 'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp})
    except: return None

# --- WIDGET WRAPPERS ---
def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

def p_text_input(label, key, default, **kwargs):
    init_state(key, default)
    def on_change(): st.session_state[key] = st.session_state[f"w_{key}"]
    return st.text_input(label, value=st.session_state[key], key=f"w_{key}", on_change=on_change, **kwargs)

def p_text_area(label, key, default, height=None, **kwargs):
    init_state(key, default)
    def on_change(): st.session_state[key] = st.session_state[f"w_{key}"]
    return st.text_area(label, value=st.session_state[key], key=f"w_{key}", on_change=on_change, height=height, **kwargs)

def p_number_input(label, key, default, **kwargs):
    init_state(key, default)
    def on_change(): st.session_state[key] = st.session_state[f"w_{key}"]
    return st.number_input(label, value=st.session_state[key], key=f"w_{key}", on_change=on_change, **kwargs)

def p_slider(label, key, min_value, max_value, default, **kwargs):
    init_state(key, default)
    def on_change(): st.session_state[key] = st.session_state[f"w_{key}"]
    return st.slider(label, min_value=min_value, max_value=max_value, value=st.session_state[key], key=f"w_{key}", on_change=on_change, **kwargs)

def p_selectbox(label, options, key, default_idx=0, **kwargs):
    default = options[default_idx]
    init_state(key, default)
    def on_change(): st.session_state[key] = st.session_state[f"w_{key}"]
    try: idx = options.index(st.session_state[key])
    except: idx = 0
    return st.selectbox(label, options, index=idx, key=f"w_{key}", on_change=on_change, **kwargs)

def p_radio(label, options, key, default_idx=0, **kwargs):
    default = options[default_idx]
    init_state(key, default)
    def on_change(): st.session_state[key] = st.session_state[f"w_{key}"]
    try: idx = options.index(st.session_state[key])
    except: idx = 0
    return st.radio(label, options, index=idx, key=f"w_{key}", on_change=on_change, **kwargs)
