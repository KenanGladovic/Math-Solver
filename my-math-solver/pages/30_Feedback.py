import streamlit as st
import streamlit.components.v1 as components
import utils  # Assuming you have this from your other page

# 1. Setup
st.set_page_config(page_title="Feedback", layout="wide")
utils.setup_page()

# --- HEADER ---
st.markdown("<h1 class='main-header'>💬 Feedback & Suggestions</h1>", unsafe_allow_html=True)

st.info("Your feedback helps improve this tool for everyone. Whether you found a bug, have a feature request, or have any other comments!")

# --- CONFIGURATION (PASTE YOUR FORM LINK HERE) ---
# Create a Google Form or Microsoft Form and paste the "Send" link here:
FORM_URL = "https://form.typeform.com/to/jhUA6722"

# --- LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Why give feedback?")
    st.markdown("""
    * **🐛 Report Bugs:** Did a specific math problem fail? Let me know the input you used.
    * **💡 Feature Requests:** Is there a specific exam topic missing?
    * **📈 Usability:** Is the interface confusing in any way?
    """)
    

with col2:
    st.subheader("Submit Feedback")
    
    # Method 1: The Modern Link Button
    st.markdown("Click the button below to open the form in a new tab:")
    st.link_button("📝 Open Feedback Form", FORM_URL, type="primary")
    
    
    
 