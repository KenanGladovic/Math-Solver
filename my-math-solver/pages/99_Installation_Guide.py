import streamlit as st
import utils

# 1. Setup
st.set_page_config(layout="wide", page_icon="💾")
utils.setup_page()

st.markdown("<h1 class='main-header'> Local Installation Guide</h1>", unsafe_allow_html=True)

# --- CONFIGURATION ---
# REPLACE THIS LINK with your actual Google Drive folder link
drive_link = "https://drive.google.com/drive/folders/1teuqC0BvSyBFGmAB-HsWI_5CWGCidxzp?usp=drive_link"

# Create two distinct columns with a large gap
left_col, right_col = st.columns(2, gap="large")

# ==========================================
# LEFT SIDE: INSTALLATION
# ==========================================
with left_col:
    st.info("### 🛠️ Part 1: Installation & Setup")
    
    st.markdown(f"""
    #### Step 1: Download Source
    1. Go to the **[Google Drive Folder]({drive_link})**.
    2. Download the folder (or `.zip` file).
    3. **Save/Unzip** the contents to a local directory (e.g., Desktop).
    """)
    
    st.divider()
    
    st.markdown("#### Step 2: Install Dependencies")
    st.write("Select your operating system method:")
    
    tab_auto, tab_manual = st.tabs(["Windows (Automatic)", "Manual Command"])
    
    with tab_auto:
        st.markdown("""
        **Recommended for Windows:**
        1. Locate the file named **`install.bat`** in the folder.
        2. Double-click it.
        3. A terminal window will appear. Wait for the success message.
        """)
    
    with tab_manual:
        st.markdown("""
        **For Mac/Linux or if Method A fails:**
        1. Open Terminal/Command Prompt in the project folder.
        2. Run the following command:
        """)
        st.code("python -m pip install -r requirements.txt", language="bash")
        st.caption("Mac users may need to use `pip3` instead of `pip`.")

# ==========================================
# RIGHT SIDE: EXECUTION
# ==========================================
with right_col:
    st.success("### 🚀 Part 2: Launching the Application")
    
    st.markdown("#### Step 3: Start the Tool")
    
    tab_run_auto, tab_run_man = st.tabs(["Windows Shortcut", "Command Line"])
    
    with tab_run_auto:
        st.markdown("""
        **Method:**
        1. Locate the file named **`run.bat`** in the folder.
        2. Double-click it.
        3. The application will launch in your default web browser.
        """)
        
    with tab_run_man:
        st.markdown("""
        **Method:**
        1. Open Terminal/Command Prompt in the project folder.
        2. Execute the following command:
        """)
        st.code("streamlit run app.py", language="bash")

    st.divider()
    
    st.markdown("#### 🛑 Troubleshooting")
    with st.expander("Browser does not open automatically?"):
        st.write("Open your web browser manually and navigate to: `http://localhost:8501`")
    
    with st.expander("'Streamlit' is not recognized?"):
        st.write("The installation process in Part 1 was not completed successfully. Please try running the installation command again.")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <b>Status Check:</b> If the application loads successfully in your browser, the local environment is correctly configured for offline use.
</div>
""", unsafe_allow_html=True)