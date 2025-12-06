import streamlit as st
import os
import random
import utils

# 1. Page Configuration
st.set_page_config(page_title="Math Solver", layout="wide", page_icon="∫")

# 2. Load CSS
utils.setup_page()

# 3. Main Title
st.markdown("<h1 class='main-header'>Welcome to Kenan's IMO calculator</h1>", unsafe_allow_html=True)

# 4. Image Section
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'background.jpg')

try:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image_path, caption='', width=600)
except:
    pass # Fail silently if image is missing

# 5. Version & Footer
st.markdown("""
<div style='text-align: center; margin-bottom: 15px;'>
    <span style='background-color: #2196F3; color: white; padding: 4px 8px; border-radius: 15px; font-size: 0.9rem; font-weight: bold;'>
        v1.0
    </span>
</div>
<div style='text-align: center; color: gray; margin-top: 10px; font-style: italic;'>
    Made by my good friend
</div>
""", unsafe_allow_html=True)

# 8. Offline Guide Section
st.markdown("<br>", unsafe_allow_html=True)
with st.container():
    st.subheader("💻 Offline Access")
    st.warning(
        "**Offline Mode Active:** You are running this tool locally using Streamlit. "
        "No internet connection is required for calculations."
    )

# --- RANDOM FUNNY TIP SECTION ---
quotes = [
    "💡 **Tip:** Keep your friends close, and your Lagrange multipliers non-negative.",
    "💡 **Tip:** Life has ups and downs. If your Hessian is Positive Definite, at least you found the bottom.",
    "💡 **Tip:** Matrix Multiplication is not a hug: Order matters! ($AB \\neq BA$)",
    "💡 **Tip:** Trust the Gradient. It usually knows the way down.",
    "💡 **Tip:** Warning: Matrices are sensitive. Don't transpose them without permission.",
    "💡 **Tip:** A local minimum is good, but a global minimum is what we really want.",
    "💡 **Tip:** If your determinant is 0, you have reached the point of no return (non-invertible).",
    "💡 **Tip:** Don't Panic. Even Newton needed a method to find his roots.",
    "💡 **Tip:** Don't be Indefinite. Pick a side (Saddle Points are uncomfortable).",
    "💡 **Tip:** Stay Positive (Definite)!",
    "💡 **Tip:** Remember to perform regular reality checks",
    "💡 **Tip:** Husk at lave lidt sjov i gaden",
    "💡 **Tip:** Не забывайте делать перерывы и разминаться!",
    "💡 **Tip:** 保持冷静，继续计算！",
    "💡 **Tip:** Remember: Don't panic! Check the Hessian for convexity first.",
]

# 1. Initialize State: Pick a random quote if one isn't already selected
if "current_quote" not in st.session_state:
    st.session_state["current_quote"] = random.choice(quotes)

# 2. Create Layout: Text takes up 85% width, Button takes 15%
col_tip, col_btn = st.columns([6, 1])

with col_tip:
    # Display the quote stored in state
    st.info(st.session_state["current_quote"])

with col_btn:
    # 3. The Button Logic
    # Use a little CSS to align the button with the box
    st.write("") # Spacer
    if st.button("🎲 Next Tip"):
        # Pick a NEW quote (ensure it's not the same as the current one)
        new_quote = random.choice([q for q in quotes if q != st.session_state["current_quote"]])
        st.session_state["current_quote"] = new_quote
        st.rerun()
