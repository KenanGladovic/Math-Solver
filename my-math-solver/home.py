import streamlit as st
import os
import utils
import datetime

# 1. Page Configuration
st.set_page_config(page_title="Math Solver", layout="wide", page_icon="∫")

# 2. Load CSS
utils.setup_page()

# 3. Main Title
st.markdown("<h1 class='main-header'>Welcome to Kenan's IMO calculator</h1>", unsafe_allow_html=True)

# 4. Countdown Timer (JavaScript Implementation)
# Target: January 23, 2026 at 10:00:00
target_date = "Jan 23, 2026 10:00:00"

st.components.v1.html(
    f"""
    <div style="text-align: center; font-family: sans-serif; color: #4B4B4B; margin-bottom: 20px;">
        <h3 style="margin-bottom: 5px;">Countdown to Exam</h3>
        <div id="countdown" style="font-size: 2.5rem; font-weight: bold; color: #d32f2f;">
            Loading...
        </div>
        </div>

    <script>
    var countDownDate = new Date("{target_date}").getTime();

    var x = setInterval(function() {{
      var now = new Date().getTime();
      var distance = countDownDate - now;

      var days = Math.floor(distance / (1000 * 60 * 60 * 24));
      var hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      var minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
      var seconds = Math.floor((distance % (1000 * 60)) / 1000);

      document.getElementById("countdown").innerHTML = days + "d " + hours + "h "
      + minutes + "m " + seconds + "s ";

      if (distance < 0) {{
        clearInterval(x);
        document.getElementById("countdown").innerHTML = "EXAM STARTED / FINISHED";
        document.getElementById("countdown").style.color = "#4CAF50";
      }}
    }}, 1000);
    </script>
    """,
    height=120 # Reduced height slightly since text was removed
)

# 5. Image Section
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = None
for ext in ['jpg', 'png', 'jpeg']:
    temp_path = os.path.join(current_dir, f'background.{ext}')
    if os.path.exists(temp_path):
        image_path = temp_path
        break

if image_path:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image_path, caption='', width=600)

# 6. Version & Footer
st.markdown("""
<div style='text-align: center; margin-top: 30px; margin-bottom: 15px;'>
    <span style='background-color: #2196F3; color: white; padding: 6px 12px; border-radius: 15px; font-size: 0.9rem; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
        v3.1
    </span>
</div>
<div style='text-align: center; color: gray; margin-top: 10px; font-style: italic;'>
    Made in collaboration with my good friend
</div>
""", unsafe_allow_html=True)