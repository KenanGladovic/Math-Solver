import streamlit as st
import os
import utils
import datetime

# --- 🛠️ UPDATE THIS DATE BEFORE SHARING ---
LAST_UPDATE = "22 Jan 2026, 14:00 CET"
VERSION = "Final Version"
# ------------------------------------------

# 1. Page Configuration
st.set_page_config(page_title="Math Solver", layout="wide", page_icon="∫")

# 2. Load CSS
utils.setup_page()

# 3. Main Title
st.markdown("<h1 class='main-header'>Welcome to Kenan's IMO calculator</h1>", unsafe_allow_html=True)

# --- 🆕 ADDED: Safe Launch Command ---
st.info("💡 **Safe Launch Command (Offline Mode)**")
st.code("py -m streamlit run App.py --browser.gatherUsageStats=false --server.headless=true --server.address=localhost", language="batch")
# -------------------------------------

# 4. Countdown Timer (JavaScript Implementation)
#    - Exam Date: Jan 23, 2026 at 10:00:00
target_date = "Jan 23, 2026 10:00:00"

st.components.v1.html(
    f"""
    <div style="text-align: center; font-family: sans-serif; color: #4B4B4B; margin-bottom: 20px;">
        <h3 style="margin-bottom: 5px;">Countdown to Exam</h3>
        <div id="countdown" style="font-size: 2.5rem; font-weight: bold; color: #d32f2f;">
            Loading...
        </div>
        <div id="panic-msg" style="font-size: 1.5rem; font-weight: bold; color: red; display: none;">
            ⚠️ START PANICKING ⚠️
        </div>
    </div>

    <script>
    var countDownDate = new Date("{target_date}").getTime();

    var x = setInterval(function() {{
      var now = new Date().getTime();
      var distance = countDownDate - now;

      // Time calculations
      var days = Math.floor(distance / (1000 * 60 * 60 * 24));
      var hours = Math.floor((distance % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      var minutes = Math.floor((distance % (1000 * 60 * 60)) / (1000 * 60));
      var seconds = Math.floor((distance % (1000 * 60)) / 1000);

      // Display the result
      document.getElementById("countdown").innerHTML = days + "d " + hours + "h "
      + minutes + "m " + seconds + "s ";

      // 2 HOURS LEFT LOGIC (2 hours * 60 mins * 60 secs * 1000 ms = 7,200,000)
      if (distance > 0 && distance < 7200000) {{
          document.getElementById("panic-msg").style.display = "block";
          document.getElementById("countdown").style.color = "#FF5722"; // Orange/Red warning
      }} else {{
          document.getElementById("panic-msg").style.display = "none";
      }}

      // EXAM START LOGIC
      if (distance < 0) {{
        clearInterval(x);
        document.getElementById("panic-msg").style.display = "none"; // Hide panic
        document.getElementById("countdown").innerHTML = "Husk at denne eksamen er gratis";
        document.getElementById("countdown").style.color = "#4CAF50"; // Green for go
        document.getElementById("countdown").style.fontSize = "2rem"; // Slightly smaller to fit text
      }}
    }}, 1000);
    </script>
    """,
    height=150
)

# 6. Version & Footer
st.markdown(f"""
<div style='text-align: center; margin-top: 30px; margin-bottom: 10px;'>
    <span style='background-color: #2196F3; color: white; padding: 6px 12px; border-radius: 15px; font-size: 0.9rem; font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.2);'>
        {VERSION}  
    </span>
</div>
<div style='text-align: center; color: #666; font-size: 0.85rem; margin-bottom: 20px;'>
    Last Updated: <b>{LAST_UPDATE}</b>
</div>
<div style='text-align: center; color: gray; font-style: italic; font-size: 0.8rem;'>
    Made in collaboration with my good friend
</div>
""", unsafe_allow_html=True)