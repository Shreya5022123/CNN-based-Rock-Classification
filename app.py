# Filename: app.py
# 🌄 Sedimentary Rock Classifier — Compact Graph + Detailed Rock Info

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Sedimentary Rock Classifier",
    layout="wide",
    page_icon="🌄",
)

# ---------------------------------
# Custom Styling (Geology-inspired)
# ---------------------------------
st.markdown("""
<style>
    body {
        background-color: #F9F7F3;
        color: #2C2C2C;
    }
    .main {
        background: linear-gradient(145deg, #F5E6CA 0%, #E9D8A6 100%);
        color: #2E2E2E;
        padding-bottom: 20px;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: #4B3832;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #5A4632;
        margin-bottom: 30px;
    }
    .prediction-box {
        background: rgba(255,255,255,0.8);
        border-radius: 15px;
        padding: 25px;
        margin-top: 15px;
        text-align: center;
        border: 1px solid #D8C292;
        box-shadow: 0 0 10px rgba(0,0,0,0.05);
    }
    .confidence {
        color: #AF5B28;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #AF5B28;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #C27E48;
        transform: scale(1.03);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load Model
# ---------------------------------
model = load_model("cnn_model.h5")
classes = ["Coal", "Limestone", "Sandstone"]

# ---------------------------------
# Header
# ---------------------------------
st.markdown("<h1 class='title'>🌄 Sedimentary Rock Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image of a rock to classify it as Coal, Limestone, or Sandstone.</p>", unsafe_allow_html=True)

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.header("📘 Instructions")
st.sidebar.markdown("""
1. Upload a **clear image** of the rock.  
2. Wait a few seconds for analysis.  
3. View classification results & rock info.  
""")
st.sidebar.markdown("---")


# ---------------------------------
# File Upload
# ---------------------------------
uploaded_file = st.file_uploader("📤 Upload Rock Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        img = image.load_img(uploaded_file, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array_expanded = np.expand_dims(img_array, axis=0)
        st.image(uploaded_file, caption="🪨 Uploaded Rock Sample", use_container_width=True)

    with col2:
        with st.spinner("⏳ Analyzing rock image..."):
            prediction = model.predict(img_array_expanded)
            predicted_class = classes[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        st.markdown(f"### 🧭 Predicted Class: **{predicted_class}**")
        st.markdown(f"### 🔹 Confidence: <span class='confidence'>{confidence:.2f}%</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------
    # Compact Probability Chart
    # -------------------------------
    st.markdown("### 📊 Class Probabilities")
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(classes, prediction[0], color=["#4B3832", "#D8C292", "#B59F78"], width=0.5)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Probability", fontsize=10, color="#F9E2E2")
    ax.set_title("Prediction Probability Distribution", fontsize=12, fontweight="bold", color="#FFE9E9")
    ax.tick_params(colors="#FFE0E0", labelsize=9)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{prediction[0][i]*100:.1f}%", ha='center', fontsize=9, color='#3C2F2F', fontweight='bold')
    fig.patch.set_alpha(0)
    ax.set_facecolor("#F9F7F3")
    st.pyplot(fig, use_container_width=False)

    # -------------------------------
    # Detailed Rock Information
    # -------------------------------
    st.markdown("### 🪨 Rock Information")

    rock_info = {
        "Coal": """
**🪵 Composition:**  
Coal is an **organic sedimentary rock** composed primarily of carbon, hydrogen, oxygen, and small amounts of sulfur and nitrogen.  

**🌿 Formation:**  
Formed from the remains of ancient plants buried in swamps, subjected to heat and pressure over millions of years.  

**⚒️ Appearance:**  
Black to dark brown, brittle, lightweight, often with a dull to shiny luster.  

**🔥 Uses:**  
- Major **fuel source** for electricity generation.  
- Used in **steel manufacturing** (coke production).  
- Source of **industrial carbon** and synthetic chemicals.  

**🌍 Fun Fact:**  
Coal is often called "**fossil sunlight**" because it stores ancient solar energy trapped in plant matter.
""",
        "Limestone": """
**🧪 Composition:** Mainly **calcium carbonate (CaCO₃)**.  
**Formation:** Precipitated in marine waters or formed from shells and skeletal fragments.  
**Uses:** Cement, construction, glass, and agriculture.  
""",
        "Sandstone": """
**Composition:** Sand-sized grains of quartz and feldspar.  
**Formation:** Deposited by wind or water in layers, compacted over time.  
**Uses:** Building stone, filters, and artwork material.
"""
    }

    st.markdown(rock_info[predicted_class])

else:
    st.info("👆 Upload an image to begin classification.")
