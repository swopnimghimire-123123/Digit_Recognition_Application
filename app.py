"""
Real-Time MNIST Digit Recognition Dashboard
Enhanced Version

Features
- Live digit prediction while drawing
- Probability distribution visualization
- Model info sidebar
- Clean dashboard UI
"""

import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_DSHOW'] = '0'

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Digit Recognition AI",
    page_icon="🔢",
    layout="wide"
)


# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:700;
text-align:center;
}

.subtitle{
text-align:center;
color:gray;
margin-bottom:30px;
}

.prediction-box{
background-color:#111;
padding:25px;
border-radius:15px;
text-align:center;
}

.big-digit{
font-size:120px;
font-weight:700;
color:#00FFB3;
}

.confidence-text{
font-size:20px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- TITLE ---------------- #

st.markdown('<div class="main-title">Real-Time Digit Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Draw a number and watch the AI predict instantly</div>', unsafe_allow_html=True)


# ---------------- LOAD MODEL ---------------- #

model = load_model("model/digit_model.h5")


# ---------------- SIDEBAR INFO ---------------- #

st.sidebar.title("Model Information")

st.sidebar.markdown("""
Dataset: MNIST  

Training Samples: **60,000**  
Test Samples: **10,000**

Training Exposure:  
~ **1.2 Million augmented samples**

Architecture

Conv2D (32)  
Conv2D (32)  
MaxPool  
Dropout  
Conv2D (64)  
Conv2D (64)  
MaxPool  
Dropout  
Dense (128)  
Dropout  
Dense (10 Softmax)
Optimizer: Adam  
Epochs: 20  

Final Test Accuracy: **99.61%**
""")


# ---------------- SESSION STATE ---------------- #

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


# ---------------- LAYOUT ---------------- #

left_col, right_col = st.columns([1,1])


# ================= DRAWING PANEL ================= #

with left_col:

    st.subheader("Draw Digit")

    brush_size = st.slider("Brush Size", 5, 40, 18)

    if st.button("Clear Canvas"):
        st.session_state.canvas_key += 1

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=brush_size,
        stroke_color="white",
        background_color="black",
        height=320,
        width=320,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
        display_toolbar=True
    )


# ================= RESULT PANEL ================= #

with right_col:

    st.subheader("Live Prediction")

    img = None

    if canvas_result.image_data is not None:

        img = canvas_result.image_data
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2GRAY)

        # detect if canvas is mostly empty
        if np.mean(img) < 5:
            img = None


    if img is None:

        st.info("Start drawing a digit...")

    else:

        # ---------------- PREPROCESS ---------------- #

        img_resized = cv2.resize(img, (28,28))
        img_resized = img_resized / 255.0
        img_resized = img_resized.reshape(1,28,28,1)


        # ---------------- PREDICTION ---------------- #

        prediction = model.predict(img_resized, verbose=0)

        predicted_digit = np.argmax(prediction)
        confidence = float(np.max(prediction))


        # ---------------- DISPLAY RESULT ---------------- #

        st.markdown(f"""
        <div class="prediction-box">
            <div class="big-digit">{predicted_digit}</div>
            <div class="confidence-text">Confidence: {confidence*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence)


        # ---------------- PROBABILITY CHART ---------------- #

        st.markdown("### Probability Distribution")

        probs = prediction[0]

        fig = plt.figure()

        plt.bar(range(10), probs)

        plt.xticks(range(10))

        plt.xlabel("Digit")
        plt.ylabel("Probability")

        plt.title("Model Confidence")

        st.pyplot(fig)


        # ---------------- TOP 3 ---------------- #

        st.markdown("### Top 3 Predictions")

        top3 = probs.argsort()[-3:][::-1]

        for i in top3:

            st.write(f"Digit {i} — {probs[i]*100:.2f}%")

            st.progress(float(probs[i]))


        # ---------------- PREVIEW ---------------- #

        st.markdown("### Processed 28x28 Input")

        st.image(img_resized.reshape(28,28), width=150)


        # ---------------- DOWNLOAD ---------------- #

        img_bytes = cv2.imencode(
            ".png",
            (img_resized.reshape(28,28)*255).astype(np.uint8)
        )[1].tobytes()

        st.download_button(
            "Download Processed Image",
            img_bytes,
            file_name="processed_digit.png",
            mime="image/png"
        )