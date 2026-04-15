import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Medical Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================
# PREMIUM MEDICAL SAAS STYLING
# =========================
st.markdown("""
<style>
/* Import Premium Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Reset & Typography */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Clinical Dark Background - Deep Navy to Slate Gradient */
.stApp {
    background: linear-gradient(135deg, #0a1128 0%, #162238 100%);
    color: #f8fafc;
}

/* Hide Streamlit Default Header */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Smooth Fade-in Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.stApp > div {
    animation: fadeIn 0.8s ease-out forwards;
}

/* Premium Gradient Text for Headers */
h1, h2, h3 {
    background: linear-gradient(90deg, #48cae4, #90e0ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 600 !important;
    letter-spacing: -0.5px;
}

/* Subtext styling */
p {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 300;
}

/* Glassmorphic File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px dashed rgba(72, 202, 228, 0.4);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(72, 202, 228, 0.8);
    box-shadow: 0 8px 32px rgba(72, 202, 228, 0.15);
}

/* Style the Uploaded Images (Rounded corners & soft shadow) */
[data-testid="stImage"] img {
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
}

/* Custom Glassmorphic Cards for Results */
.glass-card {
    background: rgba(16, 24, 43, 0.6);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px;
    margin-top: 15px;
    margin-bottom: 25px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
}

/* Clinical Alert Colors */
.diag-danger {
    border-left: 4px solid #ef476f;
    background: linear-gradient(90deg, rgba(239, 71, 111, 0.08) 0%, transparent 100%);
}
.diag-success {
    border-left: 4px solid #06d6a0;
    background: linear-gradient(90deg, rgba(6, 214, 160, 0.08) 0%, transparent 100%);
}

.diag-title {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 8px;
}
.diag-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
}

/* Progress Bar Styling */
.stProgress > div > div > div > div {
    background-image: linear-gradient(90deg, #0077b6, #48cae4);
    border-radius: 10px;
}

/* Spinner styling */
.stSpinner > div > div {
    border-top-color: #48cae4 !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# UI HEADER (Rendered immediately to prevent blank screen)
# =========================
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='font-size: 2.8rem; margin-bottom: 5px;'>🩺 PulmoVision AI</h1>
        <p>High-Fidelity Chest Radiograph Analysis System</p>
    </div>
""", unsafe_allow_html=True)

# =========================
# LAZY LOAD MODEL & TENSORFLOW
# =========================
# This prevents TensorFlow from freezing the UI before the title loads.
@st.cache_resource(show_spinner=False)
def initialize_ai_engine():
    import tensorflow as tf
    from src.gradcam import get_gradcam
    
    MODEL_PATH = "models/pneumonia_model.keras"
    model = tf.keras.models.load_model(MODEL_PATH)
    
    return model, get_gradcam

# Display the professional loading spinner while TF initializes in the background
with st.spinner("Initializing neural engine and loading clinical models. Please wait..."):
    model, get_gradcam = initialize_ai_engine()

# =========================
# FILE UPLOAD (Appears after AI engine is ready)
# =========================
uploaded_file = st.file_uploader("Drop patient X-ray here (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    
    st.markdown("<h3 style='font-size: 1.2rem; margin-top: 20px;'>Input Radiograph</h3>", unsafe_allow_html=True)
    st.image(image, use_container_width=True)

    # =========================
    # LOADING UX
    # =========================
    with st.spinner("Processing deep learning attention maps..."):
        
        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)  # smooth animation
            progress.progress(i + 1)

        # =========================
        # PREPROCESS
        # =========================
        img = np.array(image)
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        img_array = np.expand_dims(img_normalized, axis=0)

        # =========================
        # PREDICTION
        # =========================
        prediction = model.predict(img_array)[0][0]

        st.markdown("<h3 style='font-size: 1.4rem; margin-top: 30px;'>Diagnostic Assessment</h3>", unsafe_allow_html=True)

        if prediction > 0.5:
            # High-end Danger Card
            st.markdown(f"""
            <div class="glass-card diag-danger">
                <div class="diag-title" style="color: #ef476f;">⚠️ Abnormal Findings Detected</div>
                <div class="diag-value">Pneumonia</div>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 5px;">Confidence Score: <span style="color:#ffffff; font-weight: 500;">{prediction:.2%}</span></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # High-end Success Card
            st.markdown(f"""
            <div class="glass-card diag-success">
                <div class="diag-title" style="color: #06d6a0;">✅ No Abnormalities Detected</div>
                <div class="diag-value">Normal</div>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 5px;">Confidence Score: <span style="color:#ffffff; font-weight: 500;">{(1 - prediction):.2%}</span></div>
            </div>
            """, unsafe_allow_html=True)

        # =========================
        # GRAD-CAM
        # =========================
        st.markdown("<h3 style='font-size: 1.4rem; margin-top: 10px;'>Spatial Attention Map (Grad-CAM)</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.9rem; margin-bottom: 15px;'>Highlights regions of highest diagnostic significance driving the AI's decision.</p>", unsafe_allow_html=True)

        _ = model.predict(img_array)

        heatmap = get_gradcam(model, img_array, "Conv_1")

        original = cv2.resize(img, (224, 224))

        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        superimposed_img = heatmap_colored * 0.4 + original
        superimposed_img = cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB)

        st.image(superimposed_img, use_container_width=True)

        progress.empty()

    # Premium subtle success toast at the bottom instead of a blocky green alert
    st.markdown("""
        <div style="text-align: center; margin-top: 30px; padding: 10px; border-radius: 8px; background: rgba(255,255,255,0.05); color: #94a3b8; font-size: 0.85rem;">
            ✓ Analysis complete. Report generated securely.
        </div>
    """, unsafe_allow_html=True)