import streamlit as st
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Lung Disease Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# PATHS
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_densenet121.pth"

# -------------------------------------------------
# CLASS NAMES
# -------------------------------------------------
CLASS_NAMES = [
    "Covid-19",
    "Emphysema",
    "Normal",
    "Pneumonia-Bacterial",
    "Pneumonia-Viral",
    "Tuberculosis"
]

# -------------------------------------------------
# CLASS METADATA (display-only: icons + colors for the premium UI)
# -------------------------------------------------
CLASS_META = {
    "Covid-19":             {"icon": "🦠", "color": "#FF5C7A"},
    "Emphysema":             {"icon": "💨", "color": "#FFB84F"},
    "Normal":                {"icon": "✅", "color": "#00FF9D"},
    "Pneumonia-Bacterial":   {"icon": "🧫", "color": "#FF8A5C"},
    "Pneumonia-Viral":       {"icon": "🧬", "color": "#FF9DE2"},
    "Tuberculosis":          {"icon": "🫁", "color": "#7C5CFF"},
}

# -------------------------------------------------
# IMAGE TRANSFORM
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# CREATE MODEL
# -------------------------------------------------
def create_model():

    model = models.densenet121(weights=None)

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, 6)
    )

    return model

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error(f"Model not found:\n{MODEL_PATH}")
        st.stop()

    model = create_model()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model

model = load_model()

# -------------------------------------------------
# CUSTOM CSS — PREMIUM FUTURISTIC MEDICAL AI THEME
# -------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #050816;
    --card: rgba(255,255,255,0.06);
    --card-border: rgba(255,255,255,0.10);
    --accent: #4FD1FF;
    --accent2: #7C5CFF;
    --success: #00FF9D;
    --danger: #FF5C7A;
    --text-secondary: rgba(255,255,255,0.60);
}

/* ---------- GLOBAL ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(79,209,255,0.14), transparent 45%),
        radial-gradient(circle at 85% 10%, rgba(124,92,255,0.16), transparent 45%),
        radial-gradient(circle at 50% 100%, rgba(0,255,157,0.06), transparent 50%),
        var(--bg);
    color: #FFFFFF;
}

/* Hide default Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
div[data-testid="stDecoration"] {display: none;}
div[data-testid="stStatusWidget"] {display: none;}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ---------- ANIMATIONS ---------- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%      { transform: translateY(-10px); }
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 18px rgba(79,209,255,0.25); }
    50%      { box-shadow: 0 0 34px rgba(79,209,255,0.55); }
}
@keyframes shimmer {
    0%   { background-position: -300px 0; }
    100% { background-position: 300px 0; }
}
@keyframes barGrow {
    from { width: 0%; }
}

.fade-in-up { animation: fadeInUp 0.7s ease both; }

/* ---------- HERO ---------- */
.hero-wrap {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    padding: 2.6rem 2.8rem;
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(79,209,255,0.10), rgba(124,92,255,0.10));
    border: 1px solid var(--card-border);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    margin-bottom: 2rem;
    animation: fadeInUp 0.8s ease both;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: "";
    position: absolute;
    top: -60%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(79,209,255,0.25), transparent 70%);
    filter: blur(10px);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 16px;
    border-radius: 999px;
    background: rgba(79,209,255,0.12);
    border: 1px solid rgba(79,209,255,0.4);
    color: var(--accent);
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    animation: glowPulse 2.4s ease-in-out infinite;
}
.hero-badge .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 8px var(--success);
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.15;
    margin: 0 0 0.6rem 0;
    background: linear-gradient(90deg, #FFFFFF 30%, var(--accent) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-secondary);
    font-weight: 500;
    max-width: 520px;
}
.hero-icon {
    font-size: 6.5rem;
    animation: float 3.4s ease-in-out infinite;
    filter: drop-shadow(0 0 26px rgba(79,209,255,0.45));
}

/* ---------- GLASS CARD ---------- */
.glass-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 22px;
    padding: 1.8rem 2rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    transition: all 0.35s ease;
    animation: fadeInUp 0.7s ease both;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 44px rgba(79,209,255,0.18);
    border-color: rgba(79,209,255,0.35);
}

.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
}
.section-label .bar {
    width: 4px; height: 20px;
    border-radius: 4px;
    background: linear-gradient(180deg, var(--accent), var(--accent2));
    display: inline-block;
}

/* ---------- UPLOADER ---------- */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(79,209,255,0.06), rgba(124,92,255,0.06));
    border: 1.5px dashed rgba(79,209,255,0.45);
    border-radius: 20px;
    padding: 1.4rem;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent);
    background: linear-gradient(135deg, rgba(79,209,255,0.10), rgba(124,92,255,0.10));
    box-shadow: 0 0 26px rgba(79,209,255,0.18);
}
[data-testid="stFileUploader"] section {
    background: transparent;
    border: none;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg {
    fill: var(--accent);
}
[data-testid="stFileUploader"] button {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    color: #050816 !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
}

/* ---------- IMAGES ---------- */
[data-testid="stImage"] img {
    border-radius: 18px;
    border: 1px solid var(--card-border);
    box-shadow: 0 10px 34px rgba(0,0,0,0.5);
}

/* ---------- PREDICTION RESULT CARD ---------- */
.result-card {
    background: linear-gradient(135deg, rgba(0,255,157,0.10), rgba(79,209,255,0.06));
    border: 1px solid rgba(0,255,157,0.35);
    border-radius: 24px;
    padding: 2rem 2.2rem;
    text-align: center;
    animation: fadeInUp 0.6s ease both, glowPulse 3s ease-in-out infinite;
    margin-bottom: 1.5rem;
}
.result-icon { font-size: 3.6rem; margin-bottom: 0.4rem; }
.result-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0.2rem 0;
}
.result-conf {
    display: inline-block;
    margin-top: 0.6rem;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    color: var(--accent);
}

/* ---------- PROBABILITY BARS ---------- */
.prob-row {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 14px;
    animation: fadeInUp 0.6s ease both;
}
.prob-icon { font-size: 1.4rem; width: 32px; text-align: center; }
.prob-label {
    width: 175px;
    font-weight: 600;
    font-size: 0.92rem;
    color: #FFFFFF;
}
.prob-track {
    flex: 1;
    height: 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    overflow: hidden;
    position: relative;
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
    animation: barGrow 1.1s cubic-bezier(0.16, 1, 0.3, 1) both;
    background-size: 300px 100%;
    background-repeat: no-repeat;
    position: relative;
}
.prob-pct {
    width: 60px;
    text-align: right;
    font-weight: 700;
    font-size: 0.92rem;
    color: var(--text-secondary);
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070B1E 0%, #050816 100%);
    border-right: 1px solid var(--card-border);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}
.sidebar-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 1.2rem;
    color: #FFFFFF;
}
.sidebar-item {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.7rem;
    transition: all 0.25s ease;
}
.sidebar-item:hover {
    border-color: rgba(79,209,255,0.4);
    transform: translateX(3px);
}
.sidebar-item .k {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 2px;
}
.sidebar-item .v {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--accent);
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    background: rgba(0,255,157,0.12);
    border: 1px solid rgba(0,255,157,0.4);
    color: var(--success);
    font-weight: 700;
    font-size: 0.82rem;
}
.status-pill .dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 8px var(--success);
}

/* ---------- FOOTER ---------- */
.premium-footer {
    margin-top: 3rem;
    padding: 1.8rem;
    text-align: center;
    border-top: 1px solid var(--card-border);
    color: var(--text-secondary);
    font-size: 0.9rem;
}
.premium-footer .stack {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 0.7rem;
}
.stack-chip {
    padding: 5px 14px;
    border-radius: 999px;
    background: var(--card);
    border: 1px solid var(--card-border);
    font-size: 0.8rem;
    font-weight: 600;
    color: #FFFFFF;
}

/* ---------- MISC ---------- */
.stAlert { border-radius: 14px; }
hr { border-color: var(--card-border); }

/* Responsive */
@media (max-width: 900px) {
    .hero-wrap { flex-direction: column; text-align: center; padding: 1.8rem; }
    .hero-title { font-size: 2rem; }
    .hero-icon { font-size: 4rem; }
    .prob-label { width: 120px; font-size: 0.82rem; }
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR — MODEL INFO
# -------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ About This Model</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-item">
        <div class="k">Architecture</div>
        <div class="v">DenseNet121</div>
    </div>
    <div class="sidebar-item">
        <div class="k">Input Image Size</div>
        <div class="v">224 × 224</div>
    </div>
    <div class="sidebar-item">
        <div class="k">Number of Classes</div>
        <div class="v">6</div>
    </div>
    <div class="sidebar-item">
        <div class="k">Compute Device</div>
        <div class="v">{str(device).upper()}</div>
    </div>
    <div class="sidebar-item">
        <div class="k">Model Status</div>
        <div class="status-pill"><span class="dot"></span> Loaded &amp; Ready</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🫁 Detectable Classes</div>', unsafe_allow_html=True)
    for cname in CLASS_NAMES:
        meta = CLASS_META[cname]
        st.markdown(
            f'<div class="sidebar-item">{meta["icon"]} &nbsp; {cname}</div>',
            unsafe_allow_html=True
        )

# -------------------------------------------------
# HERO SECTION
# -------------------------------------------------
st.markdown("""
<div class="hero-wrap">
    <div>
        <div class="hero-badge"><span class="dot"></span> Medical AI</div>
        <div class="hero-title">🫁 AI Lung Disease Detection System</div>
        <div class="hero-sub">Powered by a DenseNet121 Deep Learning Model — upload a chest X-ray for an instant, AI-assisted diagnostic prediction.</div>
    </div>
    <div class="hero-icon">🩻</div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# UPLOAD SECTION
# -------------------------------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label"><span class="bar"></span> Upload Chest X-ray</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<br>", unsafe_allow_html=True)
    col_img, col_gap = st.columns([1, 0.001])
    with col_img:
        st.markdown('<div class="glass-card fade-in-up">', unsafe_allow_html=True)
        st.markdown('<div class="section-label"><span class="bar"></span> Uploaded Scan</div>', unsafe_allow_html=True)
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    img = transform(image)

    img = img.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(img)

        probs = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs, dim=1)

    predicted_class = CLASS_NAMES[pred.item()]
    predicted_conf = confidence.item() * 100
    meta = CLASS_META[predicted_class]

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- PREDICTION CARD ----------------
    st.markdown(f"""
    <div class="result-card">
        <div class="result-icon">{meta['icon']}</div>
        <div style="color: var(--text-secondary); font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; font-size: 0.8rem;">Prediction Result</div>
        <div class="result-name" style="color:{meta['color']};">{predicted_class}</div>
        <div class="result-conf">Confidence: {predicted_conf:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Keep native alert components too (functionality preserved 1:1)
    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{predicted_conf:.2f}%**")

    # ---------------- PROBABILITY VISUALIZATION ----------------
    st.markdown('<div class="glass-card fade-in-up">', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span class="bar"></span> Prediction Probabilities</div>', unsafe_allow_html=True)

    for i, disease in enumerate(CLASS_NAMES):

        probability = probs[0][i].item()
        pct = probability * 100
        cmeta = CLASS_META[disease]

        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-icon">{cmeta['icon']}</div>
            <div class="prob-label">{disease}</div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{pct:.2f}%; background: linear-gradient(90deg, {cmeta['color']}, {cmeta['color']}AA);"></div>
            </div>
            <div class="prob-pct">{pct:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Preserve original st.progress for exact functional parity
        st.progress(probability)

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="glass-card" style="text-align:center; color: var(--text-secondary); padding: 2.6rem;">
        🩻 Upload a chest X-ray image above to begin AI-assisted analysis.
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("""
<div class="premium-footer">
    Made with ❤️ using
    <div class="stack">
        <span class="stack-chip">🔥 PyTorch</span>
        <span class="stack-chip">🧠 DenseNet121</span>
        <span class="stack-chip">⚡ Streamlit</span>
        <span class="stack-chip">🫁 Medical AI</span>
    </div>
</div>
""", unsafe_allow_html=True)