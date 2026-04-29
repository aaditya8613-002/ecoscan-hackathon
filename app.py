import streamlit as st
import numpy as np
from PIL import Image
import tf_keras

st.set_page_config(
    page_title="EcoScan",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d2137 50%, #0a1a10 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 1rem 2rem !important; max-width: 480px !important; margin: auto; }

.hero { text-align: center; padding: 2rem 0 1.5rem; }
.hero .logo { font-size: 3.5rem; margin-bottom: 0.25rem; }
.hero h1 {
    font-size: 2rem; font-weight: 900;
    background: linear-gradient(90deg, #22c55e, #4ade80, #86efac);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.1;
}
.hero p { color: #94a3b8; font-size: 0.9rem; margin-top: 0.4rem; }

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.25rem;
    margin: 1rem 0;
    backdrop-filter: blur(12px);
}
.card-title {
    color: #94a3b8; font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 0.75rem;
}

.result-card {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(74,222,128,0.08));
    border: 1px solid rgba(34,197,94,0.35);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.result-emoji { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label { font-size: 1.8rem; font-weight: 900; color: #4ade80; }
.result-conf { color: #94a3b8; font-size: 0.85rem; margin-top: 0.2rem; }

.co2-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 50px; padding: 0.5rem 1.2rem;
    color: #86efac; font-weight: 600; font-size: 0.95rem;
    margin-top: 0.75rem;
}

.tip-card {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 16px; padding: 1rem 1.25rem;
    color: #93c5fd; font-size: 0.88rem;
    margin: 0.75rem 0;
}
.tip-card strong { color: #60a5fa; }

.stat-row { display: flex; gap: 0.75rem; margin: 0.75rem 0; }
.stat-box {
    flex: 1; background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 0.85rem 0.5rem;
    text-align: center;
}
.stat-box .val { font-size: 1.3rem; font-weight: 800; color: #4ade80; }
.stat-box .lbl { font-size: 0.7rem; color: #64748b; margin-top: 0.15rem; }

.conf-bar-wrap { margin: 0.75rem 0 0; }
.conf-bar-bg { background: rgba(255,255,255,0.06); border-radius: 50px; height: 8px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 50px; background: linear-gradient(90deg, #22c55e, #4ade80); }

[data-testid="stFileUploader"] {
    border: 2px dashed rgba(34,197,94,0.3) !important;
    border-radius: 16px !important;
    background: rgba(34,197,94,0.03) !important;
    padding: 0.5rem !important;
}
[data-testid="stCameraInput"] > div { border-radius: 16px !important; overflow: hidden; }
label { color: #94a3b8 !important; }

.footer { text-align:center; color:#334155; font-size:0.75rem; margin-top:2rem; padding-bottom:1rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = tf_keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt") as f:
        class_names = [line.strip().split(" ", 1)[1] for line in f if line.strip()]
    return model, class_names

model, class_names = load_assets()

# what we know about each waste type
WASTE_INFO = {
    "plastic":  {"emoji": "🧴", "co2": 30, "bin": "Blue Bin",     "tip": "Rinse it out first. Take the cap off."},
    "paper":    {"emoji": "📄", "co2": 17, "bin": "Blue Bin",     "tip": "Keep it dry. Rip off any tape or staples."},
    "metal":    {"emoji": "🥫", "co2": 60, "bin": "Blue Bin",     "tip": "Give it a rinse. Crush the can to save space."},
    "organic":  {"emoji": "🍂", "co2": 10, "bin": "Brown Bin",    "tip": "Perfect for home composting if you have a bin."},
    "e-waste":  {"emoji": "📱", "co2": 80, "bin": "E-Waste Drop", "tip": "Keep out of regular bins — it's toxic if landfilled."},
    "e_waste":  {"emoji": "📱", "co2": 80, "bin": "E-Waste Drop", "tip": "Keep out of regular bins — it's toxic if landfilled."},
}

def lookup(label):
    key = label.lower().replace("-", "_").replace(" ", "_")
    for k, v in WASTE_INFO.items():
        if k in key or key in k:
            return v
    return {"emoji": "🗑️", "co2": 20, "bin": "Recycling", "tip": "Check what your local council accepts."}


st.markdown("""
<div class="hero">
  <div class="logo">♻️</div>
  <h1>EcoScan</h1>
  <p>snap a photo of your rubbish — we'll tell you where it goes</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">Scan your waste</div>', unsafe_allow_html=True)

tab_cam, tab_up = st.tabs(["Camera", "Upload photo"])
img_file = None

with tab_cam:
    img_file = st.camera_input("", label_visibility="collapsed")

with tab_up:
    up = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if up:
        img_file = up

st.markdown('</div>', unsafe_allow_html=True)

if img_file:
    img = Image.open(img_file).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32).reshape(1, 224, 224, 3) / 127.5 - 1

    with st.spinner("working it out..."):
        scores = model.predict(arr, verbose=0)

    top = int(np.argmax(scores))
    label = class_names[top]
    confidence = float(scores[0][top]) * 100
    info = lookup(label)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-emoji">{info['emoji']}</div>
        <div class="result-label">{label.title()}</div>
        <div class="result-conf">{confidence:.1f}% sure</div>
        <div class="conf-bar-wrap">
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{confidence:.0f}%"></div>
            </div>
        </div>
        <div class="co2-badge">🌱 saves ~{info['co2']}g CO₂ if sorted right</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box">
            <div class="val">{info['bin']}</div>
            <div class="lbl">goes in</div>
        </div>
        <div class="stat-box">
            <div class="val">{info['co2']}g</div>
            <div class="lbl">CO₂ saved</div>
        </div>
        <div class="stat-box">
            <div class="val">{confidence:.0f}%</div>
            <div class="lbl">confidence</div>
        </div>
    </div>
    <div class="tip-card">
        <strong>tip:</strong> {info['tip']}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("see all scores"):
        for i, name in enumerate(class_names):
            s = float(scores[0][i]) * 100
            st.progress(int(s), text=f"{name.title()}: {s:.1f}%")

else:
    st.markdown("""
    <div style="text-align:center; color:#475569; padding:1.5rem 0; font-size:0.9rem;">
        use the camera tab or drop in a photo
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">EcoScan — made for the hackathon 🌍</div>', unsafe_allow_html=True)