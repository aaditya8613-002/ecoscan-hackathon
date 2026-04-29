import streamlit as st
import numpy as np
from PIL import Image
import tf_keras
from datetime import datetime
import hashlib
import json
import uuid
import os
import csv

def log_correction(predicted, actual, conf):
    file_exists = os.path.isfile("corrections_log.csv")
    with open("corrections_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Predicted_Label", "Actual_Label", "Model_Confidence"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), predicted, actual, f"{conf:.2f}%"])

st.set_page_config(page_title="EcoScan", page_icon="♻️", layout="centered", initial_sidebar_state="collapsed")

# ── persistent storage (no login — UUID in URL) ───────────────────────────────
DATA_DIR = "userdata"
os.makedirs(DATA_DIR, exist_ok=True)

def get_uid():
    params = st.query_params
    if "u" not in params:
        new_id = uuid.uuid4().hex[:10]
        st.query_params["u"] = new_id
        return new_id
    return params["u"]

def data_file(uid):
    return os.path.join(DATA_DIR, f"{uid}.json")

def load_user(uid):
    f = data_file(uid)
    if os.path.exists(f):
        try:
            d = json.load(open(f))
            d["badges"] = set(d.get("badges", []))
            return d
        except Exception:
            pass
    return dict(xp=0, streak=0, best_streak=0, history=[], total_co2=0,
                last_hash=None, badges=set(), prev_level=0)

def save_user(uid, data):
    d = dict(data)
    d["badges"] = list(d.get("badges", set()))
    json.dump(d, open(data_file(uid), "w"))

# ── init session ──────────────────────────────────────────────────────────────
if "uid" not in st.session_state:
    uid = get_uid()
    st.session_state.uid = uid
    saved = load_user(uid)
    for k, v in saved.items():
        st.session_state[k] = v
    # ensure all keys exist
    for k, v in dict(xp=0, streak=0, best_streak=0, history=[], total_co2=0,
                     last_hash=None, badges=set(), prev_level=0).items():
        if k not in st.session_state:
            st.session_state[k] = v

def persist():
    """call after any state change to save to disk"""
    save_user(st.session_state.uid, {
        k: st.session_state[k]
        for k in ["xp","streak","best_streak","history","total_co2",
                  "last_hash","badges","prev_level"]
    })


# ── levels & badges ──────────────────────────────────────────────────────────
LEVELS = [
    (0,   "Littering Larry",  "😬"),
    (50,  "Eco Rookie",       "🌱"),
    (150, "Recycler",         "♻️"),
    (300, "Green Warrior",    "⚡"),
    (500, "Eco Hero",         "🌍"),
    (800, "Planet Saver",     "🏆"),
]

def get_level(xp):
    lvl = 0
    for i, (threshold, _, _) in enumerate(LEVELS):
        if xp >= threshold:
            lvl = i
    return lvl

def level_info(xp):
    idx = get_level(xp)
    _, name, emoji = LEVELS[idx]
    next_thresh = LEVELS[idx + 1][0] if idx + 1 < len(LEVELS) else LEVELS[idx][0]
    curr_thresh = LEVELS[idx][0]
    progress = min(100, int((xp - curr_thresh) / max(1, next_thresh - curr_thresh) * 100))
    return name, emoji, progress, next_thresh, idx

BADGE_RULES = [
    ("first_scan",   "🎯 First Scan",    "scanned your first item"),
    ("streak3",      "🔥 On Fire",        "3 scan streak"),
    ("streak5",      "🌋 Unstoppable",    "5 scan streak"),
    ("co2_100",      "💨 100g CO₂ Saved", "saved 100g CO₂"),
    ("co2_500",      "🌳 500g CO₂ Saved", "saved 500g CO₂"),
    ("scans5",       "📸 Snap Happy",     "scanned 5 items"),
    ("ewaste_found", "⚠️ E-Waste Spotter","found e-waste"),
    ("all_types",    "🌈 Full House",     "scanned all 5 waste types"),
]

# ── waste data ────────────────────────────────────────────────────────────────
WASTE_INFO = {
    "plastic": {"emoji":"🧴","co2":30,"bin":"Blue Bin","tip":"Rinse it out. Caps off.","xp":10,"color":"#3b82f6"},
    "paper":   {"emoji":"📄","co2":17,"bin":"Blue Bin","tip":"Keep dry. Remove staples.","xp":10,"color":"#f59e0b"},
    "metal":   {"emoji":"🥫","co2":60,"bin":"Blue Bin","tip":"Rinse cans. Crush to save space.","xp":15,"color":"#8b5cf6"},
    "organic": {"emoji":"🍂","co2":10,"bin":"Brown Bin","tip":"Great for composting!","xp":10,"color":"#22c55e"},
    "e-waste": {"emoji":"📱","co2":80,"bin":"E-Waste Drop","tip":"Never in regular bins — toxic.","xp":20,"color":"#ef4444"},
    "e_waste": {"emoji":"📱","co2":80,"bin":"E-Waste Drop","tip":"Never in regular bins — toxic.","xp":20,"color":"#ef4444"},
    "others":  {"emoji":"🤷","co2":0, "bin":"Check locally","tip":"Could be glass, cardboard, or mixed waste — check your local council's guide.","xp":5,"color":"#f97316"},
}

def lookup(label):
    key = label.lower().replace("-","_").replace(" ","_")
    for k, v in WASTE_INFO.items():
        if k in key or key in k:
            return v
    return {"emoji":"🗑️","co2":20,"bin":"Recycling","tip":"Check local council guidelines.","xp":10,"color":"#94a3b8"}

# ── model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    m = tf_keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt") as f:
        names = [l.strip().split(" ", 1)[1] for l in f if l.strip()]
    return m, names

model, class_names = load_assets()

# ── css ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
* { font-family:'Inter',sans-serif; box-sizing:border-box; }
.stApp { background:linear-gradient(135deg,#050d1a 0%,#0a1f2e 60%,#061208 100%); min-height:100vh; }
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0.5rem 1rem 3rem!important;max-width:500px!important;margin:auto}
div[data-testid="stRadio"]>div{display:flex;gap:0.5rem;justify-content:center;flex-wrap:wrap}
div[data-testid="stRadio"] label{
    background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
    border-radius:50px;padding:0.4rem 1.2rem;color:#94a3b8;font-size:0.85rem;cursor:pointer;
    transition:all 0.2s}
div[data-testid="stRadio"] label:has(input:checked){background:rgba(34,197,94,0.2);border-color:#22c55e;color:#4ade80}
.hero{text-align:center;padding:1.2rem 0 0.5rem}
.hero h1{font-size:1.9rem;font-weight:900;background:linear-gradient(90deg,#22c55e,#4ade80,#86efac);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0}
.hero p{color:#64748b;font-size:0.8rem;margin:0.2rem 0 0}
.xp-bar-wrap{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);
    border-radius:16px;padding:0.8rem 1rem;margin:0.6rem 0}
.xp-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem}
.level-badge{font-size:0.85rem;font-weight:700;color:#4ade80}
.xp-pts{font-size:0.75rem;color:#64748b}
.bar-bg{background:rgba(255,255,255,0.06);border-radius:50px;height:7px;overflow:hidden}
.bar-fill{height:100%;border-radius:50px;background:linear-gradient(90deg,#22c55e,#4ade80)}
.stats-row{display:flex;gap:0.5rem;margin:0.6rem 0}
.stat{flex:1;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
    border-radius:14px;padding:0.7rem 0.4rem;text-align:center}
.stat .v{font-size:1.3rem;font-weight:800;color:#4ade80}
.stat .l{font-size:0.65rem;color:#475569;margin-top:0.1rem}
.result-card{background:linear-gradient(135deg,rgba(34,197,94,0.12),rgba(74,222,128,0.06));
    border:1px solid rgba(34,197,94,0.3);border-radius:20px;padding:1.4rem;text-align:center;margin:0.8rem 0}
.res-emoji{font-size:3rem}
.res-label{font-size:1.7rem;font-weight:900;color:#4ade80;margin:0.2rem 0}
.res-conf{color:#64748b;font-size:0.8rem}
.xp-pill{display:inline-block;background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.3);
    border-radius:50px;padding:0.3rem 1rem;color:#86efac;font-weight:700;font-size:0.85rem;margin-top:0.5rem}
.info-row{display:flex;gap:0.5rem;margin:0.6rem 0}
.info-box{flex:1;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
    border-radius:12px;padding:0.7rem 0.5rem;text-align:center}
.info-box .iv{font-size:1rem;font-weight:700;color:#e2e8f0}
.info-box .il{font-size:0.65rem;color:#475569;margin-top:0.1rem}
.tip-box{background:rgba(59,130,246,0.07);border:1px solid rgba(59,130,246,0.18);
    border-radius:14px;padding:0.8rem 1rem;color:#93c5fd;font-size:0.83rem;margin:0.5rem 0}
.impact-card{background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.18);
    border-radius:18px;padding:1.2rem;margin:0.7rem 0}
.impact-card h3{color:#4ade80;font-size:1rem;margin:0 0 0.8rem}
.impact-row{display:flex;gap:0.5rem;flex-wrap:wrap}
.impact-box{flex:1;min-width:100px;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
    border-radius:12px;padding:0.7rem;text-align:center}
.impact-box .icon{font-size:1.4rem}
.impact-box .num{font-size:1.1rem;font-weight:800;color:#4ade80;margin:0.2rem 0}
.impact-box .desc{font-size:0.65rem;color:#475569}
.badge-grid{display:flex;flex-wrap:wrap;gap:0.5rem;margin:0.5rem 0}
.badge{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
    border-radius:12px;padding:0.5rem 0.8rem;font-size:0.78rem;color:#94a3b8;text-align:center}
.badge.earned{background:rgba(34,197,94,0.12);border-color:rgba(34,197,94,0.3);color:#86efac}
.hist-item{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
    border-radius:14px;padding:0.7rem 0.9rem;margin:0.4rem 0;display:flex;align-items:center;gap:0.8rem}
.hist-label{color:#e2e8f0;font-weight:600;font-size:0.88rem}
.hist-meta{color:#475569;font-size:0.72rem;margin-top:0.1rem}
.hist-xp{margin-left:auto;color:#4ade80;font-weight:700;font-size:0.85rem}
.empty{text-align:center;color:#334155;padding:2rem 0;font-size:0.9rem}
[data-testid="stFileUploader"]{border:2px dashed rgba(34,197,94,0.25)!important;
    border-radius:16px!important;background:rgba(34,197,94,0.03)!important}
[data-testid="stCameraInput"]>div{border-radius:16px!important;overflow:hidden}
label{color:#64748b!important}
.stTabs [data-baseweb="tab-list"]{background:transparent;gap:0.3rem}
.stTabs [data-baseweb="tab"]{background:rgba(255,255,255,0.04);border-radius:10px;color:#64748b;
    border:1px solid rgba(255,255,255,0.07)}
.stTabs [aria-selected="true"]{background:rgba(34,197,94,0.15)!important;color:#4ade80!important;
    border-color:rgba(34,197,94,0.3)!important}
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
lv_name, lv_emoji, lv_progress, lv_next, lv_idx = level_info(st.session_state.xp)
st.markdown(f"""
<div class="hero">
  <h1>♻️ EcoScan</h1>
  <p>scan rubbish · earn XP · save the planet</p>
</div>
<div class="xp-bar-wrap">
  <div class="xp-top">
    <span class="level-badge">{lv_emoji} {lv_name}</span>
    <span class="xp-pts">{st.session_state.xp} XP · next level at {lv_next}</span>
  </div>
  <div class="bar-bg"><div class="bar-fill" style="width:{lv_progress}%"></div></div>
</div>
""", unsafe_allow_html=True)

s = st.session_state
st.markdown(f"""
<div class="stats-row">
  <div class="stat"><div class="v">{len(s.history)}</div><div class="l">scans</div></div>
  <div class="stat"><div class="v">🔥{s.streak}</div><div class="l">streak</div></div>
  <div class="stat"><div class="v">{s.total_co2}g</div><div class="l">CO₂ saved</div></div>
  <div class="stat"><div class="v">{len(s.badges)}</div><div class="l">badges</div></div>
</div>
""", unsafe_allow_html=True)

# ── nav ───────────────────────────────────────────────────────────────────────
page = st.radio("", ["📷 Scan", "🌍 Impact", "🏅 Badges", "📋 History"], horizontal=True, label_visibility="collapsed")

# ═══════════════════════════════════════════════════════════
if page == "📷 Scan":

    t1, t2 = st.tabs(["Camera", "Upload"])
    img_file = None
    with t1:
        cam = st.camera_input("Take a photo", label_visibility="collapsed")
        if cam:
            img_file = cam
    with t2:
        up = st.file_uploader("Upload image", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if up:
            img_file = up

    if img_file:
        img_bytes = img_file.getvalue()
        scan_hash = hashlib.md5(img_bytes).hexdigest()

        img = Image.open(img_file).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32).reshape(1,224,224,3) / 127.5 - 1

        with st.spinner("scanning..."):
            scores = model.predict(arr, verbose=0)

        top = int(np.argmax(scores))
        label = class_names[top]
        conf = float(scores[0][top]) * 100

        # low confidence → fall back to 'others'
        if conf < 30:
            label = "others"
            info = WASTE_INFO["others"]
        else:
            info = lookup(label)

        # ── show result ──
        card_bg = "linear-gradient(135deg,rgba(249,115,22,0.12),rgba(251,146,60,0.06))" if label == "others" else "linear-gradient(135deg,rgba(34,197,94,0.12),rgba(74,222,128,0.06))"
        card_border = "rgba(249,115,22,0.35)" if label == "others" else "rgba(34,197,94,0.3)"
        label_color = "#fb923c" if label == "others" else "#4ade80"
        conf_display = f"{conf:.1f}%" if label != "others" else "< 30% — not sure"

        st.markdown(f"""
        <div class="result-card" style="background:{card_bg};border-color:{card_border}">
          <div class="res-emoji">{info['emoji']}</div>
          <div class="res-label" style="color:{label_color}">{label.title()}</div>
          <div class="res-conf">{conf_display} confidence</div>
          <div class="xp-pill">+{info['xp']} XP</div>
        </div>
        <div class="info-row">
          <div class="info-box"><div class="iv">{info['bin']}</div><div class="il">goes in</div></div>
          <div class="info-box"><div class="iv">{info['co2']}g</div><div class="il">CO₂ saved</div></div>
          <div class="info-box"><div class="iv">{conf:.0f}%</div><div class="il">confidence</div></div>
        </div>
        <div class="tip-box">💡 {info['tip']}</div>
        """, unsafe_allow_html=True)

        # ── report wrong prediction (inspired by deep-waste-app) ──
        with st.expander("Wrong result? Tell us"):
            correct = st.selectbox("what is it actually?",
                ["plastic","paper","metal","organic","e-waste","others"],
                key=f"correct_{scan_hash}")
            if st.button("submit correction", key=f"fix_{scan_hash}"):
                correct_info = lookup(correct)
                
                # save to central CSV for ML retraining
                log_correction(label, correct, conf)
                
                st.session_state.xp += 5  # bonus for helping improve
                st.session_state.history.insert(0, {
                    "label": f"{correct.title()} (corrected)",
                    "emoji": correct_info["emoji"],
                    "conf": 100.0, "co2": correct_info["co2"],
                    "xp": 5, "time": datetime.now().strftime("%H:%M")
                })
                persist()
                st.success(f"thanks! logged as {correct.title()} · +5 XP for helping")

        # ── update state (only once per unique image) ──
        if scan_hash != st.session_state.last_hash:
            st.session_state.last_hash = scan_hash
            earned_xp = info["xp"]
            if conf >= 80:
                earned_xp += 5   # accuracy bonus
            st.session_state.streak += 1
            if st.session_state.streak > st.session_state.best_streak:
                st.session_state.best_streak = st.session_state.streak
            if st.session_state.streak >= 3:
                earned_xp = int(earned_xp * 1.5)  # streak multiplier

            st.session_state.xp += earned_xp
            st.session_state.total_co2 += info["co2"]
            st.session_state.history.insert(0, {
                "label": label.title(), "emoji": info["emoji"],
                "conf": conf, "co2": info["co2"], "xp": earned_xp,
                "time": datetime.now().strftime("%H:%M")
            })

            # ── badges ──
            new_badges = []
            h = st.session_state.history
            if len(h) == 1:
                new_badges.append("first_scan")
            if st.session_state.streak >= 3:
                new_badges.append("streak3")
            if st.session_state.streak >= 5:
                new_badges.append("streak5")
            if st.session_state.total_co2 >= 100:
                new_badges.append("co2_100")
            if st.session_state.total_co2 >= 500:
                new_badges.append("co2_500")
            if len(h) >= 5:
                new_badges.append("scans5")
            if any("waste" in item["label"].lower() for item in h):
                new_badges.append("ewaste_found")
            seen_types = set(item["label"].lower() for item in h)
            if len(seen_types) >= 5:
                new_badges.append("all_types")
            for b in new_badges:
                if b not in st.session_state.badges:
                    st.session_state.badges.add(b)

            # ── level up? ──
            new_level = get_level(st.session_state.xp)
            if new_level > st.session_state.prev_level:
                st.session_state.prev_level = new_level
                st.balloons()
            
            persist()

        # all class scores
        with st.expander("see all scores"):
            for i, name in enumerate(class_names):
                s2 = float(scores[0][i]) * 100
                st.progress(int(s2), text=f"{name.title()}: {s2:.1f}%")

    else:
        st.markdown('<div class="empty">point camera at any rubbish item<br>or drop in a photo</div>', unsafe_allow_html=True)

elif page == "🌍 Impact":

    co2 = st.session_state.total_co2
    km = round(co2 / 120, 2)
    trees_min = round(co2 / 21000, 4)
    phones = round(co2 / 8.22, 1)  # avg phone charge ~8.22g CO2

    st.markdown(f"""
    <div class="impact-card">
      <h3>your environmental impact</h3>
      <div class="impact-row">
        <div class="impact-box">
          <div class="icon">💨</div>
          <div class="num">{co2}g</div>
          <div class="desc">CO₂ diverted</div>
        </div>
        <div class="impact-box">
          <div class="icon">🚗</div>
          <div class="num">{km}km</div>
          <div class="desc">car emissions avoided</div>
        </div>
        <div class="impact-box">
          <div class="icon">📱</div>
          <div class="num">{phones:.0f}</div>
          <div class="desc">phone charges equiv.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="impact-card">
      <h3>your scan stats</h3>
      <div class="impact-row">
        <div class="impact-box">
          <div class="icon">📸</div>
          <div class="num">{len(st.session_state.history)}</div>
          <div class="desc">total scans</div>
        </div>
        <div class="impact-box">
          <div class="icon">🔥</div>
          <div class="num">{st.session_state.best_streak}</div>
          <div class="desc">best streak</div>
        </div>
        <div class="impact-box">
          <div class="icon">⚡</div>
          <div class="num">{st.session_state.xp}</div>
          <div class="desc">total XP</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # breakdown by type
    if st.session_state.history:
        from collections import Counter
        counts = Counter(item["label"] for item in st.session_state.history)
        st.markdown("**waste breakdown**")
        for name, count in counts.most_common():
            info = lookup(name)
            st.progress(int(count / len(st.session_state.history) * 100),
                        text=f"{info['emoji']} {name}: {count} scan{'s' if count>1 else ''}")
    else:
        st.markdown('<div class="empty">scan some items to see your impact!</div>', unsafe_allow_html=True)

elif page == "🏅 Badges":

    earned = st.session_state.badges
    st.markdown("**badges earned** — keep scanning to unlock more")
    st.markdown('<div class="badge-grid">', unsafe_allow_html=True)
    for key, name, desc in BADGE_RULES:
        cls = "badge earned" if key in earned else "badge"
        lock = "" if key in earned else "🔒 "
        st.markdown(f'<div class="{cls}">{lock}{name}<br><small>{desc}</small></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**level progress**")
    for i, (thresh, name, emoji) in enumerate(LEVELS):
        reached = st.session_state.xp >= thresh
        icon = "✅" if reached else "🔒"
        color = "#4ade80" if reached else "#334155"
        st.markdown(f'<div style="color:{color};padding:0.3rem 0;font-size:0.88rem">{icon} {emoji} {name} — {thresh} XP</div>',
                    unsafe_allow_html=True)

elif page == "📋 History":

    if st.session_state.history:
        st.markdown(f"**last {len(st.session_state.history)} scans**")
        for item in st.session_state.history[:20]:
            st.markdown(f"""
            <div class="hist-item">
              <span style="font-size:1.5rem">{item['emoji']}</span>
              <div>
                <div class="hist-label">{item['label']}</div>
                <div class="hist-meta">{item['conf']:.0f}% · {item['co2']}g CO₂ · {item['time']}</div>
              </div>
              <div class="hist-xp">+{item['xp']} XP</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("clear history"):
            st.session_state.history = []
            st.session_state.xp = 0
            st.session_state.streak = 0
            st.session_state.total_co2 = 0
            st.session_state.badges = set()
            st.session_state.prev_level = 0
            persist()
            st.rerun()
    else:
        st.markdown('<div class="empty">no scans yet — go sort some rubbish!</div>', unsafe_allow_html=True)