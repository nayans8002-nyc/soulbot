import streamlit as st
import ollama
import base64
import os
import json
import re
import time
from datetime import datetime

# ============================
# CONFIGURATION
# ============================
FAST_MODEL = "gemma3:4b-it-q4_K_M"
IMAGE_FILE_NAME = "background.png"
MAX_CHAT_CONTEXT_HISTORY = 6
MAX_AFFIRMATION_TOKENS = 60
MAX_MEDITATION_TOKENS = 300

st.set_page_config(page_title="SoulBot AI 🌿", layout="centered")

# ============================
# OLLAMA INITIALIZATION
# ============================
def initialize_ollama():
    st.session_state.setdefault('ollama_available', False)
    try:
        ollama.list()
        st.session_state.ollama_available = True
    except Exception as e:
        st.session_state.ollama_available = False
        st.sidebar.error(f"❌ Ollama Connection Error: {e}")
        st.sidebar.caption(f"Please ensure Ollama is running and the model {FAST_MODEL} is pulled.")

if 'ollama_available' not in st.session_state:
    initialize_ollama()

# ============================
# BACKGROUND IMAGE HANDLER
# ============================
def get_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

bin_str = get_base64(IMAGE_FILE_NAME)

if bin_str:
    st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: url("data:image/png;base64,{bin_str}") !important;
            background-size: cover;
            background-position: center;
        }}
        .main > div {{
            background: rgba(255, 255, 255, 0.68);
            border-radius: 10px;
            padding: 12px;
        }}
        </style>
    """, unsafe_allow_html=True)

# ============================
# STATES
# ============================
st.session_state.setdefault('conversation_history', [])
st.session_state.setdefault('journal_entries', [])
st.session_state.setdefault('badge_progress', {"sessions": 0, "affirmations": 0, "journals": 0})

# ============================
# HELPERS (OLLAMA WRAPPERS)
# ============================
def ollama_chat(messages, options=None):
    try:
        if options:
            return ollama.chat(model=FAST_MODEL, messages=messages, options=options)
        return ollama.chat(model=FAST_MODEL, messages=messages)
    except Exception as e:
        return {"error": str(e)}

# ============================
# EMOTION DETECTION
# ============================
def detect_emotion(user_text):
    prompt = f"""
You are an emotion classifier.
Analyze the user's message and return ONLY valid JSON.

Format:
{{
    "emotion": "happy" | "sad" | "angry" | "anxious" | "neutral" | "stressed" | "excited" | "tired"
}}

User message: \"{user_text}\"

Return JSON only.
"""
    resp = ollama_chat([{"role": "user", "content": prompt}])
    if isinstance(resp, dict) and resp.get("error"):
        # Ollama unavailable or error
        return "neutral"
    raw = resp["message"]["content"].strip()
    try:
        json_str = re.search(r"\{.*\}", raw, re.DOTALL).group()
        data = json.loads(json_str)
        return data.get("emotion", "neutral")
    except:
        # fallback: try extract a single word
        token = raw.splitlines()[0].strip().lower()
        # limited set
        for e in ["happy", "sad", "angry", "anxious", "neutral", "stressed", "excited", "tired"]:
            if e in token:
                return e
        return "neutral"

# ============================
# MAIN CHAT RESPONSE
# ============================
def generate_response(user_input):
    if not st.session_state.ollama_available:
        return "Ollama is offline. Please start the server."

    st.session_state['conversation_history'].append({"role": "user", "content": user_input})
    recent_history = st.session_state['conversation_history'][-MAX_CHAT_CONTEXT_HISTORY:]
    resp = ollama_chat(recent_history)
    if isinstance(resp, dict) and resp.get("error"):
        return f"Error: {resp['error']}"
    ai_response = resp['message']['content']
    st.session_state['conversation_history'].append({"role": "assistant", "content": ai_response})
    return ai_response

# ============================
# AFFIRMATION
# ============================
def generate_affirmation():
    prompt = "Give one short positive affirmation only."
    resp = ollama_chat([{"role": "user", "content": prompt}], options={"num_predict": MAX_AFFIRMATION_TOKENS})
    if isinstance(resp, dict) and resp.get("error"):
        return "Could not generate affirmation right now."
    st.session_state['badge_progress']["affirmations"] += 1
    return resp['message']['content']

# ============================
# MEDITATION
# ============================
def generate_meditation():
    prompt = "Give a calming 2-minute meditation script."
    resp = ollama_chat([{"role": "user", "content": prompt}], options={"num_predict": MAX_MEDITATION_TOKENS})
    if isinstance(resp, dict) and resp.get("error"):
        return "Could not generate meditation right now."
    return resp['message']['content']

# ============================
# CBT THOUGHT FIX TOOL
# ============================
def cbt_fix(thought):
    prompt = f"""
You are a CBT assistant.
User's negative thought: "{thought}"

Identify the cognitive distortion and reframe it.

Format:
Distortion: <type>
Reframed Thought: <new thought>
"""
    resp = ollama_chat([{"role": "user", "content": prompt}])
    if isinstance(resp, dict) and resp.get("error"):
        return "CBT tool unavailable."
    st.session_state['badge_progress']["sessions"] += 0  # placeholder if we want to count
    return resp['message']['content']

# ============================
# JOURNAL ENTRY
# ============================
def save_journal(text):
    entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "text": text}
    st.session_state['journal_entries'].append(entry)
    st.session_state['badge_progress']["journals"] += 1

# ============================
# BADGES
# ============================
def get_badges():
    badges = []
    p = st.session_state['badge_progress']
    if p["sessions"] >= 3:
        badges.append("🌱 Calm Starter")
    if p["affirmations"] >= 5:
        badges.append("✨ Positive Soul")
    if p["journals"] >= 3:
        badges.append("📘 Reflective Thinker")
    return badges

# ============================
# UI - Header
# ============================
st.title("🌿 SoulBot AI – Emotional Wellness Companion")
st.caption("AI-powered emotional support with CBT, journaling, meditation & more.")

# ---------------- CHAT ------------------
for msg in st.session_state['conversation_history']:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

user_message = st.chat_input("How are you feeling today?", key="chat_input_1")

if user_message:
    with st.chat_message("user"):
        st.write(user_message)

    # Count session for badges
    st.session_state['badge_progress']["sessions"] += 1

    # Emotion detection
    emotion = detect_emotion(user_message)
    st.info(f"**Detected Emotion:** {emotion}")

    with st.spinner("Thinking..."):
        reply = generate_response(user_message)

    with st.chat_message("assistant"):
        st.write(reply)

# ============================
# TOOLS - single merged panel (no duplicates)
# ============================
st.markdown("---")
st.subheader("🛠 Emotional Wellness Tools")
tools_col1, tools_col2 = st.columns(2)

with tools_col1:
    if st.button("🌟 Affirmation", key="affirm_btn_main", use_container_width=True):
        af = generate_affirmation()
        with st.chat_message("assistant"):
            st.success(af)

    # Journal input (keeps consistent key names)
    journal_text = st.text_area("📘 Write your thoughts:", key="journal_text_area")
    if st.button("Save Entry", key="save_journal_btn", use_container_width=True):
        if journal_text.strip():
            save_journal(journal_text.strip())
            st.success("Journal saved!")
        else:
            st.info("Write something before saving.")

with tools_col2:
    if st.button("🧘 Meditation", key="meditate_btn", use_container_width=True):
        med = generate_meditation()
        with st.chat_message("assistant"):
            st.write(med)

    cbt_input = st.text_input("CBT Thought Fix (type a thought):", key="cbt_input_1")
    if st.button("Fix Thought", key="cbt_fix_btn", use_container_width=True) and cbt_input:
        fx = cbt_fix(cbt_input)
        with st.chat_message("assistant"):
            st.info(fx)

# Breathing mini-card (placed after tools)
st.markdown("---")
st.subheader("🌬️ Breathing Exercises")

# breathing options (unique keys)
breathing_style = st.selectbox(
    "Choose a breathing style:",
    ["Calm (4-4)", "Deep Relax (4-7-8)", "Box Breathing (4-4-4-4)"],
    key="breath_style_select"
)

color_choice = st.selectbox(
    "Choose a calming color theme:",
    ["Blue", "Pink", "Lavender", "Mint", "Sunset"],
    key="breath_color_select"
)

# color map
colors = {
    "Blue": ["#aee6ff", "#77bfff"],
    "Pink": ["#ffccf2", "#ff99d9"],
    "Lavender": ["#e6ccff", "#c599ff"],
    "Mint": ["#ccffe6", "#88ffcc"],
    "Sunset": ["#ffd1a1", "#ff9b73"]
}
c1, c2 = colors[color_choice]

# durations
if breathing_style == "Calm (4-4)":
    inhale = 4
    hold = 0
    exhale = 4
elif breathing_style == "Deep Relax (4-7-8)":
    inhale = 4
    hold = 7
    exhale = 8
else:
    inhale = 4
    hold = 4
    exhale = 4

# CSS animation: compute total seconds for single breath cycle
total_seconds = inhale + hold + exhale
# we will use CSS animation-duration equal to total_seconds (in seconds)
# Build breathing circle CSS (unique class names to avoid collisions)
st.markdown(
    f"""
    <style>
    .soul-breathing-wrapper {{
        display:flex;
        flex-direction:column;
        align-items:center;
        gap:10px;
    }}
    .soul-breathing-circle {{
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: radial-gradient(circle, {c1} 0%, {c2} 100%);
        box-shadow: 0 0 30px {c2};
        animation: soul_breathe {total_seconds}s ease-in-out infinite;
        transform-origin: center;
    }}
    @keyframes soul_breathe {{
        0% {{ transform: scale(0.75); opacity: 0.75; }}
        {int((inhale / total_seconds) * 100)}% {{ transform: scale(1.25); opacity: 1; }}
        {int(((inhale + hold) / total_seconds) * 100)}% {{ transform: scale(1.25); opacity: 1; }}
        100% {{ transform: scale(0.75); opacity: 0.75; }}
    }}
    </style>

    <div class="soul-breathing-wrapper">
        <div class="soul-breathing-circle"></div>
        <div style="text-align:center; font-weight:600;">Follow the circle — breathe with it</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# music checkbox (unique key)
play_music = st.checkbox("🎵 Play calming background music", key="play_music_checkbox")
if play_music:
    music_path = "calm_music.mp3"
    if os.path.exists(music_path):
        st.audio(music_path, format="audio/mp3", start_time=0)
    else:
        st.info("Put a calm_music.mp3 file inside 'assets/' to enable background music.")

# breathing chime button (unique key)
if st.button("🔔 Play breathing cue", key="breathing_cue_btn", use_container_width=False):
    chime_path = "breath_music.mp3"
    if os.path.exists(chime_path):
        st.audio(chime_path, format="audio/mp3", start_time=0)
    else:
        st.info("Place 'breath_chime.mp3' inside 'assets/' to enable the cue sound.")

# Small guided run button (runs simple on-screen guide)
if st.button("▶️ Start quick guided breathing (30s)", key="guided_breath_run"):
    # We'll run a short blocking loop for 30 seconds to visually encourage breathing.
    # Note: blocking loops will freeze interaction for the duration — keep short.
    cycles = max(1, int(30 / max(1, total_seconds)))
    placeholder = st.empty()
    for cycle in range(cycles):
        # Inhale
        placeholder.markdown(f"<div style='text-align:center; font-size:18px; font-weight:600;'>Inhale for {inhale} s</div>", unsafe_allow_html=True)
        time.sleep(inhale)
        # Hold
        if hold > 0:
            placeholder.markdown(f"<div style='text-align:center; font-size:18px; font-weight:600;'>Hold for {hold} s</div>", unsafe_allow_html=True)
            time.sleep(hold)
        # Exhale
        placeholder.markdown(f"<div style='text-align:center; font-size:18px; font-weight:600;'>Exhale for {exhale} s</div>", unsafe_allow_html=True)
        time.sleep(exhale)
    placeholder.empty()
    st.success("Nice — well done! 💙")

# ============================
# BADGES & JOURNAL SUMMARY
# ============================
st.markdown("---")
st.subheader("🏅 Your Achievement Badges")
badges = get_badges()
if badges:
    st.write(" ".join(badges))
else:
    st.write("No badges yet — keep going! 🌱")

# Sidebar: journal history
st.sidebar.title("Journal History")
if st.session_state['journal_entries']:
    for entry in reversed(st.session_state['journal_entries'][-20:]):
        st.sidebar.markdown(f"**{entry['timestamp']}**\n{entry['text']}\n---")
else:
    st.sidebar.info("No journal entries yet. Use the 'Write your thoughts' box to save private notes.")

st.sidebar.caption("SoulBot AI © 2025")
