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
FAST_MODEL = "llama3.1"
IMAGE_FILE_NAME = "background.png"
JOURNAL_FILE = "journals.json"
MAX_CHAT_CONTEXT_HISTORY = 6
MAX_AFFIRMATION_TOKENS = 60
MAX_MEDITATION_TOKENS = 300
# User chose: Normal replies (~short paragraph)
FAST_CHAT_TOKENS = 120
# Hard per-message truncate to avoid huge context
MAX_MESSAGE_CHARS = 800
# Ollama fast options baseline
OLLAMA_BASE_OPTIONS = {
    "num_ctx": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "mirostat": 0,
}

st.set_page_config(page_title="🌿 SoulBot AI", layout="centered")

# ============================
# LOAD / SAVE JOURNALS
# ============================
def load_journals():
    if os.path.exists(JOURNAL_FILE):
        try:
            with open(JOURNAL_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_journals(journals):
    with open(JOURNAL_FILE, "w") as f:
        json.dump(journals, f, indent=2)

st.session_state.setdefault('journal_entries', load_journals())

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
        st.sidebar.caption(f"Ensure Ollama is running and the model {FAST_MODEL} is pulled.")

if 'ollama_available' not in st.session_state:
    initialize_ollama()

# ============================
# BACKGROUND IMAGE
# ============================

def get_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
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
        background: rgba(255, 255, 255, 0.75);
        border-radius: 12px;
        padding: 14px;
    }}
    </style>
    """, unsafe_allow_html=True)

# ============================
# STATES
# ============================
st.session_state.setdefault('conversation_history', [])
st.session_state.setdefault('badge_progress', {"sessions": 0, "affirmations": 0, "journals": 0})

# ============================
# HELPERS: truncate and prepare messages
# ============================

def truncate_message(msg, max_chars=MAX_MESSAGE_CHARS):
    if len(msg) <= max_chars:
        return msg
    # preserve start and end to keep system/user cues
    head = msg[: max_chars // 2 - 10]
    tail = msg[-(max_chars // 2 - 10) :]
    return head + "\n...\n" + tail


def prepare_recent_history(history, max_items=MAX_CHAT_CONTEXT_HISTORY):
    recent = history[-max_items:]
    # Ensure each message isn't huge
    prepared = []
    for m in recent:
        prepared.append({"role": m.get('role', 'user'), "content": truncate_message(m.get('content', ''))})
    return prepared

# ============================
# OLLAMA WRAPPER (FAST DEFAULTS)
# ============================

def ollama_chat(messages, options=None):
    try:
        # merge base options with call-specific ones
        opt = OLLAMA_BASE_OPTIONS.copy()
        if options:
            opt.update(options)
        # ensure num_predict is present for chat calls; if missing, default to FAST_CHAT_TOKENS
        if 'num_predict' not in opt:
            opt['num_predict'] = FAST_CHAT_TOKENS

        return ollama.chat(model=FAST_MODEL, messages=messages, options=opt)
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

User message: "{truncate_message(user_text, 600)}"

Return JSON only.
"""
    resp = ollama_chat([{"role": "user", "content": prompt}], options={"num_predict": 30})
    if isinstance(resp, dict) and resp.get("error"):
        return "neutral"
    raw = resp["message"]["content"].strip()
    try:
        json_str = re.search(r"\{.*\}", raw, re.DOTALL).group()
        data = json.loads(json_str)
        return data.get("emotion", "neutral")
    except Exception:
        token = raw.splitlines()[0].strip().lower()
        for e in ["happy", "sad", "angry", "anxious", "neutral", "stressed", "excited", "tired"]:
            if e in token:
                return e
        return "neutral"

# ============================
# CHAT RESPONSE (FAST & SHORT) — with continuation logic to avoid cut-offs
# ============================

def _is_likely_truncated(text):
    """Heuristic to detect if assistant response was cut off.
    We look for trailing ellipses, unfinished punctuation or abrupt line endings.
    """
    if not text or len(text) < 30:
        return False
    s = text.strip()
    if s.endswith("...") or s.endswith("…"):
        return True
    # If it doesn't end with sentence-final punctuation, assume possible truncation
    if s[-1] not in ".!?\"'”””":
        # but if it ends with a closing bracket or parenthesis, treat as finished
        if s[-1] in ")]}":
            return False
        return True
    return False


def generate_response(user_input):
    if not st.session_state.ollama_available:
        return "Ollama is offline. Please start the server."

    # Append user input and prepare trimmed history
    st.session_state['conversation_history'].append({"role": "user", "content": user_input})
    recent_history = prepare_recent_history(st.session_state['conversation_history'])

    # generation options tuned for ~short paragraph but allow more space for completeness
    options = {
        "num_predict": max(FAST_CHAT_TOKENS, 240),  # raise baseline to allow full paragraphs
        "temperature": 0.6,
        "top_p": 0.85,
        "mirostat": 0,
    }

    resp = ollama_chat(recent_history, options=options)
    if isinstance(resp, dict) and resp.get("error"):
        return f"Error: {resp['error']}"

    ai_response = resp['message']['content']

    # If the response looks truncated, attempt 1 follow-up 'continue' call to finish the output.
    attempts = 0
    while _is_likely_truncated(ai_response) and attempts < 2:
        attempts += 1
        # Ask model to continue from where it left off — give minimal context to be fast
        cont_prompt = "Please continue the previous response only, and finish the remaining content.If the previous response was complete, respond with nothing."
        cont_resp = ollama_chat(recent_history + [{"role": "user", "content": cont_prompt}], options={"num_predict": 160, "temperature": 0.6})
        if isinstance(cont_resp, dict) and cont_resp.get("error"):
            break
        cont_text = cont_resp['message']['content']
        # If continuation is non-empty, append it
        if cont_text and cont_text.strip():
            # join with a space to avoid accidental merging of words
            ai_response = ai_response.rstrip() + " " + cont_text.lstrip()
        else:
            break

    st.session_state['conversation_history'].append({"role": "assistant", "content": ai_response})
    return ai_response

# ============================

# AFFIRMATION
# ============================

def generate_affirmation():
    prompt = "Give one short positive affirmation only."
    resp = ollama_chat([{"role": "user", "content": prompt}], options={"num_predict": MAX_AFFIRMATION_TOKENS, "temperature": 0.5})
    if isinstance(resp, dict) and resp.get("error"):
        return "Could not generate affirmation right now."
    st.session_state['badge_progress']["affirmations"] += 1
    return resp['message']['content']

# ============================
# MEDITATION
# ============================

def generate_meditation():
    prompt = "Give a calming 2-minute meditation script. Keep it concise and easy to read."
    resp = ollama_chat([{"role": "user", "content": prompt}], options={"num_predict": MAX_MEDITATION_TOKENS, "temperature": 0.55})
    if isinstance(resp, dict) and resp.get("error"):
        return "Could not generate meditation right now."
    return resp['message']['content']

# ============================
# CBT THOUGHT FIX TOOL
# ============================

def cbt_fix(thought):
    prompt = f"""
You are a CBT assistant.
User's negative thought: "{truncate_message(thought, 600)}"

Identify the cognitive distortion and reframe it.

Format:
Distortion: <type>
Reframed Thought: <new thought>
"""
    resp = ollama_chat([{"role": "user", "content": prompt}], options={"num_predict": 120})
    if isinstance(resp, dict) and resp.get("error"):
        return "CBT tool unavailable."
    return resp['message']['content']

# ============================
# JOURNAL
# ============================

def save_journal(text):
    entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "text": text}
    st.session_state['journal_entries'].append(entry)
    st.session_state['badge_progress']["journals"] += 1
    save_journals(st.session_state['journal_entries'])

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
# UI - TABS (Same layout, optimized behavior)
# ============================
st.title("🌿 SoulBot AI – Emotional Wellness Companion")
st.caption("AI-powered emotional support with CBT, journaling, meditation & more.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "🧠 CBT Tool", "🧘 Meditation", "📘 Journal", "🌬 Breathing"])

# ---------------- CHAT TAB ------------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌟 Positive Affirmation"):
            af = generate_affirmation()
            st.markdown(f"<div style='background:#fff0f5; padding:15px; border-radius:10px; color:#6b2e6b;'>{af}</div>", unsafe_allow_html=True)
    with col2:
        if st.button("🧘 Meditation Guide"):
            med = generate_meditation()
            st.markdown(f"<div style='background:#e6f7ff; padding:15px; border-radius:10px; color:#004466;'>{med}</div>", unsafe_allow_html=True)

    # display chat history (trimmed messages shown)
    for msg in st.session_state['conversation_history']:
        with st.chat_message(msg['role']):
            st.write(truncate_message(msg['content'], 1000))

    user_message = st.chat_input("How are you feeling today?", key="chat_input_1")
    if user_message:
        with st.chat_message("user"):
            st.write(user_message)
        st.session_state['badge_progress']["sessions"] += 1
        emotion = detect_emotion(user_message)
        st.info(f"*Detected Emotion:* {emotion}")
        # show spinner but operations are tuned to be faster
        with st.spinner("Thinking..."):
            reply = generate_response(user_message)
        with st.chat_message("assistant"):
            st.write(reply)

# ---------------- CBT TAB ------------------
with tab2:
    cbt_input = st.text_input("Type a negative thought to fix:", key="cbt_input_tab")
    if st.button("Fix Thought", key="cbt_fix_btn_tab"):
        if cbt_input.strip():
            fx = cbt_fix(cbt_input)
            st.info(fx)

# ---------------- MEDITATION TAB ------------------
with tab3:
    if st.button("Generate Meditation Script"):
        med = generate_meditation()
        st.markdown(f"<div style='background:#e6f7ff; padding:15px; border-radius:10px; color:#004466;'>{med}</div>", unsafe_allow_html=True)

# ---------------- JOURNAL TAB ------------------
with tab4:
    journal_text = st.text_area("Write your thoughts:", key="journal_text_area_tab")
    if st.button("Save Entry", key="save_journal_btn_tab"):
        if journal_text.strip():
            save_journal(journal_text.strip())
            st.success("Journal saved!")
        else:
            st.info("Write something before saving.")
    st.subheader("Your Past Entries")
    for entry in reversed(st.session_state['journal_entries'][-20:]):
        with st.expander(f"{entry['timestamp']}"):
            st.write(entry['text'])

# ---------------- BREATHING TAB ------------------
with tab5:
    st.subheader("🌬 Guided Breathing Exercise")
    style = st.selectbox("Select breathing style", ["Calm (4-4)", "Deep Relax (4-7-8)", "Box Breathing (4-4-4-4)"])
    color_choice = st.selectbox("Select theme color", ["Blue", "Pink", "Lavender", "Mint", "Sunset"])
    play_music = st.checkbox("🎵 Play calming background music", key="play_music_checkbox")

    colors = {
        "Blue": ["#aee6ff", "#77bfff"],
        "Pink": ["#ffccf2", "#ff99d9"],
        "Lavender": ["#e6ccff", "#c599ff"],
        "Mint": ["#ccffe6", "#88ffcc"],
        "Sunset": ["#ffd1a1", "#ff9b73"]
    }
    c1, c2 = colors[color_choice]

    if style == "Calm (4-4)":
        inhale, hold, exhale = 4, 0, 4
    elif style == "Deep Relax (4-7-8)":
        inhale, hold, exhale = 4, 7, 8
    else:
        inhale, hold, exhale = 4, 4, 4

    total_sec = inhale + hold + exhale

    # Circle animation
    st.markdown(f"""
    <div style='width:150px;height:150px;margin:auto;border-radius:50%;background:radial-gradient(circle, {c1} 0%, {c2} 100%);
    box-shadow:0 0 30px {c2}; animation:breathe {total_sec}s ease-in-out infinite;'></div>
    <style>
    @keyframes breathe {{
        0% {{transform: scale(0.75); opacity:0.75;}}
        {int((inhale/total_sec)*100)}% {{transform: scale(1.25); opacity:1;}}
        {int(((inhale+hold)/total_sec)*100)}% {{transform: scale(1.25); opacity:1;}}
        100% {{transform: scale(0.75); opacity:0.75;}}
    }}
    </style>
    """, unsafe_allow_html=True)
    st.write("Follow the circle — breathe with it")

    # Music
    if play_music:
        music_path = "calm_music.mp3"
        if os.path.exists(music_path):
            st.audio(music_path, format="audio/mp3", start_time=0)
        else:
            st.info("Put a 'calm_music.mp3' file inside the 'assets/' folder to enable music.")

    # Guided Breathing Button
    if st.button("▶ Start Quick Guided Breathing (30s)"):
        placeholder = st.empty()
        cycles = max(1, int(30 / max(1, total_sec)))
        for _ in range(cycles):
            placeholder.markdown(f"<div style='text-align:center; font-size:18px; font-weight:600;'>Inhale for {inhale}s</div>", unsafe_allow_html=True)
            time.sleep(inhale)
            if hold > 0:
                placeholder.markdown(f"<div style='text-align:center; font-size:18px; font-weight:600;'>Hold for {hold}s</div>", unsafe_allow_html=True)
                time.sleep(hold)
            placeholder.markdown(f"<div style='text-align:center; font-size:18px; font-weight:600;'>Exhale for {exhale}s</div>", unsafe_allow_html=True)
            time.sleep(exhale)
        placeholder.empty()
        st.success("Nice — well done! 💙")

# ---------------- BADGES ------------------
st.markdown("---")
st.subheader("🏅 Achievement Badges")
badges = get_badges()
if badges:
    st.write(" ".join(badges))
else:
    st.write("No badges yet — keep going! 🌱")

# ---------------- SIDEBAR ------------------
st.sidebar.title("SoulBot Journal & Stats")
if st.session_state['journal_entries']:
    st.sidebar.subheader("Journal History")
    for entry in reversed(st.session_state['journal_entries'][-20:]):
        st.sidebar.markdown(f"{entry['timestamp']}\n{truncate_message(entry['text'], 300)}\n---")
st.sidebar.caption("SoulBot AI © 2025")