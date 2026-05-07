"""
app.py
════════════════════════════════════════════════════════════════
Insighto — Streamlit Web Interface

Professional ChatGPT-style UI featuring:
  • Dark-themed message bubbles (user / bot)
  • Animated typing indicator during response generation
  • Sidebar: API status, memory stats, model info, quick actions
  • Session-persistent conversation (survives page interactions)
  • Bilingual-friendly layout

Run:  streamlit run app.py
════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path

# ── Path + env ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title = "Insighto — AI Chatbot",
    page_icon  = "🧠",
    layout     = "wide",
    initial_sidebar_state = "expanded",
    menu_items = {
        "About": (
            "**Insighto** — an intelligent AI chatbot powered by Groq + LangChain + LangGraph.\n\n"
            "Built with ❤️ using a clean modular architecture."
        ),
    },
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Main content area ── */
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 5rem;
    max-width: 900px;
}

/* ── Header ── */
.insighto-header {
    text-align: center;
    padding: 18px 0 10px 0;
    margin-bottom: 8px;
}
.insighto-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.insighto-header p {
    color: #6b7280;
    font-size: 0.92rem;
    margin: 4px 0 0 0;
}

/* ── Chat bubbles ── */
.msg-row-user {
    display: flex;
    justify-content: flex-end;
    margin: 10px 0 2px 0;
}
.msg-row-bot {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 10px;
    margin: 10px 0 2px 0;
}
.bubble-user {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff;
    padding: 13px 17px;
    border-radius: 20px 20px 4px 20px;
    max-width: 78%;
    font-size: 0.95rem;
    line-height: 1.65;
    word-wrap: break-word;
    box-shadow: 0 2px 10px rgba(99,102,241,0.3);
}
.bubble-bot {
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    color: #e2e8f0;
    padding: 14px 18px;
    border-radius: 20px 20px 20px 4px;
    max-width: 84%;
    font-size: 0.95rem;
    line-height: 1.7;
    word-wrap: break-word;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.bubble-bot code   { background:#252545; padding:2px 6px; border-radius:4px; font-size:.87em; color:#a5b4fc; }
.bubble-bot pre    { background:#12121f; border-radius:10px; padding:12px; overflow-x:auto; }
.bubble-bot strong { color:#c4b5fd; }
.bubble-bot ul, .bubble-bot ol { padding-left:20px; margin:6px 0; }

.avatar-bot {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem;
    flex-shrink: 0;
    margin-top: 3px;
    box-shadow: 0 2px 8px rgba(99,102,241,0.4);
}
.msg-time {
    font-size: 0.68rem;
    color: #374151;
    margin: 1px 4px 8px 4px;
}
.msg-time-right { text-align: right; }

/* ── Typing indicator ── */
.typing-wrap {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 12px 0;
}
.typing-bubble {
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    border-radius: 20px 20px 20px 4px;
    padding: 15px 20px;
    display: flex; align-items: center; gap: 5px;
}
.tdot {
    width: 8px; height: 8px;
    background: #6366f1; border-radius: 50%;
    animation: tbounce 1.3s infinite ease-in-out;
}
.tdot:nth-child(2) { animation-delay: .22s; }
.tdot:nth-child(3) { animation-delay: .44s; }
@keyframes tbounce {
    0%,80%,100% { transform:translateY(0); opacity:.35; }
    40%          { transform:translateY(-9px); opacity:1; }
}

/* ── Welcome card ── */
.welcome-card {
    background: linear-gradient(135deg, #0a0a18 0%, #15103a 50%, #0a1528 100%);
    border: 1px solid #2a2a45;
    border-radius: 20px;
    padding: 34px 40px;
    text-align: center;
    margin-bottom: 20px;
}
.welcome-card h2 { color: #e2e8f0; font-size: 1.5rem; margin: 0 0 8px 0; font-weight: 600; }
.welcome-card p  { color: #8b949e; font-size: 0.93rem; margin: 0 0 22px 0; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.chip {
    background: rgba(99,102,241,0.13);
    border: 1px solid rgba(99,102,241,0.35);
    color: #a5b4fc;
    padding: 7px 16px;
    border-radius: 20px;
    font-size: 0.84rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d0d1f;
    border-right: 1px solid #1e1e35;
}
.sb-card {
    background: #13132a;
    border: 1px solid #1e1e35;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.sb-title {
    color: #6b7280;
    font-size: 0.71rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin: 0 0 10px 0;
}
.status-row { display:flex; align-items:center; gap:8px; margin:6px 0; color:#d1d5db; font-size:.87rem; }
.dot-g { width:9px;height:9px;border-radius:50%;background:#22c55e;box-shadow:0 0 6px #22c55e55;flex-shrink:0; }
.dot-r { width:9px;height:9px;border-radius:50%;background:#ef4444;flex-shrink:0; }
.dot-y { width:9px;height:9px;border-radius:50%;background:#f59e0b;flex-shrink:0; }
.model-badge {
    display: inline-block;
    background: rgba(99,102,241,.15);
    border: 1px solid rgba(99,102,241,.4);
    color: #a5b4fc;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: .77rem;
    font-weight: 500;
}

/* ── Input ── */
.stTextInput > div > div > input {
    background: #13132a !important;
    color: #e2e8f0 !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 14px !important;
    padding: 13px 18px !important;
    font-size: .96rem !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.2) !important;
}
div.stButton > button {
    border-radius: 12px !important;
    font-weight: 500 !important;
    transition: all .2s !important;
    font-family: 'Inter', sans-serif !important;
}
div.stButton > button:hover { transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)


# ─── Cached bot initialisation ────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_flow():
    """
    Initialise InsightoFlow once per server session (cached by Streamlit).

    Returns:
        (InsightoFlow, None) on success.
        (None, error_message_str) on failure.
    """
    try:
        from main import build_app
        flow = build_app()
        return flow, None
    except SystemExit:
        return None, (
            "GROQ_API_KEY is missing. "
            "Get a free key at https://console.groq.com and add it to `.env`."
        )
    except Exception as e:
        return None, str(e)


# ─── Session state initialisation ────────────────────────────────────────────

def init_session():
    """Initialise Streamlit session state with defaults."""
    defaults = {
        "messages":  [],       # list of {"role", "content", "ts"}
        "input_key": 0,        # increment to reset text_input widget
        "pending":   "",       # quick-prompt pending message
        "api_ok":    None,     # None = not checked yet
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Rendering helpers ────────────────────────────────────────────────────────

def render_user_msg(content: str, ts: str = ""):
    """Render a user message bubble."""
    st.markdown(
        f'<div class="msg-row-user">'
        f'  <div class="bubble-user">{content}</div>'
        f'</div>'
        f'<div class="msg-time msg-time-right">{ts}</div>',
        unsafe_allow_html=True,
    )


def render_bot_msg(content: str, meta: str = "", ts: str = ""):
    """Render a bot message bubble with proper Markdown support."""
    col_av, col_msg = st.columns([0.065, 0.935])
    with col_av:
        st.markdown('<div class="avatar-bot">🧠</div>', unsafe_allow_html=True)
    with col_msg:
        st.markdown(content)
        if meta or ts:
            st.markdown(
                f'<div class="msg-time">{meta} {ts}</div>',
                unsafe_allow_html=True,
            )


def render_typing():
    """Render the animated typing indicator."""
    return st.markdown(
        '<div class="typing-wrap">'
        '  <div class="avatar-bot">🧠</div>'
        '  <div class="typing-bubble">'
        '    <div class="tdot"></div>'
        '    <div class="tdot"></div>'
        '    <div class="tdot"></div>'
        '    <span style="color:#6b7280;font-size:.84rem;margin-left:10px;">'
        '      Insighto is thinking…'
        '    </span>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_welcome():
    """Render the welcome card on first load."""
    st.markdown("""
    <div class="welcome-card">
        <h2>Welcome to Insighto 🧠</h2>
        <p>An intelligent, analytical AI assistant powered by Groq + LangGraph</p>
        <div class="chip-row">
            <span class="chip">❓ Ask me anything</span>
            <span class="chip">🌦️ Live weather</span>
            <span class="chip">🧮 Calculations</span>
            <span class="chip">🧠 Remembers you</span>
            <span class="chip">📖 Deep analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar(flow):
    """Render the full sidebar with status, stats, and actions."""
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="text-align:center; padding:20px 0 14px 0;">
            <div style="font-size:3rem; filter:drop-shadow(0 0 12px rgba(99,102,241,.5));">🧠</div>
            <div style="font-weight:700; font-size:1.3rem; color:#e2e8f0; margin-top:4px;">Insighto</div>
            <div style="font-size:.74rem; color:#6b7280;">AI Chatbot · v1.0</div>
        </div>
        <hr style="border-color:#1e1e35; margin:0 0 14px 0;">
        """, unsafe_allow_html=True)

        # ── API Status ────────────────────────────────────────────────────────
        if flow:
            # Check API status once per session
            if st.session_state.api_ok is None:
                with st.spinner("Checking Groq API…"):
                    ok, info = flow._chain.ping()
                st.session_state.api_ok = ok
            api_ok = st.session_state.api_ok

            from tools.weather_tool import is_configured as weather_configured
            w_ok = weather_configured()

            groq_dot  = "dot-g" if api_ok else "dot-r"
            groq_lbl  = f"Connected" if api_ok else "Error — check API key"
            w_dot     = "dot-g" if w_ok else "dot-y"
            w_lbl     = "OpenWeatherMap ✓" if w_ok else "wttr.in fallback"

            st.markdown(f"""
            <div class="sb-card">
                <div class="sb-title">⚡ Status</div>
                <div class="status-row">
                    <span class="{groq_dot}"></span>
                    <span>Groq LLM: {groq_lbl}</span>
                </div>
                <div style="margin-left:17px; margin-bottom:8px;">
                    <span class="model-badge">{flow._chain.model_name}</span>
                </div>
                <div class="status-row">
                    <span class="{w_dot}"></span>
                    <span>Weather: {w_lbl}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Memory Stats ──────────────────────────────────────────────────
            stats = flow.memory.stats()
            user_name = stats["user_name"] or "Unknown"
            faiss_icon = "🟢" if stats["faiss_active"] else "🟡"

            st.markdown(f"""
            <div class="sb-card">
                <div class="sb-title">🧠 Memory</div>
                <div style="color:#d1d5db; font-size:.87rem; line-height:1.9;">
                    <div>👤 User: <strong>{user_name}</strong></div>
                    <div>💬 Turns: <strong>{stats['short_term_turns']}</strong></div>
                    <div>📚 Facts: <strong>{stats['long_term_docs']}</strong></div>
                    <div>{faiss_icon} FAISS: <strong>{"Active" if stats['faiss_active'] else "Disabled"}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sb-card">
                <div class="sb-title">⚡ Status</div>
                <div class="status-row"><span class="dot-r"></span> Not initialised</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1e1e35; margin:12px 0;'>", unsafe_allow_html=True)

        # ── Actions ───────────────────────────────────────────────────────────
        st.markdown('<div class="sb-title" style="color:#6b7280;font-size:.71rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;">Actions</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear conversation"):
                st.session_state.messages = []
                if flow:
                    flow.memory.clear()
                st.rerun()
        with col2:
            if st.button("🔄 Refresh", use_container_width=True, help="Refresh API status"):
                st.session_state.api_ok = None
                st.rerun()

        st.markdown("<hr style='border-color:#1e1e35; margin:12px 0;'>", unsafe_allow_html=True)

        # ── Quick Prompts ─────────────────────────────────────────────────────
        st.markdown('<div class="sb-title" style="color:#6b7280;font-size:.71rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;">💡 Quick Prompts</div>', unsafe_allow_html=True)

        quick_prompts = [
            ("🧠 Who are you?",          "Who are you and what can you do?"),
            ("🌦️ Weather in Cairo",      "What's the weather like in Cairo right now?"),
            ("🧮 Quick math",             "What is 15 percent of 840?"),
            ("📊 Explain AI",             "Explain artificial intelligence in simple terms"),
            ("✍️ Write a story",          "Write a short creative story about a robot discovering music"),
            ("🔍 Analyse this",           "What are the pros and cons of electric vehicles?"),
        ]

        for label, prompt in quick_prompts:
            if st.button(label, use_container_width=True, key=f"qp_{label}"):
                st.session_state.pending = prompt
                st.rerun()

        # ── Graph info ────────────────────────────────────────────────────────
        st.markdown("<hr style='border-color:#1e1e35; margin:12px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; color:#4b5563; font-size:.72rem; line-height:1.8;">
            <div>Powered by</div>
            <div><strong style="color:#6b7280;">LangChain + LangGraph</strong></div>
            <div><strong style="color:#6b7280;">Groq Llama3</strong></div>
            <div style="margin-top:4px;">Flow: Router → Nodes → Memory</div>
        </div>
        """, unsafe_allow_html=True)


# ─── Main application ─────────────────────────────────────────────────────────

def main():
    """Main Streamlit application entry point."""
    init_session()
    flow, init_error = get_flow()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    render_sidebar(flow)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="insighto-header">
        <h1>🧠 Insighto</h1>
        <p>Intelligent · Analytical · Helpful  ·  Powered by Groq + LangGraph</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Error banner ──────────────────────────────────────────────────────────
    if init_error:
        st.error(f"**Insighto could not start:** {init_error}", icon="🔑")
        st.markdown("""
```bash
# 1. Copy the environment template
cp .env.example .env

# 2. Get a FREE Groq API key at: https://console.groq.com
# 3. Edit .env and set: GROQ_API_KEY=gsk_...

# 4. Restart the app
streamlit run app.py
```
        """)
        return

    # ── Welcome card (first load) ─────────────────────────────────────────────
    if not st.session_state.messages:
        render_welcome()

    # ── Chat history ──────────────────────────────────────────────────────────
    chat_area = st.container()
    with chat_area:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                render_user_msg(msg["content"], msg.get("ts", ""))
            else:
                render_bot_msg(msg["content"], msg.get("meta", ""), msg.get("ts", ""))

    # ── Process pending quick-prompt ──────────────────────────────────────────
    if st.session_state.pending:
        pending = st.session_state.pending
        st.session_state.pending = ""
        _process_message(pending, flow, chat_area)
        st.rerun()

    # ── Input bar ─────────────────────────────────────────────────────────────
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    in_col, btn_col = st.columns([0.88, 0.12])
    with in_col:
        user_input = st.text_input(
            "message",
            label_visibility = "collapsed",
            placeholder      = "Ask Insighto anything…",
            key              = f"msg_input_{st.session_state.input_key}",
        )
    with btn_col:
        send = st.button("Send ➤", use_container_width=True, type="primary")

    # Trigger on Send button or if input changed (Enter key)
    if user_input.strip() and (
        send or user_input != st.session_state.get("_last_input", "")
    ):
        st.session_state._last_input = user_input
        st.session_state.input_key  += 1
        _process_message(user_input.strip(), flow, chat_area)
        st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center; color:#374151; font-size:.72rem; margin-top:8px;'>"
        "Insighto AI · Built with LangChain + LangGraph + Groq"
        "</div>",
        unsafe_allow_html=True,
    )


# ─── Message processing ───────────────────────────────────────────────────────

def _process_message(user_input: str, flow, chat_area):
    """
    Process a user message:
      1. Validate input.
      2. Store + display user bubble.
      3. Show typing indicator.
      4. Call InsightoFlow.
      5. Store + display bot reply.
    """
    from utils.helpers import sanitise_input, is_meaningful

    user_input = sanitise_input(user_input)
    if not user_input:
        return

    ts = time.strftime("%H:%M")

    # ── Store and render user message ─────────────────────────────────────────
    st.session_state.messages.append({
        "role":    "user",
        "content": user_input,
        "ts":      ts,
    })
    with chat_area:
        render_user_msg(user_input, ts)

    # ── Typing indicator ──────────────────────────────────────────────────────
    with chat_area:
        typing_ph = st.empty()
        typing_ph.markdown(
            '<div class="typing-wrap">'
            '  <div class="avatar-bot">🧠</div>'
            '  <div class="typing-bubble">'
            '    <div class="tdot"></div>'
            '    <div class="tdot"></div>'
            '    <div class="tdot"></div>'
            '    <span style="color:#6b7280;font-size:.84rem;margin-left:10px;">'
            '      Insighto is thinking…'
            '    </span>'
            '  </div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Generate response ─────────────────────────────────────────────────────
    t0 = time.time()
    try:
        reply = flow.run(user_input)
    except Exception as e:
        reply = f"⚠️ Error: {e}"
    duration = time.time() - t0

    typing_ph.empty()

    # ── Store and render bot reply ────────────────────────────────────────────
    meta = f"⚡ {duration:.1f}s"
    st.session_state.messages.append({
        "role":    "assistant",
        "content": reply,
        "meta":    meta,
        "ts":      time.strftime("%H:%M"),
    })
    with chat_area:
        render_bot_msg(reply, meta, time.strftime("%H:%M"))


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
