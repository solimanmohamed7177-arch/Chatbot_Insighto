<<<<<<< HEAD
# 🧠 Insighto — AI Chatbot

> *Intelligent · Analytical · Helpful*
> Built with **LangChain + LangGraph + Groq API + Streamlit**

A production-quality AI chatbot demonstrating:
- **LangGraph** state machine with intent routing
- **LangChain + ChatGroq** for LLM orchestration
- **Dual-layer memory** (short-term buffer + FAISS long-term)
- **Weather tool** integration
- **Professional Streamlit UI** (dark theme, ChatGPT-style)

---

## 🏗️ Architecture

```
Insighto/
│
├── app.py                  ← Streamlit web UI (run this)
├── main.py                 ← CLI entry + app factory (build_app())
├── requirements.txt
├── .env.example
│
├── chains/
│   └── chat_chain.py       ← LangChain ChatGroq wrapper + prompt engineering
│
├── graph/
│   └── flow.py             ← LangGraph state machine
│                             Nodes: router → weather|general|clarify → memory → END
│
├── memory/
│   └── memory.py           ← ShortTermMemory + FAISS LongTermMemory + InsightoMemory
│
├── tools/
│   └── weather_tool.py     ← OWM + wttr.in weather integration with cache
│
└── utils/
    └── helpers.py          ← Intent detection, NLP helpers, input sanitisation
```

### LangGraph Flow

```
User Input
    │
    ▼
[ROUTER NODE]  ← detect_intent() classifies the message
    │
    ├─ weather   → [WEATHER NODE]  ← fetch live weather → LLM
    ├─ general   → [GENERAL NODE]  ← standard LLM conversation
    └─ unclear   → [CLARIFY NODE]  ← ask for clarification
                        │
                        ▼
                 [MEMORY NODE]  ← update short/long-term memory
                        │
                        ▼
                       END
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
```

Edit `.env`:
```env
# REQUIRED — Get free at https://console.groq.com
GROQ_API_KEY=gsk_your_key_here

# Optional model (default: llama3-8b-8192)
# Other options: llama3-70b-8192, mixtral-8x7b-32768, gemma-7b-it
GROQ_MODEL=llama3-8b-8192

# Optional weather (free at openweathermap.org — wttr.in used if missing)
OPENWEATHER_API_KEY=
```

### 3. Launch

**Web UI (recommended):**
```bash
streamlit run app.py
```
Open http://localhost:8501

**CLI:**
```bash
python main.py
```

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ | — | Free at console.groq.com |
| `GROQ_MODEL` | ❌ | `llama3-8b-8192` | Any Groq-supported model |
| `OPENWEATHER_API_KEY` | ❌ | — | wttr.in used if missing |
| `MAX_HISTORY` | ❌ | `20` | Short-term memory window |

---

## 🤖 Groq Free Models

| Model | Context | Speed | Quality |
|-------|---------|-------|---------|
| `llama3-8b-8192` | 8K | ⚡ Fastest | Good |
| `llama3-70b-8192` | 8K | Fast | Best |
| `mixtral-8x7b-32768` | 32K | Fast | Great |
| `gemma-7b-it` | 8K | Fast | Good |

---

## 💬 Example Conversations

```
You:      My name is Sara and I love astronomy.
Insighto: Nice to meet you, Sara! Astronomy is fascinating...
          [stores name + interest in long-term memory]

You:      What's my name?
Insighto: Your name is Sara, and you mentioned you love astronomy!
          [retrieves from memory]

You:      What's the weather in Cairo?
Insighto: ☀️ In Cairo right now: 36°C, sunny, 45% humidity...
          [fetches live data via weather tool]

You:      What is 17% of 350?
Insighto: 17% of 350 = 59.5
          Here's the calculation: 350 × 0.17 = 59.5
```

---

## 🧠 Memory System

**Short-term (ConversationBuffer):**
- Keeps last 20 conversation turns
- Injected directly into every LLM call
- Cleared with the `/clear` button

**Long-term (FAISS vector store):**
- Stores learned facts (name, preferences, etc.)
- Retrieved by semantic similarity
- Persists across sessions (if FAISS initialises successfully)
- Falls back gracefully if `sentence-transformers` not available

---

## 📜 License

MIT — build freely.
=======
