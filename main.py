"""
main.py
════════════════════════════════════════════════════════════════
Insighto — CLI Entry Point & Application Bootstrap

Handles:
  • Environment loading and validation
  • Component initialisation (LLM, memory, graph)
  • Rich CLI chat interface (optional)
  • Startup self-check

Run:   python main.py
UI:    streamlit run app.py
════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Load environment FIRST — before any other module that reads env vars
from dotenv import load_dotenv
load_dotenv(override=True)

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.helpers import get_logger, sanitise_input, is_meaningful, format_duration
from memory.memory import InsightoMemory
from chains.chat_chain import InsightoChatChain
from graph.flow import InsightoFlow

logger = get_logger("insighto.main")

# ── Rich CLI (optional) ───────────────────────────────────────────────────────
try:
    from rich.console  import Console
    from rich.panel    import Panel
    from rich.markdown import Markdown
    from rich.text     import Text
    _RICH   = True
    console = Console()
except ImportError:
    _RICH   = False
    console = None  # type: ignore

_B  = "\033[1m";   _C  = "\033[96m"
_G  = "\033[92m";  _R  = "\033[91m"
_D  = "\033[2m";   _RS = "\033[0m"

BANNER = r"""
    ██╗███╗   ██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗ ██████╗
    ██║████╗  ██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝██╔═══██╗
    ██║██╔██╗ ██║███████╗██║██║  ███╗███████║   ██║   ██║   ██║
    ██║██║╚██╗██║╚════██║██║██║   ██║██╔══██║   ██║   ██║   ██║
    ██║██║ ╚████║███████║██║╚██████╔╝██║  ██║   ██║   ╚██████╔╝
    ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝
                    AI Chatbot  ·  Powered by Groq + LangGraph
"""


# ─── Application factory ──────────────────────────────────────────────────────

def build_app() -> InsightoFlow:
    """
    Initialise all Insighto components and return a ready InsightoFlow.

    Called by both main.py (CLI) and app.py (Streamlit).

    Returns:
        Compiled InsightoFlow instance.

    Raises:
        SystemExit: If critical configuration is missing.
    """
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key or groq_key == "gsk_your_groq_key_here":
        print(
            f"\n[ERROR] GROQ_API_KEY is not set or still a placeholder.\n"
            f"  1. Get a free key at: https://console.groq.com\n"
            f"  2. Copy .env.example → .env\n"
            f"  3. Set GROQ_API_KEY=gsk_...\n"
            f"  4. Run again.\n"
        )
        sys.exit(1)

    logger.info("Building Insighto components…")

    try:
        memory = InsightoMemory()
        chain  = InsightoChatChain()
        flow   = InsightoFlow(chain, memory)
        logger.info("All components ready.")
        return flow
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected init error: %s", e)
        print(f"\n[ERROR] Failed to initialise: {e}")
        sys.exit(1)


# ─── CLI interface ────────────────────────────────────────────────────────────

def _print(text: str, style: str = "") -> None:
    if _RICH:
        console.print(text, style=style)
    else:
        print(text)


def _show_banner() -> None:
    if _RICH:
        console.print(Text(BANNER, style="bold cyan"))
        console.print(Panel(
            "[bold white]Intelligent · Analytical · Helpful[/bold white]  "
            f"[dim]· Model: {os.getenv('GROQ_MODEL', 'llama3-8b-8192')}[/dim]",
            subtitle="[dim]Type /help for commands[/dim]",
            border_style="cyan",
        ))
    else:
        print(f"{_C}{_B}{BANNER}{_RS}")
        print(f"  {_B}Intelligent · Analytical · Helpful{_RS}\n")


def _show_help() -> None:
    cmds = {
        "/clear":   "Clear conversation history",
        "/memory":  "Show memory stats",
        "/quit":    "Exit Insighto",
        "/help":    "Show this help",
    }
    body = "\n".join(f"  [cyan]{k}[/cyan]  {v}" for k, v in cmds.items())
    if _RICH:
        console.print(Panel(body, title="💡 Commands", border_style="yellow"))
    else:
        for k, v in cmds.items():
            print(f"  {k:<12}  {v}")


def _run_cli(flow: InsightoFlow) -> None:
    """Interactive CLI chat loop."""
    _show_banner()

    if _RICH:
        console.print("[dim]Verifying Groq API connection…[/dim]")
        with console.status("[cyan]Connecting…[/cyan]", spinner="dots"):
            ok, info = flow._chain.ping()
    else:
        print("Connecting to Groq…", flush=True)
        ok, info = flow._chain.ping()

    if ok:
        msg = f"✓ Connected  ·  {info}"
        if _RICH:
            console.print(Panel(f"[green]{msg}[/green]", border_style="green", title="Ready"))
        else:
            print(f"{_G}{msg}{_RS}\n")
    else:
        print(f"\n[ERROR] Cannot connect to Groq: {info}")
        sys.exit(1)

    print()
    while True:
        try:
            prompt = "\n[bold cyan]You[/bold cyan] › " if _RICH else f"\n{_C}{_B}You{_RS} › "
            user_input = console.input(prompt) if _RICH else input(prompt)
            user_input = sanitise_input(user_input)
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!\n")
            break

        if not user_input:
            continue

        # Commands
        cmd = user_input.lower().strip()
        if cmd in ("/quit", "/exit", "/q"):
            print("\n👋 Goodbye! Come back soon.\n")
            break
        if cmd == "/clear":
            flow.memory.clear()
            _print("[green]✓ Conversation cleared.[/green]" if _RICH else "✓ Cleared.")
            continue
        if cmd == "/memory":
            stats = flow.memory.stats()
            info_str = (
                f"Short-term turns: {stats['short_term_turns']}\n"
                f"Long-term facts:  {stats['long_term_docs']}\n"
                f"FAISS active:     {stats['faiss_active']}\n"
                f"User name:        {stats['user_name'] or 'Unknown'}"
            )
            if _RICH:
                console.print(Panel(info_str, title="🧠 Memory", border_style="blue"))
            else:
                print(info_str)
            continue
        if cmd in ("/help", "/?"):
            _show_help()
            continue

        # Generate response
        t0 = time.time()
        if _RICH:
            with console.status("[dim italic]Insighto is thinking…[/dim italic]", spinner="dots2"):
                reply = flow.run(user_input)
        else:
            print(f"{_D}Thinking…{_RS}", flush=True)
            reply = flow.run(user_input)

        dur = format_duration(time.time() - t0)

        if _RICH:
            console.print()
            console.print(f"[bold magenta]Insighto[/bold magenta] [dim]· {dur}[/dim]")
            console.rule(style="dim")
            console.print(Markdown(reply))
        else:
            print(f"\n{_B}Insighto{_RS} {_D}· {dur}{_RS}")
            print("─" * 60)
            print(reply)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    flow = build_app()
    _run_cli(flow)
