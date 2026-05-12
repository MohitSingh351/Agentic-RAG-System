"""Interactive REPL for the agentic RAG system."""

import argparse
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()


# ANSI color codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_DIM = "\033[2m"


def _c(text: str, code: str, use_color: bool) -> str:
    return f"{code}{text}{_RESET}" if use_color else text


def _display_result(state: dict, use_color: bool) -> None:
    if state.get("answer"):
        label = _c("Answer:", _BOLD + _CYAN, use_color)
        print(f"\n{label}\n{state['answer']}\n")
    elif state.get("clarification_question"):
        label = _c("Clarifying:", _BOLD + _YELLOW, use_color)
        print(f"\n{label} {state['clarification_question']}\n")
    elif state.get("refusal_message"):
        label = _c("Cannot help:", _BOLD + _RED, use_color)
        print(f"\n{label} {state['refusal_message']}\n")


def _display_debug(state: dict, use_color: bool) -> None:
    action = state.get("action", "?")
    confidence = state.get("confidence", 0.0)
    trace = state.get("trace", [])
    info = f"[DEBUG] Action: {action} | Confidence: {confidence:.2f} | Nodes: {len(trace)}"
    print(_c(info, _DIM, use_color))


_HELP_TEXT = """Available commands:
  /quit, /exit  — exit the REPL
  /clear        — clear conversation memory
  /trace        — show last session trace nodes
  /help         — show this help message

Any other input is sent to the agent."""


def main() -> None:
    parser = argparse.ArgumentParser(description="SkyClad arXiv RAG agent")
    parser.add_argument("--debug", action="store_true", help="Show debug info after each response")
    parser.add_argument("--session-id", default=str(uuid.uuid4()), help="Session identifier")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    args = parser.parse_args()

    use_color = not args.no_color

    # Import here to avoid slow startup when just running tests
    from agent.graph import _conv_memory, run_agent

    last_state: dict | None = None

    print(_c("SkyClad arXiv RAG Agent", _BOLD + _CYAN, use_color))
    print(_c("Type /help for commands. Ctrl-C or /quit to exit.\n", _DIM, use_color))

    while True:
        try:
            user_input = input(_c("You: ", _BOLD, use_color)).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue

        if user_input in ("/quit", "/exit"):
            print("Goodbye!")
            sys.exit(0)

        if user_input == "/clear":
            _conv_memory.clear()
            print(_c("[Memory cleared]", _DIM, use_color))
            continue

        if user_input == "/trace":
            if last_state is None:
                print(_c("[No trace yet]", _DIM, use_color))
            else:
                for entry in last_state.get("trace", []):
                    print(_c(f"  {entry.get('node', '?')}", _DIM, use_color))
            continue

        if user_input == "/help":
            print(_HELP_TEXT)
            continue

        try:
            state = run_agent(user_input, debug=args.debug)
            last_state = state
            _display_result(state, use_color)
            if args.debug:
                _display_debug(state, use_color)
        except Exception as exc:
            print(_c(f"[Error] {exc}", _RED, use_color))


if __name__ == "__main__":
    main()
