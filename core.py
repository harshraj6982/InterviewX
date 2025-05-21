import threading
from typing import Callable

from llama_cpp import Llama

# === Configuration & Flags ===
SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. "
    "Do NOT reveal internal reasoning or chain-of-thought—only output final answers."
)

enable_auto_submit = True
AUTO_SUBMIT_IDLE_TIMEOUT = 10  # seconds

# === Internal State ===
_is_generating = False
_allow_always_input = True
_submit_lock = threading.Lock()
_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# === Model Initialization ===
_llm = Llama(
    model_path=r"Qwen3-4B-Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,
    verbose=False,
)


def is_submission_allowed() -> bool:
    """Return True if user may submit new input."""
    return not _is_generating and _allow_always_input


def get_block_reason() -> str:
    """
    If submission is disallowed, return the user-facing reason;
    otherwise return empty string.
    """
    if not _is_generating and not _allow_always_input:
        return "❌ Input is currently disabled."
    if _is_generating and _allow_always_input:
        return "⚠️ Assistant is still replying…"
    if _is_generating and not _allow_always_input:
        return "⚠️ Assistant is replying. Input is locked."
    return ""


def handle_user_submission(
    user_text: str,
    write_output: Callable[[str], None],
    show_status: Callable[[str], None],
    clear_input: Callable[[], None],
) -> None:
    """
    Send user_text to the LLM, streaming results via write_output.
    Use show_status() to display block reasons, and clear_input() only
    when submission actually proceeds.
    """
    global _is_generating

    if not is_submission_allowed():
        show_status(get_block_reason())
        return

    if _submit_lock.locked():
        write_output("\n[Wait: assistant is still replying...]\n")
        return

    show_status("")
    clear_input()

    def _generate():
        global _is_generating
        with _submit_lock:
            _is_generating = True
            _messages.append({"role": "user", "content": user_text})
            write_output(f"\nYou: {user_text}\nAI: ")

            assistant_text = ""
            try:
                for chunk in _llm.create_chat_completion(
                    messages=_messages, stream=True
                ):
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content", "")
                    assistant_text += token
                    write_output(token)
            except Exception:
                write_output("\n[Generation interrupted]\n")

            _messages.append({"role": "assistant", "content": assistant_text})
            _is_generating = False
            show_status("")

    threading.Thread(target=_generate, daemon=True).start()