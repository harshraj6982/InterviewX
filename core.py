import threading
from typing import Callable

from llama_cpp import Llama
from config import LLMConfig, CoreFlags


class ChatManager:
    """
    Contains all pure chat logic and LLM orchestration.
    No UI code here—just methods you can call with
    abstract handlers for I/O and status.
    """

    def __init__(self):
        # === Configuration & Flags (moved to config.py) ===
        self.SYSTEM_PROMPT = LLMConfig.system_prompt
        self.enable_auto_submit = CoreFlags.enable_auto_submit
        self.AUTO_SUBMIT_IDLE_TIMEOUT = CoreFlags.auto_submit_idle_timeout

        # === Internal State ===
        self._is_generating = False
        self._allow_always_input = CoreFlags.allow_always_input
        self._submit_lock = threading.Lock()
        self._messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # === New Control Flags ===
        # self.is_user_speaking: bool = False  # True while VAD+MediaPipe detect speech
        self.can_interrupt_generation: bool = CoreFlags.can_interrupt_generation
        # True if current input originates from speech
        # self.input_via_speech: bool = CoreFlags.input_via_speech
        self._cancel_requested: bool = False  # internal flag to cancel generation

        # === Model Initialization ===
        self._llm = Llama(
            model_path=LLMConfig.model_path,
            n_ctx=LLMConfig.n_ctx,
            n_gpu_layers=LLMConfig.n_gpu_layers,
            verbose=LLMConfig.verbose,
            use_mmap=LLMConfig.use_mmap,
        )

    def is_submission_allowed(self) -> bool:
        """Return True if user may submit new input."""
        if not self._allow_always_input:
            return False
        if self._is_generating:
            return self.can_interrupt_generation
        return True

    def get_block_reason(self) -> str:
        """
        If submission is disallowed, return the user-facing reason;
        otherwise return empty string.
        """
        if not self._is_generating and not self._allow_always_input:
            return "❌ Input is currently disabled."
        if self._is_generating and not self.can_interrupt_generation:
            return "⚠️ Assistant is replying (interrupt disabled)."
        if self._is_generating and self._allow_always_input:
            return "⚠️ Assistant is still replying…"
        if self._is_generating and not self._allow_always_input:
            return "⚠️ Assistant is replying. Input is locked."
        return ""

    def cancel_generation(self) -> None:
        """Signal an in-progress generation thread to abort."""
        self._cancel_requested = True

    def handle_user_submission(
        self,
        user_text: str,
        write_output: Callable[[str], None],
        show_status: Callable[[str], None],
        clear_input: Callable[[], None],
        via_speech: bool = False,
    ) -> None:
        """
        Send user_text to the LLM, streaming results via write_output.
        Use show_status() to display block reasons, and clear_input() only
        when submission actually proceeds.
        """
        # mark this as keyboard-originated input
        CoreFlags.input_via_speech = via_speech
        # self.input_via_speech = via_speech
        # reset any previous cancel request
        self._cancel_requested = False

        if not self.is_submission_allowed():
            show_status(self.get_block_reason())
            return

        if self._submit_lock.locked():
            if self.can_interrupt_generation:
                self.cancel_generation()
            else:
                write_output("\n[Wait: assistant is still replying...]\n")
                return

        show_status("")
        clear_input()

        def _generate():
            with self._submit_lock:
                self._is_generating = True
                self._messages.append(
                    {"role": "user", "content": user_text if CoreFlags.is_thinking_enabled else user_text + " /no_think"})
                write_output(f"\nYou: {user_text}\nAI: ")

                assistant_text = ""
                try:
                    for chunk in self._llm.create_chat_completion(
                        messages=self._messages, stream=True
                    ):
                        if self._cancel_requested:
                            if hasattr(self._llm, "stop"):
                                self._llm.stop()
                            break
                        delta = chunk["choices"][0]["delta"]
                        token = delta.get("content", "")
                        assistant_text += token
                        write_output(token)
                    if self._cancel_requested:
                        write_output("\n[Generation canceled]\n")
                except Exception:
                    write_output("\n[Generation interrupted]\n")

                self._messages.append(
                    {"role": "assistant", "content": assistant_text})
                self._is_generating = False
                self._cancel_requested = False
                show_status("")

        threading.Thread(target=_generate, daemon=True).start()