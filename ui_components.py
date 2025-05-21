import threading
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Optional
import re

from tts_engine import TTSManager
from stt_engine import STTManager  # new import


class ChatOutput:
    """Scrollable text area with </think> filtering, buffered TTS, and smart sentence detection."""

    FLUSH_TIMEOUT_MS = 2000  # flush incomplete buffer after 2s of silence

    def __init__(self, master: tk.Widget, filter_think_tag: bool = True):
        self.master = master
        self.widget = ScrolledText(
            master, wrap=tk.WORD, state="disabled", font=("Courier", 11)
        )
        self.widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.filter_think_tag = filter_think_tag
        self._buffer = ""
        self._filtering = filter_think_tag
        self._ai_prefix_written = False

        # TTS manager and buffering
        try:
            self.tts_manager = TTSManager()
        except Exception:
            self.tts_manager = None
        self._tts_buffer = ""
        self._flush_timer_id: Optional[str] = None

    def write(self, text: str) -> None:
        """Append text and auto-scroll."""
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")

    def _schedule_flush(self):
        """Schedule a timer to flush TTS buffer after timeout."""
        if self._flush_timer_id:
            self.widget.after_cancel(self._flush_timer_id)
        self._flush_timer_id = self.widget.after(
            self.FLUSH_TIMEOUT_MS, self._flush_tts_buffer
        )

    def _flush_tts_buffer(self):
        """Force-speak any leftover TTS buffer."""
        if self.tts_manager and self._tts_buffer.strip():
            self.tts_manager.speak(self._tts_buffer.strip())
        self._tts_buffer = ""
        self._flush_timer_id = None

    def _handle_tts(self, text: str) -> None:
        """Accumulate text and speak only complete sentences or after timeout."""
        if not self.tts_manager:
            return

        self._tts_buffer += text

        # Split on end-of-sentence punctuation, handling quotes/ellipses
        parts = re.split(r'(?<=[.?!…])(?=\s+["“”]?[A-Z])', self._tts_buffer)

        # Speak all complete segments
        for sentence in parts[:-1]:
            self.tts_manager.speak(sentence.strip())

        # If the buffer ends with punctuation, speak the last too
        if self._tts_buffer.rstrip().endswith(('.', '?', '!', '…')):
            last = parts[-1].strip()
            if last:
                self.tts_manager.speak(last)
            self._tts_buffer = ""
            if self._flush_timer_id:
                self.widget.after_cancel(self._flush_timer_id)
                self._flush_timer_id = None
        else:
            # otherwise, keep the tail and schedule a flush
            self._tts_buffer = parts[-1]
            self._schedule_flush()

    def stream(self, token: str) -> None:
        """
        Stream tokens, filter out <think> blocks, and send to TTS in buffered sentences.
        """
        if not self.filter_think_tag:
            self.write(token)
            self._handle_tts(token)
            return

        self._buffer += token
        if self._filtering:
            idx = self._buffer.find("</think>")
            if idx != -1:
                content = self._buffer[idx + len("</think>"):].lstrip()
                if not self._ai_prefix_written:
                    self.write("AI:")
                    self._ai_prefix_written = True
                self.write(content)
                self._handle_tts(content)
                self._buffer = ""
                self._filtering = False
        else:
            if not self._ai_prefix_written:
                self.write("AI:")
                self._ai_prefix_written = True
            self.write(token)
            self._handle_tts(token)

    def reset_filter(self) -> None:
        """Reset for new response and flush any leftover TTS buffer immediately."""
        if self._flush_timer_id:
            self.widget.after_cancel(self._flush_timer_id)
        if self.tts_manager and self._tts_buffer.strip():
            self.tts_manager.speak(self._tts_buffer.strip())
        self._tts_buffer = ""
        self._buffer = ""
        self._filtering = self.filter_think_tag
        self._ai_prefix_written = False

    # --- TTS Settings UI ---
    def open_tts_settings(self):
        if not self.tts_manager:
            return

        # Prevent multiple dialogs
        if hasattr(self, "_settings_win") and self._settings_win.winfo_exists():
            self._settings_win.lift()
            return

        win = tk.Toplevel(self.master)
        win.title("TTS Settings")
        self._settings_win = win

        # Enable toggle
        enabled_var = tk.BooleanVar(value=self.tts_manager.tts_enabled)
        tk.Checkbutton(win, text="Enable TTS", var=enabled_var).pack(
            anchor="w", padx=10, pady=5)

        # Voice selector
        voices = self.tts_manager.get_voices()
        voice_var = tk.StringVar(value=voices[self.tts_manager.voice_index])
        tk.Label(win, text="Voice:").pack(anchor="w", padx=10)
        voice_menu = tk.OptionMenu(win, voice_var, *voices)
        voice_menu.pack(fill="x", padx=10, pady=5)

        # Rate slider
        rate_var = tk.IntVar(value=self.tts_manager.rate)
        tk.Label(win, text="Rate:").pack(anchor="w", padx=10)
        tk.Scale(win, from_=50, to=300, orient="horizontal",
                 var=rate_var).pack(fill="x", padx=10, pady=5)

        def apply_and_close():
            self.tts_manager.enable(enabled_var.get())
            idx = voices.index(voice_var.get())
            self.tts_manager.set_voice(idx)
            self.tts_manager.set_rate(rate_var.get())
            win.destroy()

        tk.Button(win, text="Apply", command=apply_and_close).pack(pady=10)


class ChatStatus:
    """Single-line label for status or error messages."""

    def __init__(self, master: tk.Widget):
        self.widget = tk.Label(master, text="", fg="red", font=("Courier", 10))
        self.widget.pack(padx=10, anchor="w")

    def show(self, text: str) -> None:
        """Set status text (or clear if empty)."""
        self.widget.config(text=text)


class ChatInput:
    """
    Entry + Send button combo, with Settings button for TTS and a Record button for STT.
    """

    def __init__(self, master: tk.Widget, chat_output: ChatOutput):
        frame = tk.Frame(master)
        frame.pack(padx=10, pady=(0, 10), fill=tk.X)

        self.entry = tk.Entry(frame, font=("Courier", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.send_btn = tk.Button(frame, text="Send")
        self.send_btn.pack(side=tk.RIGHT, padx=(5, 0))

        self.settings_btn = tk.Button(
            frame, text="⚙", command=chat_output.open_tts_settings)
        self.settings_btn.pack(side=tk.RIGHT)

        # STT manager and Record button
        try:
            self.stt_manager = STTManager()
        except Exception:
            self.stt_manager = None

        # -- new toggle logic fields --
        self._listening = False
        self._silence_timer_id: Optional[str] = None

        self.mic_btn = tk.Button(
            frame, text="🎙️", command=self._toggle_listening)
        self.mic_btn.pack(side=tk.RIGHT, padx=(5, 0))

    def _toggle_listening(self):
        """Toggle live listening on/off."""
        if not self.stt_manager:
            return

        if not self._listening:
            # start live stream: append into entry, auto-submit after silence
            self.stt_manager.start_stream(
                append_callback=self._append_text,
                submit_callback=self._on_silence_submit,
                silence_timeout=10
            )
            self.mic_btn.config(text="⏹️")  # stop icon
            self._listening = True
        else:
            # stop live stream
            self.stt_manager.stop_stream()
            self.mic_btn.config(text="🎙️")
            self._clear_silence_timer()
            self._listening = False

    def _append_text(self, text: str):
        """Append transcribed text into the entry field."""
        # insert at end and reset auto-submit timer
        self.entry.insert(tk.END, text)
        self._reset_silence_timer()

    def _reset_silence_timer(self):
        """Reset the 10s silence auto-submit timer."""
        if self._silence_timer_id:
            self.entry.after_cancel(self._silence_timer_id)
        self._silence_timer_id = self.entry.after(
            10_000, self._on_silence_submit)

    def _clear_silence_timer(self):
        if self._silence_timer_id:
            self.entry.after_cancel(self._silence_timer_id)
            self._silence_timer_id = None

    def _on_silence_submit(self):
        """Called after 10s silence to auto-submit."""
        if self._listening:
            # simulate pressing Send
            self.entry.event_generate("<Return>")

    def bind_submit(self, callback: Callable[[], None]) -> None:
        """Call callback() on Enter or button click."""
        self.entry.bind("<Return>", lambda e: callback())
        self.send_btn.config(command=callback)

    def bind_key(self, callback: Callable[[], None]) -> None:
        """Call callback() on any keypress in entry."""
        self.entry.bind("<Key>", lambda e: callback())

    def get(self) -> str:
        """Get trimmed text."""
        return self.entry.get().strip()

    def clear(self) -> None:
        """Remove all text."""
        self.entry.delete(0, tk.END)

    def focus(self) -> None:
        """Focus the entry."""
        self.entry.focus()


class DebugOutput:
    """
    Optional slide-in debug window showing raw stream.
    Shows every token immediately, regardless of main filtering.
    """

    def __init__(
        self,
        master: tk.Tk,
        width: int = 400,
        height: int = 600,
        debug_enabled: bool = False,
    ):
        self.debug_enabled = debug_enabled
        self.width = width
        self.height = height
        self.window: Optional[tk.Toplevel] = None
        self.text: Optional[ScrolledText] = None

        if not self.debug_enabled:
            return

        # Create top-level debug window off to the right
        master.update_idletasks()
        x0 = master.winfo_x() + master.winfo_width()
        y0 = master.winfo_y()
        self.window = tk.Toplevel(master)
        self.window.title("Debug Output")
        self.window.geometry(f"{width}x{height}+{x0}+{y0}")
        self.text = ScrolledText(
            self.window, wrap=tk.WORD, state="disabled", font=("Courier", 10)
        )
        self.text.pack(fill=tk.BOTH, expand=True)

        # Start sliding in
        master.after(100, self._slide_in, x0)

    def _slide_in(self, current_x: int) -> None:
        assert self.window is not None
        master = self.window.master  # type: ignore
        master.update_idletasks()
        target_x = master.winfo_x() + master.winfo_width() - self.width
        if current_x > target_x:
            current_x -= 20
            self.window.geometry(
                f"{self.width}x{self.height}+{current_x}+{master.winfo_y()}"
            )
            self.window.after(10, self._slide_in, current_x)

    def write(self, text: str) -> None:
        """Append raw text to debug window."""
        if not self.debug_enabled or self.text is None:
            return
        self.text.configure(state="normal")
        self.text.insert(tk.END, text)
        self.text.see(tk.END)
        self.text.configure(state="disabled")