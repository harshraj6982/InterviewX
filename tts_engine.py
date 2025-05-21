import pyttsx3
import threading
import queue
from config import TTSConfig


class TTSManager:
    """
    Offline TTS using pyttsx3, with a background queue to avoid overlap.
    Provides runtime enable/disable, voice selection, and rate adjustment.
    """

    def __init__(self, voice_index: int = TTSConfig.default_voice_index, rate: int = TTSConfig.default_rate):
        self.engine = pyttsx3.init()
        self.queue = queue.Queue()
        self._lock = threading.Lock()

        # Settings
        self.tts_enabled = True
        self._voices = self.engine.getProperty('voices')
        self.voice_index = voice_index if 0 <= voice_index < len(
            self._voices) else 0
        self.rate = rate

        # Apply initial settings
        self.engine.setProperty('voice', self._voices[self.voice_index].id)
        self.engine.setProperty('rate', self.rate)

        # Start background worker
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            text = self.queue.get()
            if not self.tts_enabled:
                continue
            with self._lock:
                self.engine.say(text)
                self.engine.runAndWait()

    def speak(self, text: str) -> None:
        """Queue text for speaking (if enabled)."""
        if self.tts_enabled:
            self.queue.put(text)

    def enable(self, flag: bool) -> None:
        """Turn TTS on or off."""
        self.tts_enabled = flag

    def get_voices(self) -> list[str]:
        """Return list of human-readable voice names."""
        return [v.name for v in self._voices]

    def set_voice(self, index: int) -> None:
        """Switch to a different voice by index."""
        if 0 <= index < len(self._voices):
            self.voice_index = index
            self.engine.setProperty('voice', self._voices[index].id)

    def set_rate(self, rate: int) -> None:
        """Adjust speaking rate."""
        self.rate = rate
        self.engine.setProperty('rate', rate)