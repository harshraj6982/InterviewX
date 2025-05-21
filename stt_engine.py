import os
os.environ["HF_HUB_CACHE"] = r"/Users/harsh/Documents/huggingface_cache"

from faster_whisper import WhisperModel
import webrtcvad
import pyaudio
import numpy as np
import threading
import wave
import tempfile


class STTManager:
    """
    Offline STT using faster-whisper with VAD-based recording.
    Records only when speech is detected, stops after silence.
    """

    def __init__(
        self,
        model_size: str = "base",
        compute_type: str = "int8",
        samplerate: int = 16000,
        vad_mode: int = 1,
        frame_duration_ms: int = 30,
        silence_duration_ms: int = 500,
    ):
        # Whisper initialization
        self.whisper_model = WhisperModel(
            model_size, compute_type=compute_type)
        self.samplerate = samplerate

        # VAD setup
        self.vad = webrtcvad.Vad(vad_mode)
        self.frame_duration_ms = frame_duration_ms
        # samples per frame
        self.frame_size = int(self.samplerate * frame_duration_ms / 1000)
        self.bytes_per_sample = 2  # paInt16
        self.silence_frame_limit = int(silence_duration_ms / frame_duration_ms)

        # PyAudio
        self.audio_interface = pyaudio.PyAudio()

    def _read_frame(self, stream) -> bytes:
        """Read a single frame from the stream."""
        return stream.read(self.frame_size, exception_on_overflow=False)

    def listen(self) -> str:
        """
        Record only when speech is present; stop after sustained silence.
        Returns the transcribed text.
        """
        # 1. Open stream
        stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.samplerate,
            input=True,
            frames_per_buffer=self.frame_size,
        )

        frames = []
        recording = False
        silence_counter = 0

        # 2. VAD-driven loop
        while True:
            frame = self._read_frame(stream)
            is_speech = self.vad.is_speech(frame, self.samplerate)

            if is_speech:
                if not recording:
                    recording = True
                frames.append(frame)
                silence_counter = 0
            else:
                if recording:
                    silence_counter += 1
                    if silence_counter > self.silence_frame_limit:
                        break

        # 3. Teardown
        stream.stop_stream()
        stream.close()

        # 4. Write WAV to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(
                    self.audio_interface.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.samplerate)
                wf.writeframes(b''.join(frames))

            # 5. Transcribe
            segments, _ = self.whisper_model.transcribe(tmp.name, beam_size=5)

        # 6. Aggregate and return
        return " ".join(seg.text for seg in segments)