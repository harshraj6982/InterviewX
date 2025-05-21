import time
import threading
import wave
import numpy as np
import pyaudio
import webrtcvad
from faster_whisper import WhisperModel
from pywhispercpp.model import Model
import os
os.environ["HF_HUB_CACHE"] = r"/Users/harsh/Documents/huggingface_cache"


def ensure_min_duration(samples: np.ndarray, sample_rate: int, min_ms: int = 1000) -> np.ndarray:
    """
    Pad the audio with zeros if it's shorter than min_ms.
    """
    required_len = int((min_ms / 1000) * sample_rate)
    if samples.shape[0] < required_len:
        pad_len = required_len - samples.shape[0]
        samples = np.concatenate(
            [samples, np.zeros(pad_len, dtype=samples.dtype)])
    return samples


class STTManager:
    """
    Offline & live STT using:
      - faster-whisper for full-file transcription
      - pywhispercpp for sub-second live streaming with VAD
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
        # faster-whisper for offline
        self.whisper_model = WhisperModel(
            model_size, compute_type=compute_type)
        self.samplerate = samplerate

        # webrtc VAD
        self.vad = webrtcvad.Vad(vad_mode)
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(self.samplerate * frame_duration_ms / 1000)
        self.bytes_per_sample = 2  # 16-bit PCM
        self.silence_frame_limit = silence_duration_ms // frame_duration_ms

        # audio I/O
        self.audio_interface = pyaudio.PyAudio()

        # lock for thread-safe access
        self._buffer_lock = threading.Lock()

        # live-streaming state
        self._streaming = False
        self._stream_thread = None
        self._append_callback = None
        self._submit_callback = None
        self._silence_timeout = None
        self._last_speech_time = None

        # pywhispercpp for live chunks
        self.live_model = Model(
            "base.en",
            print_realtime=False,
            print_progress=False,
        )

    def _read_frame(self, stream) -> bytes:
        return stream.read(self.frame_size, exception_on_overflow=False)

    def listen(self) -> str:
        """
        Offline: record a chunk around speech + silence, then transcribe end-to-end.
        No temp file: buffer → NumPy → whispercpp.transcribe().
        """
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

        # VAD-driven capture
        while True:
            frame = self._read_frame(stream)
            is_speech = self.vad.is_speech(frame, self.samplerate)

            if is_speech:
                recording = True
                frames.append(frame)
                silence_counter = 0
            elif recording:
                silence_counter += 1
                if silence_counter > self.silence_frame_limit:
                    break

        stream.stop_stream()
        stream.close()

        # raw bytes → int16 numpy
        audio = np.frombuffer(b"".join(frames), dtype=np.int16)

        # pad to ≥1 s
        audio = ensure_min_duration(audio, self.samplerate, min_ms=1000)

        # transcribe with faster-whisper (or switch to live_model if you prefer)
        segments, _ = self.whisper_model.transcribe(
            audio,
            beam_size=5,
            # note: faster-whisper accepts numpy directly
        )
        return " ".join(seg.text for seg in segments)

    # ---------------------------------------------------------------------
    # Live Streaming API
    # ---------------------------------------------------------------------

    def start_stream(self, append_callback, submit_callback, silence_timeout: int = 10):
        """
        append_callback(text)  ➡️ on each new segment
        submit_callback()      ➡️ after silence_timeout seconds of no speech
        """
        if self._streaming:
            return

        self._append_callback = append_callback
        self._submit_callback = submit_callback
        self._silence_timeout = silence_timeout
        self._last_speech_time = time.time()
        self._streaming = True

        self._stream_thread = threading.Thread(
            target=self._stream_worker, daemon=True)
        self._stream_thread.start()

    def stop_stream(self):
        self._streaming = False
        if self._stream_thread:
            self._stream_thread.join()
        self._stream_thread = None

    def _stream_worker(self):
        stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.samplerate,
            input=True,
            frames_per_buffer=self.frame_size,
        )

        buffer = bytearray()

        def on_segment(seg):
            # callback from pywhispercpp
            self._append_callback(seg.text)
            self._last_speech_time = time.time()

        while self._streaming:
            frame = self._read_frame(stream)
            is_speech = self.vad.is_speech(frame, self.samplerate)

            if is_speech:
                with self._buffer_lock:
                    buffer.extend(frame)
                self._last_speech_time = time.time()
            else:
                # silence after speech → flush buffer
                with self._buffer_lock:
                    if buffer:
                        audio_np = np.frombuffer(buffer, dtype=np.int16)
                        audio_np = ensure_min_duration(
                            audio_np, self.samplerate, min_ms=1000
                        )

                        # real-time callback transcription
                        self.live_model.transcribe(
                            audio_np,
                            new_segment_callback=on_segment
                        )
                        buffer.clear()

                # auto‐submit after long silence
                if time.time() - self._last_speech_time > self._silence_timeout:
                    self._submit_callback()
                    # prevent repeat-calls until next speech
                    self._last_speech_time = time.time() + 1e6

        stream.stop_stream()
        stream.close()