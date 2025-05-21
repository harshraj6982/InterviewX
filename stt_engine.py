from __future__ import annotations
import contextlib
import datetime as dt
import logging
import sys
import threading
import urllib.request
import wave
from collections import deque
from pathlib import Path
from typing import Deque, List, Callable

from config import CoreFlags, STTConfig
import mediapipe as mp
import numpy as np
import pyaudio
import webrtcvad
from mediapipe.tasks.python.audio.audio_classifier import AudioClassifier, AudioClassifierOptions
from mediapipe.tasks.python.components.containers.audio_data import AudioData
from pywhispercpp.model import Model


class STTManager:
    """
    Offline STT using pywhispercpp with VAD + MediaPipe gating.
    Records only when human speech is detected, stops after silence.
    """

    # Audio & VAD settings
    RATE: int = STTConfig.RATE          # 16 kHz
    CHANNELS: int = STTConfig.CHANNELS
    FORMAT = pyaudio.paInt16     # 16-bit PCM
    FRAME_MS: int = STTConfig.FRAME_MS           # frame duration for VAD
    FRAME_LEN: int = STTConfig.FRAME_LEN
    SILENCE_TIMEOUT_MS: int = STTConfig.SILENCE_TIMEOUT_MS  # 2 s silence to stop

    # MediaPipe classifier settings
    CLASSIFIER_WINDOW_MS: int = STTConfig.CLASSIFIER_WINDOW_MS
    CLASSIFIER_THRESHOLD: float = STTConfig.CLASSIFIER_THRESHOLD

    # MediaPipe model download
    MODEL_URL: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "audio_classifier/yamnet/float32/1/yamnet.tflite"
    )
    MODEL_PATH: Path = Path(__file__).parent / STTConfig.MODEL_NAME

    def __init__(self) -> None:
        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # Whisper backend
        self.whisper_model = Model(
            # model="small",
            model=STTConfig.WHISPER_MODEL,
            params_sampling_strategy=0,
            print_progress=False,
            print_realtime=False,
        )

        # Ensure MediaPipe model is present
        self.ensure_model(self.MODEL_PATH, self.MODEL_URL)

        # MediaPipe AudioClassifier
        base_opts = mp.tasks.BaseOptions(model_asset_path=str(self.MODEL_PATH))
        clf_opts = AudioClassifierOptions(
            base_options=base_opts,
            max_results=1,
            score_threshold=self.CLASSIFIER_THRESHOLD,
        )
        self.classifier = AudioClassifier.create_from_options(clf_opts)

        # VAD
        self.vad = webrtcvad.Vad(1)

        # PyAudio
        self.audio_interface = pyaudio.PyAudio()
        self._listening = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @staticmethod
    def ensure_model(path: Path, url: str) -> None:
        if path.exists():
            return
        logging.info("Model not found → downloading: %s", url)
        try:
            urllib.request.urlretrieve(url, path)
            logging.info("Model saved to %s", path)
        except Exception as exc:
            logging.error("Model download failed: %s", exc)
            sys.exit(1)

    @staticmethod
    def bytes_to_float32_pcm(data: bytes) -> np.ndarray:
        pcm16 = np.frombuffer(data, dtype=np.int16)
        return pcm16.astype(np.float32) / 32768.0

    @staticmethod
    def iso_timestamp() -> str:
        return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    def save_wave(self, frames: List[bytes], out_dir: Path = Path(".")) -> Path:
        filename = out_dir / f"{self.iso_timestamp()}.wav"
        with contextlib.closing(wave.open(str(filename), "wb")) as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio_interface.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b"".join(frames))
        logging.info("Saved → %s", filename)
        return filename

    def save_and_transcribe(self, frames: List[bytes]) -> str:
        audio_frames = frames.copy()
        filename = self.save_wave(audio_frames)
        segments = self.whisper_model.transcribe(
            str(filename),
            language="en",
            translate=False,
            greedy={"best_of": 1},
            beam_search={"beam_size": 1, "patience": 1},
        )
        transcription = " ".join(s.text.strip() for s in segments)
        logging.info("Transcription: %s", transcription)
        return transcription

    def is_human_voice(self, window: Deque[bytes]) -> bool:
        if not window:
            return False
        clip = b"".join(window)
        audio_data = AudioData.create_from_array(
            self.bytes_to_float32_pcm(clip), self.RATE
        )
        result = self.classifier.classify(audio_data)[0]
        categories = result.classifications and result.classifications[0].categories
        return bool(categories and categories[0].category_name.lower() == "speech")

    def _read_frame(self, stream) -> bytes:
        return stream.read(self.FRAME_LEN, exception_on_overflow=False)

    def listen(self) -> str:
        """
        Record a single speech segment and return its transcription.
        """
        stream = self.audio_interface.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.FRAME_LEN,
        )
        frames: List[bytes] = []
        recent: Deque[bytes] = deque(
            maxlen=self.CLASSIFIER_WINDOW_MS // self.FRAME_MS)
        recording = False
        silence_ms = 0

        while True:
            frame = self._read_frame(stream)
            recent.append(frame)
            if self.vad.is_speech(frame, self.RATE) and (recording or self.is_human_voice(recent)):
                if not recording:
                    recording = True
                    frames.extend(recent)
                    recent.clear()
                    logging.info("▶ Recording started")
                frames.append(frame)
                silence_ms = 0
            elif recording:
                silence_ms += self.FRAME_MS
                if silence_ms >= self.SILENCE_TIMEOUT_MS:
                    break

        stream.stop_stream()
        stream.close()

        # Save & transcribe synchronously
        filename = self.save_wave(frames)
        segments = self.whisper_model.transcribe(
            str(filename),
            language="en",
            translate=False,
            greedy={"best_of": 1},
            beam_search={"beam_size": 1, "patience": 1},
        )
        return " ".join(s.text.strip() for s in segments)

    def start_listening(self, on_segment: Callable[[str], None]) -> None:
        """
        Start continuous segmented listening; for each speech chunk,
        transcribe and call on_segment(text).
        """
        if self._listening:
            return
        self._listening = True

        def _loop():
            stream = self.audio_interface.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.FRAME_LEN,
            )
            while self._listening:
                frames: List[bytes] = []
                recent: Deque[bytes] = deque(
                    maxlen=self.CLASSIFIER_WINDOW_MS // self.FRAME_MS
                )
                silence_ms = 0

                # Wait for speech
                while self._listening:
                    frame = self._read_frame(stream)
                    recent.append(frame)
                    if self.vad.is_speech(frame, self.RATE) and self.is_human_voice(recent):
                        frames.extend(recent)
                        recent.clear()
                        logging.info("▶ Recording started")
                        CoreFlags.is_user_speaking = True
                        break

                # Record until silence
                while self._listening:
                    frame = self._read_frame(stream)
                    frames.append(frame)
                    if self.vad.is_speech(frame, self.RATE):
                        silence_ms = 0
                    else:
                        silence_ms += self.FRAME_MS
                        if silence_ms >= self.SILENCE_TIMEOUT_MS:
                            break

                # Transcribe and callback
                with self._lock:
                    transcription = self.save_and_transcribe(frames)
                CoreFlags.input_via_speech = True
                CoreFlags.is_user_speaking = False
                on_segment(transcription)

            stream.stop_stream()
            stream.close()

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_listening(self) -> None:
        """Stop continuous listening."""
        self._listening = False
        CoreFlags.is_user_speaking = False


if __name__ == "__main__":
    manager = STTManager()
    # Example: synchronous listen
    print(manager.listen())