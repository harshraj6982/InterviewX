#!/usr/bin/env python3
"""
stt_engine.py • Offline STT with VAD + MediaPipe + pywhispercpp
--------------------------------------------------------------
• Listens continuously on the microphone.
• Starts recording only when *real* human speech is detected
  (webrtcvad ➞ MediaPipe AudioClassifier).
• Stops after 2 s of silence, then writes <ISO timestamp>.wav and transcribes.
• Auto-downloads 'yamnet.tflite' on first run.

Python ≥ 3.9 • PEP 8 compliant • Graceful error handling.
"""

from __future__ import annotations
from queue import Empty, Queue
import time
from multiprocessing import freeze_support
import contextlib
import datetime as dt
import logging
import sys
from threading import Thread, Event, Lock
import urllib.request
import wave
from collections import deque
from pathlib import Path
from typing import Deque, List, Callable

import mediapipe as mp
import numpy as np
import pyaudio
import webrtcvad
from mediapipe.tasks.python.audio.audio_classifier import (
    AudioClassifier,
    AudioClassifierOptions,
)
from mediapipe.tasks.python.components.containers.audio_data import AudioData
from pywhispercpp.model import Model

import os

# Determine base path for bundled resources (PyInstaller or normal run)
if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS
else:
    # __file__ refers to this config.py location
    base_path = os.path.abspath(os.path.dirname(__file__))


class STTConfig:
    """
    Configuration for STT and audio settings, adjusted for PyInstaller bundles.
    """
    # Audio settings
    RATE: int = 16_000           # 16 kHz
    CHANNELS: int = 1
    FRAME_MS: int = 30           # frame duration for VAD
    FRAME_LEN: int = RATE * FRAME_MS // 1000
    SILENCE_TIMEOUT_MS: int = 2_000  # 2 s silence to stop

    # MediaPipe classifier settings
    CLASSIFIER_WINDOW_MS: int = 960
    CLASSIFIER_THRESHOLD: float = 0.5

    # MediaPipe model
    MODEL_NAME: str = str(Path(base_path) / "yamnet.tflite")

    # Whisper model settings
    # Options: tiny, base, small, small.en, medium, large, large-v3, large-turbo-v3
    WHISPER_MODEL: str = "medium"  # or "small", "large", etc.
    WHISPER_MODEL_PATH: str = str(Path(base_path) / "ggml-small.bin")

    # Delay before printing transcripts (sliding window)
    PRINT_DELAY_MS: int = 5_000   # 5 seconds

    SAVE_WAV: bool = False  # Save WAV files after transcription


class RealListener:
    """
    Captures raw PCM frames from the system microphone and
    enqueues them into an audio_queue for STTEngine to consume.
    """

    def __init__(self) -> None:
        self.audio_interface = pyaudio.PyAudio()

    def run(
        self,
        audio_queue: Queue,
        stop_event: Event,  # type: ignore
    ) -> None:
        stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=STTConfig.CHANNELS,
            rate=STTConfig.RATE,
            input=True,
            frames_per_buffer=STTConfig.FRAME_LEN,
        )
        try:
            while True:
                if stop_event.is_set():
                    time.sleep(0.1)  # Mute: wait until unmuted
                    continue

                frame = stream.read(
                    STTConfig.FRAME_LEN, exception_on_overflow=False
                )
                audio_queue.put(frame)

        except OSError as e:
            logging.error("Microphone input failed: %s", e)

        finally:
            stream.stop_stream()
            stream.close()
            self.audio_interface.terminate()


class TranscriptReciver:
    """
    Receives transcriptions from STTEngine and prints them to the console.
    """

    def __init__(self, transcript_queue: Queue[List], send_reanscript_to_llm: Callable) -> None:
        self.transcript_queue = transcript_queue
        self.send_reanscript_to_llm = send_reanscript_to_llm

    def run(self) -> None:
        while True:
            try:
                transcript_list = self.transcript_queue.get(timeout=0.1)
            except Empty:
                continue
            if transcript_list is not None:
                transcript = " ".join(transcript_list)
            self.send_reanscript_to_llm(transcript)
            print(transcript)


class STTEngine:
    """
    Consumes raw PCM frames from audio_queue, applies VAD + MediaPipe gating,
    saves WAVs, transcribes via Whisper, and prints transcripts with a 5 s
    sliding-window delay.
    """

    def __init__(self) -> None:
        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # Whisper backend
        self.whisper_model = Model(
            model=STTConfig.WHISPER_MODEL_PATH,
            # model=STTConfig.WHISPER_MODEL,
            params_sampling_strategy=0,
            print_progress=False,
            print_realtime=False,
        )

        # Ensure MediaPipe model is present
        self.MODEL_PATH = Path(__file__).parent / STTConfig.MODEL_NAME

        # MediaPipe AudioClassifier
        base_opts = mp.tasks.BaseOptions(model_asset_path=str(self.MODEL_PATH))
        clf_opts = AudioClassifierOptions(
            base_options=base_opts,
            max_results=1,
            score_threshold=STTConfig.CLASSIFIER_THRESHOLD,
        )
        self.classifier = AudioClassifier.create_from_options(clf_opts)

        # VAD
        self.vad = webrtcvad.Vad(1)

        # For WAV saving
        self.audio_interface = pyaudio.PyAudio()
        self._lock = Lock()

    @staticmethod
    def _bytes_to_float32_pcm(data: bytes) -> np.ndarray:
        pcm16 = np.frombuffer(data, dtype=np.int16)
        return pcm16.astype(np.float32) / 32768.0

    @staticmethod
    def _iso_timestamp() -> str:
        return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    def _save_wave(self, frames: List[bytes], out_dir: Path = Path(".")) -> Path:
        filename = out_dir / f"{self._iso_timestamp()}.wav"
        with contextlib.closing(wave.open(str(filename), "wb")) as wf:
            wf.setnchannels(STTConfig.CHANNELS)
            wf.setsampwidth(
                self.audio_interface.get_sample_size(pyaudio.paInt16)
            )
            wf.setframerate(STTConfig.RATE)
            wf.writeframes(b"".join(frames))
        logging.info("Saved → %s", filename)
        return filename

    def _save_and_transcribe(self, frames: List[bytes]) -> str:
        audio_frames = frames.copy()
        filename = self._save_wave(audio_frames)
        segments = self.whisper_model.transcribe(
            str(filename),
            language="en",
            translate=False,
            greedy={"best_of": 1},
            beam_search={"beam_size": 1, "patience": 1},
        )
        text = " ".join(s.text.strip() for s in segments)
        logging.info("Transcription: %s", text)
        if not STTConfig.SAVE_WAV:
            try:
                os.remove(filename)
            except OSError:
                self.logger.warning("Could not delete %s", filename)
        return text

    def _is_human_voice(self, window: Deque[bytes]) -> bool:
        if not window:
            return False
        clip = b"".join(window)
        audio_data = AudioData.create_from_array(
            self._bytes_to_float32_pcm(clip), STTConfig.RATE
        )
        result = self.classifier.classify(audio_data)[0]
        cats = (
            result.classifications and result.classifications[0].categories) or []
        # Scan all categories for speech above threshold
        for cat in cats:
            if cat.category_name.lower() == "speech" and cat.score >= STTConfig.CLASSIFIER_THRESHOLD:
                return True
        return False

    def run(
        self,
        audio_queue: Queue[bytes],
        transcript_queue: Queue[List],
        stop_event: Event,  # type: ignore
        stop_tts_gen_event: Event,  # type: ignore
        stop_speak_event: Event,  # type: ignore
    ) -> None:
        pending: list[str] = []
        last_received = 0.0
        delay_s = STTConfig.PRINT_DELAY_MS / 1000.0

        segment_frames: list[bytes] = []
        recent: Deque[bytes] = deque(
            maxlen=STTConfig.CLASSIFIER_WINDOW_MS // STTConfig.FRAME_MS
        )
        recording = False
        silence_ms = 0

        MIN_SEGMENT_FRAMES = (750 // STTConfig.FRAME_MS)  # drop <300ms

        while not stop_event.is_set():
            try:
                frame = audio_queue.get(timeout=0.1)
            except Empty:
                continue

            # VAD + MediaPipe gating on every frame
            recent.append(frame)
            vad_flag = self.vad.is_speech(frame, STTConfig.RATE)
            class_flag = self._is_human_voice(recent)

            if vad_flag and class_flag:
                # both signals must be true to start/continue
                if not recording:
                    #######################
                    stop_tts_gen_event.set()
                    stop_speak_event.set()
                    #######################
                    recording = True
                    segment_frames.extend(recent)
                    recent.clear()
                    logging.info("▶ Recording started")
                segment_frames.append(frame)
                silence_ms = 0
                if pending:
                    last_received = time.time()
            elif recording:
                # accumulate silence while recording
                silence_ms += STTConfig.FRAME_MS
                if silence_ms >= STTConfig.SILENCE_TIMEOUT_MS:
                    # Only transcribe if segment is long enough
                    if len(segment_frames) >= MIN_SEGMENT_FRAMES:
                        transcription = self._save_and_transcribe(
                            segment_frames)
                        # discard empty or leading-'[' transcripts
                        trimmed = transcription.strip()
                        if trimmed and not trimmed.startswith('['):
                            pending.append(transcription)
                            last_received = time.time()
                    # reset for next segment
                    segment_frames.clear()
                    recent.clear()
                    recording = False
                    silence_ms = 0

            # Sliding-window print
            if pending and (time.time() - last_received) >= delay_s:
                ###############################
                stop_tts_gen_event.clear()
                stop_speak_event.clear()
                time.sleep(0.5)
                ###############################
                transcript_queue.put(pending)
                for txt in pending:
                    print(txt)
                pending.clear()

        # Flush any remaining on shutdown
        for txt in pending:
            print(txt)
        self.audio_interface.terminate()


# --- Top-level spawn functions, so we never pickle un-pickleable state -------

# type: ignore
def run_real_listener(audio_queue: Queue[bytes], stop_event: Event) -> None:
    listener = RealListener()
    listener.run(audio_queue, stop_event)


def run_stt_engine(
    audio_queue: Queue[bytes],
    transcript_queue: Queue[List],
    stop_event: Event,  # type: ignore
    stop_tts_gen_event: Event,  # type: ignore
    stop_speak_event: Event,  # type: ignore
) -> None:
    engine = STTEngine()
    engine.run(audio_queue, transcript_queue, stop_event,
               stop_tts_gen_event, stop_speak_event)


def run_transcript_reciver(transcript_queue: Queue[List], send_reanscript_to_llm: Callable) -> None:
    reciver = TranscriptReciver(transcript_queue, send_reanscript_to_llm)
    reciver.run()


if __name__ == "__main__":
    freeze_support()

    # IPC primitives
    audio_queue: Queue[bytes] = Queue()
    transcript_queue: Queue[List] = Queue()
    stop_listener_event: Event = Event()    # type: ignore
    stop_engine_event: Event = Event()      # type: ignore

    listener_proc = Thread(
        target=run_real_listener,
        args=(audio_queue, stop_listener_event),
        daemon=False,
    )
    engine_proc = Thread(
        target=run_stt_engine,
        args=(audio_queue, transcript_queue, stop_engine_event),
        daemon=False,
    )
    transcript_proc = Thread(
        target=run_transcript_reciver,
        args=(transcript_queue,),
        daemon=False,
    )

    listener_proc.start()
    engine_proc.start()
    transcript_proc.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_listener_event.set()
        stop_engine_event.set()
        listener_proc.join()
        engine_proc.join()

    print("Shutdown complete.")
