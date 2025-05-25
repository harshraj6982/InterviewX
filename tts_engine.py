#!/usr/bin/env python3
"""
Debug-enabled cross-platform Text-to-Speech (TTS) engine with stepwise audio
file saving.

Pipeline
--------
  1. Raw TTS output
  2. Pre-playback copy
  3. Post-read (via soundfile)
  4. In-memory chunk streaming for real-time playback
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
import threading
import time
import uuid
from queue import Empty, Queue
from typing import Optional

import sounddevice as sd
import soundfile as sf

# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #


class TTSConfig:
    """
    Central place for runtime-adjustable constants.
    """
    KOKORO_MODEL_PATH: str = "kokoro-v1.0.onnx"
    KOKORO_VOICES_PATH: str = "voices-v1.0.bin"

    SAVE_WAV: bool = False


# --------------------------------------------------------------------------- #
#  macOS: Kokoro primary, 'say' fallback                                      #
# --------------------------------------------------------------------------- #

try:  # Lazy import for non-Darwin hosts
    from kokoro_onnx import SAMPLE_RATE as KOKORO_RATE, Kokoro  # type: ignore
except ModuleNotFoundError:                                     # noqa: PERF203
    # type: ignore[misc,assignment]
    Kokoro = None
    KOKORO_RATE = 44_100

# --------------------------------------------------------------------------- #
#  TTS Engine                                                                 #
# --------------------------------------------------------------------------- #


class TTSEngine:
    """
    Core TTS engine running in its own thread / process.

    • Accepts strings on *text_queue*  
    • Emits `(samplerate, chunk)` tuples on *audio_queue*  
    • Drops every created file name onto *audio_file_queue* for debugging
    """
    EXIT_SIGNAL = "__EXIT__"
    ENABLE_SIGNAL = "__ENABLE__"
    CHUNK_END = (None, None)

    # --------------------------------------------------------------------- #
    #  Construction & voice discovery                                       #
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        text_queue: Queue,
        audio_queue: Queue,
        audio_file_queue: Queue,
        stop_tts_gen_event: threading.Event,  # type: ignore
    ) -> None:
        self.system = platform.system()
        self.text_queue = text_queue
        self.audio_queue = audio_queue
        self.audio_file_queue = audio_file_queue
        self.stop_event = stop_tts_gen_event

        self._lock = threading.Lock()
        self._tts_thread: Optional[threading.Thread] = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.tts_enabled: bool = True
        self.rate: float = 1.0

        self._kokoro: Optional[Kokoro] = None  # lazy-loaded for macOS
        self._voices: list[str] = self._detect_voices()
        self.voice_index: Optional[int] = 0 if self._voices else None
        self.active_voice_name: Optional[str] = None

    # ------------------------------------------------------------------ #
    #  Helper: initialise Kokoro once                                     #
    # ------------------------------------------------------------------ #

    def _ensure_kokoro(self) -> None:
        if self._kokoro is None:
            if Kokoro is None:  # pragma: no cover – handled via fallback
                raise RuntimeError("Kokoro runtime is not available")
            self._kokoro = Kokoro(
                TTSConfig.KOKORO_MODEL_PATH, TTSConfig.KOKORO_VOICES_PATH
            )

    # ------------------------------------------------------------------ #
    #  Voice enumeration                                                 #
    # ------------------------------------------------------------------ #

    def _detect_voices(self) -> list[str]:
        if self.system == "Darwin":
            try:
                self._ensure_kokoro()
                return self._kokoro.get_voices()  # type: ignore[union-attr]
            except Exception as e:  # noqa: BLE001
                self.logger.warning("Kokoro voice discovery failed: %s", e)
            return self._get_macos_voices()

        if self.system == "Windows":
            return self._get_windows_voices()

        return self._get_linux_voices()

    def _get_macos_voices(self) -> list[str]:
        try:
            output = subprocess.check_output(
                ["say", "-v", "?"], text=True, stderr=subprocess.DEVNULL
            )
            return [line.split()[0] for line in output.splitlines() if line.strip()]
        except Exception as e:  # noqa: BLE001
            self.logger.error("Failed to get macOS voices: %s", e)
            return []

    def _get_windows_voices(self) -> list[str]:
        try:
            ps_cmd = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.GetInstalledVoices() | "
                "ForEach-Object { $_.VoiceInfo.Name }"
            )
            output = subprocess.check_output(
                ["PowerShell", "-Command", ps_cmd],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return [line.strip() for line in output.splitlines() if line.strip()]
        except Exception as e:  # noqa: BLE001
            self.logger.error("Failed to get Windows voices: %s", e)
            return []

    def _get_linux_voices(self) -> list[str]:
        try:
            output = subprocess.check_output(
                ["espeak", "--voices"], text=True, stderr=subprocess.DEVNULL
            )
            lines = [l for l in output.splitlines(
            ) if l and not l.lower().startswith("pty")]
            return [line.split()[3] for line in lines[1:]]
        except Exception as e:  # noqa: BLE001
            self.logger.error("Failed to get Linux voices: %s", e)
            return []

    # ------------------------------------------------------------------ #
    #  Core synthesis                                                    #
    # ------------------------------------------------------------------ #

    def _perform_tts(self, text: str) -> None:
        voice = (
            self._voices[self.voice_index]
            if self.system != "Darwin" and self.voice_index is not None
            else None
        )
        raw_filename = f"tts_raw_{uuid.uuid4().hex}.wav"

        try:
            # ======================================================== #
            #  macOS  ➜  primary: Kokoro · fallback: say               #
            # ======================================================== #
            if self.system == "Darwin":
                try:
                    self._ensure_kokoro()
                    kokoro_voice = (
                        # type: ignore[operator,union-attr]
                        "af_heart" if "af_heart" in self._kokoro.get_voices() else voice
                    )
                    samples, samplerate = self._kokoro.create(  # type: ignore[union-attr]
                        text,
                        voice=kokoro_voice,
                        speed=self.rate,
                        lang="en-us",
                    )
                    sf.write(raw_filename, samples, samplerate)
                    self.active_voice_name = kokoro_voice
                except Exception as e:  # noqa: BLE001
                    self.logger.error(
                        "Kokoro synthesis failed: %s – falling back to 'say'", e
                    )
                    intermediate_aiff = raw_filename.replace(".wav", ".aiff")
                    cmd = [
                        "say",
                        "-r",
                        str(int(self.rate * 180)),
                        text,
                        "-o",
                        intermediate_aiff,
                    ]
                    if voice:
                        cmd[1:1] = ["-v", voice]
                    subprocess.run(cmd, check=True)
                    subprocess.run(
                        [
                            "afconvert",
                            intermediate_aiff,
                            "-f",
                            "WAVE",
                            "-d",
                            "LEI16",
                            raw_filename,
                        ],
                        check=True,
                    )
                    os.remove(intermediate_aiff)
                    samplerate = 44_100
            # ======================================================== #
            #  Windows                                                 #
            # ======================================================== #
            elif self.system == "Windows":
                try:
                    import comtypes.client  # type: ignore
                    import win32com.client  # type: ignore

                    dll_path = r"C:\Windows\System32\Speech\Common\sapi.dll"
                    if not os.path.exists(dll_path):
                        raise FileNotFoundError(
                            f"SAPI DLL not found at {dll_path}"
                        )

                    comtypes.client.GetModule(dll_path)
                    from comtypes.gen import SpeechLib  # type: ignore

                    speaker = win32com.client.Dispatch("SAPI.SpVoice")
                    stream = win32com.client.Dispatch("SAPI.SpFileStream")
                    stream.Format.Type = SpeechLib.SAFT44kHz16BitStereo
                    stream.Open(raw_filename, SpeechLib.SSFMCreateForWrite)

                    if voice:
                        for v in speaker.GetVoices():
                            if voice.lower() in v.GetDescription().lower():
                                speaker.Voice = v
                                break

                    speaker.AudioOutputStream = stream
                    speaker.Rate = int(self.rate)
                    speaker.Speak(text)
                    stream.Close()
                    samplerate = 44_100
                except Exception as e:  # noqa: BLE001
                    self.logger.warning(
                        "SAPI COM failed: %s – trying PowerShell", e
                    )
                    safe_text = text.replace("'", "''")
                    ps_script = (
                        "Add-Type -AssemblyName System.Speech; "
                        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    )
                    if voice:
                        ps_script += f"$s.SelectVoice('{voice}'); "
                    ps_script += (
                        f"$s.Rate={int(self.rate)}; "
                        f"$s.SetOutputToWaveFile('{raw_filename}'); "
                        f"$s.Speak('{safe_text}');"
                    )
                    subprocess.run(
                        ["PowerShell", "-Command", ps_script], check=True)
                    samplerate = 44_100
            # ======================================================== #
            #  Linux (espeak)                                          #
            # ======================================================== #
            else:
                cmd = [
                    "espeak",
                    "-s",
                    str(int(self.rate * 175)),
                    text,
                    "-w",
                    raw_filename,
                ]
                if voice:
                    cmd[1:1] = ["-v", voice]
                subprocess.run(cmd, check=True)
                samplerate = 44_100

            # -------------------------------------------------------- #
            #  Post-read & chunking                                    #
            # -------------------------------------------------------- #
            self.audio_file_queue.put(raw_filename)
            data, samplerate_rd = sf.read(raw_filename, dtype="float32")
            samplerate = samplerate_rd
            if TTSConfig.SAVE_WAV:
                sf.write(
                    raw_filename.replace("tts_raw_", "tts_postread_"),
                    data,
                    samplerate,
                )

            chunk_size = 2048
            for start in range(0, len(data), chunk_size):
                if self.stop_event.is_set():
                    break
                self.audio_queue.put((samplerate, data[start: start + chunk_size]))  # noqa: E203
            self.audio_queue.put(self.CHUNK_END)

        except subprocess.CalledProcessError as e:  # noqa: BLE001
            self.logger.error(
                "TTS generation failed (returncode=%s): %s", e.returncode, e
            )
        except Exception as e:  # noqa: BLE001
            self.logger.error("Unexpected error during TTS generation: %s", e)

    # ------------------------------------------------------------------ #
    #  Engine main loop                                                  #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        count = 1
        while True:
            try:
                msg = self.text_queue.get(timeout=0.1)
            except Empty:
                if self.stop_event.is_set() and self.text_queue.empty():
                    self.stop_event.clear()
                continue

            if msg == self.EXIT_SIGNAL:
                break

            if isinstance(msg, str) and msg.startswith(f"{self.ENABLE_SIGNAL}:"):
                try:
                    self.tts_enabled = bool(int(msg.split(":", 1)[1]))
                except ValueError:
                    self.logger.warning("Invalid enable flag: %s", msg)
                continue

            if not self.tts_enabled:
                continue

            if self.stop_event.is_set():
                if self.text_queue.empty():
                    self.stop_event.clear()
                    continue
                while not self.text_queue.empty():
                    try:
                        self.text_queue.get_nowait()
                    except Empty:
                        break
                self.stop_event.clear()
                continue

            if self._tts_thread and self._tts_thread.is_alive():
                self._tts_thread.join()

            with self._lock:
                if not self.stop_event.is_set():
                    self.logger.info("Processing message %d: %s", count, msg)
                    count += 1
                    self._tts_thread = threading.Thread(
                        target=self._perform_tts, args=(msg,)
                    )
                    self._tts_thread.start()


# --------------------------------------------------------------------------- #
#  Speaker                                                                    #
# --------------------------------------------------------------------------- #


class RealSpeaker:
    """Plays `(samplerate, chunk)` tuples from *audio_queue* with stop support."""

    EXIT_SIGNAL = "__EXIT__"
    CHUNK_END = TTSEngine.CHUNK_END

    def __init__(self, audio_queue: Queue, stop_speak_event: threading.Event):  # type: ignore
        self.audio_queue = audio_queue
        self.stop_event = stop_speak_event

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> None:
        stream: Optional[sd.OutputStream] = None

        while True:
            if self.stop_event.is_set():
                try:
                    self.audio_queue.get(timeout=0.1)
                except Empty:
                    self.logger.info("Stop event while idle – clearing flag.")
                    self.stop_event.clear()
                continue

            item = self.audio_queue.get()

            if item == self.EXIT_SIGNAL:
                break
            if isinstance(item, str):
                continue
            if item == self.CHUNK_END:
                if stream:
                    stream.stop()
                    stream.close()
                    stream = None
                continue

            samplerate, chunk = item

            if stream is None:
                channels = chunk.ndim if hasattr(
                    chunk, "ndim") and chunk.ndim > 1 else 1
                try:
                    stream = sd.OutputStream(
                        samplerate=samplerate, channels=channels, dtype="float32"
                    )
                    stream.start()
                except Exception as e:  # noqa: BLE001
                    self.logger.error("Failed to open audio stream: %s", e)
                    continue

            try:
                stream.write(chunk)
            except Exception as e:  # noqa: BLE001
                self.logger.error("Playback failed: %s", e)

        if stream:
            stream.stop()
            stream.close()


# --------------------------------------------------------------------------- #
#  Helpers for multiprocessing                                                #
# --------------------------------------------------------------------------- #


def run_tts_engine(
    text_queue: Queue,
    audio_queue: Queue,
    audio_file_queue: Queue,
    stop_event: threading.Event,  # type: ignore
) -> None:
    TTSEngine(text_queue, audio_queue, audio_file_queue, stop_event).run()


def run_speaker(audio_queue: Queue, stop_event: threading.Event) -> None:  # type: ignore
    RealSpeaker(audio_queue, stop_event).run()