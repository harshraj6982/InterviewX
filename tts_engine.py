#!/usr/bin/env python3
"""
Debug-enabled cross-platform Text-to-Speech (TTS) engine with stepwise audio file saving.
Generates and saves audio at each pipeline step:
  1. Raw TTS output
  2. Pre-playback copy
  3. Post-read (via soundfile)
  4. In-memory chunk streaming for real-time playback
"""
import logging
import multiprocessing
import platform
from queue import Empty
import shutil
import subprocess
import threading
import time
import uuid
import os

import sounddevice as sd
import soundfile as sf

# --- New: AppKit-based voice listing ---


def list_macos_voices_appkit() -> list[str]:
    try:
        from AppKit import NSSpeechSynthesizer
        return NSSpeechSynthesizer.availableVoices()
    except Exception:
        return []

# --- New: AppKit-based speaking (direct, no file save) ---


def speak_with_nsspeech(text: str, voice: str, rate: float) -> bool:
    try:
        from AppKit import NSSpeechSynthesizer
        synth = NSSpeechSynthesizer.alloc().initWithVoice_(voice)
        synth.setRate_(rate)
        synth.startSpeakingString_(text)
        while synth.isSpeaking():
            time.sleep(0.1)
        return True
    except Exception:
        return False


class TTSEngine:
    """
    Core TTS engine running in a separate process.
    Listens on a multiprocessing.Queue for text or control messages,
    generates audio files via system TTS, and sends audio chunks to audio queue.
    """
    EXIT_SIGNAL = "__EXIT__"
    ENABLE_SIGNAL = "__ENABLE__"
    CHUNK_END = (None, None)

    def __init__(
        self,
        text_queue: multiprocessing.Queue,
        audio_queue: multiprocessing.Queue,
        audio_file_queue: multiprocessing.Queue,
        stop_tts_gen_event: multiprocessing.Event,  # type: ignore

    ) -> None:
        self.system = platform.system()
        self.text_queue = text_queue
        self.audio_queue = audio_queue
        self.audio_file_queue = audio_file_queue
        self.stop_event = stop_tts_gen_event
        self._lock = threading.Lock()
        self._tts_thread: threading.Thread | None = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.tts_enabled = True
        self.rate = 1

        self.active_voice_name = None   # <--- New: record which voice was actually used
        self._voices = self._detect_voices()
        self.voice_index = 0 if self._voices else None

    def _detect_voices(self) -> list[str]:
        if self.system == "Darwin":
            # Try AppKit first
            voices = list_macos_voices_appkit()
            if voices:
                return voices
            # Fallback to say-based list
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
        except Exception as e:
            self.logger.error("Failed to get macOS voices: %s", e)
            return []

    def _get_windows_voices(self) -> list[str]:
        try:
            ps_cmd = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }"
            )
            output = subprocess.check_output(
                ["PowerShell", "-Command", ps_cmd],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return [line.strip() for line in output.splitlines() if line.strip()]
        except Exception as e:
            self.logger.error("Failed to get Windows voices: %s", e)
            return []

    def _get_linux_voices(self) -> list[str]:
        try:
            output = subprocess.check_output(
                ["espeak", "--voices"], text=True, stderr=subprocess.DEVNULL
            )
            lines = [
                l for l in output.splitlines()
                if l and not l.lower().startswith("pty")
            ]
            return [line.split()[3] for line in lines[1:]]
        except Exception as e:
            self.logger.error("Failed to get Linux voices: %s", e)
            return []

    def _perform_tts(self, text: str) -> None:
        # if self.stop_event.is_set():
        #     return

        voice = "com.apple.speech.synthesis.voice.samantha" if self.system == "Darwin" else (
            self._voices[self.voice_index] if self.voice_index is not None else None
        )
        file_ext = "aiff" if self.system == "Darwin" else "wav"
        raw_filename = f"tts_raw_{uuid.uuid4().hex}.{file_ext}"

        try:
            # generate raw file as before
            if self.system == "Darwin":
                # --- New: try AppKit primary ---
                if voice and speak_with_nsspeech(text, voice, self.rate * 180.0):
                    self.active_voice_name = voice
                    # create an empty wav so downstream logic can chunk
                    raw_filename = f"tts_raw_{uuid.uuid4().hex}.wav"
                    sf.write(raw_filename, [], 44100)
                else:
                    # --- Fallback to say ---
                    intermediate_aiff = raw_filename
                    raw_filename = raw_filename.replace(".aiff", ".wav")
                    cmd = ["say", "-r", str(self.rate), text,
                           "-o", intermediate_aiff]
                    if voice:
                        cmd[1:1] = ["-v", voice]
                    proc = subprocess.Popen(cmd)
                    proc.wait()
                    convert_cmd = ["afconvert", intermediate_aiff,
                                   "-f", "WAVE", "-d", "LEI16", raw_filename]
                    subprocess.run(convert_cmd, check=True)
                    os.remove(intermediate_aiff)
                    self.active_voice_name = voice

            elif self.system == "Windows":
                try:
                    import comtypes.client
                    import win32com.client
                    dll_path = r"C:\Windows\System32\Speech\Common\sapi.dll"
                    if not os.path.exists(dll_path):
                        raise FileNotFoundError(
                            f"SAPI DLL not found at {dll_path}")
                    comtypes.client.GetModule(dll_path)
                    from comtypes.gen import SpeechLib

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
                    speaker.Rate = self.rate
                    speaker.Speak(text)
                    stream.Close()

                except Exception as e:
                    self.logger.warning(
                        "Primary SAPI COM method failed: %s", e)
                    safe_text = text.replace("'", "''")
                    ps_script = (
                        "Add-Type -AssemblyName System.Speech; "
                        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    )
                    if voice:
                        ps_script += f"$s.SelectVoice('{voice}'); "
                    ps_script += (
                        f"$s.Rate={self.rate}; "
                        f"$s.SetOutputToWaveFile('{raw_filename}'); "
                        f"$s.Speak('{safe_text}');"
                    )
                    proc = subprocess.Popen(
                        ["PowerShell", "-Command", ps_script])
                    # while proc.poll() is None:
                    #     if self.stop_event.is_set():
                    #         proc.terminate()
                    #         return

            else:
                cmd = ["espeak", "-s", str(self.rate),
                       text, "-w", raw_filename]
                if voice:
                    cmd[1:1] = ["-v", voice]
                proc = subprocess.Popen(cmd)

            # 1) put raw filename for any external debug
            # self.logger.info("Saved raw TTS output: %s", raw_filename)

            # 3) read and save post-read copy
            self.audio_file_queue.put(raw_filename)
            data, samplerate = sf.read(raw_filename, dtype='float32')
            postread_filename = raw_filename.replace(
                "tts_raw_", "tts_postread_")
            sf.write(postread_filename, data, samplerate)
            self.logger.info("Saved post-read file: %s", postread_filename)

            # 4) split into chunks and stream
            chunk_size = 2048
            for start in range(0, len(data), chunk_size):
                if self.stop_event.is_set():
                    break
                chunk = data[start:start + chunk_size]
                self.audio_queue.put((samplerate, chunk))
                # print(f"Chunk {start // chunk_size + 1}")
            # signal end of this sentence’s stream
            self.audio_queue.put(self.CHUNK_END)

        except subprocess.CalledProcessError as e:
            self.logger.error(
                "TTS generation failed (returncode=%s): %s", e.returncode, e)
        except Exception as e:
            self.logger.error("Unexpected error during TTS generation: %s", e)

    def run(self) -> None:
        count = 1
        while True:
            # — Attempt to get a message with timeout to stay responsive to stop_event
            try:
                # print("Waiting for message...")
                # <-- Changed: add timeout to avoid blocking forever
                msg = self.text_queue.get(timeout=0.1)
                print(f"122233333Received message: {msg}")
            except Empty:
                print("Queue is empty")
                # No new text; if stop_event set and queue empty, clear and continue
                if self.stop_event.is_set():
                    print("Stop event detected. Clearing stop flag.")
                    if self.text_queue.empty():
                        print("Queue is empty, clearing event now")
                        self.stop_event.clear()  # <-- Changed: clear even when idle
                continue

            # --- Old blocking get branch (commented out):
            # msg = self.text_queue.get()
            print("--1")
            if msg == self.EXIT_SIGNAL:
                break
            print("--2")
            if (
                isinstance(msg, str)
                and msg.startswith(f"{self.ENABLE_SIGNAL}:")
            ):
                try:
                    self.tts_enabled = bool(int(msg.split(":", 1)[1]))
                except ValueError:
                    self.logger.warning("Invalid enable flag: %s", msg)
                continue
            print("--3")    
            if not self.tts_enabled:
                continue
            print("--4")
            if self.stop_event.is_set():
                if self.text_queue.empty():
                    self.stop_event.clear()
                    continue

                while not self.text_queue.empty():
                    try:
                        msgg = self.text_queue.get_nowait()
                        print(f" --- Processing message count {count}: {msgg}")
                        count += 1
                    except Empty:
                        break
                self.stop_event.clear()
                continue
                # try:
                #     while self.text:
                #         msgg = self.text_queue.get_nowait()
                #         print(f" --- Processing message count {count}: {msgg}")
                #         count += 1
                # except Empty:
                #     # print("Queue is empty")
                #     self.stop_event.clear()
                #     # pass
            print("--5")
            if self._tts_thread and self._tts_thread.is_alive():
                self._tts_thread.join()
            print("--6")
            with self._lock:
                if not self.stop_event.is_set():
                    print(f"Processing message {count}: {msg}")
                    count += 1
                    self._tts_thread = threading.Thread(
                        target=self._perform_tts, args=(msg,)
                    )
                    self._tts_thread.start()


class RealSpeaker:
    """
    Plays audio chunks from a queue with immediate-stop support.
    """
    EXIT_SIGNAL = "__EXIT__"
    CHUNK_END = TTSEngine.CHUNK_END

    def __init__(
        self,
        audio_queue: multiprocessing.Queue,
        stop_speak_event: multiprocessing.Event  # type: ignore
    ) -> None:
        self.audio_queue = audio_queue
        self.stop_event = stop_speak_event

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> None:
        stream = None

        while True:
            # —— New Idle-state stop check ——
            if self.stop_event.is_set():
                try:
                    self.audio_queue.get(timeout=0.1)
                except Empty:
                    self.logger.info(
                        "Stop event detected while idle. Clearing stop flag."
                    )
                    self.stop_event.clear()
                    # Skip straight to next loop iteration
                continue

            # 1) If stop requested during playback: halt & drain queue fully
            if self.stop_event.is_set():
                self.logger.info(
                    "Stop event detected. Stopping playback and draining queue."
                )

                sd.stop()  # Immediately halt any sounddevice playback

                if stream:
                    stream.stop()
                    stream.close()
                    stream = None

                # Drain until truly empty
                while True:
                    try:
                        self.audio_queue.get_nowait()
                    except Empty:
                        time.sleep(0.05)
                        if self.audio_queue.empty():
                            break

                self.stop_event.clear()
                continue

            # 2) Fetch next queue item
            item = self.audio_queue.get()

            # 3) Handle shutdown signal
            if item == self.EXIT_SIGNAL:
                break

            # 4) Ignore any debug filenames
            if isinstance(item, str):
                continue

            # 5) End-of-sentence marker: close stream
            if item == self.CHUNK_END:
                if stream:
                    stream.stop()
                    stream.close()
                    stream = None
                continue

            # 6) Regular audio chunk: unpack
            samplerate, chunk = item

            # 7) Open stream if first chunk
            if stream is None:
                channels = chunk.ndim if hasattr(
                    chunk, "ndim") and chunk.ndim > 1 else 1
                try:
                    stream = sd.OutputStream(
                        samplerate=samplerate,
                        channels=channels,
                        dtype="float32",
                    )
                    stream.start()
                except Exception as e:
                    self.logger.error("Failed to open audio stream: %s", e)
                    continue

            # 8) Write chunk (may block briefly)
            try:
                stream.write(chunk)
            except Exception as e:
                self.logger.error("Playback failed during write: %s", e)

        # 9) Final cleanup on exit
        if stream:
            stream.stop()
            stream.close()


def run_tts_engine(
    text_queue: multiprocessing.Queue,
    audio_queue: multiprocessing.Queue,
    audio_file_queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event  # type: ignore
) -> None:
    TTSEngine(text_queue, audio_queue, audio_file_queue, stop_event).run()


def run_speaker(
    audio_queue: multiprocessing.Queue,
    stop_event: multiprocessing.Event  # type: ignore
) -> None:
    RealSpeaker(audio_queue, stop_event).run()
