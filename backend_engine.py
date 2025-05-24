# backend_engine.py

import asyncio
from functools import partial
import json
from multiprocessing import Process, Queue, Event, freeze_support
from pathlib import Path
from queue import Empty
import threading
import wave
import webbrowser

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from llm_engine import run_consumer, run_llm_engine
from tts_engine import TTSEngine, run_tts_engine, run_speaker
from stt_engine import STTConfig, run_real_listener, run_stt_engine, run_transcript_reciver
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

def send_transcript_to_llm(transcript: str, llm_input_queue: Queue):
    """
    Send the transcript to the LLM engine.
    """
    user_transcript_str = transcript.strip()
    print("Sending transcript to LLM engine... ", user_transcript_str)
    print(type(user_transcript_str))
    llm_input_queue.put(user_transcript_str)


def send_transcript_to_tts(
    text: str,
    text_queue: Queue,
):
    """
    Send the transcript to the TTS engine.
    """
    print("\n"*2, "Sending transcript to TTS engine... ", text, "\n"*2)
    text_queue.put(text)


class BackendEngine:
    def __init__(self):
        freeze_support()

        # Create queues & stop‐events for TTS
        self.text_queue = Queue()
        self.audio_queue = Queue()
        self.audio_file_queue = Queue()
        self.stop_tts_gen_event = Event()
        self.stop_speak_event = Event()

        # Spawn TTS engine & speaker processes
        self.engine_proc = Process(
            target=run_tts_engine,
            args=(self.text_queue, self.audio_queue,
                  self.audio_file_queue, self.stop_tts_gen_event),
            daemon=False
        )
        # self.speaker_proc = Process(
        #     target=run_speaker,
        #     args=(self.audio_queue, self.stop_speak_event),
        #     daemon=False
        # )

        # STT integration: set up queue & events, start STT engine immediately
        self.stt_audio_queue = Queue()
        self.transcript_queue = Queue()
        self.stop_listener_event = Event()
        self.stop_engine_event = Event()
        self.stt_engine_proc = Process(
            target=run_stt_engine,
            args=(
                self.stt_audio_queue,
                self.transcript_queue,
                self.stop_engine_event,
                self.stop_tts_gen_event,
                self.stop_speak_event
            ),
            daemon=False
        )

        # LLM queues & processes
        self.llm_input_queue: Queue = Queue()
        self.llm_output_queue: Queue = Queue()
        self.llm_engine_proc = Process(
            target=run_llm_engine,
            args=(self.llm_input_queue, self.llm_output_queue),
            daemon=False,
        )

        # Consumer of LLM output → TTS
        self.send_to_tts_func = partial(
            send_transcript_to_tts, text_queue=self.text_queue)
        self.buffre_llm_genrated_consumer_proc = Process(
            target=run_consumer,
            args=(self.llm_output_queue, self.send_to_tts_func),
            daemon=False,
        )

        # STT transcript receiver → LLM
        self.send_to_llm_func = partial(
            send_transcript_to_llm, llm_input_queue=self.llm_input_queue)
        self.transcribe_reciver_proc = Process(
            target=run_transcript_reciver,
            args=(self.transcript_queue, self.send_to_llm_func),
            daemon=False
        )

        # Real‐time listener
        # self.listener_proc = Process(
        #     target=run_real_listener,
        #     args=(self.stt_audio_queue, self.stop_listener_event),
        #     daemon=False
        # )

        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.mount(
            "/static", StaticFiles(directory="static"), name="static")
        self._register_routes()


    # def _register_routes(self):
    #     @self.app.get("/")
    #     async def root():
    #         return {"status": "up"}
        

    def _register_routes(self):

        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            html_path = Path("static/index.html")
            return HTMLResponse(content=html_path.read_text(), status_code=200)
        
        # Endpoint for receiving audio from client (STT input)
        @self.app.websocket("/ws/audio_in")
        async def audio_in(websocket: WebSocket):
            await websocket.accept()
            buffer = bytearray()
            # Calculate bytes per 30ms frame: samples * channels * 2 bytes/sample
            frame_bytes = STTConfig.FRAME_LEN * STTConfig.CHANNELS * 2
            try:
                while True:
                    chunk = await websocket.receive_bytes()  # Receive incoming PCM bytes
                    buffer.extend(chunk)

                    while len(buffer) >= frame_bytes:
                        frame = bytes(buffer[:frame_bytes])
                        del buffer[:frame_bytes]
                        # direct, non-blocking put on an unbounded queue
                        self.stt_audio_queue.put(frame)
                        # print(f"Received audio frame of size {len(frame)} bytes")

                        # Original echo logic replaced by streaming from queue

            except WebSocketDisconnect:
                print("WebSocket disconnected (audio_in)")

        # Endpoint for sending audio to client (TTS output)


        @self.app.websocket("/ws/audio_out")
        async def audio_out(websocket: WebSocket):
            await websocket.accept()
            loop = asyncio.get_event_loop()

            def tts_worker():
                """Stream TTS audio from queue to client in a separate thread"""
                metadata_sent = False

                try:
                    while True:
                        if self.stop_speak_event.is_set():
                            # Drain queue if stop is triggered
                            while not self.audio_queue.empty():
                                try:
                                    self.audio_queue.get_nowait()
                                except Empty:
                                    break
                            self.stop_speak_event.clear()
                            continue

                        item = self.audio_queue.get()

                        if item == "__EXIT__":
                            break

                        if isinstance(item, str):
                            continue

                        if item == TTSEngine.CHUNK_END:
                            continue

                        # Unpack sample rate and pcm chunk
                        samplerate, chunk = item

                        # ── Send metadata once ────────────────────────────────────
                        if not metadata_sent:
                            meta = {
                                "type": "metadata",
                                "sample_rate": samplerate,
                                "channels": chunk.ndim if getattr(chunk, "ndim", 1) > 1 else 1,
                                # <— new field (float-32 little-endian)
                                "format": "f32le"
                            }
                            # schedule metadata send on the asyncio loop
                            asyncio.run_coroutine_threadsafe(
                                websocket.send_text(json.dumps(meta)),
                                loop
                            ).result()
                            metadata_sent = True

                        # Ensure little-endian float32 ordering
                        chunk = chunk.astype("<f4")

                        try:
                            # schedule the send on the main asyncio loop
                            future = asyncio.run_coroutine_threadsafe(
                                websocket.send_bytes(chunk.tobytes()), loop
                            )
                            future.result()
                        except Exception as e:
                            print(f"Failed to send chunk: {e}")
                            break
                except Exception as e:
                    print(f"Streaming error: {e}")

            # start TTS thread
            t = threading.Thread(target=tts_worker, daemon=True)
            t.start()

            try:
                # keep the WS handler alive until client disconnects
                while True:
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                print("WebSocket disconnected (audio_out)")


        # ... inside your FastAPI app class ...
        @self.app.websocket("/ws/audio_file")
        async def audio_file(websocket: WebSocket):
            await websocket.accept()
            loop = asyncio.get_event_loop()

            def file_worker():
                """Stream full WAV files from queue to client."""
                try:
                    while True:
                        # handle stop event: clear queue and wait
                        if self.stop_speak_event.is_set():
                            while not self.audio_file_queue.empty():
                                try:
                                    self.audio_file_queue.get_nowait()
                                except Empty:
                                    break
                            self.stop_speak_event.clear()
                            continue

                        # get next filepath
                        filepath = self.audio_file_queue.get()
                        print(f"Processing file: {filepath}")
                        if filepath == "__EXIT__":
                            break

                        try:
                            # read metadata from WAV
                            with wave.open(filepath, "rb") as wf:
                                sample_rate = wf.getframerate()
                                channels = wf.getnchannels()

                            # read full WAV bytes
                            with open(filepath, "rb") as f:
                                wav_bytes = f.read()

                            # send metadata for decodeAudioData()
                            meta = {
                                "type": "metadata",
                                "sample_rate": sample_rate,
                                "channels": channels
                            }
                            asyncio.run_coroutine_threadsafe(
                                websocket.send_text(json.dumps(meta)),
                                loop
                            ).result()

                            # send entire WAV file in one frame
                            asyncio.run_coroutine_threadsafe(
                                websocket.send_bytes(wav_bytes),
                                loop
                            ).result()

                        except Exception as e:
                            print(f"Error streaming file {filepath}: {e}")

                except Exception as e:
                    print(f"Streaming error: {e}")

            # start file-stream worker
            t = threading.Thread(target=file_worker, daemon=True)
            t.start()

            try:
                # keep connection alive
                while True:
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                print("WebSocket disconnected (audio_file)")


    def start(self):
        # Start TTS
        self.engine_proc.start()
        # self.speaker_proc.start()

        # Start LLM
        self.llm_engine_proc.start()
        self.buffre_llm_genrated_consumer_proc.start()

        # Start STT
        self.stt_engine_proc.start()
        self.transcribe_reciver_proc.start()

        # Start real listener
        # self.listener_proc.start()
        # print("STT listening process started.")

        # Prime TTS with a single initial sentence
        initial = ["Hello, this is the first test sentence."]
        print(len(initial))
        for s in initial:
            self.text_queue.put(s)
            
        # --- Start the FastAPI/uvicorn server in a background thread ---
        def _serve():
            uvicorn.run(self.app, host="0.0.0.0", port=8000, log_level="info", loop="asyncio")

        self.server_thread = threading.Thread(target=_serve, daemon=True)
        self.server_thread.start()

        threading.Timer(2.0, lambda: webbrowser.open(
            "http://localhost:8000")).start()

    def toggle_listening(self):
        """
        Toggles between mute and unmute for the active STT listener.
        """
        if self.stop_listener_event.is_set():
            # Unmute: clear stop signal
            self.stop_listener_event.clear()
            print("STT unmuted.")
            return True
        else:
            # Mute: send stop signal
            self.stop_listener_event.set()
            print("STT muted.")
            return False

    def stop(self):
        # Shutdown TTS
        self.stop_tts_gen_event.set()
        self.stop_speak_event.set()
        if self.engine_proc.is_alive():
            self.engine_proc.terminate()
            self.engine_proc.join()
        # if self.speaker_proc.is_alive():
        #     self.speaker_proc.terminate()
        #     self.speaker_proc.join()

        # Shutdown STT listener
        # if self.listener_proc.is_alive():
        #     self.listener_proc.terminate()
        #     self.listener_proc.join()

        # Shutdown STT engine
        self.stop_engine_event.set()
        if self.stt_engine_proc.is_alive():
            self.stt_engine_proc.terminate()
            self.stt_engine_proc.join()

        if self.transcribe_reciver_proc.is_alive():
            self.transcribe_reciver_proc.terminate()
            self.transcribe_reciver_proc.join()

        # Shutdown LLM engine
        if self.llm_engine_proc.is_alive():
            self.llm_engine_proc.terminate()
            self.llm_engine_proc.join()

        if self.buffre_llm_genrated_consumer_proc.is_alive():
            self.buffre_llm_genrated_consumer_proc.terminate()
            self.buffre_llm_genrated_consumer_proc.join()


if __name__ == "__main__":
    engine = BackendEngine()
    engine.start()
