# backend_engine.py

import logging
import random
import re
import time
import uuid
from fastapi.responses import HTMLResponse, JSONResponse
import os
import sys
from fastapi.websockets import WebSocketState
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from stt_engine import STTConfig, run_real_listener, run_stt_engine, run_transcript_reciver
from tts_engine import TTSConfig, TTSEngine, run_tts_engine, run_speaker
from llm_engine import run_consumer, run_llm_engine
import numpy as np
from fastapi.staticfiles import StaticFiles
import asyncio
from functools import partial
import json
import threading
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event
import wave
import webbrowser



logger = logging.getLogger(__name__)
# Alias Process to use threading instead of multiprocessing, preserving interface


class Process(Thread):
    def __init__(self, target, args=(), daemon=False):
        super().__init__(target=target, args=args, daemon=daemon)

    def terminate(self):
        # Threads cannot be terminated externally; no-op
        pass


if getattr(sys, 'frozen', False):
    # Running in PyInstaller bundle
    base_path = sys._MEIPASS
else:
    # Running in normal Python environment
    base_path = os.path.abspath(".")

static_dir = os.path.join(base_path, "static")
assets_path = os.path.join(static_dir, "assets")


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
    text_queue.put(text)



def wait_until_ready(path: Path, timeout: float = 5.0, step: float = 0.05) -> None:
    """
    Block until `path` exists **and** its size has stopped growing.
    A minimal size of 44 B (RIFF header) is required.
    """
    start = time.monotonic()
    last_size = -1
    while time.monotonic() - start < timeout:
        if not path.exists():
            time.sleep(step)
            continue
        size = path.stat().st_size
        if size == last_size and size > 44:
            return
        last_size = size
        time.sleep(step)
    raise TimeoutError(f"{path} not ready after {timeout:.1f}s")


async def safe_send(ws: WebSocket, data, *, binary: bool = False) -> None:
    """Guarded send that raises WebSocketDisconnect if the socket is no longer open."""
    if ws.client_state != WebSocketState.CONNECTED:
        raise WebSocketDisconnect(code=1000)
    if binary:
        await ws.send_bytes(data)
    else:
        await ws.send_text(data)

AUDIO_PATTERNS = (
    re.compile(r"tts_raw_[0-9a-f]{32}\.wav$"),
    re.compile(r"tts_postread_[0-9a-f]{32}\.wav$"),
    re.compile(r"\d{8}T\d{6}Z\.wav$"),             # STT ISO stamp
)    


def cleanup_audio(dir_: Path = Path(".")) -> int:
    """
    Delete every WAV file created by this app in *dir_*.
    Returns the number of files removed.
    """
    removed = 0
    for wav in dir_.glob("*.wav"):
        if any(pat.match(wav.name) for pat in AUDIO_PATTERNS):
            try:
                wav.unlink()
                removed += 1
            except OSError as exc:                  # permissions, race, …
                logging.warning("Could not delete %s: %s", wav, exc)
    return removed


class BackendEngine:
    def __init__(self):

        self.call_ended = 0
        self.rank_queue: Queue = Queue()
        self.jobs = {}

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
        self.sessoion_end_event: Event = Event()
        self.call_end_event: Event = Event()
        self.llm_engine_proc = Process(
            target=run_llm_engine,
            args=(self.llm_input_queue, self.llm_output_queue, self.rank_queue, self.sessoion_end_event, self.call_end_event),
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
            "/static", StaticFiles(directory=static_dir), name="static")

        self.app.mount(
            "/assets", StaticFiles(directory=assets_path), name="assets")

        # Serve vite.svg or any root-level static files
        self.app.mount(
            "/favicon.svg", StaticFiles(directory=static_dir), name="favicon-svg")
        self._register_routes()

        self.rank_listener = threading.Thread(
            target=self._rank_listener,
            name="rank-listener",
            daemon=True
        )
        self.rank_listener.start()

    def _rank_listener(self) -> None:
        """Pull { 'rank': int } dicts from rank_queue and file them."""
        while True:
            data = self.rank_queue.get()            # blocks
            try:
                rank = int(data["rank"])
            except Exception:
                logger.warning("Malformed rank payload: %s", data)
                continue
            for job_id, info in self.jobs.items():
                if info["status"] == "processing":
                    self.jobs[job_id] = {"status": "complete", "result": rank}
                    break

    async def process_job(self, job_id: str, timeout: float=30.0) -> None:
        """
        Polls *self.jobs[job_id]* until the rank-listener thread
        flips it to 'complete' or until *timeout* seconds elapse.
        """
        elapsed, step = 0.0, 0.1
        while elapsed < timeout:
            await asyncio.sleep(step)
            elapsed += step
            if self.jobs[job_id]["status"] == "complete":
                return
        # nothing arrived – mark failure
        self.jobs[job_id] = {
            "status": "failed",
            "error": "LLM rank not received within timeout",
        }

    def _register_routes(self):

        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            index_file = Path(static_dir) / "index.html"
            if not index_file.exists():
                # better to fail explicitly with a clear error
                raise HTTPException(
                    status_code=500, detail=f"Index file not found at {index_file!s}")
            content = index_file.read_text(encoding="utf-8")
            return HTMLResponse(content=content, status_code=200)

        @self.app.post("/start-call")
        async def start_call():
            try:

                print("Received data: ", "start call")
                self.call_ended = 0
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid request data: {str(e)}")

            return JSONResponse(content={"message": "Call started status received"}, status_code=200)
        
        @self.app.post("/end-call")
        async def end_call(request: Request):
            try:
                data = await request.json()
                self.call_ended = data.get("status", 0)
                self.call_end_event.set()  # Signal that the call has ended

                # ── stop audio threads ───────────────────────────────
                self.stop_speak_event.set()
                self.stop_tts_gen_event.set()

                # ── run janitor ──────────────────────────────────────
                removed = cleanup_audio(Path("."))   # or Path(base_path)

                # ── graceful shutdown (optional) ─────────────────────
                # Stop child threads / processes that your own stop() already handles
                # self.stop()
                # # Give FastAPI time to send the JSON response, then hard-exit
                # threading.Thread(
                #     target=lambda: (time.sleep(0.2), os._exit(0)),
                #     daemon=True
                # ).start()
            except Exception as e:
                raise HTTPException(status_code=400,
                                    detail=f"Invalid request data: {e}") from e

            return JSONResponse(content={"message": "Call ended status received"}, status_code=200)

        @self.app.post("/report")
        async def report():
            if self.call_ended != 1:
                return JSONResponse(content={"message": "Call not ended yet"}, status_code=400)

            job_id = str(uuid.uuid4())
            self.jobs[job_id] = {"status": "processing", "result": None}

            # Start background task
            asyncio.create_task(self.process_job(job_id))

            return JSONResponse(content={
                "message": "Job started",
                "job_id": job_id,
                "status_url": f"/job-status/{job_id}"
            }, status_code=202)

        @self.app.get("/job-status/{job_id}")
        async def job_status(job_id: str):
            job = self.jobs.get(job_id)
            if not job:
                return JSONResponse(content={"message": "Invalid job ID"}, status_code=404)

            if job["status"] == "complete":
                return JSONResponse(content={"number": job["result"]}, status_code=200)

            return JSONResponse(content={"status": job["status"]}, status_code=202)

        @self.app.post("/audio-buffer-empty")
        async def audio_buffer_empty():
            logger.debug("\n\nAudio buffer empty check\n\n")

            return JSONResponse(
                content={"message": "Audio buffer is empty"}, status_code=200)

        @self.app.post("/end-session")
        async def end_session():
            logger.debug("\n\nEnd session called\n\n")
            self.sessoion_end_event.set()

            return JSONResponse(
                content={"message": "Session ended"}, status_code=200)
        
        # Endpoint for receiving audio from client (STT input)
        @self.app.websocket("/ws/audio_in")
        async def audio_in(websocket: WebSocket):
            await websocket.accept()
            buffer = bytearray()
            # Calculate bytes per 30ms frame: samples * channels * 2 bytes/sample
            frame_bytes = STTConfig.FRAME_LEN * STTConfig.CHANNELS * 2
            try:
                while True:
                    if self.call_ended == 1:
                        print("Call ended, closing audio_in WebSocket.")
                        await websocket.close()
                        break
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


        @self.app.websocket("/ws/audio_file")                    # pylint: disable=unused-variable
        async def audio_file(websocket: WebSocket) -> None:
            """Stream WAV files that appear in `self.audio_file_queue`."""
            await websocket.accept()
            loop = asyncio.get_event_loop()

            def async_send(data, *, binary: bool = False):
                """Thread-friendly wrapper around safe_send()."""
                fut = asyncio.run_coroutine_threadsafe(
                    safe_send(websocket, data, binary=binary), loop
                )
                return fut.result()

            def worker() -> None:                                # runs in daemon thread
                try:
                    while True:
                        # honour “stop speaking” flag
                        if self.stop_speak_event.is_set():
                            with self.audio_file_queue.mutex:
                                self.audio_file_queue.queue.clear()
                            self.stop_speak_event.clear()
                            continue

                        filepath = self.audio_file_queue.get()
                        logger.debug("Processing file: %s", filepath)

                        if filepath == "__EXIT__":
                            break

                        path = Path(filepath)
                        try:
                            wait_until_ready(path)
                        except TimeoutError as exc:
                            logger.warning("%s – skipping", exc)
                            continue

                        try:
                            # ── metadata ───────────────────────────────────────────────
                            with wave.open(str(path), "rb") as wf:   # ← cast to str
                                meta = {
                                    "type": "metadata",
                                    "sample_rate": wf.getframerate(),
                                    "channels": wf.getnchannels(),
                                }
                            async_send(json.dumps(meta))

                            # ── payload ───────────────────────────────────────────────
                            async_send(path.read_bytes(), binary=True)

                             # ── cleanup ───────────────────────────────────────────────
                            if not TTSConfig.SAVE_WAV:
                                path.unlink(missing_ok=True)

                        except (WebSocketDisconnect, RuntimeError):
                            logger.info(
                                "Client disconnected while streaming %s", filepath)
                            break
                        except Exception:                           # noqa: BLE001
                            logger.exception("Error streaming file %s", filepath)

                finally:
                    logger.debug("audio-file worker exiting")

            # ── launch worker thread ──────────────────────────────────────
            t = threading.Thread(target=worker, name="audio-file-worker", daemon=True)
            t.start()

            try:
                while True:                                        # keep coroutine alive
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected (audio_file)")
            finally:
                # clear queue so the next session starts clean
                with self.audio_file_queue.mutex:
                    self.audio_file_queue.queue.clear()
                self.audio_file_queue.put("__EXIT__")              # stop worker
                t.join(timeout=1.0)

                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.close(code=1000)

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
        initial = ["Hello! Welcome. To begin the interview, please say 'start'."]
        print(len(initial))
        for s in initial:
            self.text_queue.put(s)

        # --- Start the FastAPI/uvicorn server in a background thread ---
        def _serve():
            uvicorn.run(self.app, host="0.0.0.0", port=8000,
                        log_level="debug", loop="asyncio")

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

    try:
        # block forever until Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nReceived exit signal. Shutting down...")
        engine.stop()
        sys.exit(0)