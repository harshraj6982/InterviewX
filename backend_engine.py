# backend_engine.py

from functools import partial
from multiprocessing import Process, Queue, Event, freeze_support
from llm_engine import run_consumer, run_llm_engine
from tts_engine import run_tts_engine, run_speaker
from stt_engine import run_real_listener, run_stt_engine, run_transcript_reciver


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


class BackendEngine:
    def __init__(self):
        freeze_support()

        # Create queues & stop‐events for TTS
        self.text_queue = Queue()
        self.audio_queue = Queue()
        self.stop_tts_gen_event = Event()
        self.stop_speak_event = Event()

        # Spawn TTS engine & speaker processes
        self.engine_proc = Process(
            target=run_tts_engine,
            args=(self.text_queue, self.audio_queue, self.stop_tts_gen_event),
            daemon=False
        )
        self.speaker_proc = Process(
            target=run_speaker,
            args=(self.audio_queue, self.stop_speak_event),
            daemon=False
        )

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
        self.listener_proc = Process(
            target=run_real_listener,
            args=(self.stt_audio_queue, self.stop_listener_event),
            daemon=False
        )

    def start(self):
        # Start TTS
        self.engine_proc.start()
        self.speaker_proc.start()

        # Start LLM
        self.llm_engine_proc.start()
        self.buffre_llm_genrated_consumer_proc.start()

        # Start STT
        self.stt_engine_proc.start()
        self.transcribe_reciver_proc.start()

        # Start real listener
        self.listener_proc.start()
        print("STT listening process started.")

        # Prime TTS with a single initial sentence
        initial = ["Hello, this is the first test sentence."]
        print(len(initial))
        for s in initial:
            self.text_queue.put(s)

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
        if self.speaker_proc.is_alive():
            self.speaker_proc.terminate()
            self.speaker_proc.join()

        # Shutdown STT listener
        if self.listener_proc.is_alive():
            self.listener_proc.terminate()
            self.listener_proc.join()

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
