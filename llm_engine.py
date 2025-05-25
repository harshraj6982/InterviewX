#!/usr/bin/env python3
from threading import Event
from typing import Callable
import json
from asyncio import Event
import logging
from queue import Empty, Queue
import threading
import re
from typing import Callable, List

from llama_cpp import Llama
import sys
import os
from pathlib import Path
logger = logging.getLogger(__name__)

# Determine base path for bundled resources (PyInstaller or normal run)
if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS
else:
    # __file__ refers to this config.py location
    base_path = os.path.abspath(os.path.dirname(__file__))


class LLMConfig:
    """
    Configuration for LLM model loading, adjusted for PyInstaller bundles.
    """
    # Path to your local GGUF model
    model_path: str = str(Path(base_path) / "Qwen3-4B-Q4_K_M.gguf")
    n_ctx: int = 6144
    n_gpu_layers: int = -1
    verbose: bool = False
    use_mmap: bool = True

    # Optional: toggle thinking indicator
    is_thinking_enabled: bool = False

    # System prompt for the assistant
    system_prompt: str = (
        "Act as a Smart Interviewer, who takes interview of freshers Computer Science Graduate for Software Engineering role."
        "First the user will introduce themselves, use that context for asking questions."
        "Then, Ask 10-15 questions and also ask follow up question if necessary."
        "Only ask one question at a time by analyzing the previous response."
        "Use filler words like 'Hmm', 'Interesting', 'Okay' to make the conversation more natural."
        "Don't genrate any text in markdown format, just plain text. Never generate anything with * , ** , or any other special characters."
        "Don't give any feedback or suggestions in between the interview."
        "Don't repeat the question. Move on to the next question if user gives a long answer or don't know the answer."
        "If user asks for feedback or suggestion, then say that you will give feedback at the end of interview."
        "At the end of interview, ask the user to disconnect the call in order to get feedback."
    )

    feedback_system_prompt: str = (
        "You are an AI interview coach. Based on the dialogue provided, generate concise spoken feedback that helps the candidate improve. Don't genrate any text in markdown format, just plain text. Never generate anything with * , ** , or any other special characters."
    )

    ranking_system_prompt: str = (
        "You are an HR evaluator. Using the dialogue provided, output a single JSON object with exactly one key 'rank' whose value is 1 (poor), 2 (average) or 3 (good) indicating the candidate's hireability. Don't genrate any text in markdown format, just plain text. Never generate anything with * , ** , or any other special characters."
    )

# Alias Process to use threading instead of multiprocessing, preserving interface


class Process(threading.Thread):
    def __init__(self, target, args=(), daemon=False):
        super().__init__(target=target, args=args, daemon=daemon)

    def terminate(self):
        # Threads cannot be terminated externally; no-op
        pass

logger = logging.getLogger(__name__)

class LlmEngine:
    """
    Runs in its own process, listens on input_queue for user text,
    streams tokens into output_queue, one request at a time.
    """
    EXIT_SIGNAL = "__EXIT__"
    RESPONSE_END = "__END__"

    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        rank_queue: Queue,
        sessoion_end_event: Event = None,
        call_end_event: Event = None,
    ) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.rank_queue = rank_queue
        self.sessoion_end_event = sessoion_end_event
        self.call_end_event = call_end_event

        # Concurrency controls
        self._submit_lock = threading.Lock()
        self._generating_event = threading.Event()

        if self.sessoion_end_event is None:
            self.sessoion_end_event = threading.Event()
        if self.call_end_event is None:
            self.call_end_event = threading.Event()

        # ── launch watcher that reacts as soon as the flag is set ───────────
        self._watcher_thread = threading.Thread(
            target=self._watch_session_end,
            name="session-watcher",
            daemon=True,
        )
        self._watcher_thread.start()

        # ── launch watcher that handles post‑call processing ───────────────
        self._call_end_thread = threading.Thread(
            target=self._watch_call_end,
            name="call-watcher",
            daemon=True,
        )
        self._call_end_thread.start()

        # Message history
        self._messages = [
            {"role": "system", "content": LLMConfig.system_prompt}]

        # LLM model
        self._llm = Llama(
            model_path=LLMConfig.model_path,
            n_ctx=LLMConfig.n_ctx,
            n_gpu_layers=LLMConfig.n_gpu_layers,
            verbose=LLMConfig.verbose,
            use_mmap=LLMConfig.use_mmap,
        )

    @property
    def is_generating(self) -> bool:
        """Whether the engine is currently generating a response."""
        return self._generating_event.is_set()

    def handle_user_submission(
        self,
        user_text: str,
        write_output: Callable[[str], None],
    ) -> None:
        """
        Streams tokens from the model into write_output.
        Uses an Event to signal generation state.
        """
        with self._submit_lock:
            # Signal start
            self._generating_event.set()
            prompt = (
                user_text
                if LLMConfig.is_thinking_enabled
                else user_text + " /no_think"
            )
            self._messages.append({"role": "user", "content": prompt})

            assistant_text = ""
            try:
                for chunk in self._llm.create_chat_completion(
                    messages=self._messages, stream=True
                ):
                    token = chunk["choices"][0]["delta"].get("content", "")
                    assistant_text += token
                    write_output(token)
            except Exception as exc:
                write_output(f"\n[Generation interrupted: {exc}]\n")
            finally:
                # Record response and clear event
                self._messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
                self._generating_event.clear()
                write_output(self.RESPONSE_END)
    
    # ------------------------------------------------------------------ #
    #  Session‑end handling                                              #
    # ------------------------------------------------------------------ #
    def _dump_and_reset(self) -> None:
        logger.debug("Session ended, printing messages:")
        for m in self._messages:
            print(f"{m['role'].capitalize()}: {m['content']}")
        self._messages = [
            {"role": "system", "content": LLMConfig.system_prompt}]
        logger.debug("Messages cleaned for next session.")
        logger.debug("Fresh session started: %s", self._messages)

    def _watch_session_end(self) -> None:
        while True:
            self.sessoion_end_event.wait()
            with self._submit_lock:
                self._dump_and_reset()
                self.sessoion_end_event.clear()

    # ------------------------------------------------------------------ #
    #  Call‑end handling (feedback + ranking)                            #
    # ------------------------------------------------------------------ #
    def _conversation_as_text(self) -> str:
        """Return conversation lines as annotated text without /no_think."""
        lines = []
        for m in self._messages:
            if m["role"] == "user":
                txt = m["content"].replace("/no_think", "").strip()
                lines.append(f"candidate: {txt}")
            elif m["role"] == "assistant":
                txt = m["content"].replace("/no_think", "").strip()
                lines.append(f"ai_interviewer: {txt}")
        return "\n".join(lines)


    def _stream_feedback(self, convo: str, to_console: bool = False) -> None:
        """Generate and stream audio feedback."""
        messages = [
            {"role": "system", "content": LLMConfig.feedback_system_prompt},
            {"role": "user",   "content": convo},
        ]
        # tag so you can see where feedback begins
        if to_console:
            print("\n[FEEDBACK START]\n", end="", flush=True)

        try:
            for chunk in self._llm.create_chat_completion(messages=messages,
                                                        stream=True):
                token = chunk["choices"][0]["delta"].get("content", "")
                if to_console:
                    # real-time console stream
                    print(token, end="", flush=True)
                else:
                    # keep old behaviour
                    self.output_queue.put(token)
        except Exception as exc:
            msg = f"[Feedback generation error: {exc}]\n"
            if to_console:
                print(msg, end="", flush=True)
            else:
                self.output_queue.put(msg)
        finally:
            if to_console:
                print("\n[FEEDBACK END]\n", flush=True)
            else:
                self.output_queue.put(self.RESPONSE_END)


    def _rank_candidate(self, convo: str) -> None:
        """Generate ranking JSON, parse it, and forward result."""
        messages = [
            {"role": "system", "content": LLMConfig.ranking_system_prompt},
            {"role": "user", "content": convo},
        ]
        try:
            # request non‑stream for easier parse
            resp = self._llm.create_chat_completion(
                messages=messages, stream=False)
            text = resp["choices"][0]["message"]["content"].strip()
            logger.debug("Ranking response text: %s", text)
            # remove code fences first
            cleaned = re.sub(r"```json|```", "", text).strip()
            # try direct parse
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                # fallback: extract first {{...}}
                match = re.search(r"\{[\s\S]*?\}", cleaned)
                if not match:
                    raise
                data = json.loads(match.group(0))
            if "rank" not in data or not isinstance(data["rank"], int):
                raise ValueError(f"Invalid JSON schema: {data}")
        except Exception as exc:
            logger.error("Ranking generation or parse failed: %s", exc)
            data = {"error": str(exc)}
        print("Interview result:", data)
        try:
            self.rank_queue.put({"rank": data["rank"]})
        except Exception:
            logger.error("Failed to put rank data into queue: %s", data)
        # Forward to external route via output_queue
        # self.output_queue.put(f"[JOB_STATUS]{json.dumps(data)}")

    def _watch_call_end(self) -> None:
        while True:
            self.call_end_event.wait()
            with self._submit_lock:
                conversation_text = self._conversation_as_text()
                self._stream_feedback(conversation_text)
                self._rank_candidate(conversation_text)
                self.call_end_event.clear()

    # ------------------------------------------------------------------ #
    #  Main engine loop                                                 #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """
        Loop forever, processing one input at a time from input_queue.
        """
        while True:
            msg = self.input_queue.get()
            print(f"LLM engine received: {msg}")
            if msg == self.EXIT_SIGNAL:
                break
            
            if self.sessoion_end_event.is_set():
                # Print and empty the _messages
                logger.debug("Session ended, printing messages:")
                for m in self._messages:
                    role = m["role"]
                    content = m["content"]
                    print(f"{role.capitalize()}: {content}")

                # clean message
                self._messages = [
                    {"role": "system", "content": LLMConfig.system_prompt}]

                logger.debug("Messages cleaned for next session.")
                logger.debug("Fresh session started.", self._messages)
                self.sessoion_end_event.clear()

            if self.is_generating:
                # refuse overlap
                self.output_queue.put("[Wait: still generating...]\n")
                self.output_queue.put(self.RESPONSE_END)
                continue

            # callback pushes tokens into output_queue
            def write_output(tok: str) -> None:
                self.output_queue.put(tok)

            # process
            self.handle_user_submission(msg, write_output)

        # clean shutdown
        self.output_queue.put(self.EXIT_SIGNAL)


def run_llm_engine(
    input_queue: Queue,
    output_queue: Queue,
    rank_queue: Queue,
    sessoion_end_event: Event = None,
    call_end_event: Event = None,
) -> None:
    LlmEngine(input_queue, output_queue, rank_queue,
              sessoion_end_event, call_end_event).run()


def run_consumer(output_queue: Queue, send_to_tts_func: Callable) -> None:
    consumer = BufferedPrintConsumer(output_queue, send_to_tts_func)
    while True:
        # consumer.consume_response()
        consumer.consume_response_string()


class BufferedPrintConsumer:
    """
    Reads tokens from an output_queue, buffers until </think>,
    then either prints (consume_response) or collects sentences (consume_response_string).
    """

    def __init__(self, llm_output_queue: Queue, send_to_tts_func: Callable) -> None:
        self.llm_output_queue = llm_output_queue
        self._buffer = ""
        self._flushing = False
        self._think_tag = "</think>"
        self.send_to_tts_func = send_to_tts_func

    def consume_response(self) -> None:
        while True:
            tok = self.llm_output_queue.get()
            if tok == LlmEngine.EXIT_SIGNAL:
                print("\n[LLM process exited]")
                break
            if tok == LlmEngine.RESPONSE_END:
                self._buffer = ""
                self._flushing = False
                break

            if not self._flushing:
                self._buffer += tok
                idx = self._buffer.find(self._think_tag)
                if idx != -1:
                    start = idx + len(self._think_tag)
                    tail = self._buffer[start:]
                    if tail:
                        print(tail, end="", flush=True)
                    self._flushing = True
                    self._buffer = ""
            else:
                print(tok, end="", flush=True)

    def consume_response_string(self) -> None:
        """
        Reads tokens from the queue, buffers until </think>,
        then starts collecting complete sentences using add_text().
        Returns the list of sentences.
        """
        # sentences = []
        sentences_queue = Queue()

        while True:
            # Try reading from the sentences_queue (non-blocking)
            try:
                item = sentences_queue.get_nowait()
                self.send_to_tts_func(item)
                print("&&&&&&", item)
            except Empty:
                pass  # no message right now

            # Now handle the self.queue (blocking)
            try:
                tok = self.llm_output_queue.get(timeout=0.1)
            except Empty:
                continue  # no token to process yet

            if tok == LlmEngine.EXIT_SIGNAL:
                print("\n[LLM process exited]")
                break

            if tok == LlmEngine.RESPONSE_END:
                tail = self.flush()
                if tail:
                    # print("****", tail)
                    sentences_queue.put(tail)
                self._buffer = ""
                self._flushing = False
                break

            if not self._flushing:
                self._buffer += tok
                idx = self._buffer.find(self._think_tag)
                if idx != -1:
                    start = idx + len(self._think_tag)
                    tail = self._buffer[start:]
                    if tail:
                        new_sentences = self.add_text(tail)
                        for s in new_sentences:
                            sentences_queue.put(s)
                        self._flushing = True
                        self._buffer = ""
            else:
                new_sentences = self.add_text(tok)
                for s in new_sentences:
                    sentences_queue.put(s)

    def add_text(self, text: str) -> List[str]:
        """
        Add new text and return complete sentences if any.
        """
        self._buffer += text
        parts = re.split(r'(?<=[.?!…])(?=\s+["“”]?[A-Z])', self._buffer)

        if not self._buffer.rstrip().endswith(('.', '?', '!', '…')):
            self._buffer = parts[-1]
            complete_sentences = parts[:-1]
        else:
            complete_sentences = parts
            self._buffer = ""

        return [s.strip() for s in complete_sentences if s.strip()]

    def flush(self) -> str:
        """
        Flush any remaining text in the buffer.
        """
        tail = self._buffer.strip()
        self._buffer = ""
        return tail


def main() -> None:
    """
    Parent process: spawns LlmEngine in another process,
    sends user input over input_queue, and uses BufferedPrintConsumer
    to print each response.
    """
    input_queue: Queue = Queue()
    output_queue: Queue = Queue()

    llm_proc = Process(
        target=run_llm_engine,
        args=(input_queue, output_queue),
        daemon=True,
    )
    llm_proc.start()

    consumer = BufferedPrintConsumer(output_queue)

    print("LlmEngine process started. Type your message (exit to quit):")
    while True:
        user_input = input("\n>> ")
        if user_input.strip().lower() in ("exit", "quit"):
            input_queue.put(LlmEngine.EXIT_SIGNAL)
            break

        input_queue.put(user_input)
        # consumer.consume_response()
        consumer.consume_response_string()

    llm_proc.join()


if __name__ == "__main__":
    main()
