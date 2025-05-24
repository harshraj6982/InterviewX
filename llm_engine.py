#!/usr/bin/env python3
from queue import Empty, Queue
import multiprocessing
import re
import threading
from typing import Callable, List

from llama_cpp import Llama


class LLMConfig:
    # Path to your local GGUF model
    model_path: str = r"./Qwen3-4B-Q4_K_M.gguf"
    n_ctx: int = 6144
    n_gpu_layers: int = -1
    verbose: bool = False
    use_mmap: bool = True

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
        "At the end of interview, give a strong feedback to user in one long paragraph so the that they can improve."
    )


class LlmEngine:
    """
    Runs in its own process, listens on input_queue for user text,
    streams tokens into output_queue, one request at a time.
    """
    EXIT_SIGNAL = "__EXIT__"
    RESPONSE_END = "__END__"

    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        output_queue: multiprocessing.Queue,
    ) -> None:
        self.input_queue = input_queue
        self.output_queue = output_queue

        # Concurrency controls
        self._submit_lock = threading.Lock()
        self._generating_event = threading.Event()

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

    def run(self) -> None:
        """
        Loop forever, processing one input at a time from input_queue.
        """
        while True:
            msg = self.input_queue.get()
            print(f"LLM engine received: {msg}")
            if msg == self.EXIT_SIGNAL:
                break

            if self.is_generating:
                # refuse overlap
                self.output_queue.put("[Wait: still generating...]\n")
                self.output_queue.put(self.RESPONSE_END)
                continue

            # callback pushes tokens into output_queue
            def write_output(tok: str) -> None:
                self.output_queue.put(tok)
                # print(tok, end="", flush=True)

            # process
            self.handle_user_submission(msg, write_output)

        # clean shutdown
        self.output_queue.put(self.EXIT_SIGNAL)


def run_llm_engine(
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
) -> None:
    LlmEngine(input_queue, output_queue).run()


def run_consumer(output_queue: multiprocessing.Queue, send_to_tts_func: Callable) -> None:
    consumer = BufferedPrintConsumer(output_queue, send_to_tts_func)
    while True:
        # consumer.consume_response()
        consumer.consume_response_string()


class BufferedPrintConsumer:
    """
    Reads tokens from an output_queue, buffers until </think>,
    then either prints (consume_response) or collects sentences (consume_response_string).
    """

    def __init__(self, llm_output_queue: multiprocessing.Queue, send_to_tts_func: Callable) -> None:
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

    # def consume_response_string(self) -> None:
    #     """
    #     Reads tokens from the queue, buffers until </think>,
    #     then starts collecting complete sentences using add_text().
    #     Returns the list of sentences.
    #     """
    #     sentences = []
    #     while True:
    #         tok = self.queue.get()
    #         if tok == LlmEngine.EXIT_SIGNAL:
    #             print("\n[LLM process exited]")
    #             break

    #         if tok == LlmEngine.RESPONSE_END:
    #             tail = self.flush()
    #             if tail:
    #                 print("****",tail)
    #                 sentences.append(tail)
    #             self._buffer = ""
    #             self._flushing = False
    #             break

    #         if not self._flushing:
    #             self._buffer += tok
    #             idx = self._buffer.find(self._think_tag)
    #             if idx != -1:
    #                 start = idx + len(self._think_tag)
    #                 tail = self._buffer[start:]
    #                 if tail:
    #                     new_sentences = self.add_text(tail)
    #                     for s in new_sentences:
    #                         print("######",s)
    #                     sentences.extend(new_sentences)
    #                 self._flushing = True
    #                 self._buffer = ""
    #         else:
    #             new_sentences = self.add_text(tok)
    #             for s in new_sentences:
    #                 print("/////",s)
    #             sentences.extend(new_sentences)

    #     print("----",sentences)
    #     print(len(sentences))

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
                    # sentences.append(tail)
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
                            # print("######", s)
                            sentences_queue.put(s)
                        # sentences.extend(new_sentences)
                    self._flushing = True
                    self._buffer = ""
            else:
                new_sentences = self.add_text(tok)
                for s in new_sentences:
                    # print("/////", s)
                    sentences_queue.put(s)
                # sentences.extend(new_sentences)

        # print("----", sentences)
        # print(len(sentences))

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
    input_queue: multiprocessing.Queue = multiprocessing.Queue()
    output_queue: multiprocessing.Queue = multiprocessing.Queue()

    llm_proc = multiprocessing.Process(
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