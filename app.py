import tkinter as tk
from typing import Optional

import core
from ui_components import ChatOutput, ChatStatus, ChatInput, DebugOutput


def main():
    root = tk.Tk()
    root.title("Chat with LLM")
    root.geometry("800x600")

    output = ChatOutput(root, filter_think_tag=True)
    status = ChatStatus(root)
    user_input = ChatInput(root, output)
    debug = DebugOutput(root, debug_enabled=True)

    def write_output(token: str) -> None:
        debug.write(token)
        output.stream(token)

    def show_status(text: str) -> None:
        status.show(text)

    def clear_input() -> None:
        user_input.clear()

    def on_submit() -> None:
        txt = user_input.get()
        if not txt:
            return
        if txt.lower() == "exit":
            root.quit()
            return
        output.write(f"\nYou: {txt}\n")
        output.reset_filter()
        core.handle_user_submission(
            txt, write_output, show_status, clear_input)

    auto_submit_timer: Optional[str] = None

    def auto_submit() -> None:
        txt = user_input.get()
        if txt:
            on_submit()

    def reset_timer() -> None:
        nonlocal auto_submit_timer  # type: ignore
        if not core.enable_auto_submit:
            return
        if auto_submit_timer:
            root.after_cancel(auto_submit_timer)
        auto_submit_timer = root.after(
            core.AUTO_SUBMIT_IDLE_TIMEOUT * 1000, auto_submit
        )

    user_input.bind_submit(on_submit)
    user_input.bind_key(reset_timer)

    output.write("Welcome! Type your question below. Type 'exit' to quit.\n")
    user_input.focus()

    root.mainloop()


if __name__ == "__main__":
    main()