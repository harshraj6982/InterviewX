import tkinter as tk
from typing import Optional

from core import ChatManager
from ui_components import ChatOutput, ChatStatus, ChatInput, DebugOutput


def main():
    root = tk.Tk()
    root.title("Chat with LLM")
    root.geometry("800x600")

    # Instantiate the new ChatManager
    chat_manager = ChatManager()

    output = ChatOutput(root, filter_think_tag=True)
    status = ChatStatus(root)
    # Pass chat_manager into ChatInput
    user_input = ChatInput(root, output, chat_manager)
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
        # Use the ChatManager method
        chat_manager.handle_user_submission(
            txt, write_output, show_status, clear_input,
            via_speech=chat_manager.input_via_speech
        )
        chat_manager.input_via_speech = False

    # Only bind manual submit and voice-triggered auto-submit
    user_input.bind_submit(on_submit)
    user_input.bind_auto_submit(on_submit)

    output.write("Welcome! Type your question below. Type 'exit' to quit.\n")
    user_input.focus()

    root.mainloop()


if __name__ == "__main__":
    main()