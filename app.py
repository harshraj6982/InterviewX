# app.py

import tkinter as tk
from backend_engine import BackendEngine


def main() -> None:
    backend = BackendEngine()
    backend.start()

    # Build GUI
    root = tk.Tk()
    root.title("Single Button App")
    root.geometry("300x250")

    def toggle_listening():
        running = backend.toggle_listening()
        if running:
            listen_button.config(text="Stop Listening")
        else:
            listen_button.config(text="Start Listening")

    def on_close() -> None:
        print("Closing application...")
        backend.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    listen_button = tk.Button(
        root,
        text="Start Listening",
        command=toggle_listening
    )
    listen_button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
