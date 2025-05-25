import logging
import io
import sys

utf8_stream = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="backslashreplace")
handler = logging.StreamHandler(utf8_stream)
handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
logging.root.handlers.clear()
logging.root.addHandler(handler)
logging.root.setLevel(logging.DEBUG)
from backend_engine import BackendEngine

def main():
    # Initialize the backend engine
    backend_engine = BackendEngine()

    # Start the backend engine
    backend_engine.start()

if __name__ == "__main__":
    main()