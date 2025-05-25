# import os
# import glob
# import atexit
# import faulthandler; faulthandler.enable()
from backend_engine import BackendEngine

# def delete_wav_files():
#     for file_path in glob.glob("*.wav"):  # adjust path if needed
#         try:
#             os.remove(file_path)
#             print(f"Deleted: {file_path}")
#         except Exception as e:
#             print(f"Failed to delete {file_path}: {e}")

def main():
    # Register the cleanup function
    # atexit.register(delete_wav_files)
    
    # Initialize the backend engine
    backend_engine = BackendEngine()

    # Start the backend engine
    backend_engine.start()

if __name__ == "__main__":
    main()