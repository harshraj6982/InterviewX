# InterviewX

InterviewX is an AI-powered interview assistant that helps you automate and analyze interviews using cutting-edge speech-to-text, voice processing, and language models. This project brings together multiple Python libraries for real-time audio, transcription, and AI inference. A candidate can give unlimited interviews all without internet. Internet is only required for the first time.

---

## Requirements

- **Python:** 3.11.x (Strictly required)
- **OS Support:** Windows & macOS

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/harshraj6982/InterviewX.git
cd InterviewX
```

---

### 2. Create and Activate a Virtual Environment

#### **For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

#### **For macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Set CMAKE_ARGS (for hardware acceleration)

#### **For Windows:**
```bash
set CMAKE_ARGS="-DGGML_VULKAN=on"
echo %CMAKE_ARGS%
```

#### **For macOS:**
```bash
export CMAKE_ARGS="-DGGML_METAL=on -DLLAMA_METAL=on -DLLAMA_NATIVE=on"
echo $CMAKE_ARGS
```

---

### 4. Install Python Dependencies

#### Common requirements (all platforms):
```bash
pip install fastapi 'uvicorn[standard]'
pip install --upgrade --verbose --force-reinstall --no-cache-dir llama-cpp-python
pip install git+https://github.com/absadiki/pywhispercpp
pip install mediapipe pyaudio webrtcvad sounddevice soundfile pyinstaller
```

#### **For macOS only:**
```bash
pip install kokoro_onnx
pip install "numpy<2" --upgrade
```

#### **For Windows only:**
```bash
pip install pywin32 comtypes
```

---

### 5. Download Required Model Files

Download the following files from the Internet and place them in the root directory of the project:

- `Qwen3-4B-Q4_K_M.gguf`
- `ggml-small.bin`

#### **For macOS only:**
- `kokoro-v1.0.onnx`
- `voices-v1.0.bin`

---

## Running the Project

Simply run:

```bash
python main.py
```

---

## Notes

- **Python version must be 3.11.x** – other versions are not guaranteed to work.
- **Model files** must be downloaded separately (links not provided for copyright reasons). Ensure they are in the root directory.
- For best performance, use the appropriate hardware acceleration flags (`Vulkan` for Windows, `Metal` for macOS).
- If you encounter missing dependencies or errors, ensure you are using the correct virtual environment and Python version.

---

## License

This project is for educational and research purposes. See [LICENSE](LICENSE) for more details.

---

## Contributing

Pull requests and issues are welcome! Feel free to connect with me on [@harshraj_dev](https://x.com/harshraj_dev)
