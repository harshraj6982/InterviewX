# config.py

class LLMConfig:
    # Path to your local GGUF model
    model_path: str = r"Qwen3-8B-Q4_K_M.gguf"
    n_ctx: int = 6144
    # n_ctx: int = 1024
    # n_gpu_layers: int = 20
    n_gpu_layers: int = -1
    verbose: bool = False
    use_mmap: bool = True

    # System prompt for the assistant
    system_prompt: str = (
        "You are a concise, helpful assistant. "
        "Do NOT reveal internal reasoning or chain-of-thought—only output final answers."
    )


class CoreFlags:
    # Auto-submit settings
    enable_auto_submit: bool = True
    auto_submit_idle_timeout: int = 5  # seconds

    # Input/generation flags
    allow_always_input: bool = True
    can_interrupt_generation: bool = True

    is_user_speaking: bool = False  # True while VAD+MediaPipe detect speech
    input_via_speech: bool = False  # True if current input originates from speech

    is_thinking_enabled: bool = False # True to enable the "thinking" state, false to disable /no_think


class STTConfig:
    # Audio & VAD settings
    RATE: int = 16_000           # 16 kHz
    CHANNELS: int = 1
    FRAME_MS: int = 30           # frame duration for VAD
    FRAME_LEN: int = RATE * FRAME_MS // 1000
    SILENCE_TIMEOUT_MS: int = 2_000  # 2 s silence to stop

    # MediaPipe classifier settings
    CLASSIFIER_WINDOW_MS: int = 960
    CLASSIFIER_THRESHOLD: float = 0.5

    # MediaPipe model download
    MODEL_URL: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "audio_classifier/yamnet/float32/1/yamnet.tflite"
    )
    MODEL_NAME: str = "yamnet.tflite"

    # Wisper model settings
    WHISPER_MODEL: str = "large-v3"  # Options: tiny, base, small, small.en, medium, large, large-v3, large-turbo-v3


class TTSConfig:
    default_voice_index: int = 0
    default_rate: int = 150