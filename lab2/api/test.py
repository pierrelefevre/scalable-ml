import torch
import numpy as np
from transformers import pipeline
from subprocess import CalledProcessError, run


def load_audio(file: str, sr: int):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


# read test.mp3 and print the text
def detect_gpu():
    if not torch.cuda.is_available():
        print("No GPU device found")
        exit()

    print(f"Found {torch.cuda.device_count()} GPU device(s)")
    print(f"Using {torch.cuda.get_device_name(0)}")


detect_gpu()

pipe = pipeline(
    "automatic-speech-recognition", model="pierrelf/whisper-small-sv", device=0
)


stream = load_audio("test.mp3")


output = pipe({"sampling_rate": 48000, "raw": stream})["text"]

print(output)
