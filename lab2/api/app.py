import torch
import requests
import time
import shutil
import numpy as np
from transformers import pipeline
import subprocess
import os
from flask import Flask, send_from_directory
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)


captions = {}
stations = {
    "SR P1": {"bitrate": 64000, "url": "https://http-live.sr.se/p1-mp3-64"},
    "SR P2": {"bitrate": 64000, "url": "https://http-live.sr.se/p2-mp3-64"},
    "SR P3": {"bitrate": 64000, "url": "https://http-live.sr.se/p3-mp3-64"},
    "SR P4 Stockholm": {
        "bitrate": 64000,
        "url": "https://http-live.sr.se/p4stockholm-mp3-64",
    },
    "Europa Plus": {
        "bitrate": 128000,
        "url": "https://ep128.hostingradio.ru:8030/ep128",
    },
}

for station in stations:
    captions[station] = ""


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
        "-loglevel",
        "quiet",
        "-ar",
        str(sr),
        "-",
    ]
    # fmt: on
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        return None

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def get_stream(url):
    return requests.get(url, stream=True)


def detect_gpu():
    if not torch.cuda.is_available():
        print("No GPU device found")
        exit()

    print(f"Found {torch.cuda.device_count()} GPU device(s)")
    print(f"Using {torch.cuda.get_device_name(0)}")


def ffmpeg_worker(station):
    station_info = stations[station]
    station_dir = f"/root/scalable/scalable-ml/lab2/api/radios/{station}/"

    shutil.rmtree(station_dir, ignore_errors=True)
    os.makedirs(station_dir)

    cmd = f'/usr/bin/ffmpeg -i {station_info["url"]} -c copy -map 0 -segment_time 00:00:05 -f segment -reset_timestamps 1 -loglevel quiet "{station_dir}/%d.mp3"'
    subprocess.run(cmd, shell=True, cwd=station_dir, stdout=subprocess.DEVNULL)


def stream_worker(station):
    global captions

    print(f"Starting stream worker for {station}")
    pipe = pipeline(model="pierrelf/whisper-small-sv", device=0)

    while True:
        all_files = os.listdir(f"radios/{station}")
        if len(all_files) < 15:
            time.sleep(1)
            continue

        # Find second last file in folder
        all_files = [f for f in all_files if f.endswith(".mp3")]
        all_files = [f.split(".")[0] for f in all_files]
        all_files = sorted(all_files, key=lambda x: int(x))
        max_num = int(all_files[-15])

        # load audio
        audio = load_audio(
            f"radios/{station}/{max_num}.mp3", stations[station]["bitrate"]
        )

        if audio is None:
            continue

        # run inference
        text = pipe({"sampling_rate": 48000, "raw": audio})["text"]
        captions[station] = text

        all_files = os.listdir(f"radios/{station}")
        # remove file and every other file with lower number
        for file in all_files:
            num = int(file.split(".")[0])
            if num <= max_num:
                os.remove(f"radios/{station}/{file}")


@app.route("/")
def root():
    return send_from_directory("static", "index.html")


@app.route("/radio/<radio>")
def radio(radio):
    if radio not in stations:
        return f"Unknown station {radio}"

    return captions[radio]


@app.route("/<path:path>")
def index(path):
    return send_from_directory("static", path)


if __name__ == "__main__":
    detect_gpu()

    for station in stations.keys():
        t = threading.Thread(target=ffmpeg_worker, args=(station,))
        t.start()

        time.sleep(1)

        t = threading.Thread(target=stream_worker, args=(station,))
        t.start()

    app.run(host="0.0.0.0", port=8080)
