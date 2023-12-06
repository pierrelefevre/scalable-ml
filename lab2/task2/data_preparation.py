import os
import huggingface_hub
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
)
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import multiprocessing

load_dotenv()


def detect_gpu():
    if not torch.cuda.is_available():
        print("No GPU device found")
        exit()

    print(f"Found {torch.cuda.device_count()} GPU device(s)")
    print(f"Using {torch.cuda.get_device_name(0)}")


def get_num_cpus():
    return multiprocessing.cpu_count()


def login_to_huggingface_hub():
    print("Login to HuggingFace Hub...")
    huggingface_hub.login(
        token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=True
    )


def load_data():
    common_voice = DatasetDict()

    common_voice["train"] = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "sv-SE",
        split="train+validation",
        token=True,
    )
    common_voice["test"] = load_dataset(
        "mozilla-foundation/common_voice_11_0", "sv-SE", split="test", token=True
    )
    common_voice = common_voice.remove_columns(
        [
            "accent",
            "age",
            "client_id",
            "down_votes",
            "gender",
            "locale",
            "path",
            "segment",
            "up_votes",
        ]
    )
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    return common_voice


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def get_or_process_common_voice():
    try:
        common_voice = DatasetDict.load_from_disk("common_voice_sv")
    except:
        common_voice = None
    if common_voice:
        print("Dataset loaded from cache")
    else:
        print("Dataset not found in cache, loading from source...")
        dataset = load_data()
        print(dataset)

        # Preprocess dataset
        common_voice = dataset.map(
            prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=8
        )

        # Write common_voice to disk
        common_voice.save_to_disk("common_voice_sv")
        print("Dataset saved to cache")
    return common_voice


detect_gpu()

login_to_huggingface_hub()

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-medium", language="sv", task="transcribe"
)
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-medium", language="sv", task="transcribe"
)

print("Loading dataset...")
common_voice = get_or_process_common_voice()
