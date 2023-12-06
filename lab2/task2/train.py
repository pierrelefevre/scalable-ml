import os
import huggingface_hub
import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from dotenv import load_dotenv
import evaluate
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


def get_voice():
    try:
        common_voice = DatasetDict.load_from_disk("common_voice_sv")
        return common_voice
    except:
        print("No dataset found, please run data_preparation.py")
        exit()


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


# ============================================


detect_gpu()

login_to_huggingface_hub()

print("Loading pretrained tools...")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-medium", language="sv", task="transcribe"
)
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-medium", language="sv", task="transcribe"
)

print("Loading dataset...")
common_voice = get_voice()


# Data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Evaluate
metric = evaluate.load("wer")

# Load pretrained checkpoint
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
# pierrelf/whisper-medium-sv
# Define training configuration
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-medium-sv",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

# Launch training
trainer.train()

# Save model
kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
    "dataset_args": "config: sv, split: test",
    "language": "swe",
    "model_name": "Whisper Swedish fine tuned",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-medium",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
}
trainer.push_to_hub(**kwargs)
trainer.save_model(f"{training_args.output_dir}-finetuned")
