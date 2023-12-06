from transformers import GenerationConfig, WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("pierrelf/whisper-small-sv")
generation_config = GenerationConfig.from_pretrained(
    "openai/whisper-base"
)  # if you are using a multilingual model
model.generation_config = generation_config
model.push_to_hub(
    "pierrelf/whisper-small-sv",
)
