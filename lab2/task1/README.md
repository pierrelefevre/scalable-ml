# ID2223 Scalable ML - Lab 2, Task 1 - Fine tuning whisper
![man whispering swedish at computer](../assets/task1.png)
by Emil Karlsson, Pierre Le Fevre

## Task
Fine tune the OpenAI Whisper model with the Mozilla swedish dataset.

## Training

## Frontend
The frontend is divided in three parts:

### Live transcription

The application is hosted on Hugging Face and available at [Whisper](https://huggingface.co/spaces/pierrelf/whisper)

### Captions from file

The application is hosted on Hugging Face and available at [Whisper](https://huggingface.co/spaces/pierrelf/whisper-live)

###
exposes a way to listen to Sveriges Radio by reading live transcripts of the radio.

## Results
### Whisper small
{'eval_loss': 0.3259948492050171, 'eval_wer': 117.89411416740609, 'eval_runtime': 3432.6219, 'eval_samples_per_second': 1.477, 'eval_steps_per_second': 0.185}

### Whisper medium
{'eval_loss': 0.2513550817966461, 'eval_wer': 99.79296066252587, 'eval_runtime': 1774.9156, 'eval_samples_per_second': 2.856, 'eval_steps_per_second': 0.357, 'epoch': 5.17}

## Discussion

### Troubles with XXX

## Conclusion
