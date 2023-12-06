# ID2223 Scalable ML - Lab 2
![scientist ](assets/lab2.png)
by Emil Karlsson, Pierre Le Fevre

This lab report is split in two parts, one for each task of the lab.

# Task 1 - Fine tuning Whisper
![man whispering at big server](./assets/task1.png)

## Task
Fine tune the OpenAI Whisper model with the Mozilla swedish dataset. Create an innovative application that uses the model (bonus points).

## Training
Cleaning the data and training proved quite simple, however it was time consuming even on our RTX A6000 GPU. We mostly used the code from the Colab notebook provided in the assignment. 

As we noticed the WER for the small model was atrocious, we tried training the medium model too, which ended up being a bit better. However we kept the small model for the Huggingface Spaces applications as they are hosted on CPU only machines.

These are the results from 
### Whisper small
{'eval_loss': 0.3259948492050171, 'eval_wer': 117.89411416740609, 'eval_runtime': 3432.6219, 'eval_samples_per_second': 1.477, 'eval_steps_per_second': 0.185}

### Whisper medium
{'eval_loss': 0.2513550817966461, 'eval_wer': 99.79296066252587, 'eval_runtime': 1774.9156, 'eval_samples_per_second': 2.856, 'eval_steps_per_second': 0.357, 'epoch': 5.17}

## Frontend
The frontend is divided in three parts:

### Live transcription
Using Gradio we were able to 
The application is hosted on Hugging Face and available at [Whisper](https://huggingface.co/spaces/pierrelf/whisper-live)

### Captions from file
The application is hosted on Hugging Face and available at [Whisper](https://huggingface.co/spaces/pierrelf/whisper)

### Radio streaming
Adds live captions to various radio stations. Hosted on a VM with RTX A6000 GPU. The application is available at [Radio](https://radio.vm-app.cloud.cbh.kth.se/)

## Discussion

### Troubles with the models
We thought the model result was absolutely atrocious, with WER well over 100. We thought the whisper-large-v3 could be cool to train since we would have the A6000 for inference anyway it could run it. 

However, it seems our training tooling from the Colab notebook does not work with the v3 model, however training the medium worked fine.

### Troubles with audio streaming
Getting the live radio streaming to work properly with our model, running on the A6000 GPU, and multiple stations simultaneously was quite difficult. 

We had most troubles with the stream rate which was supplied as a random magic number in all the examples we found. It turns out this number was supplied within the stream, and could be retrieved using `ffmpeg -i <url>`. We then used this number to set the rate of the stream, and it worked perfectly.

We use chunks of 5 seconds to transcribe the audio, which works quite well apart from the words which get cut in half, but for this demo it was not a major concern.

# Task 2 - Improve scalability
![man whispering at big server](./assets/task2.png)


## Task
Improve the scalability of our Whisper fine tuning pipeline.

## Discussion
### Model-centric approach
- Hyperparameter tuning\
We could improve the model by using hyperparamter tuning, where we could adjust parameters such as rate, batch size and number of epochs. This could lead to significant improvments in the model performance. We could use methods such as Grid search (exhaustive, not very efficient) and Bayesian optimization (probabalistic model, more efficient).

- Regularization techniques\
We could use techniques such as dropout and early stopping to prevent overfitting (since the Swedish dataset was relatively small)

- Larger base model\
We could use a larger model, as a base for our fine tuning which could improve the model performance.\
This is what we implemented, and it did improve the model performance, from a WER of 117 to 99. However, we were not able to train the large model, which could have improved the model performance even more.

### Data-centric approach
- More data\
The obvious answer when using the Swedish dataset, is that more data could definitely improve the model. The Swedish dataset was relatively small, and many words were probably not in the dataset. 

- Augmentation\
We could augment the data to add noise, varying pitches, and speed. This could improve various use-cases that isn't just a crispy clear podcast recording. This would make the model more robust.

- Balanced dataset\
We could ensure that the dataset is balanced across multiple accents, since some accents pronounce words completely differently (and could even have their own vocabulary).\
We discovered this issue when a person from Skåne was interviewed in a Swedish radio program. 


## Conclusion
We started with the small model, fine tuning it with the swedish dataset. We noticed a high WER, and tried to fine tune the medium model instead. This improved the WER, but we were not able to train the large v3 model.

We then created a frontend, which was hosted on Hugging Face Spaces, and a radio streaming application, which was hosted on a VM with an A6000 GPU, where a user can get live subtitles for some popular radios using our fine-tuned model.

We then discussed how we could improve the model, and the training pipeline. We concluded that we could improve the model by using hyperparameter tuning, regularization techniques, and a larger base model. We could also improve the model by using more data, augmenting the data, and ensuring that the dataset is balanced across multiple accents.