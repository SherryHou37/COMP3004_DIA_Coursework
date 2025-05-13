# Health Chat Box

This is a health chat box

## Enviornment requirement
- To git clone this code, please add `LFS` command
- To run this, please use anaconda, with both Tensorflow and pytorch install. It would be better if `cuda` is supported on your machine.
- If wants to run the code, please run `Health_chat_main.ipynb`

## key component
### Intent recognition agent
-	Determines if a user’s query is related to mental health or physical health.
###	Mental Health Agent(LLM)
-	Handles psychological consultations
###	Physical Health Agent(LLM)
-	Provides responses for common diseases, symptoms, and medications
### GUI

## Technick
### LLM: LSTM based seq2seq model, Bidirectional LSTM model，GRU based seq2seq model
- MiniTransformer -- meet mode collapse problem
- MiniTransformer -- merge input/output embeddings and share weights with the output layer fc_out; introduce LayerNorm and Dropout; add pre-normalization to fc_out
- BiGRU 
- BiLSTM

## Data set
- mental health: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
- physical health: https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k

