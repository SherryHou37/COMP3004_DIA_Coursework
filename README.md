# Dia_Yixin -- Health Chat Box

This is a health chat box

## key component
### Intent recognition agent
-	Determines if a user’s query is related to mental health or physical health.
###	Mental Health Agent(LLM)
-	Handles psychological consultations
###	Physical Health Agent(LLM)
-	Provides responses for common diseases, symptoms, and medications
###	Audit & Safety Agent
-	Detects extreme inputs, including self-harm, drug overdose queries
### GUI


## Technick
### LLM: LSTM based seq2seq model, Bidirectional LSTM model，GRU based seq2seq model
- MiniTransformer -- meet mode collapse problem
- MiniTransformer -- merge input/output embeddings and share weights with the output layer fc_out; introduce LayerNorm and Dropout; add pre-normalization to fc_out
- GRU LSTM

## Data set
- mental health: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
- physical health: https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k
- Data set for extreme inputs: https://huggingface.co/datasets/jquiros/suicide

## Testing
- BEU: bilingual evaluation understudy,
- average responsed time


## xxxx
- LSTM 和 GRU都使用了attention来优化推理效果，但没有使用dropout来防止过拟合