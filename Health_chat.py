import torch
from typing import List
import torch.nn.functional as F
import re
import json
import unicodedata
from sentence_transformers import SentenceTransformer, util


from myTokenizer import myTokenizer
from models.BiLSTM import EncoderBiLSTM, DecoderLSTM, Seq2Seq, Attention
from models.BiGRU import EncoderBiGRU, DecoderGRU, Seq2SeqGRU, BahdanauAttention
from models.miniTransformer import generate_square_subsequent_mask, TransformerChat as TransformerChatV1
from models.miniTransformerV2 import TransformerChat as TransformerChatV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LEN = 100
INTENT_TRAINING_NUM = 16



class HealthChat:
    def __init__(self):
        thisTokenizer = myTokenizer()
        self.intent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.template_cache = {}  # for speed up json file loading
        self.tokenizerForPhysicalHealth = thisTokenizer.load_tokenizer('/tokenizer/tokenizerForHealthCare.pkl')
        self.tokenizerForMentalHealth = thisTokenizer.load_tokenizer('/tokenizer/tokenizerForMentalHealth.pkl')
        self.models = dict()
        self.all_models = {1:'biLSTM', 2: 'biGRU', 3: 'transformerV1', 4: 'transformerV2', 5: 'transformerV2_mentalHealth'}
        self.current_model = 1
        self.intent_matching_method = {1: 'keyWorld', 2: 'CosineSim'}
        self.current_intent_matching_method_id = 1
        self.intent_training_num = INTENT_TRAINING_NUM
        self.load_all_models()
        for i in [16, 30, 50]:
            self.load_templates(i)

    def set_current_model(self, model_id):
        """"model_id: 1 for biLSTM, 2 for biGRU, 3 for transformerV1, 4 for transformerV2, 5 for transformerV2_mentalHealth"""
        if model_id in self.all_models:
            self.current_model = model_id
        else:
            raise ValueError(f"Invalid model ID: {model_id}. Available models: {self.all_models}")
        
    def get_current_model(self):
        return self.current_model, self.all_models[self.current_model]
    
    def set_model_weight(self, model_id, weight_path):
        if model_id in self.all_models:
            model = self.models[self.all_models[model_id]]
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            self.models[self.all_models[model_id]] = model
        else:
            raise ValueError(f"Invalid model ID: {model_id}. Available models: {self.all_models}")
        
    def set_intent_matching_method(self, method_id, intent_training_num=None):
        if intent_training_num != None:
            if intent_training_num in [16, 30, 50]:
                self.intent_training_num = intent_training_num
            else:
                raise ValueError(f"Invalid intent_training_num: {intent_training_num}. Available methods: {[16, 30, 50]}")

        if method_id in self.intent_matching_method:
            self.current_intent_matching_method_id = method_id
        else:
            raise ValueError(f"Invalid method ID: {method_id}. Available methods: {self.intent_matching_method}")
        
    def get_current_intent_matching_method(self):
        return self.current_intent_matching_method_id, self.intent_matching_method[self.current_intent_matching_method_id], self.intent_training_num
    


    def load_all_models(self):
        vocab_size = self.tokenizerForPhysicalHealth.num_words + 1  
        embedding_dim = 256
        hidden_dim = 512
        pad_idx = 0
        output_dim = vocab_size  # 生成任务，输出词表大小和输入相同
        # ------- parameters for LSTM -------
        attn = Attention(hidden_dim)
        encoder = EncoderBiLSTM(vocab_size, embedding_dim, hidden_dim, pad_idx)
        decoder = DecoderLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, attn)
        model_BiLSTM = Seq2Seq(encoder, decoder, pad_idx=pad_idx, device=DEVICE).to(DEVICE)
        model_BiLSTM.load_state_dict(torch.load('checkpoint/weight_biLSTM_550.pth', map_location=DEVICE))
        self.models['biLSTM'] = model_BiLSTM
        # ------- parameters for GRU -------
        encoder = EncoderBiGRU(vocab_size, embedding_dim, hidden_dim, pad_idx)
        attention = BahdanauAttention(hidden_dim)
        decoder = DecoderGRU(vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, attention)
        model_BiGRU = Seq2SeqGRU(encoder, decoder, pad_idx, DEVICE).to(DEVICE)
        model_BiGRU.load_state_dict(torch.load('checkpoint/weight_biGRU_1550.pth', map_location=DEVICE))
        self.models['biGRU'] = model_BiGRU
        # ------- parameters for TransformerV1 -------
        model_transformerV1 = TransformerChatV1(input_vocab_size=vocab_size, target_vocab_size=vocab_size)
        model_transformerV1.load_state_dict(torch.load('checkpoint/weight_transformer_3550.pth', map_location='cuda'))
        model_transformerV1 = model_transformerV1.to('cuda')  
        self.models['transformerV1'] = model_transformerV1
        # ------- parameters for TransformerV2 -------
        model_transformerV2 = TransformerChatV2(vocab_size=vocab_size)
        model_transformerV2.load_state_dict(torch.load('checkpoint/weight_transformerV2_3550.pth', map_location='cuda')) 
        model_transformerV2 = model_transformerV2.to('cuda')  
        self.models['transformerV2'] = model_transformerV2
        # ------- parameters for TransformerV2_mentalHealth -------
        vocab_size = self.tokenizerForMentalHealth.num_words + 1
        model_transformerV2_mentalHealth = TransformerChatV2(vocab_size=vocab_size)
        model_transformerV2_mentalHealth.load_state_dict(torch.load('checkpoint/weight_transformerV2_JSON_1550.pth', map_location='cuda')) 
        model_transformerV2_mentalHealth = model_transformerV2_mentalHealth.to('cuda') 
        self.models['transformerV2_mentalHealth'] = model_transformerV2_mentalHealth

    # -------- response functions -----------
    def chat(self, input_text):
        input_text = self.clean_text(input_text)
        # 1. intent matching
        if self.current_intent_matching_method_id == 1:
            intent = self.match_intent_keyWorld(input_text)
        elif self.current_intent_matching_method_id == 2:
            intent = self.match_intent_CosineSim(input_text)
        else:
            raise ValueError(f"Invalid method ID: {self.current_intent_matching_method_id}. Available methods: {self.intent_matching_method}")

        # 2. response generation
        if intent == "physiological_agent":
            return self.get_model_response_physical(input_text)
        elif intent == "psychological_agent":
            return self.get_model_response_mental(input_text)
        else:
            raise ValueError(f"Invalid intent: {intent}. Available intents: physiological_agent, psychological_agent")
        
    def get_model_response_mental(self, input_text):
        output = self.decode_with_topk_penalty(self.models['transformerV2_mentalHealth'], self.tokenizerForMentalHealth, input_text, max_len=MAX_LEN, device=DEVICE)
        return output, "mental" ,"transformerV2h"

    def get_model_response_physical(self, input_text):
        word2idx = self.tokenizerForPhysicalHealth.word_index
        idx2word = {idx: word for word, idx in word2idx.items()}
        pad_idx = 0
        word2idx["<pad>"] = pad_idx
        idx2word[pad_idx] = "<pad>"
        src_vocab = word2idx
        trg_vocab = word2idx
        sos_idx = self.tokenizerForPhysicalHealth.word_index.get("<start>", "<Not found>")
        eos_idx = self.tokenizerForPhysicalHealth.word_index.get("<end>", "<Not found>")
        src_tokens = [src_vocab.get(tok, src_vocab['<UNKNOWN>']) for tok in input_text.split()]
        src_tensor = torch.LongTensor(src_tokens)
        if self.current_model == 1:
            output_ids = self.response_LSTM(src_tensor, src_vocab, trg_vocab, DEVICE)
            output_words = [idx2word[idx] for idx in output_ids]
            return ' '.join(output_words), "physical", "biLSTM"
        elif self.current_model == 2:
            model = self.models['biGRU']
            output = self.response_GRU(sentence=input_text, model=model, tokenizer_src=self.tokenizerForPhysicalHealth, tokenizer_trg=self.tokenizerForPhysicalHealth, sos_idx=sos_idx, eos_idx=eos_idx)
            return output,  "physical", "biGRU"
        elif self.current_model == 3:
            output = self.greedy_decode(self.models['transformerV1'], self.tokenizerForPhysicalHealth, input_text, max_output_len=MAX_LEN, device=DEVICE)
            return output, "physical", "transformerV1"
        elif self.current_model == 4:
            output = self.decode_with_topk_penalty(self.models['transformerV2'], self.tokenizerForPhysicalHealth, input_text, max_len=MAX_LEN, device=DEVICE)
            return output, "physical", "transformerV2"
        elif self.current_model == 5:
            return "" # this model is not be used in this method
        else:
            raise ValueError(f"Invalid model ID: {self.current_model}. Available models: {self.all_models}")
        
    # --------- intent matching functions -----------

    def match_intent_keyWorld(self, user_input):
        physiological_intent_patterns = [
            r"(pain|fever|headache|cough|cold|diarrhea|rash|sore throat)",
            r"(i have.*(symptoms|pain|inflammation))"
        ]

        psychological_intent_patterns = [
            r"(anxiety|depression|stress|emotion|psychological|emotional|suicide|loneliness|insomnia|fear|tension)",
            r"(i feel.*(sad|helpless|afraid|sorrow))"
        ]

        user_input = user_input.lower()  # 将输入转换为小写

        # 统计各类意图匹配的数量
        physiological_count = sum(bool(re.search(pattern, user_input)) for pattern in physiological_intent_patterns)
        psychological_count = sum(bool(re.search(pattern, user_input)) for pattern in psychological_intent_patterns)
        # print(f"physiological_count: {physiological_count}, psychological_count: {psychological_count}")
        
        # 根据匹配数量分配意图
        if physiological_count >= psychological_count:
            return "physiological_agent"
        else:
            return "psychological_agent"  # 当两类意图匹配数量相同时
        
    def match_intent_CosineSim(self, input_text):
        # 加载预先计算的模板嵌入
        # with open("model_intent/template_embeddings"+str(self.intent_training_num)+".json", "r", encoding="utf-8") as file:
        #     templates = json.load(file)
        templates = self.load_templates(self.intent_training_num)

        # 计算用户输入的嵌入向量
        user_embedding =  self.intent_model.encode(input_text)

        best_score = -1
        best_category = "physiological"  # default category

        # 与所有模板嵌入计算相似度
        for template in templates:
            template_embedding = template['embedding']
            score = util.cos_sim(user_embedding, template_embedding).item()
            if score > best_score:
                best_score = score
                best_category = template['category']

        return f"{best_category}_agent"
    
    def load_templates(self, intent_training_num):
        if intent_training_num in self.template_cache:
            return self.template_cache[intent_training_num]
        
        with open(f"model_intent/template_embeddings{intent_training_num}.json", "r", encoding="utf-8") as file:
            templates = json.load(file)
            self.template_cache[intent_training_num] = templates
            return templates
        
    # -------- All kinds of response methods --------
        
    def response_LSTM(self, src_tensor, src_vocab, trg_vocab, device, max_len=MAX_LEN):
        model = self.models['biLSTM']
        model.eval()
        src_tensor = src_tensor.unsqueeze(0).to(DEVICE)  # [1, src_len]
        mask = model.create_mask(src_tensor)

        with torch.no_grad():
            encoder_outputs, hidden, cell = model.encoder(src_tensor)

            # 处理 hidden, cell 初始化
            hidden = torch.tanh(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
            cell = torch.tanh(torch.cat((cell[0:1], cell[1:2]), dim=2))
            hidden = hidden[:, :, :model.decoder.decoder_hidden_dim]
            cell = cell[:, :, :model.decoder.decoder_hidden_dim]

        # 第一个 decoder 输入是 <sos>
        trg_indices = [trg_vocab['<start>']]

        for _ in range(max_len):
            prev_input = torch.LongTensor([trg_indices[-1]]).to(DEVICE)  # [1]
            
            with torch.no_grad():
                output, hidden, cell, _ = model.decoder(prev_input, hidden, cell, encoder_outputs, mask)

            next_token = output.argmax(1).item()
            trg_indices.append(next_token)

            if next_token == trg_vocab['<end>']:
                break

        # 去除 <sos> 和 <eos>
        return trg_indices[1:-1]
    
    @torch.inference_mode()      # PyTorch ≥1.12，自动关掉梯度
    def response_GRU(
        self,
        sentence,
        model,
        tokenizer_src,
        tokenizer_trg,
        device = DEVICE,
        max_len: int = MAX_LEN,
        sos_idx: int = 18,
        eos_idx: int = 19,
    ) -> str:
        model.eval()

        # 1. 预处理 —— 分词 → id → tensor
        src_ids = tokenizer_src.texts_to_sequences([sentence])[0] 
        
        src_ids = [sos_idx] + src_ids + [eos_idx]

        src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, src_len]

        # 2. Encoder
        encoder_outs, enc_hidden = model.encoder(src_tensor)             # [1, src_len, 2H], [1, 2H]  :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
        dec_hidden = torch.tanh(model.bridge(enc_hidden))                # [1, H]          :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
        dec_hidden = dec_hidden.unsqueeze(0)                             # [1, 1, H]      :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}

        # 3. Decoder – 逐步生成，greedy search
        trg_indices: List[int] = [sos_idx]                               # 先放 <sos>
        for _ in range(max_len):
            # 上一步输出（或 <sos>）作为当前输入
            last_token = torch.tensor([trg_indices[-1]], device=device)  # [1]
            output, dec_hidden, _ = model.decoder(
                last_token, dec_hidden, encoder_outs
            )                                                            # output: [1, vocab]  :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}

            next_token = int(output.argmax(1))                           # greedy
            trg_indices.append(next_token)

            if next_token == eos_idx:
                break

        # 4. 去掉首尾标记，转回文本
        trg_tokens = [tokenizer_trg.index_word[i] for i in trg_indices[1:-1]]  # id → token
        translation = " ".join(trg_tokens)  
        return translation.strip()
    
    def greedy_decode(self, model, tokenizer, input_text, max_output_len=100, device=DEVICE):
        model.eval()

        # Step 1: 清洗并编码输入文本
        cleaned = myTokenizer.clean_text(input_text)
        # print(f"cleaned: {cleaned}")
        input_seq = tokenizer.texts_to_sequences([cleaned])
        # print(f"input_seq: {input_seq}")
        input_tensor = torch.tensor(input_seq).to(device)

        # Step 2: 准备 decoder 输入（以 <start> 开头）
        start_token_id = tokenizer.word_index.get('<start>', 1)
        # print(f"start_token_id: {start_token_id}")
        end_token_id = tokenizer.word_index.get('<end>', 2)
        decoder_input = torch.tensor([[start_token_id]], device=device)

        # Step 3: 编码器输出
        with torch.no_grad():
            src_emb = model.pos_encoder(model.src_embedding(input_tensor))
            memory = model.transformer.encoder(src_emb)

        # Step 4: 逐步生成 token
        for _ in range(max_output_len):
            tgt_emb = model.pos_encoder(model.tgt_embedding(decoder_input))
            tgt_mask = generate_square_subsequent_mask(decoder_input.size(1)).to(device)

            # with torch.no_grad():
            #     output = model.transformer.decoder(
            #         tgt_emb, memory, tgt_mask=tgt_mask
            #     )
            #     logits = model.fc_out(output[:, -1, :])  # 最后一个 token 的输出
            #     next_token = logits.argmax(dim=-1).unsqueeze(0)

            with torch.no_grad():
                output = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = model.fc_out(output[:, -1, :])
                # print(logits)
                # 惩罚重复 <start>（避免死循环）
                if decoder_input[0, -1].item() == start_token_id:
                    logits[0, start_token_id] -= 1.0

                next_token = logits.argmax(dim=-1).unsqueeze(0)

            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if next_token.item() == end_token_id:
                break

        # Step 5: 解码输出为文本
        output_tokens = decoder_input.squeeze().tolist()[1:]  # 去掉 <start>
        words = [tokenizer.index_word.get(tok, '<UNK>') for tok in output_tokens if tok != end_token_id]
        return ' '.join(words)
    
    def decode_with_topk_penalty(self, model, tokenizer, input_text, k=10, max_len=100, temperature=1.2, penalty=1.5, device=DEVICE):
        model.eval()
        start_id = tokenizer.word_index['<start>']
        end_id = tokenizer.word_index['<end>']

        cleaned = myTokenizer.clean_text(input_text)
        input_ids = tokenizer.texts_to_sequences([cleaned])
        input_tensor = torch.tensor(input_ids).to(device)

        with torch.no_grad():
            src_emb = model.pos_encoder(model.embedding(input_tensor))
            memory = model.transformer.encoder(model.norm(src_emb))

        decoder_input = torch.tensor([[start_id]], device=device)

        for _ in range(max_len):
            tgt_emb = model.pos_encoder(model.embedding(decoder_input))
            tgt_mask = generate_square_subsequent_mask(decoder_input.size(1)).to(device)

            with torch.no_grad():
                output = model.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                logits = model.fc_out(model.norm(output[:, -1, :])) / temperature

            # Repetition penalty
            for tok in set(decoder_input[0].tolist()):
                logits[0, tok] -= penalty

            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(probs, k)
            next_token = topk_ids[0, torch.multinomial(topk_probs[0], 1)]

            decoder_input = torch.cat([decoder_input, next_token.view(1, 1)], dim=1)
            if next_token.item() == end_id:
                break

        output_tokens = decoder_input.squeeze().tolist()[1:]
        return ' '.join([tokenizer.index_word.get(i, '<UNK>') for i in output_tokens if i != end_id])

# ---------- input cleaning function ----------
    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def clean_text(self, text):
        text = self.unicode_to_ascii(text.lower().strip())
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)

        text = re.sub(r"([?.!,])", r" \1 ", text)  

        text =  "<start> " +  text + " <end>"
        return text