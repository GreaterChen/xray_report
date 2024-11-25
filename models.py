import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from transformers import AutoModel, AutoTokenizer
import sentencepiece as spm
import math
        
# --- Transformer Modules ---
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, input, query, pad_mask=None, att_mask=None):
        input = input.permute(1,0,2) # (V,B,E)
        query = query.permute(1,0,2) # (Q,B,E)
        embed, att = self.attention(query, input, input, key_padding_mask=pad_mask, attn_mask=att_mask) # (Q,B,E), (B,Q,V)
        
        embed = self.normalize(embed + query) # (Q,B,E)
        embed = embed.permute(1,0,2) # (B,Q,E)
        return embed, att # (B,Q,E), (B,Q,V)
    
class PointwiseFeedForward(nn.Module):
    def __init__(self, emb_dim, fwd_dim, dropout=0.0):
        super().__init__()
        self.fwd_layer = nn.Sequential(
            nn.Linear(emb_dim, fwd_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fwd_dim, emb_dim),
        )
        self.normalize = nn.LayerNorm(emb_dim)

    def forward(self, input):
        output = self.fwd_layer(input) # (B,L,E)
        output = self.normalize(output + input) # (B,L,E)
        return output

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.0):
        super().__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout)
        self.fwd_layer = PointwiseFeedForward(embed_dim, fwd_dim, dropout)

    def forward(self, input, pad_mask=None, att_mask=None):
        emb, att = self.attention(input,input,pad_mask,att_mask)
        emb = self.fwd_layer(emb)
        return emb, att

class TNN(nn.Module):
    def __init__(self, embed_dim, num_heads, fwd_dim, dropout=0.1, num_layers=1,
                num_tokens=1, num_posits=1, token_embedding=None, posit_embedding=None):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, embed_dim) if not token_embedding else token_embedding
        self.posit_embedding = nn.Embedding(num_posits, embed_dim) if not posit_embedding else posit_embedding
        self.transform = nn.ModuleList([TransformerLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, token_index=None, token_embed=None, pad_mask=None, pad_id=-1, att_mask=None):
        if token_index != None:
            if pad_mask == None:
                pad_mask = (token_index == pad_id) # (B,L)
            posit_index = torch.arange(token_index.shape[1]).unsqueeze(0).repeat(token_index.shape[0],1).to(token_index.device) # (B,L)
            posit_embed = self.posit_embedding(posit_index) # (B,L,E)
            token_embed = self.token_embedding(token_index) # (B,L,E)
            final_embed = self.dropout(token_embed + posit_embed) # (B,L,E)
        elif token_embed != None:
            posit_index = torch.arange(token_embed.shape[1]).unsqueeze(0).repeat(token_embed.shape[0],1).to(token_embed.device) # (B,L)
            posit_embed = self.posit_embedding(posit_index) # (B,L,E)
            final_embed = self.dropout(token_embed + posit_embed) # (B,L,E)
        else:
            raise ValueError('token_index or token_embed must not be None')

        for i in range(len(self.transform)):
            final_embed = self.transform[i](final_embed, pad_mask, att_mask)[0]
            
        return final_embed # (B,L,E)

# --- Convolution Modules ---
class CNN(nn.Module):
    def __init__(self, model, model_type='resnet'):
        super().__init__()
        if 'res' in model_type.lower(): # resnet, resnet-50, resnest-50, ...
            modules = list(model.children())[:-1] # Drop the FC layer
            self.feature = nn.Sequential(*modules[:-1])
            self.average = modules[-1]
        elif 'dense' in model_type.lower(): # densenet, densenet-121, densenet121, ...
            modules = list(model.features.children())[:-1]
            self.feature = nn.Sequential(*modules)
            self.average = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError('Unsupported model_type!')
        
    def forward(self, input):
        wxh_features = self.feature(input) # (B,2048,W,H)
        avg_features = self.average(wxh_features) # (B,2048,1,1)
        avg_features = avg_features.view(avg_features.shape[0], -1) # (B,2048)
        return avg_features, wxh_features

class MVCNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        img = input[0] # (B,V,C,W,H)
        pos = input[1] # (B,V)
        B,V,C,W,H = img.shape

        img = img.view(B*V,C,W,H)
        avg, wxh = self.model(img) # (B*V,F), (B*V,F,W,H)
        avg = avg.view(B,V,-1) # (B,V,F)
        wxh = wxh.view(B,V,wxh.shape[-3],wxh.shape[-2],wxh.shape[-1]) # (B,V,F,W,H)
        
        msk = (pos == -1) # (B,V)
        msk_wxh = msk.view(B,V,1,1,1).float() # (B,V,1,1,1) * (B,V,F,C,W,H)
        msk_avg = msk.view(B,V,1).float() # (B,V,1) * (B,V,F)
        wxh = msk_wxh * (-1) + (1-msk_wxh) * wxh
        avg = msk_avg * (-1) + (1-msk_avg) * avg

        wxh_features = wxh.max(dim=1)[0] # (B,F,W,H)
        avg_features = avg.max(dim=1)[0] # (B,F)
        return avg_features, wxh_features

# --- Main Moduldes ---
class Classifier(nn.Module):
    def __init__(self, num_topics, num_states, cnn=None, tnn=None,
                fc_features=2048, embed_dim=128, num_heads=1, dropout=0.1):
        super().__init__()
        
        # For img & txt embedding and feature extraction
        self.cnn = cnn
        self.tnn = tnn
        self.img_features = nn.Linear(fc_features, num_topics * embed_dim) if cnn != None else None
        self.txt_features = MultiheadAttention(embed_dim, num_heads, dropout) if tnn != None else None
        
        # For classification
        self.topic_embedding = nn.Embedding(num_topics, embed_dim)
        self.state_embedding = nn.Embedding(num_states, embed_dim)
        self.attention = MultiheadAttention(embed_dim, num_heads)
        
        # Some constants
        self.num_topics = num_topics
        self.num_states = num_states
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(embed_dim)

    def forward(self, img=None, txt=None, lbl=None, txt_embed=None, pad_mask=None, pad_id=3, threshold=0.5, get_embed=False, get_txt_att=False):
        # --- Get img and txt features ---
        if img != None: # (B,C,W,H) or ((B,V,C,W,H), (B,V))
            img_features, wxh_features = self.cnn(img) # (B,F), (B,F,W,H)
            img_features = self.dropout(img_features) # (B,F)
            
        if txt != None:
            if pad_id >= 0 and pad_mask == None:
                pad_mask = (txt == pad_id)
            txt_features = self.tnn(token_index=txt, pad_mask=pad_mask) # (B,L,E)
        
        elif txt_embed != None:
            txt_features = self.tnn(token_embed=txt_embed, pad_mask=pad_mask) # (B,L,E)

        # --- Fuse img and txt features ---
        if img != None and (txt != None or txt_embed != None):
            topic_index = torch.arange(self.num_topics).unsqueeze(0).repeat(img_features.shape[0],1).to(img_features.device) # (B,T)
            state_index = torch.arange(self.num_states).unsqueeze(0).repeat(img_features.shape[0],1).to(img_features.device) # (B,C)
            topic_embed = self.topic_embedding(topic_index) # (B,T,E)
            state_embed = self.state_embedding(state_index) # (B,C,E)
            
            img_features = self.img_features(img_features).view(img_features.shape[0], self.num_topics, -1) # (B,F) --> (B,T*E) --> (B,T,E)   
            txt_features, txt_attention = self.txt_features(txt_features, topic_embed, pad_mask) # (B,T,E), (B,T,L)
            final_embed = self.normalize(img_features + txt_features) # (B,T,E)
            
        elif img != None:
            topic_index = torch.arange(self.num_topics).unsqueeze(0).repeat(img_features.shape[0],1).to(img_features.device) # (B,T)
            state_index = torch.arange(self.num_states).unsqueeze(0).repeat(img_features.shape[0],1).to(img_features.device) # (B,C)
            topic_embed = self.topic_embedding(topic_index) # (B,T,E)
            state_embed = self.state_embedding(state_index) # (B,C,E)

            img_features = self.img_features(img_features).view(img_features.shape[0], self.num_topics, -1) # (B,F) --> (B,T*E) --> (B,T,E)   
            final_embed = img_features # (B,T,E)
            
        elif txt != None or txt_embed != None:
            topic_index = torch.arange(self.num_topics).unsqueeze(0).repeat(txt_features.shape[0],1).to(txt_features.device) # (B,T)
            state_index = torch.arange(self.num_states).unsqueeze(0).repeat(txt_features.shape[0],1).to(txt_features.device) # (B,C)
            topic_embed = self.topic_embedding(topic_index) # (B,T,E)
            state_embed = self.state_embedding(state_index) # (B,C,E)

            txt_features, txt_attention = self.txt_features(txt_features, topic_embed, pad_mask) # (B,T,E), (B,T,L)
            final_embed = txt_features # (B,T,E)
            
        else:
            raise ValueError('img and (txt or txt_embed) must not be all none')
        
        # Classifier output
        emb, att = self.attention(state_embed, final_embed) # (B,T,E), (B,T,C)
        
        if lbl != None: # Teacher forcing
            emb = self.state_embedding(lbl) # (B,T,E)
        else:
            emb = self.state_embedding((att[:,:,1] > threshold).long()) # (B,T,E)
            
        if get_embed:
            return att, final_embed + emb # (B,T,C), (B,T,E)
        elif get_txt_att and (txt != None or txt_embed != None):
            return att, txt_attention # (B,T,C), (B,T,L)
        else:
            return att # (B,T,C)

class Generator(nn.Module):
    def __init__(self, num_tokens, num_posits, embed_dim=128, num_heads=1, fwd_dim=256, dropout=0.1, num_layers=12):
        super().__init__()
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)
        self.posit_embedding = nn.Embedding(num_posits, embed_dim)
        self.transform = nn.ModuleList([TransformerLayer(embed_dim, num_heads, fwd_dim, dropout) for _ in range(num_layers)])
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.num_tokens = num_tokens
        self.num_posits = num_posits
        
    def forward(self, source_embed, token_index=None, source_pad_mask=None, target_pad_mask=None, max_len=300, top_k=1, bos_id=1, pad_id=3, mode='eye'):
        if token_index != None: # --- Training/Testing Phase ---
            # Adding token embedding and posititional embedding.
            posit_index = torch.arange(token_index.shape[1]).unsqueeze(0).repeat(token_index.shape[0],1).to(token_index.device) # (1,L) --> (B,L)
            posit_embed = self.posit_embedding(posit_index) # (B,L,E)
            token_embed = self.token_embedding(token_index) # (B,L,E)
            target_embed = token_embed + posit_embed # (B,L,E)
            
            # Make embedding, attention mask, pad mask for Transformer Decoder
            final_embed = torch.cat([source_embed,target_embed], dim=1) # (B,T+L,E)
            if source_pad_mask == None:
                source_pad_mask = torch.zeros((source_embed.shape[0],source_embed.shape[1]),device=source_embed.device).bool() # (B,T)
            if target_pad_mask == None:
                target_pad_mask = torch.zeros((target_embed.shape[0],target_embed.shape[1]),device=target_embed.device).bool() # (B,L)
            pad_mask = torch.cat([source_pad_mask,target_pad_mask], dim=1) # (B,T+L)
            att_mask = self.generate_square_subsequent_mask_with_source(source_embed.shape[1], target_embed.shape[1], mode).to(final_embed.device) # (T+L,T+L)

            # Transformer Decoder
            for i in range(len(self.transform)):
                final_embed = self.transform[i](final_embed,pad_mask,att_mask)[0]

            # Make prediction for next tokens
            token_index = torch.arange(self.num_tokens).unsqueeze(0).repeat(token_index.shape[0],1).to(token_index.device) # (1,K) --> (B,K)
            token_embed = self.token_embedding(token_index) # (B,K,E)
            emb, att = self.attention(token_embed,final_embed) # (B,T+L,E), (B,T+L,K)
            
            # Truncate results from source_embed
            emb = emb[:,source_embed.shape[1]:,:] # (B,L,E)
            att = att[:,source_embed.shape[1]:,:] # (B,L,K)
            return att, emb
        
        else: # --- Inference Phase ---
            return self.infer(source_embed, source_pad_mask, max_len, top_k, bos_id, pad_id)

    def infer(self, source_embed, source_pad_mask=None, max_len=100, top_k=1, bos_id=1, pad_id=3):
        outputs = torch.ones((top_k, source_embed.shape[0], 1), dtype=torch.long).to(source_embed.device) * bos_id # (K,B,1) <s>
        scores = torch.zeros((top_k, source_embed.shape[0]), dtype=torch.float32).to(source_embed.device) # (K,B)

        for _ in range(1,max_len):
            possible_outputs = []
            possible_scores = []

            for k in range(top_k):
                output = outputs[k] # (B,L)
                score = scores[k] # (B)
                
                att, emb = self.forward(source_embed, output, source_pad_mask=source_pad_mask, target_pad_mask=(output == pad_id))
                val, idx = torch.topk(att[:,-1,:], top_k) # (B,K)
                log_val = -torch.log(val) # (B,K)
                
                for i in range(top_k):
                    new_output = torch.cat([output, idx[:,i].view(-1,1)], dim=-1) # (B,L+1)
                    new_score = score + log_val[:,i].view(-1) # (B)
                    possible_outputs.append(new_output.unsqueeze(0)) # (1,B,L+1)
                    possible_scores.append(new_score.unsqueeze(0)) # (1,B)
            
            possible_outputs = torch.cat(possible_outputs, dim=0) # (K^2,B,L+1)
            possible_scores = torch.cat(possible_scores, dim=0) # (K^2,B)

            # Pruning the solutions
            val, idx = torch.topk(possible_scores, top_k, dim=0) # (K,B)
            col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0],1) # (K,B)
            outputs = possible_outputs[idx,col_idx] # (K,B,L+1)
            scores = possible_scores[idx,col_idx] # (K,B)

        val, idx = torch.topk(scores, 1, dim=0) # (1,B)
        col_idx = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0).repeat(idx.shape[0],1) # (K,B)
        output = outputs[idx,col_idx] # (1,B,L)
        score = scores[idx,col_idx] # (1,B)
        return output.squeeze(0) # (B,L)

    def generate_square_subsequent_mask_with_source(self, src_sz, tgt_sz, mode='eye'):
        mask = self.generate_square_subsequent_mask(src_sz + tgt_sz)
        if mode == 'one': # model can look at surrounding positions of the current index ith
            mask[:src_sz, :src_sz] = self.generate_square_mask(src_sz)
        elif mode == 'eye': # model can only look at the current index ith
            mask[:src_sz, :src_sz] = self.generate_square_identity_mask(src_sz)
        else: # model can look at surrounding positions of the current index ith with some patterns
            raise ValueError('Mode must be "one" or "eye".')
        mask[src_sz:, src_sz:] = self.generate_square_subsequent_mask(tgt_sz)
        return mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_square_identity_mask(self, sz):
        mask = (torch.eye(sz) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask 

    def generate_square_mask(self, sz):
        mask = (torch.ones(sz,sz) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# --- Full Models ---
class ClsGen(nn.Module):
    def __init__(self, classifier, generator, num_topics, embed_dim):
        super().__init__()
        self.classifier = classifier
        self.generator = generator
        self.label_embedding = nn.Embedding(num_topics, embed_dim)

    def forward(self, image, history=None, caption=None, label=None, threshold=0.15, bos_id=1, eos_id=2, pad_id=3, max_len=300, get_emb=False):
        
        # image 有两个元素，第一个是multi-view的图像，第二个是vpos，代表是哪个视角的图像
        # history 是 source_info, 包含['INDICATION:', 'HISTORY:', 'CLINICAL HISTORY:', 'REASON FOR EXAM:', 'REASON FOR EXAMINATION:', 'CLINICAL INFORMATION:', 'CLINICAL INDICATION:', 'PATIENT HISTORY:']
        # caption是FINDINGS， 最大token长度为1000，后面用0填充。
        # label 有两个元素，第一个是官方提供的各种病是否存在，第二个是预先构建了一个keyword词典，用0-1表征当前caption中是否存在这些keyword
        label = label.long() if label != None else label
        img_mlc, img_emb = self.classifier(img=image, txt=history, lbl=label, threshold=threshold, pad_id=pad_id, get_embed=True) # (B,T,C), (B,T,E)
        lbl_idx = torch.arange(img_emb.shape[1]).unsqueeze(0).repeat(img_emb.shape[0],1).to(img_emb.device) # (B,T)
        lbl_emb = self.label_embedding(lbl_idx) # (B,T,E)
        
        if caption != None:
            src_emb = img_emb + lbl_emb
            pad_mask = (caption == pad_id)
            cap_gen, cap_emb = self.generator(source_embed=src_emb, token_index=caption, target_pad_mask=pad_mask) # (B,L,S), (B,L,E)
            if get_emb:
                return cap_gen, img_mlc, cap_emb
            else:
                return cap_gen, img_mlc
        else:
            src_emb = img_emb + lbl_emb
            cap_gen = self.generator(source_embed=src_emb, token_index=caption, max_len=max_len, bos_id=bos_id, pad_id=pad_id) # (B,L,S)
            return cap_gen, img_mlc
        
class CXR_BERT_FeatureExtractor(nn.Module):
    def __init__(self, word_translator, model_name='microsoft/BiomedVLP-CXR-BERT-specialized', device='cuda',):
        super(CXR_BERT_FeatureExtractor, self).__init__()
        self.device = device
        # 加载预训练的 CXR-BERT 模型和对应的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()  # 设置模型为评估模式
        self.word_translator = word_translator

    def forward(self, inputs):
        """
        texts: 输入的文本列表，每个元素为一个字符串
        返回:
        features: 文本特征张量，形状为 (B, hidden_size)
        """
        # 使用origin的映射方式输出文本
        texts = self.word_translator.decode(inputs)
        # 对输入文本进行编码
        inputs = self.tokenizer(texts, max_length=1000, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 获取 [CLS] 标记的嵌入表示
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings


class SwinFeatureExtractor(nn.Module):
    def __init__(self, image_encoder_name='swin_large_patch4_window7_224', pretrained=True):
        super().__init__()
        # 加载预训练的 Swin Transformer
        self.image_encoder = create_model(image_encoder_name, pretrained=pretrained, features_only=True)
        
        # 映射到低维视觉特征 Fv
        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.image_encoder.feature_info[-1]['num_chs'], 512, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 提取全局特征
        )
    
    def forward(self, image):
        """
        image: 输入的图像，形状为 (B, C, H, W)
        返回:
        Fv: 视觉特征，形状为 (B, 512)
        """
        # TODO 处理多视角
        image = image[:, 0, :, :, :]

        # 提取图像的多层特征
        features = self.image_encoder(image)
        
        # 获取最后一层特征并调整维度顺序 (B, C, H, W)
        features_last = features[-1].permute(0, 3, 1, 2)
        
        # 仅使用最后一层特征进行降维和处理
        fv = self.feature_proj(features_last)  # 输出形状 (B, 512, 1, 1)
        fv = fv.squeeze(-1).squeeze(-1)      # 输出形状 (B, 512)
        
        return fv
    
class DiseaseFeatureProjector(nn.Module):
    def __init__(self, input_dim, num_diseases, feature_dim):
        """
        Args:
            input_dim: 输入视觉特征 x 的维度（Swin Transformer 输出的维度 C）。
            num_diseases: 疾病数量 N_v。
            feature_dim: 每种疾病的特征维度 C_v。
        """
        super().__init__()
        self.num_diseases = num_diseases
        self.feature_dim = feature_dim
        
        # 定义可学习的 A_i 和 b_i
        self.A = nn.Parameter(torch.randn(num_diseases, input_dim, feature_dim))  # A 的形状 (N_v, C, C_v)
        self.b = nn.Parameter(torch.randn(num_diseases, feature_dim))  # b 的形状 (N_v, C_v)

    def forward(self, x):
        """
        Args:
            x: 输入的视觉特征，形状为 (B, C)。
        Returns:
            Fv: 疾病特征矩阵，形状为 (B, N_v, C_v)。
        """
        # 扩展 x 的维度以匹配 A 的形状
        # x 的形状 (B, C)，A 的形状 (N_v, C, C_v)
        # A @ x 结果形状为 (B, N_v, C_v)
        F_v = torch.einsum('bc,ncf->bnf', x, self.A) + self.b  # (B, N_v, C_v)
        
        # 对每个疾病的特征进行 softmax 归一化
        F_v = torch.softmax(F_v, dim=-1)  # 对每个疾病的特征向量进行归一化
        return F_v
    
class TextDecoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, vocab_size=1000, num_layers=1, max_len=1000, bos_id=1, eos_id=2, pad_id=3):
        super(TextDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)  # 词嵌入
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8), num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)  # 输出词汇分布
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        # 生成 Sinusoidal 位置编码
        self.register_buffer("positional_encoding", self._get_sinusoidal_encoding(max_len, hidden_dim))

    def _get_sinusoidal_encoding(self, max_len, d_model):
        """
        创建 Sinusoidal 位置编码。
        Args:
            max_len: 最大序列长度。
            d_model: 隐藏层维度。
        Returns:
            position_encoding: 形状 (max_len, d_model)
        """
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # (d_model // 2)
        
        # 计算 sin 和 cos 编码
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用 cos
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, fv, target_sequence=None):
        """
        Args:
            fv: 疾病特征矩阵，形状 (B, N_v, C_v)，作为 memory。
            target_sequence: 编码的目标序列，形状 (B, max_len)，仅在训练阶段提供。
        Returns:
            output: 生成的词汇分布，形状 (B, max_len, vocab_size)。
            F_t: 文本特征矩阵，形状 (B, max_len, hidden_dim)。
        """
        batch_size = fv.size(0)
        memory = fv.permute(1, 0, 2)  # 转换为 (N_v, B, C_v)

        if target_sequence is not None:  # 训练阶段
            seq_len = target_sequence.size(1)

            # 嵌入目标序列并加上 Sinusoidal 位置编码
            target_embed = self.embedding(target_sequence)  # (B, seq_len, hidden_dim)
            position_encoding = self.positional_encoding[:, :seq_len, :].to(target_embed.device)  # 动态调整长度    # (1, seq_len, hidden_dim)
            target_embed = target_embed + position_encoding  # (B, seq_len, hidden_dim)

            # 自回归掩码
            target_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(target_embed.device)

            # Transformer 解码器
            decoder_output = self.transformer_decoder(
                tgt=target_embed.permute(1, 0, 2),  # (seq_len, B, hidden_dim)
                memory=memory,                     # (N_v, B, C_v)
                tgt_mask=target_mask               # 自回归掩码
            )  # 输出 (seq_len, B, hidden_dim)

            decoder_output = decoder_output.permute(1, 0, 2)  # 转换回 (B, seq_len, hidden_dim)

            # 生成词汇分布
            output = self.fc_out(decoder_output)  # (B, seq_len, vocab_size)

            # 生成 F_t：对每个隐藏状态应用 softmax
            F_t = F.softmax(decoder_output, dim=-1)  # (B, seq_len, hidden_dim)

        else:  # 测试阶段
            outputs = torch.zeros(batch_size, self.max_len, dtype=torch.long).fill_(self.pad_id).to(fv.device)
            outputs[:, 0] = self.bos_id  # 设置起始标记

            F_t_list = []

            for t in range(1, self.max_len):
                # 嵌入当前序列并加上位置编码
                target_embed = self.embedding(outputs[:, :t])  # (B, t, hidden_dim)
                position_encoding = self.positional_encoding[:, :t, :].to(target_embed.device)  # 动态调整长度
                target_embed = target_embed + position_encoding

                # 自回归掩码
                target_mask = torch.triu(torch.ones(t, t), diagonal=1).bool().to(target_embed.device)

                # Transformer 解码器
                decoder_output = self.transformer_decoder(
                    tgt=target_embed.permute(1, 0, 2),  # (t, B, hidden_dim)
                    memory=memory,                     # (N_v, B, C_v)
                    tgt_mask=target_mask               # 自回归掩码
                )  # 输出 (t, B, hidden_dim)

                decoder_output = decoder_output.permute(1, 0, 2)  # 转换回 (B, t, hidden_dim)

                # 预测下一个词
                output_t = self.fc_out(decoder_output[:, -1, :])  # (B, vocab_size)
                next_token = output_t.argmax(dim=-1)  # (B)

                outputs[:, t] = next_token

                # 生成 F_t：对每个隐藏状态应用 softmax
                F_t_t = F.softmax(decoder_output[:, -1, :], dim=-1)  # (B, hidden_dim)
                F_t_list.append(F_t_t)

                # 如果所有序列都生成了 <eos>，提前停止
                if (next_token == self.eos_id).all():
                    break

            output = outputs  # 返回生成的序列 (B, max_len)
            F_t = torch.stack(F_t_list, dim=1)  # 拼接 F_t，形状 (B, max_len, hidden_dim)

        return output, F_t
    
class FindingsGenerator(nn.Module):
    def __init__(self, text_decoder):
        """
        Findings Generator 封装 TextDecoder 实现。
        Args:
            text_decoder: 一个 TextDecoder 实例，用于解码器生成。
        """
        super(FindingsGenerator, self).__init__()
        self.text_decoder = text_decoder

    def forward(self, F_v, target_sequence=None):
        """
        Args:
            F_v: 输入的视觉特征，形状 (B, N_v, C_v)。
            target_sequence: 目标序列，形状 (B, max_len)，仅在训练时提供。
        Returns:
            output: 生成的词汇分布，形状 (B, max_len, vocab_size)。
            F_t_decoded: 解码器的隐藏状态，形状 (B, max_len, hidden_dim)。
        """
        # 使用 TextDecoder 进行生成
        output, F_t_decoded = self.text_decoder(F_v, target_sequence=target_sequence)

        return output, F_t_decoded
    
class CoAttentionBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        """
        单个 Co-Attention Block 的实现，包含以下步骤：
        1. 自注意力 (Self-Attention)
        2. 标准 Cross-Attention (Cross-Attention-1)
        3. 非对称 Cross-Attention (Cross-Attention-2)
        Args:
            embed_dim: 特征嵌入维度。
            num_heads: 注意力头的数量。
        """
        super(CoAttentionBlock, self).__init__()
        # 自注意力层
        self.self_attn_text = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.self_attn_visual = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        # 标准 Cross-Attention 层
        self.cross_attn_text_to_visual = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.cross_attn_visual_to_text = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        # 非对称 Cross-Attention 层
        self.cross_attn_asym_text = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.cross_attn_asym_visual = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        # 残差连接和归一化层
        self.norm_text1 = nn.LayerNorm(embed_dim)
        self.norm_visual1 = nn.LayerNorm(embed_dim)
        self.norm_text2 = nn.LayerNorm(embed_dim)
        self.norm_visual2 = nn.LayerNorm(embed_dim)
        self.norm_text3 = nn.LayerNorm(embed_dim)
        self.norm_visual3 = nn.LayerNorm(embed_dim)

    def forward(self, F_t, F_v):
        """
        Args:
            F_t: 文本特征，形状 (B, N_t, C_t)。
            F_v: 视觉特征，形状 (B, N_v, C_v)。
        Returns:
            F_t': 增强的文本特征，形状 (B, N_t, C_t)。
            F_v': 增强的视觉特征，形状 (B, N_v, C_v)。
        """
        # 转置到 (seq_len, batch_size, embed_dim) 格式，符合 MultiheadAttention 要求
        F_t = F_t.transpose(0, 1)  # (N_t, B, C_t)
        F_v = F_v.transpose(0, 1)  # (N_v, B, C_v)

        # Step 1: 自注意力
        F_t1, _ = self.self_attn_text(F_t, F_t, F_t)  # 文本自注意力
        F_t1 = self.norm_text1(F_t + F_t1)  # 残差连接 + 归一化

        F_v1, _ = self.self_attn_visual(F_v, F_v, F_v)  # 视觉自注意力
        F_v1 = self.norm_visual1(F_v + F_v1)  # 残差连接 + 归一化

        # Step 2: 标准 Cross-Attention
        F_t2, _ = self.cross_attn_visual_to_text(F_t1, F_v1, F_v1)  # 文本 -> 视觉
        F_t2 = self.norm_text2(F_t1 + F_t2)  # 残差连接 + 归一化

        F_v2, _ = self.cross_attn_text_to_visual(F_v1, F_t1, F_t1)  # 视觉 -> 文本
        F_v2 = self.norm_visual2(F_v1 + F_v2)  # 残差连接 + 归一化

        # Step 3: 非对称 Cross-Attention
        F_t3, _ = self.cross_attn_asym_text(F_t2, F_t2, F_v2)  # Query 和 Key 是文本，Value 是视觉
        F_t3 = self.norm_text3(F_t2 + F_t3)  # 残差连接 + 归一化

        F_v3, _ = self.cross_attn_asym_visual(F_v2, F_v2, F_t2)  # Query 和 Key 是视觉，Value 是文本
        F_v3 = self.norm_visual3(F_v2 + F_v3)  # 残差连接 + 归一化

        # 转回 (B, seq_len, embed_dim) 格式
        F_t3 = F_t3.transpose(0, 1)  # (B, N_t, C_t)
        F_v3 = F_v3.transpose(0, 1)  # (B, N_v, C_v)

        return F_t3, F_v3



class CoAttentionModule(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_blocks=6):
        """
        Co-Attention 模块，由多个 Co-Attention Block 组成。
        Args:
            embed_dim: 特征嵌入维度。
            num_heads: 注意力头的数量。
            num_blocks: Co-Attention Block 的数量。
        """
        super(CoAttentionModule, self).__init__()
        self.blocks = nn.ModuleList([CoAttentionBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(num_blocks)])

    def forward(self, F_t, F_v):
        """
        Args:
            F_t: 文本特征，形状 (B, N_t, C_t)。
            F_v: 视觉特征，形状 (B, N_v, C_v)。
        Returns:
            F_t': 增强的文本特征，形状 (B, N_t, C_t)。
            F_v': 增强的视觉特征，形状 (B, N_v, C_v)。
        """
        for block in self.blocks:
            F_t, F_v = block(F_t, F_v)
        return F_t, F_v

class ImpressionGenerator(nn.Module):
    def __init__(self, text_decoder):
        """
        Impression Generator 复用 TextDecoder 实现。
        Args:
            text_decoder: 一个 TextDecoder 实例，用于解码器生成。
        """
        super(ImpressionGenerator, self).__init__()
        self.text_decoder = text_decoder

    def forward(self, F_v_prime, F_t_prime, F_t, target_sequence=None):
        """
        Args:
            F_v_prime: 增强的视觉特征 (B, N_v, C_v)。
            F_t_prime: 增强的文本特征 (B, N_t, C_t)。
            F_t: 原始的文本特征 (B, N_t, C_t)。
            target_sequence: 目标序列 (B, max_len)，仅在训练时提供。
        Returns:
            output: 生成的词汇分布 (B, max_len, vocab_size)。
            F_t_decoded: 解码过程中的隐藏状态 (B, max_len, hidden_dim)。
        """
        # 拼接 F_v', F_t', F_t -> memory
        memory = torch.cat([F_v_prime, F_t_prime, F_t], dim=1)  # (B, N_v + 2 * N_t, C)

        # 使用 TextDecoder 进行生成
        output, _ = self.text_decoder(memory, target_sequence=target_sequence)

        return output
    
class WordTranslator:
    def __init__(self, model_file_path):
        """
        初始化 TextDecoder 类。

        参数：
        - model_file_path (str): SentencePiece 模型文件的路径。
        """
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"模型文件 {model_file_path} 不存在。")
        
        self.vocab = spm.SentencePieceProcessor(model_file=model_file_path)
        self.pad_id = self.vocab.pad_id()
        self.bos_id = self.vocab.bos_id()
        self.eos_id = self.vocab.eos_id()

    def decode(self, result_findings):
        """
        将模型输出的结果映射回文本。

        参数：
        - result_findings (torch.Tensor): 模型的输出，形状为 (batch_size, seq_len, vocab_size)。

        返回：
        - decoded_texts (list): 解码后的文本列表。
        """
        if not isinstance(result_findings, torch.Tensor):
            raise TypeError("输入的 result_findings 必须是 torch.Tensor 类型。")
        
        if result_findings.dim() != 3:
            raise ValueError("输入的 result_findings 必须是三维张量，形状为 (batch_size, seq_len, vocab_size)。")
        
        # 获取每个时间步的预测标记 ID
        predicted_ids = torch.argmax(result_findings, dim=-1)  # 形状为 (batch_size, seq_len)

        decoded_texts = []
        for ids in predicted_ids:
            # 将 Tensor 转换为列表
            ids = ids.tolist()
            # 移除特殊标记
            ids = [id for id in ids if id not in (self.pad_id, self.bos_id, self.eos_id)]
            # 解码为文本
            text = self.vocab.decode(ids)
            decoded_texts.append(text)
        
        return decoded_texts
    

class HiMrGn(nn.Module):
    def __init__(self, image_encoder, features_projector, findings_decoder, co_attention_module, impression_decoder, cxr_bert_feature_extractor):
        super().__init__()
        self.image_encoder = image_encoder
        self.features_projector = features_projector
        self.findings_decoder = findings_decoder
        self.co_attention_module = co_attention_module
        self.impression_decoder = impression_decoder
        self.cxr_bert_feature_extractor = cxr_bert_feature_extractor
        
    def forward(self, image, vpos=None, findings=None, impression=None, threshold=0.15, bos_id=1, eos_id=2, pad_id=3, max_len=1000, get_emb=False):
        x = self.image_encoder(image)   # (B, C)

        F_v = self.features_projector(x)    # (B, Nv, Cv)

        findings, F_t = self.findings_decoder(F_v, findings)    # (B, max_len, vocab_size), (B, max_len, hidden_dim)         

        F_t_prime, F_v_prime = self.co_attention_module(F_t, F_v)   # (B, max_len, hidden_dim), (B, Nv, Cv)      

        impression = self.impression_decoder(F_v_prime, F_t_prime, F_t, target_sequence=impression) # (B, max_len, vocab_size)

        F_F = self.cxr_bert_feature_extractor(findings)     #（B, 768）
        F_I = self.cxr_bert_feature_extractor(impression)   # (B, 768)


        return findings, impression


class ClsGenInt(nn.Module):
    def __init__(self, clsgen, interpreter, freeze_evaluator=True):
        super().__init__()
        self.clsgen = clsgen
        self.interpreter = interpreter
            
        # Freeze evaluator's paramters
        if freeze_evaluator:
            for param in self.interpreter.parameters():
                param.requires_grad = False

    def forward(self, image, history=None, caption=None, label=None, threshold=0.15, bos_id=1, eos_id=2, pad_id=3, max_len=300):        
        if caption != None:
            pad_mask = (caption == pad_id)
            cap_gen, img_mlc, cap_emb = self.clsgen(image, history, caption, label, threshold, bos_id, eos_id, pad_id, max_len, True)
            cap_mlc = self.interpreter(txt_embed=cap_emb, pad_mask=pad_mask)
            return cap_gen, img_mlc, cap_mlc
        else:
            return self.clsgen(image, history, caption, label, threshold, bos_id, eos_id, pad_id, max_len, False)