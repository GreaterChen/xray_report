import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from transformers import AutoModel, AutoTokenizer
import sentencepiece as spm
import math
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import Transformer

class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 tokenizer,
                 hidden_dim=768,
                 prompt='',
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.prompt = prompt
        
        # 加载BERT配置
        decoder_config = BertConfig.from_json_file('/home/chenlb/xray_report_generation/configs/bert_config.json')
        decoder_config.encoder_width = hidden_dim
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        
        # 初始化解码器
        self.text_decoder = BertLMHeadModel.from_pretrained(
            'microsoft/BiomedVLP-CXR-BERT-specialized',
            config=decoder_config
        )
        
        # 调整词表大小
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

    def forward(self, encoder_hidden_states, target_embed=None):
        """
        Args:
            encoder_hidden_states: 视觉特征，形状 (B, N_v, C_v)
            target_embed: 目标文本的token ids，形状 (B, max_len)
        Returns:
            output: 生成的词汇分布，形状 (B, max_len, vocab_size)
            hidden_states: 文本特征，形状 (B, max_len, hidden_dim)
            decoded_texts: 生成的文本列表
        """
        if target_embed is not None:
            # 构建输入
            attention_mask = (target_embed != self.tokenizer.pad_token_id).long()
            
            # 设置decoder targets，忽略padding
            decoder_targets = target_embed.clone()
            decoder_targets[decoder_targets == self.tokenizer.pad_token_id] = -100
            
            # 前向传播
            outputs = self.text_decoder(
                input_ids=target_embed,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                labels=decoder_targets,
                return_dict=True
            )
            
            # 获取logits和隐藏状态
            logits = outputs.logits  # (B, max_len, vocab_size) 
            hidden_states = outputs.hidden_states[-1]  # (B, max_len, hidden_dim)
            
            # 解码生成的文本
            pred_tokens = torch.argmax(logits, dim=-1)
            decoded_texts = []
            for tokens in pred_tokens:
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                decoded_texts.append(text)
            
            return logits, hidden_states, decoded_texts
        
        else:
            # 使用generate方法生成文本
            captions = self.generate(encoder_hidden_states)
            return None, None, captions

    def generate(self, image_embeds, sample=False, num_beams=3, max_length=100, min_length=10, top_p=0.9, repetition_penalty=1.0):
        batch_size = image_embeds.size(0)
        
        # 创建attention mask
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        
        # 初始化输入
        input_ids = torch.ones((batch_size, 1), dtype=torch.long).to(image_embeds.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        
        # 生成文本
        outputs = self.text_decoder.generate(
            input_ids=input_ids,
            min_length=min_length,
            max_new_tokens=max_length,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            **model_kwargs
        )
        
        # 解码生成的文本
        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption)
            
        return captions



        
class CXR_BERT_FeatureExtractor(nn.Module):
    def __init__(self, tokenizer, device='cuda'):
        super(CXR_BERT_FeatureExtractor, self).__init__()
        self.device = device
        # 加载预训练的 CXR-BERT 模型和对应的分词器
        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized', trust_remote_code=True).to(self.device)

        # Freeze all parameters in the BERT model (no training of these parameters)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()  # 设置模型为评估模式

    def forward(self, inputs):
        """
        inputs: 输入的文本概率列表，形状为 (B, seq_len, vocab_size)
        返回:
        features: 文本特征张量，形状为 (B, hidden_size)
        """
        # 从概率分布中选择每个位置最大概率的 token ID
        # 通过 torch.argmax 获取每个序列位置最可能的 token
        token_ids = torch.argmax(inputs, dim=-1)  # 选择每个位置最大概率的 token ID，形状为 (B, seq_len)
        
        # 使用 tokenizer 对 token IDs 进行解码
        texts = [self.tokenizer.decode(token_id, skip_special_tokens=True) for token_id in token_ids]
        
        # 对输入文本进行编码
        inputs = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 获取 [CLS] 标记的嵌入表示
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        return cls_embeddings, texts


class SwinFeatureExtractor(nn.Module):
    def __init__(self, image_encoder_name='swin_large_patch4_window7_224', hidden_dim=768, pretrained=True):
        super().__init__()
        # 加载预训练的 Swin Transformer
        self.image_encoder = create_model(image_encoder_name, pretrained=pretrained, features_only=True)
        
        # 映射到低维视觉特征 Fv
        self.feature_proj = nn.Sequential(
            nn.Conv2d(self.image_encoder.feature_info[-1]['num_chs'], hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 提取全局特征
        )
    
    def forward(self, image):
        """
        image: 输入的图像，形状为 (B, C, H, W)
        返回:
        Fv: 视觉特征，形状为 (B, hidden_dim)
        """
        # 提取图像的多层特征
        features = self.image_encoder(image)
        
        # 获取最后一层特征并调整维度顺序 (B, C, H, W)
        features_last = features[-1].permute(0, 3, 1, 2)    # (2,1536,7,7)
        
        # 仅使用最后一层特征进行降维和处理
        fv = self.feature_proj(features_last)  
        fv = fv.squeeze(-1).squeeze(-1)      # 输出形状 (B, hidden_dim)
        
        return fv
    
class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        # 加载预训练的 ViT base版本 (768维)
        self.image_encoder = create_model(model_name, pretrained=pretrained)
        
    def forward(self, image):
        """
        image: 输入的图像，形状为 (B, C, H, W)
        返回:
        features: ViT最后一层的输出特征，形状为 (B, 768)
        """
        # 获取ViT最后一层[CLS] token的输出
        features = self.image_encoder.forward_features(image)
        
        return features
    
class DiseaseFeatureProjector(nn.Module):
    def __init__(self, input_dim=768, num_diseases=512, feature_dim=768):
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
    
class ModalityFusion(nn.Module):
    def __init__(self, d_model=768, input_dim=256+197, nhead=8, num_encoder_layers=6, dropout=0.1):
        super().__init__()
        
        # Transform layer parameters
        self.d_model = d_model  # 特征维度
        self.nhead = nhead      # 注意力头数
        
        # Position encoding
        self.pos_encoder = nn.Parameter(torch.randn(input_dim, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, image_features, text_features):
        """
        Args:
            image_features: shape (batch_size, 256, 768)
            text_features: shape (batch_size, 256, 768)
        Returns:
            fused_features: shape (batch_size, 512, 768)
        """
        batch_size = image_features.size(0)
        
        # Concatenate features
        fused_features = torch.cat([image_features, text_features], dim=1)  # (batch_size, 512, 768)
        
        # Add positional encoding
        fused_features = fused_features + self.pos_encoder.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer encoder
        fused_features = self.transformer_encoder(fused_features)
        
        # Apply layer normalization
        fused_features = self.norm(fused_features)
        
        return fused_features
    
class TextDecoder(nn.Module):
    def __init__(self, tokenizer, input_dim=512, hidden_dim=768, num_head=8, num_layers=6, max_len=256):
        super(TextDecoder, self).__init__()
        bert = AutoModel.from_pretrained(
            'microsoft/BiomedVLP-CXR-BERT-specialized', 
            trust_remote_code=True
        )
        self.embedding_layer = bert.bert.embeddings
        self.tokenizer = tokenizer
        # 冻结原始embedding
        self.embedding_layer.word_embeddings.weight.requires_grad = False
        self.output_weight = nn.Parameter(
            self.embedding_layer.word_embeddings.weight.clone()
        )

        self.vocab_size = self.tokenizer.vocab_size
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_head), num_layers=num_layers
        ) 
        self.fc_out = nn.Linear(hidden_dim, self.vocab_size, bias=False)  # 输出词汇分布
        self.fc_out.weight = self.output_weight

        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # 获取 CXR-BERT 特殊标记
        self.pad_id = self.tokenizer.pad_token_id
        self.bos_id = self.tokenizer.cls_token_id  # 使用 CLS 作为 BOS
        self.eos_id = self.tokenizer.sep_token_id  # 使用 SEP 作为 EOS

        # 生成 Sinusoidal 位置编码
        self.register_buffer("positional_encoding", self._get_sinusoidal_encoding(max_len, hidden_dim))

    def _get_sinusoidal_encoding(self, max_len, d_model):
        """
        创建 Sinusoidal 位置编码。
        """
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # (d_model // 2)
        
        # 计算 sin 和 cos 编码
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用 cos
        
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, fv, target_embed=None, current_step=None, total_steps=None):
        """
        Args:
            fv: 疾病特征矩阵，形状 (B, N_v, C_v)，作为 memory。
            target_embed: 已经是embedding形式的目标序列，形状 (B, max_len, hidden_dim)
        Returns:
            output: 生成的词汇分布，形状 (B, max_len, vocab_size)。
            F_t: 文本特征矩阵，形状 (B, max_len, hidden_dim)。
        """
        batch_size = fv.size(0)
        memory = fv.permute(1, 0, 2)  # 转换为 (N_v, B, C_v)

        if self.training:
            seq_len = target_embed.size(1)
            position_encoding = self.positional_encoding[:, :seq_len, :].to(target_embed.device)
            
            # 初始化输出序列
            outputs = []
            F_t_list = []
            
            # 第一个token的embedding加上位置编码
            current_input = target_embed[:, 0:1, :] + position_encoding[:, 0:1, :]
            current_sequence = current_input  # 用于存储完整序列
            
            # 计算teacher forcing比率
            min_ratio = 0.2
            max_ratio = 1.0
            current_ratio = max(min_ratio, max_ratio - (max_ratio - min_ratio) * current_step / total_steps)
            
            # 逐token生成
            for t in range(1, seq_len):
                # Transformer解码
                # 注意：current_sequence已经包含了position encoding
                target_mask = torch.triu(torch.ones(t, t), diagonal=1).bool().to(target_embed.device)
                
                decoder_output = self.transformer_decoder(
                    tgt=current_sequence.permute(1, 0, 2),
                    memory=memory,
                    tgt_mask=target_mask
                )
                decoder_output = decoder_output.permute(1, 0, 2)
                
                # 生成当前时间步的输出
                current_output = self.fc_out(decoder_output[:, -1:, :])  # (B, 1, vocab_size)
                outputs.append(current_output)
                
                # 计算F_t
                F_t_t = F.softmax(decoder_output[:, -1:, :], dim=-1)
                F_t_list.append(F_t_t)
                
                # Scheduled Sampling
                use_teacher_forcing = (torch.rand(1).item() < current_ratio)
                
                if use_teacher_forcing:
                    # 使用ground truth
                    next_embed = target_embed[:, t:t+1, :]
                else:
                    # 使用模型预测
                    pred_token = torch.argmax(current_output, dim=-1)  # (B, 1)
                    next_embed = self.embedding_layer(
                        input_ids=pred_token,
                        token_type_ids=torch.zeros_like(pred_token)
                    )
                
                # 为新token添加位置编码
                next_input = next_embed + position_encoding[:, t:t+1, :]
                
                # 更新序列
                current_sequence = torch.cat([current_sequence, next_input], dim=1)
            
            # 合并所有输出
            output = torch.cat(outputs, dim=1)  # (B, seq_len-1, vocab_size)
            F_t = torch.cat(F_t_list, dim=1)    # (B, seq_len-1, hidden_dim)
            
            # 生成文本用于打印
            decoded_texts = []
            pred_tokens = torch.argmax(output, dim=-1)
            for tokens in pred_tokens:
                try:
                    sep_pos = tokens.tolist().index(self.eos_id)
                    text = self.tokenizer.decode(tokens[:sep_pos], skip_special_tokens=True)
                except:
                    text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                decoded_texts.append(text)
            print(decoded_texts[0])

            return output, F_t, decoded_texts

        else:  # 测试阶段
            outputs_probs = torch.zeros(batch_size, self.max_len, self.vocab_size).to(fv.device)
            outputs_probs[:, 0, self.bos_id] = 1.0
            current_tokens = torch.full((batch_size, 1), self.bos_id, dtype=torch.long).to(fv.device)
            
            F_t_list = []
            decoded_texts = [''] * batch_size
            
            # 初始化重复惩罚记录器
            repetition_penalty = 1.2  # 惩罚系数，大于1
            token_counts = torch.ones(batch_size, self.vocab_size).to(fv.device)  # 平滑处理，避免除0
            
            for t in range(1, self.max_len):
                # 获取当前序列的 embedding
                with torch.no_grad():
                    current_embed = self.embedding_layer(
                        input_ids=current_tokens,
                        token_type_ids= torch.zeros_like(current_tokens)
                    )
                
                # 添加位置编码
                position_encoding = self.positional_encoding[:, :t, :].to(current_embed.device)
                target_embed = current_embed + position_encoding
                
                # 创建自回归掩码
                target_mask = torch.triu(torch.ones(t, t), diagonal=1).bool().to(target_embed.device)
                
                # Transformer 解码
                decoder_output = self.transformer_decoder(
                    tgt=target_embed.permute(1, 0, 2),
                    memory=memory,
                    tgt_mask=target_mask
                )
                decoder_output = decoder_output.permute(1, 0, 2)
                
                # 计算当前时间步的词汇分布
                temperature = 0.7
                output_t = self.fc_out(decoder_output[:, -1, :]) / temperature
                
                # 应用重复惩罚
                for i in range(batch_size):
                    # 获取已生成的token
                    generated_tokens = current_tokens[i]
                    # 更新token计数
                    for token in generated_tokens:
                        token_counts[i, token] += 1
                    
                    # 计算惩罚项
                    penalty = torch.ones_like(output_t[i])
                    penalty.scatter_(0, generated_tokens, repetition_penalty)
                    
                    # 应用惩罚
                    output_t[i] = torch.where(
                        output_t[i] > 0,
                        output_t[i] / penalty,
                        output_t[i] * penalty
                    )
                
                # 计算概率分布
                probs = F.softmax(output_t, dim=-1)
                outputs_probs[:, t, :] = probs
                
                # 动态调整采样范围
                top_k = max(5, int(self.vocab_size * (1 - t/self.max_len)))  # 随着生成过程推进逐渐减小采样范围
                top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
                
                # 应用重复度过滤
                filtered_probs = top_k_probs.clone()
                for i in range(batch_size):
                    # 计算token的重复率
                    repeat_rates = token_counts[i, top_k_indices[i]] / t
                    # 对重复率高的token降低其概率
                    filtered_probs[i] = top_k_probs[i] * torch.exp(-repeat_rates)
                    # 重新归一化
                    filtered_probs[i] = filtered_probs[i] / filtered_probs[i].sum()
                
                # 在过滤后的概率分布中采样
                next_token_idx = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
                next_token = top_k_indices[torch.arange(batch_size), next_token_idx]
                
                # 更新当前序列
                current_tokens = torch.cat([current_tokens, next_token.unsqueeze(1)], dim=1)
                
                # 解码文本
                for i in range(batch_size):
                    decoded_texts[i] += self.tokenizer.decode(next_token[i].item(), skip_special_tokens=True)
                
                # 计算并存储 F_t
                F_t_t = F.softmax(decoder_output[:, -1, :], dim=-1)
                F_t_list.append(F_t_t)
                
                # 检查序列是否应该结束
                if (next_token == self.eos_id).all():
                    break
            
            output = outputs_probs
            F_t = torch.stack(F_t_list, dim=1)  # (B, seq_len, hidden_dim)

            return output, F_t, decoded_texts
        

class FindingsGenerator(nn.Module):
    def __init__(self, text_decoder):
        """
        Findings Generator 使用 BLIP_Decoder 实现。
        Args:
            text_decoder: BLIP_Decoder 实例
        """
        super(FindingsGenerator, self).__init__()
        self.text_decoder = text_decoder

    def forward(self, F_v, target_embed=None):
        """
        Args:
            F_v: 输入的视觉特征，形状 (B, N_v, C_v)
            target_embed: 目标文本的token ids，形状 (B, max_len)
        Returns:
            output: 生成的词汇分布，形状 (B, max_len, vocab_size)
            F_t: 解码器的隐藏状态，形状 (B, max_len, hidden_dim)
            findings_text: 生成的文本列表
        """
        # 使用 BLIP_Decoder 进行生成
        output, F_t, findings_text = self.text_decoder(F_v, target_embed)
        
        return output, F_t, findings_text


    
class CoAttentionBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
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
    def __init__(self, embed_dim=768, num_heads=8, num_blocks=6):
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

class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=384):
        """
        多标签分类器
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
        """
        super(MultiLabelClassifier, self).__init__()
        
        # 全局平均池化，用于将序列特征压缩为单个向量
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器网络
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加层归一化提高稳定性
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 14),  # 14个类别
        )

    def forward(self, memory):
        """
        Args:
            memory: 拼接后的特征 (B, seq_len, hidden_dim)
        Returns:
            class_logits: 14个类别的预测概率 (B, 14)
        """
        # 全局平均池化
        pooled_features = self.global_pool(memory.transpose(1, 2)).squeeze(-1)  # (B, hidden_dim)
        
        # 分类预测
        class_logits = self.classifier(pooled_features)  # (B, 14)
        
        return class_logits


class ImpressionGenerator(nn.Module):
    def __init__(self, text_decoder):
        """
        Impression Generator 使用 BLIP_Decoder 实现。
        Args:
            text_decoder: BLIP_Decoder 实例
        """
        super(ImpressionGenerator, self).__init__()
        self.text_decoder = text_decoder

    def forward(self, F_v_prime, F_t_prime, F_t, target_embed=None):
        """
        Args:
            F_v_prime: 增强的视觉特征 (B, N_v, C_v)
            F_t_prime: 增强的文本特征 (B, N_t, C_t)
            F_t: 原始的文本特征 (B, N_t, C_t)
            target_embed: 目标文本的token ids，形状 (B, max_len)
        Returns:
            memory: 拼接后的特征 (B, N_v + 2*N_t, C)
            output: 生成的词汇分布 (B, max_len, vocab_size)
        """
        # 拼接特征
        memory = torch.cat([F_v_prime, F_t_prime, F_t], dim=1)  # (B, N_v + 2*N_t, C)

        # 使用 BLIP_Decoder 进行生成
        output, _, _ = self.text_decoder(memory, target_embed)

        return memory, output
    
class HiMrGn(nn.Module):
    def __init__(self, image_encoder, features_projector, modality_fusion, findings_decoder, multi_label_classifier, co_attention_module, impression_decoder, cxr_bert_feature_extractor):
        super().__init__()
        self.image_encoder = image_encoder
        self.features_projector = features_projector
        self.modality_fusion = modality_fusion
        self.findings_decoder = findings_decoder
        self.co_attention_module = co_attention_module
        self.impression_decoder = impression_decoder
        self.cxr_bert_feature_extractor = cxr_bert_feature_extractor
        self.multi_label_classifier = multi_label_classifier

    def forward(self, image, findings=None, impression=None, history=None, train_stage=2):
        if train_stage == 1:
            x = self.image_encoder(image[0])   # (B, C)
            F_v = x

            fusion_features = self.modality_fusion(F_v, history)

            findings, _, findings_text = self.findings_decoder(fusion_features, findings)    

            return {
                "findings": findings, 
                "findings_text": findings_text,
                "impression": None, 
                "impression_text": None,
                "F_F": None, 
                "F_I": None,
                "class_logits": None
            }
        
        elif train_stage == 2:
            x = self.image_encoder(image[0])   # (B, C)
            F_v = x

            fusion_features = self.modality_fusion(F_v, history)

            findings, F_t = self.findings_decoder(fusion_features, findings)         

            F_t_prime, F_v_prime = self.co_attention_module(F_t, F_v)  

            memory, impression = self.impression_decoder(F_v_prime, F_t_prime, F_t, impression)

            class_logits = self.multi_label_classifier(memory)
            
            F_F, findings_text = self.cxr_bert_feature_extractor(findings)
            F_I, impression_text = self.cxr_bert_feature_extractor(impression)

            return {
                "findings": findings, 
                "findings_text": findings_text,
                "impression": impression, 
                "impression_text": impression_text,
                "F_F": F_F, 
                "F_I": F_I,
                "class_logits": class_logits
            }


