import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from transformers import AutoModel, AutoTokenizer
import sentencepiece as spm
import math
        
class CXR_BERT_FeatureExtractor(nn.Module):
    def __init__(self, model_name='microsoft/BiomedVLP-CXR-BERT-specialized', device='cuda'):
        super(CXR_BERT_FeatureExtractor, self).__init__()
        self.device = device
        # 加载预训练的 CXR-BERT 模型和对应的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

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
    def __init__(self, tokenizer_model_name='microsoft/BiomedVLP-CXR-BERT-specialized', input_dim=512, hidden_dim=768, num_head=8, num_layers=6, max_len=256):
        super(TextDecoder, self).__init__()
        self.embedding_layer = AutoModel.from_pretrained(
            tokenizer_model_name, 
            trust_remote_code=True
        ).bert.embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, trust_remote_code=True)

        self.vocab_size = self.tokenizer.vocab_size
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_head), num_layers=num_layers
        ) 
        self.fc_out = nn.Linear(hidden_dim, self.vocab_size)  # 输出词汇分布
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

    def forward(self, fv, target_embed=None):
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

        if self.training:  # 训练阶段
            seq_len = target_embed.size(1)

            # 嵌入目标序列并加上 Sinusoidal 位置编码
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

            # 生成文本
            decoded_texts = []
            pred_tokens = torch.argmax(output, dim=-1)  # (B, seq_len)
            
            for tokens in pred_tokens:
                try:
                    sep_pos = tokens.tolist().index(self.eos_id)
                    text = self.tokenizer.decode(tokens[:sep_pos], skip_special_tokens=True)
                except:
                    text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                decoded_texts.append(text)


            return output, F_t, decoded_texts

        else:  # 测试阶段
            outputs_probs = torch.zeros(batch_size, self.max_len, self.vocab_size).to(fv.device)
            outputs_probs[:, 0, self.bos_id] = 1.0
            current_tokens = torch.full((batch_size, 1), self.bos_id, dtype=torch.long).to(fv.device)
            
            F_t_list = []
            decoded_texts = [''] * batch_size
            
            for t in range(1, self.max_len):
                # 获取当前序列的 embedding
                encoded = {
                    'input_ids': current_tokens,
                    'token_type_ids': torch.zeros_like(current_tokens)
                }
                
                with torch.no_grad():
                    current_embed = self.embedding_layer(
                        input_ids=encoded['input_ids'],
                        token_type_ids=encoded['token_type_ids']
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
                output_t = self.fc_out(decoder_output[:, -1, :])  # (B, vocab_size)
                probs = F.softmax(output_t / 0.7, dim=-1)
                outputs_probs[:, t, :] = probs

                # 选择最可能的 token
                next_token = output_t.argmax(dim=-1)
                # next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # 更新当前序列
                current_tokens = torch.cat([current_tokens, next_token.unsqueeze(1)], dim=1)
                # 解码文本
                for i in range(batch_size):
                    decoded_texts[i] += self.tokenizer.decode(next_token[i].item(), skip_special_tokens=True)
                
                # 计算并存储 F_t
                F_t_t = F.softmax(decoder_output[:, -1, :], dim=-1)
                F_t_list.append(F_t_t)
                
                if (next_token == self.eos_id).all():
                    break
            
            output = outputs_probs
            F_t = torch.stack(F_t_list, dim=1)  # (B, seq_len, hidden_dim)

            return output, F_t, decoded_texts
    
class FindingsGenerator(nn.Module):
    def __init__(self, text_decoder):
        """
        Findings Generator 封装 TextDecoder 实现。
        Args:
            text_decoder: 一个 TextDecoder 实例，用于解码器生成。
        """
        super(FindingsGenerator, self).__init__()
        self.text_decoder = text_decoder

    def forward(self, F_v, target_embed=None):
        """
        Args:
            F_v: 输入的视觉特征，形状 (B, N_v, C_v)。
            target_embed: 已经是embedding形式的目标序列，形状 (B, max_len, hidden_dim)，仅在训练时提供。
        Returns:
            output: 生成的词汇分布，形状 (B, max_len, vocab_size)。
            F_t_decoded: 解码器的隐藏状态，形状 (B, max_len, hidden_dim)。
        """
        # 使用 TextDecoder 进行生成
        output, F_t_decoded, findings_text = self.text_decoder(F_v, target_embed=target_embed)

        return output, F_t_decoded, findings_text


    
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
        Impression Generator 复用 TextDecoder 实现。
        Args:
            text_decoder: 一个 TextDecoder 实例，用于解码器生成。
        """
        super(ImpressionGenerator, self).__init__()
        self.text_decoder = text_decoder

    def forward(self, F_v_prime, F_t_prime, F_t, target_embed=None):
        """
        Args:
            F_v_prime: 增强的视觉特征 (B, N_v, C_v)。
            F_t_prime: 增强的文本特征 (B, N_t, C_t)。
            F_t: 原始的文本特征 (B, N_t, C_t)。
            target_embed: 目标序列 (B, max_len, hidden_dim)，仅在训练时提供。
        Returns:
            output: 生成的词汇分布 (B, max_len, vocab_size)。
            F_t_decoded: 解码过程中的隐藏状态 (B, max_len, hidden_dim)。
        """
        # 拼接 F_v', F_t', F_t -> memory
        memory = torch.cat([F_v_prime, F_t_prime, F_t], dim=1)  # (B, N_v + 2 * N_t, C)

        # 使用 TextDecoder 进行生成
        output, _ = self.text_decoder(memory, target_embed=target_embed)

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

    def forward(self, image, findings=None, impression=None, history=None, train_stage=2, idx=None):
        if train_stage == 1:
            x = self.image_encoder(image[0])   # (B, C)

            # F_v = self.features_projector(x)    # (B, Nv, Cv)
            F_v = x

            fusion_features = self.modality_fusion(F_v, history)

            findings, _, findings_text = self.findings_decoder(fusion_features, findings)    # (B, max_len, vocab_size), (B, max_len, hidden_dim)

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

            # F_v = self.features_projector(x)    # (B, Nv, Cv)
            F_v = x

            fusion_features = self.modality_fusion(F_v, history)

            findings, F_t = self.findings_decoder(fusion_features, findings)    # (B, max_len, vocab_size), (B, max_len, hidden_dim)         

            F_t_prime, F_v_prime = self.co_attention_module(F_t, F_v)   # (B, max_len, hidden_dim), (B, Nv, Cv)  

            memory, impression = self.impression_decoder(F_v_prime, F_t_prime, F_t, target_embed=impression) # (B, max_len, vocab_size)

            class_logits = self.multi_label_classifier(memory)
            
            F_F, findings_text = self.cxr_bert_feature_extractor(findings)     #（B, 768）
            F_I, impression_text = self.cxr_bert_feature_extractor(impression)   # (B, 768)

            return {
                "findings": findings, 
                "findings_text": findings_text,
                "impression": impression, 
                "impression_text": impression_text,
                "F_F": F_F, 
                "F_I": F_I,
                "class_logits": class_logits
            }


