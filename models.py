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
        # 提取图像的多层特征
        features = self.image_encoder(image)
        
        # 获取最后一层特征并调整维度顺序 (B, C, H, W)
        features_last = features[-1].permute(0, 3, 1, 2)    # (2,1536,7,7)
        
        # 仅使用最后一层特征进行降维和处理
        fv = self.feature_proj(features_last)  
        features_last = features_last.squeeze(-1).squeeze(-1)      # 输出形状 (B, 512)
        
        return features_last
    
class DiseaseFeatureProjector(nn.Module):
    def __init__(self, input_dim=512, num_diseases=512, feature_dim=512):
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
    def __init__(self, tokenizer_model_name='microsoft/BiomedVLP-CXR-BERT-specialized', input_dim=512, hidden_dim=512, num_head=8, num_layers=6, max_len=512):
        super(TextDecoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, trust_remote_code=True)  # 使用 CXR-BERT 的 tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, hidden_dim)  # 词嵌入
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
    
class HiMrGn(nn.Module):
    def __init__(self, image_encoder, features_projector, findings_decoder, co_attention_module, impression_decoder, cxr_bert_feature_extractor):
        super().__init__()
        self.image_encoder = image_encoder
        self.features_projector = features_projector
        self.findings_decoder = findings_decoder
        self.co_attention_module = co_attention_module
        self.impression_decoder = impression_decoder
        self.cxr_bert_feature_extractor = cxr_bert_feature_extractor
        
    def forward(self, image, vpos=None, findings=None, impression=None):
        x = self.image_encoder(image[0])   # (B, C)

        F_v = self.features_projector(x)    # (B, Nv, Cv)

        findings, F_t = self.findings_decoder(F_v, findings)    # (B, max_len, vocab_size), (B, max_len, hidden_dim)         

        F_t_prime, F_v_prime = self.co_attention_module(F_t, F_v)   # (B, max_len, hidden_dim), (B, Nv, Cv)      

        impression = self.impression_decoder(F_v_prime, F_t_prime, F_t, target_sequence=impression) # (B, max_len, vocab_size)

        F_F, findings_text = self.cxr_bert_feature_extractor(findings)     #（B, 768）
        F_I, impression_text = self.cxr_bert_feature_extractor(impression)   # (B, 768)


        return {
            "findings": findings, 
            "impression": impression, 
            "F_F": F_F, 
            "F_I": F_I
        }


