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
import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import Transformer
from transformers import SwinModel

class BLIP_Decoder(nn.Module):
    def __init__(self,
                 args,
                 tokenizer,
                 hidden_dim=768,
                 prompt='',
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.prompt = prompt
        
        # 加载BERT配置
        decoder_config = BertConfig.from_json_file(os.path.join(args.root_dir, 'configs/bert_config.json'))
        decoder_config.encoder_width = hidden_dim
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        
        # 初始化解码器
        self.text_decoder = BertLMHeadModel.from_pretrained(
            'bert-base-uncased',
            config=decoder_config
        )
        
        # 调整词表大小
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

    def forward(self, encoder_hidden_states, text=None):
        """
        Args:
            encoder_hidden_states: 视觉特征，形状 (B, N_v, C_v)
            target_ids: 目标文本的token ids，形状 (B, max_len)
        Returns:
            output: 生成的词汇分布，形状 (B, max_len, vocab_size)
            hidden_states: 文本特征，形状 (B, max_len, hidden_dim)
            decoded_texts: 生成的文本列表
        """
        if text is not None:
            decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)
            # 前向传播
            outputs = self.text_decoder(
                input_ids=text.input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                labels=decoder_targets,
                output_hidden_states=True,  # 添加这个参数以获取隐藏状态
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

            loss_lm = outputs.loss
            
            return logits, hidden_states, decoded_texts, loss_lm
        
        else:
            # 使用generate方法生成文本
            logits, hidden_states, captions = self.generate(encoder_hidden_states)
            return logits, hidden_states, captions, None

    def generate(self, image_embeds, sample=False, num_beams=3, max_length=196, min_length=100, top_p=0.9, repetition_penalty=1.0):
        batch_size = image_embeds.size(0)
        
        # 创建attention mask
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        model_kwargs = {
            "encoder_hidden_states": image_embeds, 
            "encoder_attention_mask": image_atts,
            # "output_hidden_states": True,  # 添加这个参数以获取隐藏状态
            # "return_dict_in_generate": True,  # 确保返回字典格式
            # "output_scores": True,  # 获取生成的分数
        }
        
        # 初始化输入
        input_ids = torch.ones((batch_size, 1), dtype=torch.long).to(image_embeds.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        
        # 生成文本
        outputs = self.text_decoder.generate(
            input_ids=input_ids,
            # min_length=min_length,
            max_new_tokens=max_length - 1,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            **model_kwargs
        )
        
        # 获取生成的序列
        # generated_tokens = outputs.sequences  # (batch_size, seq_len)
        generated_tokens = outputs
        
        # 由于beam search生成不会返回hidden states，我们需要手动计算
        # 使用forward pass获取hidden states
        attention_mask = (generated_tokens != self.tokenizer.pad_token_id).long()
        decoder_outputs = self.text_decoder(
            input_ids=generated_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = decoder_outputs.hidden_states[-1]  # 获取最后一层的hidden states
        
        # 解码生成的文本
        captions = []
        for tokens in generated_tokens:
            caption = self.tokenizer.decode(tokens, skip_special_tokens=True)
            captions.append(caption)
            
        return None, hidden_states, captions

        
class CXR_BERT_FeatureExtractor(nn.Module):
    def __init__(self, tokenizer, device='cuda'):
        super(CXR_BERT_FeatureExtractor, self).__init__()
        self.device = device
        # 加载预训练的 CXR-BERT 模型和对应的分词器
        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained('bert-base-uncased', trust_remote_code=True).to(self.device)

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
        self.image_encoder = create_model(image_encoder_name, pretrained=True, features_only=True)

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
        Fv: 视觉特征，形状为 (B, 196, 768)
        """
        # 提取图像的多层特征
        features = self.image_encoder(image)

        """
        Swin Transformer 
        最后一层输出形状 (B, 7, 7, 1536)
        倒数第二层输出形状 (B, 14, 14, 768)

        """
        
        # 获取最后一层特征并调整维度顺序 (B, C, H, W)
        features_last = features[-2].permute(0, 3, 1, 2)    # (2,768,14,14)
        b,c,h,w = features_last.size()
        fv = features_last.view(b, c, h*w).permute(0, 2, 1)
        
        return fv
    
class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()

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
            image_features: shape (batch_size, 196, 768)
            text_features: shape (batch_size, 256, 768)
        Returns:
            fused_features: shape (batch_size, 452, 768)
        """
        batch_size = image_features.size(0)
        
        # Concatenate features
        fused_features = torch.cat([image_features, text_features], dim=1)  # (batch_size, 452, 768)
        
        # Add positional encoding
        fused_features = fused_features + self.pos_encoder.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply transformer encoder
        fused_features = self.transformer_encoder(fused_features)
        
        # Apply layer normalization
        fused_features = self.norm(fused_features)
        
        return fused_features
    
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
        output, F_t, findings_text, loss_lm = self.text_decoder(F_v, target_embed)
        
        return output, F_t, findings_text, loss_lm


    
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

    def forward(self, image, findings=None, impression=None, history=None, train_stage=2, mode='train'):
        if train_stage == 1:
            F_v = self.image_encoder(image)

            # fusion_features = self.modality_fusion(F_v, history)
            fusion_features = F_v

            if mode == 'train':
                findings, _, findings_text, loss_lm = self.findings_decoder(fusion_features, findings)
            else:
                _, _, findings_text, loss_lm = self.findings_decoder(fusion_features, None)

            return {
                "findings": findings, 
                "findings_text": findings_text,
                "loss_lm": loss_lm,
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


class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        model = getattr(models, 'resnet101')(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        modules = list(model.children())[:-3]
        self.model = nn.Sequential(*modules)
        
        # 添加线性映射层，将1024维降到768维
        self.dim_reduction = nn.Sequential(
            nn.Linear(1024, 512),  # 降到中间的低维度
            nn.ReLU(),             # 非线性激活
            nn.Linear(512, 768)    # 再升到目标维度
        )
        
    def forward(self, x):
        patch_feats = self.model(x)  # (batch_size, 1024, 14, 14)
        batch_size, feat_size, _, _ = patch_feats.shape
        
        # 重排并降维
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)  # (batch_size, 196, 1024)
        patch_feats = self.dim_reduction(patch_feats)  # (batch_size, 196, 768)
        
        return patch_feats