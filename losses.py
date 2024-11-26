import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.KLLoss = nn.KLDivLoss()

	def forward(self, output, target):
		'''
		Output: (N,*) \n
		Target: (N,*) \n
		'''
		output = torch.log(output)  # Invert softmax
		# target = torch.log(target) # Invert softmax
		# How output distribution differs from target distribution
		return self.KLLoss(output, target)


class CELoss(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_index)

	def forward(self, output, target):
		'''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
		output = torch.log(output)  # Invert softmax
		output = output.reshape(-1, output.shape[-1])  # (*,C)
		target = target.reshape(-1).long()  # (*)
		return self.CELoss(output, target)


class CELossSame(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_index)

	def forward(self, outputs, target):
		'''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
		output_img = torch.log(outputs[0]) # Invert softmax
		output_txt = torch.log(outputs[1])
		output_sen = torch.log(outputs[2])

		output_img = output_img.reshape(-1, output_img.shape[-1]) # (*,C)
		output_txt = output_txt.reshape(-1, output_txt.shape[-1]) # (*,C)
		output_sen = output_sen.reshape(-1, output_sen.shape[-1]) # (*,C)
		target = target.reshape(-1).long() # (*)
		return self.CELoss(output_img, target) + self.CELoss(output_txt, target) + self.CELoss(output_sen, target)

class CELossShift(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = CELoss(ignore_index=ignore_index)

	def forward(self, output, target):
		'''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
		output = output[:,:-1,:] # (* - 1,C)
		target = target[:,1:] # (* - 1)
		return self.CELoss(output, target)

class CELossTotal(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = CELoss()
		self.CELossShift = CELossShift(ignore_index=ignore_index)

	def forward(self, output, target):
		return self.CELossShift(output[0], target[0]) + self.CELoss(output[1], target[1])

class CELossTotalEval(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = CELoss()
		self.CELossShift = CELossShift(ignore_index=ignore_index)

	def forward(self, output, target):
		return self.CELossShift(output[0], target[0]) + self.CELoss(output[1], target[1]) + self.CELoss(output[2], target[1])

class CELossTransfer(nn.Module):
	def __init__(self, ignore_index=-1):
		super().__init__()
		self.CELoss = CELoss()
		self.CELossShift = CELossShift(ignore_index=ignore_index)

	def forward(self, output, target):
		return self.CELossShift(output[0], target[0]) # + self.CELoss(output[1], target[1])
	


class ContrastiveLearningLoss(nn.Module):
    def __init__(self, feature_dim=768, projection_dim=128, hidden_dim=256):
        """
        对比学习损失模块，基于 SimSiam。
        Args:
            feature_dim: 输入特征维度 (F_F 和 F_I)。
            projection_dim: 投影空间维度。
            hidden_dim: 预测 MLP 的隐藏层维度。
        """
        super(ContrastiveLearningLoss, self).__init__()
        # Projection MLP
        self.projection_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        # Prediction MLP
        self.prediction_mlp = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, F_F, F_I):
        """
        Args:
            F_F: Findings 的特征，形状 (B, feature_dim)。
            F_I: Impression 的特征，形状 (B, feature_dim)。
        Returns:
            loss_sim: 对比学习损失 (标量)。
        """
        # Step 1: Projection
        Z_F = self.projection_mlp(F_F)  # (B, projection_dim)
        Z_I = self.projection_mlp(F_I)  # (B, projection_dim)

        # Step 2: Prediction
        P_F = self.prediction_mlp(Z_F)  # (B, projection_dim)
        P_I = self.prediction_mlp(Z_I)  # (B, projection_dim)

        # Step 3: Normalize features for cosine similarity
        Z_F = F.normalize(Z_F, dim=-1)
        Z_I = F.normalize(Z_I, dim=-1)
        P_F = F.normalize(P_F, dim=-1)
        P_I = F.normalize(P_I, dim=-1)

        # Step 4: Cosine Similarity Loss
        loss_sim = (- F.cosine_similarity(P_F, Z_I.detach(), dim=-1).mean() \
            		- F.cosine_similarity(P_I, Z_F.detach(), dim=-1).mean()) / 2

        return loss_sim
	

class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self, pad_id=3):
        """
        计算生成序列的交叉熵损失。
        Args:
            pad_id: 填充值的索引，用于忽略填充标记的损失。
        """
        super(SequenceCrossEntropyLoss, self).__init__()
        self.pad_id = pad_id

    def forward(self, predictions, targets):
        """
        Args:
            predictions: 模型输出的词汇分布概率，形状 (B, seq_len, vocab_size)。
            targets: 目标序列，形状 (B, seq_len)。
        Returns:
            loss: 归一化的交叉熵损失 (标量)。
        """
        # 将预测的概率分布展平，适应交叉熵损失的输入
        batch_size, seq_len, vocab_size = predictions.size()
        predictions = predictions.view(-1, vocab_size)  # (B * seq_len, vocab_size)
        targets = targets.view(-1)  # (B * seq_len)

        # 忽略填充标记的损失
        loss = F.cross_entropy(predictions, targets, ignore_index=self.pad_id, reduction="mean")
        return loss
	
class CombinedLoss(nn.Module):
    def __init__(self, pad_id=3, feature_dim=768, projection_dim=128, hidden_dim=256, lambda_contrastive=1.0):
        """
        综合损失函数，包括：
        1. Findings 和 Impression 的交叉熵损失。
        2. Findings 和 Impression 特征的对比学习损失。
        
        Args:
            pad_id: 填充值的索引，用于忽略填充标记的交叉熵损失。
            feature_dim: 输入特征维度 (F_F 和 F_I)。
            projection_dim: 投影空间维度，用于对比学习。
            hidden_dim: 预测 MLP 的隐藏层维度。
            lambda_contrastive: 对比损失的权重因子。
        """
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = SequenceCrossEntropyLoss(pad_id=pad_id)
        self.contrastive_loss = ContrastiveLearningLoss(feature_dim=feature_dim, 
                                                        projection_dim=projection_dim, 
                                                        hidden_dim=hidden_dim)
        self.lambda_contrastive = lambda_contrastive

    def forward(self, output, targets):
        """
        Args:
            output: 模型输出，包含：
                - findings: 模型生成的 findings 概率分布 (B, seq_len, vocab_size)。
                - impression: 模型生成的 impression 概率分布 (B, seq_len, vocab_size)。
                - F_F: Findings 的特征 (B, feature_dim)。
                - F_I: Impression 的特征 (B, feature_dim)。
            targets: 目标值，包含：
                - findings_gt: Findings 的目标序列 (B, seq_len)。
                - impression_gt: Impression 的目标序列 (B, seq_len)。
        
        Returns:
            total_loss: 综合损失 (标量)。
            losses: 包含各项损失的字典。
        """
        findings, impression, F_F, F_I = output['findings'], output['impression'], output['F_F'], output['F_I']  # 解包 output
        findings_gt, impression_gt = targets['findings'], targets['impression']  # 解包 targets

        # 计算交叉熵损失
        findings_loss = self.cross_entropy_loss(findings, findings_gt)  # Findings 的交叉熵损失
        impression_loss = self.cross_entropy_loss(impression, impression_gt)  # Impression 的交叉熵损失

        # 计算对比学习损失
        contrastive_loss = self.contrastive_loss(F_F, F_I)

        # 综合损失
        total_loss = findings_loss + impression_loss + self.lambda_contrastive * contrastive_loss

        # 返回总损失和各项损失
        losses = {
            "findings_loss": findings_loss,
            "impression_loss": impression_loss,
            "contrastive_loss": contrastive_loss,
            "total_loss": total_loss
        }
        return total_loss, losses