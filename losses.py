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