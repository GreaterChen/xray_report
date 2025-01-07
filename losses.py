import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.Linear(hidden_dim, projection_dim),
        )
        # Prediction MLP
        self.prediction_mlp = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
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
        loss_sim = (
            -F.cosine_similarity(P_F, Z_I.detach(), dim=-1).mean()
            - F.cosine_similarity(P_I, Z_F.detach(), dim=-1).mean()
        ) / 2

        return loss_sim


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self, pad_id=0):
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
        targets = targets.input_ids.view(-1)  # (B * seq_len)

        # 忽略填充标记的损失
        loss = F.cross_entropy(
            predictions, targets, ignore_index=self.pad_id, reduction="mean"
        )
        return loss


class StageOneLoss(nn.Module):
    def __init__(self, pad_id):
        super(StageOneLoss, self).__init__()
        self.cross_entropy_loss = SequenceCrossEntropyLoss(pad_id=pad_id)

    def forward(self, output, targets):
        findings = output["findings"]
        findings_gt = targets["findings"]

        findings_loss = self.cross_entropy_loss(findings, findings_gt)

        losses = {
            "findings_loss": findings_loss,
        }

        return findings_loss, losses


class MultiLabelBCELoss(nn.Module):
    def __init__(self, num_classes=14, pos_weight=None):
        """
        计算多标签分类的二元交叉熵损失。
        Args:
            num_classes: 类别数量，默认14个疾病类别
            pos_weight: 正样本权重，用于处理类别不平衡，形状 (num_classes,)
        """
        super(MultiLabelBCELoss, self).__init__()
        self.num_classes = num_classes
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")

    def forward(self, predictions, targets):
        """
        Args:
            predictions: 模型输出的类别logits，形状 (B, num_classes)
            targets: 目标标签，形状 (B, num_classes)，每个位置是0或1
        Returns:
            loss: 归一化的二元交叉熵损失 (标量)
        """
        # 确保输入形状正确
        batch_size = predictions.size(0)
        predictions = predictions.view(batch_size, self.num_classes)  # (B, num_classes)
        targets = targets.view(batch_size, self.num_classes)  # (B, num_classes)

        # 计算二元交叉熵损失
        loss = self.criterion(predictions, targets)
        return loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        pad_id=0,
        feature_dim=768,
        projection_dim=128,
        hidden_dim=256,
        lambda_contrastive=1.0,
        lambda_class=1.0,
    ):
        """
        综合损失函数，包括：
        1. Findings 和 Impression 的交叉熵损失。
        2. Findings 和 Impression 特征的对比学习损失。
        3. 多标签分类损失。
        Args:
            pad_id: 填充值的索引，用于忽略填充标记的交叉熵损失。
            feature_dim: 输入特征维度 (F_F 和 F_I)。
            projection_dim: 投影空间维度，用于对比学习。
            hidden_dim: 预测 MLP 的隐藏层维度。
            lambda_contrastive: 对比损失的权重因子。
            lambda_class: 多标签分类损失的权重因子。
        """
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = SequenceCrossEntropyLoss(pad_id=pad_id)
        self.contrastive_loss = ContrastiveLearningLoss(
            feature_dim=feature_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
        )
        self.multi_label_loss = MultiLabelBCELoss()
        self.lambda_contrastive = lambda_contrastive
        self.lambda_class = lambda_class

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
                - label_gt: 14个疾病类别的标签 (B, 14)。

        Returns:
            total_loss: 综合损失 (标量)。
            losses: 包含各项损失的字典。
        """
        findings, impression, F_F, F_I, class_logits = (
            output["findings"],
            output["impression"],
            output["F_F"],
            output["F_I"],
            output["class_logits"],
        )  # 解包 output
        findings_gt, impression_gt, class_logits_gt = (
            targets["findings"],
            targets["impression"],
            targets["label"],
        )  # 解包 targets

        # 计算交叉熵损失
        findings_loss = self.cross_entropy_loss(
            findings, findings_gt
        )  # Findings 的交叉熵损失
        impression_loss = self.cross_entropy_loss(
            impression, impression_gt
        )  # Impression 的交叉熵损失

        # 计算分类损失
        class_loss = self.multi_label_loss(class_logits, class_logits_gt)

        # 计算对比学习损失
        contrastive_loss = self.contrastive_loss(F_F, F_I)

        # 综合损失
        total_loss = (
            findings_loss
            + impression_loss
            + self.lambda_contrastive * contrastive_loss
            + self.lambda_class * class_loss
        )

        # 返回总损失和各项损失
        losses = {
            "findings_loss": findings_loss,
            "impression_loss": impression_loss,
            "contrastive_loss": contrastive_loss,
            "class_loss": class_loss,
            "total_loss": total_loss,
        }
        return total_loss, losses
