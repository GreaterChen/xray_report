import json
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from datetime import datetime

# --- PyTorch packages ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# --- Helper Packages ---
from tqdm import tqdm
import argparse

# --- Project Packages ---
from utils import *
from datasets import MIMIC
from losses import *
from models.model import *
from metrics import compute_scores
from tools.optims import *

logger = setup_logger(log_dir="logs")


# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", default=False, help="Debug mode.")
    parser.add_argument("--CLS", default=False, help="Classifier.")
    parser.add_argument("--CO", default=False, help="Co-attention.")
    parser.add_argument("--CL", default=False, help="Constrained Learning.")
    parser.add_argument(
        "--co_num_heads", type=int, default=1, help="Number of heads for co-attention."
    )
    parser.add_argument(
        "--co_num_blocks",
        type=int,
        default=1,
        help="Number of blocks for co-attention.",
    )

    # Data input settings
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/chenlb/xray_report_generation/",
        help="Root directory.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/chenlb/datasets/mimic_cxr/",
        help="Path to the directory.",
    )
    parser.add_argument(
        "--ann_dir",
        type=str,
        # default="/mnt/chenlb/datasets/mimic_cxr/mimic_annotation_impression-full.json",
        default="/mnt/chenlb/datasets/mimic_cxr/mimic_annotation.json",
        help="Path to the annotation file.",
    )

    parser.add_argument("--image_size", type=int, default=224, help="Input image size.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MIMIC",
        choices=["MIMIC", "NIHCXR", "NLMCXR"],
        help="Dataset name to use.",
    )
    parser.add_argument(
        "--max_len_findings",
        type=int,
        default=196,
        help="Maximum length of the input text.",
    )
    parser.add_argument(
        "--max_len_impression",
        type=int,
        default=100,
        help="Maximum length of the input text.",
    )
    parser.add_argument(
        "--max_len_history",
        type=int,
        default=50,
        help="Maximum length of the input text.",
    )

    parser.add_argument(
        "--tokenizer_max_len",
        type=int,
        default=30523,
        help="Maximum length of the tokenizer.",
    )

    # Model settings
    parser.add_argument(
        "--model_name", type=str, default="HiMrGn", help="Name of the model to use."
    )
    parser.add_argument(
        "--backbone_name", type=str, default="ResNet101", help="Backbone model name."
    )

    # HiMrGn-specific settings
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=["image", "findings", "impression", "history"],
        help="List of source inputs for the model (e.g., image, findings, impression).",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["findings", "impression", "label"],
        help="List of target outputs for the model (e.g., findings, impression).",
    )
    parser.add_argument(
        "--kw_src",
        type=str,
        nargs="+",
        default=["image", "findings", "impression", "history"],
        help="Keyword arguments for the source inputs of the model (e.g., image, findings, impression).",
    )
    parser.add_argument(
        "--kw_tgt",
        type=str,
        nargs="+",
        default=["findings", "impression", "label"],
        help="Keyword arguments for the target outputs of the model (e.g., findings, impression).",
    )
    parser.add_argument(
        "--kw_out",
        type=str,
        default=None,
        help="Keyword arguments for the output settings of the model (default: None).",
    )

    # Training settings
    parser.add_argument(
        "--phase",
        type=str,
        default="TRAIN_STAGE_1",
        choices=["TRAIN_STAGE_1", "TRAIN_STAGE_2", "TRAIN_STAGE_3", "TEST", "INFER"],
        help="Phase of the program",
    )

    # TRAIN OR TEST
    parser.add_argument(
        "--mode",
        type=str,
        default="TRAIN",
        choices=["TRAIN", "TEST"],
        help="Train or Test",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=8, help="Batch size for validation."
    )
    parser.add_argument(
        "--num_workers", type=int, default=6, help="Number of workers for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs for training."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="Minimum learning rate."
    )
    parser.add_argument(
        "--warmup_lr", type=float, default=5e-6, help="Warmup learning rate."
    )
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps.")
    parser.add_argument(
        "--wd", type=float, default=0.01, help="Weight decay (L2 regularization)."
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    # Device settings
    parser.add_argument(
        "--cuda_visible_devices", type=str, default="0", help="CUDA visible devices."
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility."
    )

    # Reload settings
    parser.add_argument(
        "--reload", action="store_true", help="Reload from a checkpoint."
    )
    parser.add_argument(
        "--checkpoint_path_from",
        type=str,
        # default="/home/chenlb/xray_report_generation/results/stage1/49visual_extend/epoch_12_BLEU_1_0.3919511280957931.pth",
        # default="/home/chenlb/xray_report_generation/results/resnet/stage1/epoch_19_BLEU_1_0.41520361597936345.pth",
        # default="/mnt/chenlb/results/MRG/ours/stage2/no_co/epoch_15_BLEU_1_0.2563890373855226.pth",
        # default="/home/chenlb/xray_report_generation/results/stage3/baseline/epoch_2_BLEU_1_0.4009259151415555_0.18074402611477508.pth",
        # default="/home/chenlb/xray_report_generation/results/stage1/49visual/epoch_8_BLEU_1_0.3736232743510843.pth",
        # default="/home/chenlb/xray_report_generation/results/stage2/cxr_bert/epoch_8_BLEU_1_0.19664481187635488.pth",
        default=None,
        help="Path to load the checkpoint from.",
    )
    parser.add_argument(
        "--checkpoint_path_to",
        type=str,
        default="/home/chenlb/xray_report_generation/results/stage1/total",
        help="Path to save the checkpoint to.",
    )
    args = parser.parse_args()

    # Convert args to dictionary
    args_dict = vars(args)

    # Create save path
    from pathlib import Path

    save_dir = Path(args.checkpoint_path_to)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "args_config.json"

    # Save to JSON file
    with open(save_path, "w") as f:
        json.dump(args_dict, f, indent=4)

    return args


# --- Main Program ---
if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    torch.manual_seed(args.seed)

    # Dataset-specific settings
    if args.dataset_name == "MIMIC":
        input_size = (args.image_size, args.image_size)

        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=True
        )
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        vocab_size = len(tokenizer)
        pad_id = tokenizer.pad_token_id

        # 设置训练阶段和调试模式
        train_stage = (
            1
            if args.phase == "TRAIN_STAGE_3"
            else 2 if args.phase == "TRAIN_STAGE_2" else 3
        )

        MIMIC.load_shared_data(args.data_dir, args.ann_dir, train_stage, args.mode)
        # 创建训练、验证和测试数据集
        train_data = MIMIC(
            args.data_dir,
            input_size,
            random_transform=True,
            train_stage=train_stage,
            tokenizer=tokenizer,
            mode="train",
            subset_size=50 if args.debug else None,
        )

        val_data = MIMIC(
            args.data_dir,
            input_size,
            random_transform=False,
            train_stage=train_stage,
            tokenizer=tokenizer,
            mode="val",
            subset_size=10 if args.phase.startswith("TRAIN") else 100,
        )

        test_data = MIMIC(
            args.data_dir,
            input_size,
            random_transform=False,
            train_stage=train_stage,
            tokenizer=tokenizer,
            mode="test",
            subset_size=10 if args.debug else None,
        )

        comment = f"Stage{args.phase}"
    else:
        raise ValueError("Invalid dataset_name")

    # Model-specific settings
    if args.model_name == "HiMrGn":

        resnet101 = ResNet101()

        history_encoder = HistoryEncoder(args)

        modality_fusion = ModalityFusion(hidden_size=768)

        findings_decoder = BLIP_Decoder(
            args, tokenizer=tokenizer, max_length=args.max_len_findings
        )
        findings_generator = FindingsGenerator(findings_decoder)

        co_attention_module = CoAttentionModule(
            embed_dim=768, num_heads=args.co_num_heads, num_blocks=args.co_num_blocks
        )
        multi_label_classifier = MultiLabelClassifier(input_dim=768, hidden_dim=384)

        impression_decoder = BLIP_Decoder(
            args, tokenizer=tokenizer, max_length=args.max_len_impression
        )
        impression_generator = ImpressionGenerator(impression_decoder)

        cxr_bert_feature_extractor = CXR_BERT_FeatureExtractor()

        model = HiMrGn(
            args=args,
            image_encoder=resnet101,
            history_encoder=history_encoder,
            modality_fusion=modality_fusion,
            findings_decoder=findings_generator,
            multi_label_classifier=multi_label_classifier,
            co_attention_module=co_attention_module,
            impression_decoder=impression_generator,
            cxr_bert_feature_extractor=cxr_bert_feature_extractor,
        )

        # Compute parameters for each module
        module_parameters = {
            "ResNet101": count_parameters(resnet101),
            "Findings Generator": count_parameters(findings_generator),
            "Modality Fusion": count_parameters(modality_fusion),
            "Co-Attention Module": count_parameters(co_attention_module),
            "Multi-Label Classifier": count_parameters(multi_label_classifier),
            "Impression Generator": count_parameters(impression_generator),
            "CXR BERT Feature Extractor": count_parameters(cxr_bert_feature_extractor),
        }

        # Print results
        for module_name, param_count in module_parameters.items():
            logger.info(f"{module_name}: {param_count} parameters")

    else:
        raise ValueError("Invalid model_name")

    # Data loaders
    train_loader = data.DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = data.DataLoader(
        test_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 打印数量
    logger.info(f"Train Data Size: {len(train_data)}")
    logger.info(f"Val Data Size: {len(val_data)}")
    logger.info(f"Test Data Size: {len(test_data)}")

    # model = nn.DataParallel(model).cuda()
    model = model.cuda()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer,
        args.epochs,
        args.min_lr,
        args.lr,
        decay_rate=None,
        warmup_start_lr=args.warmup_lr,
        warmup_steps=args.warmup_steps,
    )

    logger.info(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")

    last_epoch = -1
    best_metric = -1e9

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")

    # Load checkpoint if needed
    if args.reload and args.checkpoint_path_from:
        last_epoch, (best_metric, test_metric) = load(
            args.checkpoint_path_from, model, optimizer, scheduler
        )
        logger.info(
            f"Reloaded from {args.checkpoint_path_from}: Last Epoch {last_epoch}, Best Metric {best_metric}, Test Metric {test_metric}"
        )

    metrics = compute_scores

    # Training phase
    if args.phase == "TRAIN_STAGE_1" and args.mode == "TRAIN":
        if args.checkpoint_path_from:
            _, _ = load(args.checkpoint_path_from, model, optimizer, scheduler)

        criterion = None
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(last_epoch + 1, args.epochs):
            print(f"Epoch: {epoch}")
            train_loss = train(
                args,
                train_loader,
                model,
                optimizer,
                criterion,
                scheduler=scheduler,
                num_epochs=args.epochs,
                current_epoch=epoch,
                device="cuda",
                kw_src=args.kw_src,
                kw_tgt=args.kw_tgt,
                kw_out=args.kw_out,
                scaler=scaler,
                train_stage=1,
            )

            test_loss, result = test(
                args,
                test_loader,
                model,
                logger,
                mode="test",
                metric_ftns=metrics,
                criterion=criterion,
                device="cuda",
                kw_src=args.kw_src,
                kw_tgt=args.kw_tgt,
                kw_out=args.kw_out,
                train_stage=1,
                epoch=epoch,
            )

            # 记录测试结果
            log_metrics(logger, epoch, train_loss, test_loss, result)

            # 保存检查点时使用BLEU_1分数
            save_path = os.path.join(
                args.checkpoint_path_to,
                f'epoch_{epoch}_BLEU_1_{result["metrics_df"]["findings_BLEU_1"].iloc[0]}.pth',
            )
            save(
                save_path,
                model,
                optimizer,
                scheduler,
                epoch,
                (test_loss, result),
            )
            logger.info(f"Saved To: {save_path}")

    elif args.phase == "TRAIN_STAGE_2" and args.mode == "TRAIN":

        _, _ = load(args.checkpoint_path_from, model, optimizer, scheduler)
        logger.info(f"从 {args.checkpoint_path_from} 加载模型权重")

        # model.impression_decoder.load_state_dict(model.findings_decoder.state_dict())

        # 冻结指定模块的参数
        # for param in model.image_encoder.parameters():
        #     param.requires_grad = False

        # for param in model.history_encoder.parameters():
        #     param.requires_grad = False

        # for param in model.modality_fusion.parameters():
        #     param.requires_grad = False

        # for param in model.findings_decoder.parameters():
        #     param.requires_grad = False

        logger.info(
            "已冻结 image_encoder, history_encoder, modality_fusion, findings_decoder 的参数"
        )

        # optimizer = optim.AdamW(
        #     [
        #         {"params": model.image_encoder.parameters(), "lr": 1e-7},
        #         {"params": model.history_encoder.parameters(), "lr": 1e-7},
        #         {"params": model.modality_fusion.parameters(), "lr": 1e-7},
        #         {"params": model.findings_decoder.parameters(), "lr": 1e-7},
        #         {"params": model.multi_label_classifier.parameters()},
        #         {"params": model.co_attention_module.parameters()},
        #         {"params": model.impression_decoder.parameters()},
        #         {"params": model.cxr_bert_feature_extractor.parameters()},
        #     ],
        #     lr=args.lr,
        #     weight_decay=args.wd,
        # )



        criterion = CombinedLoss(pad_id=pad_id).cuda()
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(last_epoch + 1, args.epochs):
            print(f"Epoch: {epoch}")
            train_loss = train(
                args,
                train_loader,
                model,
                optimizer,
                criterion,
                scheduler=scheduler,
                num_epochs=args.epochs,
                current_epoch=epoch,
                device="cuda",
                kw_src=args.kw_src,
                kw_tgt=args.kw_tgt,
                kw_out=args.kw_out,
                scaler=scaler,
                train_stage=2,
            )

            test_loss, result = test(
                args,
                test_loader,
                model,
                logger,
                mode="test",
                metric_ftns=metrics,
                criterion=criterion,
                device="cuda",
                kw_src=args.kw_src,
                kw_tgt=args.kw_tgt,
                kw_out=args.kw_out,
                train_stage=2,
                epoch=epoch,
            )

            # 记录测试结果
            log_metrics(logger, epoch, train_loss, test_loss, result)

            # 保存检查点时使用BLEU_1分数
            save_path = os.path.join(
                args.checkpoint_path_to,
                f'epoch_{epoch}_BLEU_1_{result["metrics_df"]["impression_BLEU_1"].iloc[0]}.pth',
            )
            save(
                save_path,
                model,
                optimizer,
                scheduler,
                epoch,
                (test_loss, result),
            )

            logger.info(f"Saved To: {save_path}")

    elif args.phase == "TRAIN_STAGE_3" and args.mode == "TRAIN":
        # # 加载第一个checkpoint文件
        # ckpt1 = torch.load('/home/chenlb/xray_report_generation/results/resnet/stage1/epoch_19_BLEU_1_0.41520361597936345.pth')
        # model_state_dict1 = ckpt1['model_state_dict']

        # # 加载第二个checkpoint文件 
        # ckpt2 = torch.load('/home/chenlb/xray_report_generation/results/stage2/test_cls/epoch_10_BLEU_1_0.19510754670748817.pth')
        # model_state_dict2 = ckpt2['model_state_dict']

        # # 创建一个新的state_dict来存储合并后的权重
        # new_state_dict = {}

        # # 从第一个checkpoint加载指定组件的权重
        # components1 = ['image_encoder', 'history_encoder', 'modality_fusion', 'findings_decoder']
        # for k, v in model_state_dict1.items():
        #     if any(comp in k for comp in components1):
        #         new_state_dict[k] = v

        # # 从第二个checkpoint加载指定组件的权重
        # components2 = ['co_attention_module', 'impression_decoder']
        # for k, v in model_state_dict2.items():
        #     if any(comp in k for comp in components2):
        #         new_state_dict[k] = v

        # # 加载合并后的权重到模型
        # model.load_state_dict(new_state_dict, strict=False)

        # _, _ = load(args.checkpoint_path_from, model, optimizer, scheduler)
        # logger.info(f"从 {args.checkpoint_path_from} 加载模型权重")

        criterion = CombinedLoss(pad_id=pad_id).cuda()
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(last_epoch + 1, args.epochs):
            print(f"Epoch: {epoch}")
            train_loss = train(
                args,
                train_loader,
                model,
                optimizer,
                criterion,
                scheduler=scheduler,
                num_epochs=args.epochs,
                current_epoch=epoch,
                device="cuda",
                kw_src=args.kw_src,
                kw_tgt=args.kw_tgt,
                kw_out=args.kw_out,
                scaler=scaler,
                train_stage=3,
            )

            test_loss, result = test(
                args,
                test_loader,
                model,
                logger,
                mode="test",
                metric_ftns=metrics,
                criterion=criterion,
                device="cuda",
                kw_src=args.kw_src,
                kw_tgt=args.kw_tgt,
                kw_out=args.kw_out,
                train_stage=3,
                epoch=epoch,
            )

            # 记录测试结果
            log_metrics(logger, epoch, train_loss, test_loss, result)

            # 保存检查点时使用BLEU_1分数
            save_path = os.path.join(
                args.checkpoint_path_to,
                f'epoch_{epoch}_BLEU_1_{result["metrics_df"]["findings_BLEU_1"].iloc[0]}_{result["metrics_df"]["impression_BLEU_1"].iloc[0]}.pth',
            )
            save(
                save_path,
                model,
                optimizer,
                scheduler,
                epoch,
                (test_loss, result),
            )

            logger.info(f"Saved To: {save_path}")

        


            

    elif args.mode == "TEST":
        # 确保提供了checkpoint路径
        if not args.checkpoint_path_from:
            raise ValueError("必须提供checkpoint路径用于测试!")

        # 加载模型权重
        _, _ = load(args.checkpoint_path_from, model, optimizer, scheduler)
        logger.info(f"从 {args.checkpoint_path_from} 加载模型权重")

        # 保存生成结果
        save_generations(
            args,
            test_loader,
            model,
            logger,
            save_dir=os.path.join(args.checkpoint_path_to, "generations"),
            mode="test",
            train_stage=train_stage,
            device="cuda",
            kw_src=args.kw_src,
            kw_tgt=args.kw_tgt,
            kw_out=args.kw_out,
        )

    else:
        raise ValueError("Invalid phase")
