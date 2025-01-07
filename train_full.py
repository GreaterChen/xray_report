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


# --- Helper Functions ---
def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    return threshold[ix]


def infer(data_loader, model, device="cpu", threshold=None):
    model.eval()
    outputs = []
    targets = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for i, (source, target) in enumerate(prog_bar):
            source = data_to_device(source, device)
            target = data_to_device(target, device)

            # Use single input if there is no clinical history
            if threshold != None:
                output = model(image=source[0], history=source[3], threshold=threshold)
            else:
                output = model(source[0])

            outputs.append(data_to_device(output))
            targets.append(data_to_device(target))

        outputs = data_concatenate(outputs)
        targets = data_concatenate(targets)

    return outputs, targets


# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument("--debug", default=False, help="Debug mode.")

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
        default=196,
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
        choices=["TRAIN_STAGE_1", "TRAIN_STAGE_2", "TEST", "INFER"],
        help="Phase of the program: TRAIN, TEST, or INFER.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=48, help="Batch size for training."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=8, help="Batch size for validation."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training."
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument(
        "--min_lr", type=float, default=5e-6, help="Minimum learning rate."
    )
    parser.add_argument(
        "--warmup_lr", type=float, default=5e-7, help="Warmup learning rate."
    )
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps.")
    parser.add_argument(
        "--wd", type=float, default=0.05, help="Weight decay (L2 regularization)."
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
        default=None,
        help="Path to load the checkpoint from.",
    )
    parser.add_argument(
        "--checkpoint_path_to",
        type=str,
        default="/home/chenlb/xray_report_generation/results/new_data/stage_1",
        help="Path to save the checkpoint to.",
    )

    return parser.parse_args()


# --- Main Program ---
if __name__ == "__main__":
    args = parse_args()

    # Set environment variables
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
        train_stage = 1 if args.phase == "TRAIN_STAGE_1" else 2

        MIMIC.load_shared_data(args.data_dir)
        # 创建训练、验证和测试数据集
        train_data = MIMIC(
            args.data_dir,
            input_size,
            random_transform=True,
            train_stage=train_stage,
            tokenizer=tokenizer,
            mode="train",
            subset_size=200 if args.debug else None,
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

        # swin_transformer = SwinFeatureExtractor(hidden_dim=768)
        # vit_transformer = ViTFeatureExtractor(model_name='vit_base_patch16_224', pretrained=True)
        # features_projector = DiseaseFeatureProjector(input_dim=768, num_diseases=256, feature_dim=768)

        history_encoder = HistoryEncoder(args)

        modality_fusion = ModalityFusion(hidden_size=768)

        findings_decoder = BLIP_Decoder(args, tokenizer=tokenizer)
        findings_generator = FindingsGenerator(findings_decoder)

        co_attention_module = CoAttentionModule(embed_dim=768)
        multi_label_classifier = MultiLabelClassifier(input_dim=768, hidden_dim=384)

        impression_decoder = BLIP_Decoder(args, tokenizer=tokenizer)
        impression_generator = ImpressionGenerator(impression_decoder)

        cxr_bert_feature_extractor = CXR_BERT_FeatureExtractor(tokenizer=tokenizer)

        model = HiMrGn(
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
    if args.phase == "TRAIN_STAGE_1":
        criterion = StageOneLoss(pad_id=pad_id).cuda()
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
            val_loss, val_met = test(
                args,
                val_loader,
                model,
                logger,
                mode="val",
                metric_ftns=metrics,
                criterion=criterion,
                device="cuda",
                kw_src=args.kw_src,
                kw_tgt=args.kw_tgt,
                kw_out=args.kw_out,
                return_results=False,
                train_stage=1,
            )
            test_loss, test_met = test(
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
                return_results=False,
                train_stage=1,
            )

            logger.info(f"epoch: {epoch}")
            for k, v in val_met.items():
                logger.info(f"val_{k}: {v}")

            for k, v in test_met.items():
                logger.info(f"test_{k}: {v}")

            save_path = os.path.join(
                args.checkpoint_path_to, f'epoch_{epoch}_BLEU_1_{val_met["BLEU_1"]}.pth'
            )
            save(
                save_path,
                model,
                optimizer,
                scheduler,
                epoch,
                (val_loss, test_loss, val_met),
            )
            logger.info(f"Saved To: {save_path}")

    elif args.phase == "TRAIN_STAGE_2":
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
                device="cuda",
                kw_src=args.kw_src,
                kw_tgt=args.kw_tgt,
                kw_out=args.kw_out,
                scaler=scaler,
                train_stage=2,
            )
            val_loss = test(
                args,
                val_loader,
                model,
                metric_ftns=metrics,
                criterion=criterion,
                device="cuda",
                kw_src=args.kw_src,
                kw_tgt=args.kw_tgt,
                kw_out=args.kw_out,
                return_results=False,
                train_stage=2,
            )
            test_loss = test(
                args,
                test_loader,
                model,
                metric_ftns=metrics,
                criterion=criterion,
                device="cuda",
                kw_src=args.kw_src,
                kw_tgt=args.kw_tgt,
                kw_out=args.kw_out,
                return_results=False,
                train_stage=2,
            )

            scheduler.step()
            if best_metric > val_loss:  # TODO 修改评测指标
                best_metric = val_loss
                save(
                    args.checkpoint_path_to,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    (val_loss, test_loss),
                )
                logger.info(f"New Best Metric: {best_metric}")
                logger.info(f"Saved To: {args.checkpoint_path_to}")

    elif args.phase == "TEST":
        # Output the file list for inspection
        out_file_img = open(
            "outputs/{}_{}_{}_{}_Img.txt".format(
                args.dataset_name, args.model_name, args.backbone_name, comment
            ),
            "w",
        )
        for i in range(len(test_data.idx_pidsid)):
            out_file_img.write(
                test_data.idx_pidsid[i][0] + " " + test_data.idx_pidsid[i][1] + "\n"
            )

    elif args.phase == "INFER":
        # txt_test_outputs, txt_test_targets = infer(test_loader, model, device='cuda', threshold=0.25)
        txt_test_outputs, txt_test_targets = infer(test_loader, model, device="cuda")

        gen_outputs = txt_test_outputs
        gen_targets = txt_test_targets
        # gen_outputs = txt_test_outputs[0]
        # gen_targets = txt_test_targets[0]

        out_file_ref = open(
            "outputs/x_{}_{}_{}_{}_Ref.txt".format(
                args.dataset_name, args.model_name, args.backbone_name, comment
            ),
            "w",
        )
        out_file_hyp = open(
            "outputs/x_{}_{}_{}_{}_Hyp.txt".format(
                args.dataset_name, args.model_name, args.backbone_name, comment
            ),
            "w",
        )
        out_file_lbl = open(
            "outputs/x_{}_{}_{}_{}_Lbl.txt".format(
                args.dataset_name, args.model_name, args.backbone_name, comment
            ),
            "w",
        )

        for i in range(len(gen_outputs)):
            candidate = ""
            for j in range(len(gen_outputs[i])):
                tok = test_data.tokenizer.id_to_piece(int(gen_outputs[i, j]))
                if tok == "</s>":
                    break  # Manually stop generating token after </s> is reached
                elif tok == "<s>":
                    continue
                elif tok == "▁":  # space
                    if len(candidate) and candidate[-1] != " ":
                        candidate += " "
                elif tok in [",", ".", "-", ":"]:  # or not tok.isalpha():
                    if len(candidate) and candidate[-1] != " ":
                        candidate += " " + tok + " "
                    else:
                        candidate += tok + " "
                else:  # letter
                    candidate += tok
            out_file_hyp.write(candidate + "\n")

            reference = ""
            for j in range(len(gen_targets[i])):
                tok = test_data.tokenizer.id_to_piece(int(gen_targets[i, j]))
                if tok == "</s>":
                    break
                elif tok == "<s>":
                    continue
                elif tok == "▁":  # space
                    if len(reference) and reference[-1] != " ":
                        reference += " "
                elif tok in [",", ".", "-", ":"]:  # or not tok.isalpha():
                    if len(reference) and reference[-1] != " ":
                        reference += " " + tok + " "
                    else:
                        reference += tok + " "
                else:  # letter
                    reference += tok
            out_file_ref.write(reference + "\n")

    else:
        raise ValueError("Invalid PHASE")
