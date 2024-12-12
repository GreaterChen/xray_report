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
from datasets import NIHCXR, MIMIC, NLMCXR
from losses import *
from models import *

# --- Helper Functions ---
def find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)
    return threshold[ix]

def infer(data_loader, model, device='cpu', threshold=None):
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
    parser.add_argument('--dir', type=str, default='/mnt/chenlb/datasets/mimic_cxr/',
                        help='Path to the directory.')
    parser.add_argument('--image_dir', type=str, default='/mnt/chenlb/datasets/mimic_cxr/images/',
                        help='Path to the directory containing the image data.')
    parser.add_argument('--ann_path', type=str, default='/mnt/chenlb/datasets/mimic_cxr/mimic_annotation_promptmrg_new.json',
                        help='Path to the annotation file.')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size.')
    parser.add_argument('--dataset_name', type=str, default='MIMIC', choices=['MIMIC', 'NIHCXR', 'NLMCXR'],
                        help='Dataset name to use.')
    
    # Model settings
    parser.add_argument('--model_name', type=str, default='HiMrGn', help='Name of the model to use.')
    parser.add_argument('--backbone_name', type=str, default='SwinT', help='Backbone model name.')

    # HiMrGn-specific settings
    parser.add_argument('--sources', type=str, nargs='+', default=['image', 'findings', 'impression', 'history'],
                        help='List of source inputs for the model (e.g., image, findings, impression).')
    parser.add_argument('--targets', type=str, nargs='+', default=['findings', 'impression', 'label'],
                        help='List of target outputs for the model (e.g., findings, impression).')
    parser.add_argument('--kw_src', type=str, nargs='+', default=['image', 'findings', 'impression', 'history'],
                        help='Keyword arguments for the source inputs of the model (e.g., image, findings, impression).')
    parser.add_argument('--kw_tgt', type=str, nargs='+', default=['findings', 'impression', 'label'],
                        help='Keyword arguments for the target outputs of the model (e.g., findings, impression).')
    parser.add_argument('--kw_out', type=str, default=None,
                        help='Keyword arguments for the output settings of the model (default: None).')

    # Training settings
    parser.add_argument('--phase', type=str, default='TRAIN_STAGE_1', choices=['TRAIN_STAGE_1', 'TRAIN_STAGE_2', 'TEST', 'INFER'],
                        help='Phase of the program: TRAIN, TEST, or INFER.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-2, help='Weight decay (L2 regularization).')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')

    # Device settings
    parser.add_argument('--cuda_visible_devices', type=str, default="0", help='CUDA visible devices.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility.')

    # Reload settings
    parser.add_argument('--reload', action='store_true', help='Reload from a checkpoint.')
    parser.add_argument('--checkpoint_path_from', type=str, default=None, help='Path to load the checkpoint from.')
    parser.add_argument('--checkpoint_path_to', type=str, default="/home/chenlb/xray_report_generation/results/test/stage_1", help='Path to save the checkpoint to.')

    return parser.parse_args()

# --- Main Program ---
if __name__ == "__main__":
    args = parse_args()

    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    torch.manual_seed(args.seed)

    # Dataset-specific settings
    if args.dataset_name == 'MIMIC':
        input_size = (args.image_size, args.image_size)
        max_views = 2
        num_labels = 114
        num_classes = 2
        view_pos = ['AP']

        if args.phase == "TRAIN_STAGE_1":
            dataset = MIMIC(args.dir, input_size, view_pos=view_pos, max_views=max_views,
                            sources=args.sources, targets=args.targets, train_stage=1)
        else:
            dataset = MIMIC(args.dir, input_size, view_pos=view_pos, max_views=max_views,
                            sources=args.sources, targets=args.targets, train_stage=2)
        train_data, val_data, test_data = dataset.get_subsets(seed=args.seed)

        vocab_size = dataset.tokenizer.vocab_size
        posit_size = dataset.max_len
        pad_id = dataset.tokenizer.pad_token_id
        comment = f'Stage{args.phase}'

    elif args.dataset_name == 'NLMCXR':
        input_size = (256, 256)
        max_views = 2
        num_labels = 114
        num_classes = 2

        dataset = NLMCXR(args.image_dir, input_size, view_pos=['AP', 'PA', 'LATERAL'], max_views=max_views,
                         sources=['image', 'findings', 'impression'], targets=['findings', 'impression'])
        train_data, val_data, test_data = dataset.get_subsets(seed=args.seed)

        vocab_size = len(dataset.vocab)
        posit_size = dataset.max_len
        comment = f'MaxView{max_views}_NumLabel{num_labels}_NoHistory'

    else:
        raise ValueError('Invalid dataset_name')

    # Model-specific settings
    if args.model_name == 'HiMrGn':
        swin_transformer = SwinFeatureExtractor(hidden_dim=768)
        features_projector = DiseaseFeatureProjector(input_dim=768, num_diseases=256, feature_dim=768)
        modality_fusion = ModalityFusion(d_model=768, nhead=8, num_encoder_layers=6, dropout=0.1)
        findings_decoder = TextDecoder(input_dim=256, hidden_dim=768)
        findings_generator = FindingsGenerator(findings_decoder)
        co_attention_module = CoAttentionModule(embed_dim=768)
        multi_label_classifier = MultiLabelClassifier(input_dim=768, hidden_dim=384)
        impression_decoder = TextDecoder(input_dim=256, hidden_dim=768)
        impression_generator = ImpressionGenerator(impression_decoder)
        cxr_bert_feature_extractor = CXR_BERT_FeatureExtractor()

        model = HiMrGn(image_encoder=swin_transformer,
                       features_projector=features_projector,
                       modality_fusion=modality_fusion,
                       findings_decoder=findings_generator,
                       multi_label_classifier=multi_label_classifier,
                       co_attention_module=co_attention_module,
                       impression_decoder=impression_generator,
                       cxr_bert_feature_extractor=cxr_bert_feature_extractor)

        # Compute parameters for each module
        module_parameters = {
            "Swin Transformer": count_parameters(swin_transformer),
            "Features Projector": count_parameters(features_projector),
            "Findings Generator": count_parameters(findings_generator),
            "Modality Fusion": count_parameters(modality_fusion),
            "Co-Attention Module": count_parameters(co_attention_module),
            "Multi-Label Classifier": count_parameters(multi_label_classifier),
            "Impression Generator": count_parameters(impression_generator),
            "CXR BERT Feature Extractor": count_parameters(cxr_bert_feature_extractor),
        }

        # Print results
        for module_name, param_count in module_parameters.items():
            print(f"{module_name}: {param_count} parameters")

    else:
        raise ValueError('Invalid model_name')

    # Data loaders
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_loader = data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_loader = data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=1)

    model = nn.DataParallel(model).cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40, 55, 70, 85])

    print('Total Parameters:', sum(p.numel() for p in model.parameters()))
    
    last_epoch = -1
    best_metric = 1e9

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    
    # Load checkpoint if needed
    if args.reload and args.checkpoint_path_from:
        last_epoch, (best_metric, test_metric) = load(args.checkpoint_path_from, model, optimizer, scheduler)
        print(f'Reloaded from {args.checkpoint_path_from}: Last Epoch {last_epoch}, Best Metric {best_metric}, Test Metric {test_metric}')

    # Training phase
    if args.phase == 'TRAIN_STAGE_1':
        criterion = StageOneLoss(pad_id=pad_id).cuda()
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(last_epoch+1, args.epochs):
            print(f'Epoch: {epoch}')
            train_loss = train(train_loader, model, optimizer, criterion, device='cuda', kw_src=args.kw_src, kw_tgt=args.kw_tgt, kw_out=args.kw_out, scaler=scaler, train_stage=1)
            val_loss = test(val_loader, model, criterion=criterion, device='cuda', kw_src=args.kw_src, kw_tgt=args.kw_tgt, kw_out=args.kw_out, return_results=False, train_stage=1)
            test_loss = test(test_loader, model, criterion=criterion, device='cuda', kw_src=args.kw_src, kw_tgt=args.kw_tgt, kw_out=args.kw_out, return_results=False, train_stage=1)
            
            scheduler.step()
            if best_metric > val_loss:
                best_metric = val_loss
                save(args.checkpoint_path_to, model, optimizer, scheduler, epoch, (val_loss, test_loss))
                print(f'New Best Metric: {best_metric}')
                print(f'Saved To: {args.checkpoint_path_to}')

    elif args.phase == 'TRAIN_STAGE_2':
        criterion = CombinedLoss(pad_id=pad_id).cuda()
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(last_epoch + 1, args.epochs):
            print(f'Epoch: {epoch}')
            train_loss = train(train_loader, model, optimizer, criterion, device='cuda', kw_src=args.kw_src, kw_tgt=args.kw_tgt, kw_out=args.kw_out, scaler=scaler, train_stage=2)
            val_loss = test(val_loader, model, criterion=criterion, device='cuda', kw_src=args.kw_src, kw_tgt=args.kw_tgt, kw_out=args.kw_out, return_results=False, train_stage=2)
            test_loss = test(test_loader, model, criterion=criterion, device='cuda', kw_src=args.kw_src, kw_tgt=args.kw_tgt, kw_out=args.kw_out, return_results=False, train_stage=2)
            
            scheduler.step()
            if best_metric > val_loss:
                best_metric = val_loss
                save(args.checkpoint_path_to, model, optimizer, scheduler, epoch, (val_loss, test_loss))
                print(f'New Best Metric: {best_metric}')
                print(f'Saved To: {args.checkpoint_path_to}')
    
    elif args.phase == 'TEST':
        # Output the file list for inspection
        out_file_img = open('outputs/{}_{}_{}_{}_Img.txt'.format(args.dataset_name, args.model_name, args.backbone_name, comment), 'w')
        for i in range(len(test_data.idx_pidsid)):
            out_file_img.write(test_data.idx_pidsid[i][0] + ' ' + test_data.idx_pidsid[i][1] + '\n')
            
    elif args.phase == 'INFER':
        # txt_test_outputs, txt_test_targets = infer(test_loader, model, device='cuda', threshold=0.25)
        txt_test_outputs, txt_test_targets = infer(test_loader, model, device='cuda')

        gen_outputs = txt_test_outputs
        gen_targets = txt_test_targets
        # gen_outputs = txt_test_outputs[0]
        # gen_targets = txt_test_targets[0]
        
        out_file_ref = open('outputs/x_{}_{}_{}_{}_Ref.txt'.format(args.dataset_name, args.model_name, args.backbone_name, comment), 'w')
        out_file_hyp = open('outputs/x_{}_{}_{}_{}_Hyp.txt'.format(args.dataset_name, args.model_name, args.backbone_name, comment), 'w')
        out_file_lbl = open('outputs/x_{}_{}_{}_{}_Lbl.txt'.format(args.dataset_name, args.model_name, args.backbone_name, comment), 'w')
        
        for i in range(len(gen_outputs)):
            candidate = ''
            for j in range(len(gen_outputs[i])):
                tok = dataset.vocab.id_to_piece(int(gen_outputs[i,j]))
                if tok == '</s>':
                    break # Manually stop generating token after </s> is reached
                elif tok == '<s>':
                    continue
                elif tok == '▁': # space
                    if len(candidate) and candidate[-1] != ' ':
                        candidate += ' '
                elif tok in [',', '.', '-', ':']: # or not tok.isalpha():
                    if len(candidate) and candidate[-1] != ' ':
                        candidate += ' ' + tok + ' ' 
                    else:
                        candidate += tok + ' '
                else: # letter
                    candidate += tok       
            out_file_hyp.write(candidate + '\n')
            
            reference = ''
            for j in range(len(gen_targets[i])):
                tok = dataset.vocab.id_to_piece(int(gen_targets[i,j]))
                if tok == '</s>':
                    break
                elif tok == '<s>':
                    continue
                elif tok == '▁': # space
                    if len(reference) and reference[-1] != ' ':
                        reference += ' '
                elif tok in [',', '.', '-', ':']: # or not tok.isalpha():
                    if len(reference) and reference[-1] != ' ':
                        reference += ' ' + tok + ' ' 
                    else:
                        reference += tok + ' '
                else: # letter
                    reference += tok    
            out_file_ref.write(reference + '\n')

        # for i in tqdm(range(len(test_data))):
        #     target = test_data[i][1] # caption, label
        #     out_file_lbl.write(' '.join(map(str,target[1])) + '\n')
                
    else:
        raise ValueError('Invalid PHASE')