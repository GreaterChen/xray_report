   # --- Base packages ---
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

# --- Project Packages ---
from utils import save, load, train, test, data_to_device, data_concatenate
from datasets import NIHCXR, MIMIC, NLMCXR
from losses import CombinedLoss
from models import *
from baselines.transformer.models import LSTM_Attn, Transformer, GumbelTransformer
from baselines.rnn.models import ST

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
                # output = model(image=source[0], threshold=threshold)
                # output = model(image=source[0], history=source[3], label=source[2])
                # output = model(image=source[0], label=source[2])
            else:
                # output = model(source[0], source[1])
                output = model(source[0])
                
            outputs.append(data_to_device(output))
            targets.append(data_to_device(target))

        outputs = data_concatenate(outputs)
        targets = data_concatenate(targets)
    
    return outputs, targets
#
# --- Hyperparameters ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["OMP_NUM_THREADS"] = "4"
# torch.set_num_threads(4)
torch.manual_seed(seed=123)

RELOAD = False # True / False
PHASE = 'TRAIN' # TRAIN / TEST / INFER
DATASET_NAME = 'MIMIC' # NIHCXR / NLMCXR / MIMIC
BACKBONE_NAME = 'SwinT'
MODEL_NAME = 'HiMrGn' # HiMrGn

if DATASET_NAME == 'MIMIC':
    EPOCHS = 100 # Start overfitting after 20 epochs
    BATCH_SIZE =  2 if PHASE == 'TRAIN' else 32 # 192 # Fit 4 GPUs
    MILESTONES = [25, 40, 55, 70, 85] # Reduce LR by 10 after reaching milestone epochs
    
elif DATASET_NAME == 'NLMCXR':
    EPOCHS = 50 # Start overfitting after 20 epochs
    BATCH_SIZE = 64 if PHASE == 'TRAIN' else 64 # Fit 4 GPUs
    MILESTONES = [25] # Reduce LR by 10 after reaching milestone epochs
    
else:
    raise ValueError('Invalid DATASET_NAME')

if __name__ == "__main__":
    # --- Choose Inputs/Outputs
    if MODEL_NAME in ['ClsGen', 'ClsGenInt']:
        SOURCES = ['image','caption','label','history']
        TARGETS = ['caption','label']
        KW_SRC = ['image','caption','label','history']
        KW_TGT = None
        KW_OUT = None
                
    elif MODEL_NAME == 'VisualTransformer':
        SOURCES = ['image','caption']
        TARGETS = ['caption']
        KW_SRC = ['image','caption'] # kwargs of Classifier
        KW_TGT = None
        KW_OUT = None
        
    elif MODEL_NAME == 'GumbelTransformer':
        SOURCES = ['image','caption','caption_length']
        TARGETS = ['caption','label']
        KW_SRC = ['image','caption','caption_length'] # kwargs of Classifier
        KW_TGT = None
        KW_OUT = None

    elif MODEL_NAME == 'HiMrGn':
        SOURCES = ['image', 'findings', 'impression']
        TARGETS = ['findings','impression']
        KW_SRC = ['image', 'findings', 'impression'] # kwargs of Classifier
        KW_TGT = ['findings', 'impression']
        KW_OUT = None
        
    else:
        raise ValueError('Invalid BACKBONE_NAME')
        
    # --- Choose a Dataset ---
    if DATASET_NAME == 'MIMIC':
        INPUT_SIZE = (224,224)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 2
        VIEW_POS = ['AP']
        
        dataset = MIMIC('/mnt/chenlb/mimic/', INPUT_SIZE, view_pos=VIEW_POS, max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(seed=123)
        
        # if not os.path.exists('/home/chenlb/xray_report_generation/checkpoints/train_data.pt'):
        #     torch.save(train_data, '/home/chenlb/xray_report_generation/checkpoints/train_data.pt')
        #     torch.save(val_data, '/home/chenlb/xray_report_generation/checkpoints/val_data.pt')
        #     torch.save(test_data, '/home/chenlb/xray_report_generation/checkpoints/test_data.pt')
        # else:
        #     train_data = torch.load('/home/chenlb/xray_report_generation/checkpoints/train_data.pt')
        #     val_data = torch.load('/home/chenlb/xray_report_generation/checkpoints/val_data.pt')
        #     test_data = torch.load('/home/chenlb/xray_report_generation/checkpoints/test_data.pt')

        VOCAB_SIZE = dataset.tokenizer.vocab_size
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')
            
    elif DATASET_NAME == 'NLMCXR':
        INPUT_SIZE = (256,256)
        MAX_VIEWS = 2
        NUM_LABELS = 114
        NUM_CLASSES = 2

        dataset = NLMCXR('/home/LAB/liudy/Datasets/NLMCXR/', INPUT_SIZE, view_pos=['AP','PA','LATERAL'], max_views=MAX_VIEWS, sources=SOURCES, targets=TARGETS)
        train_data, val_data, test_data = dataset.get_subsets(seed=123)
        
        VOCAB_SIZE = len(dataset.vocab)
        POSIT_SIZE = dataset.max_len
        COMMENT = 'MaxView{}_NumLabel{}_{}History'.format(MAX_VIEWS, NUM_LABELS, 'No' if 'history' not in SOURCES else '')
        
    else:
        raise ValueError('Invalid DATASET_NAME')

    # --- Choose a Model ---
    if MODEL_NAME == 'HiMrGn':
        LR = 5e-4 # Fastest LR
        # LR = 3e-4 # Fastest LR
        WD = 1e-2 # Avoid overfitting with L2 regularization
        DROPOUT = 0.1 # Avoid overfitting
        NUM_EMBEDS = 256
        FWD_DIM = 256
        
        NUM_HEADS = 8
        NUM_LAYERS = 1

        swin_transformer = SwinFeatureExtractor()

        features_projector = DiseaseFeatureProjector()

        findings_decoder = TextDecoder()
        findings_generator = FindingsGenerator(findings_decoder)

        co_attention_module = CoAttentionModule()

        impression_decoder = TextDecoder()
        impression_generator = ImpressionGenerator(impression_decoder)

        # word_translator = WordTranslator(model_file_path="/mnt/chenlb/mimic/mimic_unigram_1000.model")
        cxr_bert_feature_extractor = CXR_BERT_FeatureExtractor()


        model = HiMrGn(image_encoder=swin_transformer, 
                       features_projector=features_projector, 
                       findings_decoder=findings_generator, 
                       co_attention_module=co_attention_module,
                       impression_decoder=impression_generator,
                       cxr_bert_feature_extractor=cxr_bert_feature_extractor)
        
        criterion = CombinedLoss().cuda()
        
    else:
        raise ValueError('Invalid MODEL_NAME')
    
    # --- Main program ---
    train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = nn.DataParallel(model).cuda()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES)

    print('Total Parameters:', sum(p.numel() for p in model.parameters()))
    
    last_epoch = -1
    best_metric = 1e9

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    # checkpoint_path_from = 'checkpoints/2023-02-16_22:46:43_{}_{}_{}_{}.pt'.format(DATASET_NAME,MODEL_NAME,BACKBONE_NAME,COMMENT)
    checkpoint_path_from = 'checkpoints/{}_{}_{}_{}_{}.pt'.format(date_time, DATASET_NAME,MODEL_NAME,BACKBONE_NAME,COMMENT)
    checkpoint_path_to = 'checkpoints/{}_{}_{}_{}_{}.pt'.format(date_time, DATASET_NAME,MODEL_NAME,BACKBONE_NAME,COMMENT)
    
    if RELOAD:
        last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model, optimizer, scheduler) # Reload
        # last_epoch, (best_metric, test_metric) = load(checkpoint_path_from, model) # Fine-tune
        print('Reload From: {} | Last Epoch: {} | Validation Metric: {} | Test Metric: {}'.format(checkpoint_path_from, last_epoch, best_metric, test_metric))

    if PHASE == 'TRAIN':
        scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(last_epoch+1, EPOCHS):
            print('Epoch:', epoch)
            train_loss = train(train_loader, model, optimizer, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, scaler=scaler)
            val_loss = test(val_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, return_results=False)
            test_loss = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, return_results=False)
            
            scheduler.step()
            
            if best_metric > val_loss:
                best_metric = val_loss
                save(checkpoint_path_to, model, optimizer, scheduler, epoch, (val_loss, test_loss))
                print('New Best Metric: {}'.format(best_metric)) 
                print('Saved To:', checkpoint_path_to)
    
    elif PHASE == 'TEST':
        # Output the file list for inspection
        out_file_img = open('outputs/{}_{}_{}_{}_Img.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        for i in range(len(test_data.idx_pidsid)):
            out_file_img.write(test_data.idx_pidsid[i][0] + ' ' + test_data.idx_pidsid[i][1] + '\n')
            
        # test_loss, test_outputs, test_targets = test(test_loader, model, criterion, device='cuda', kw_src=KW_SRC, kw_tgt=KW_TGT, kw_out=KW_OUT, select_outputs=[1])
        
        # test_auc = []
        # test_f1 = []
        # test_prc = []
        # test_rec = []
        # test_acc = []
        
        # threshold = 0.25
        # NUM_LABELS = 14
        # for i in range(NUM_LABELS):
        #     try:
        #         test_auc.append(metrics.roc_auc_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1]))
        #         test_f1.append(metrics.f1_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
        #         test_prc.append(metrics.precision_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
        #         test_rec.append(metrics.recall_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
        #         test_acc.append(metrics.accuracy_score(test_targets.cpu()[...,i], test_outputs.cpu()[...,i,1] > threshold))
                
        #     except:
        #         print('An error occurs for label', i)
                
        # test_auc = np.mean([x for x in test_auc if str(x) != 'nan'])
        # test_f1 = np.mean([x for x in test_f1 if str(x) != 'nan'])
        # test_prc = np.mean([x for x in test_prc if str(x) != 'nan'])
        # test_rec = np.mean([x for x in test_rec if str(x) != 'nan'])
        # test_acc = np.mean([x for x in test_acc if str(x) != 'nan'])
        
        # print('Accuracy       : {}'.format(test_acc))
        # print('Macro AUC      : {}'.format(test_auc))
        # print('Macro F1       : {}'.format(test_f1))
        # print('Macro Precision: {}'.format(test_prc))
        # print('Macro Recall   : {}'.format(test_rec))
        # print('Micro AUC      : {}'.format(metrics.roc_auc_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1], average='micro')))
        # print('Micro F1       : {}'.format(metrics.f1_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
        # print('Micro Precision: {}'.format(metrics.precision_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
        # print('Micro Recall   : {}'.format(metrics.recall_score(test_targets.cpu()[...,:NUM_LABELS] == 1, test_outputs.cpu()[...,:NUM_LABELS,1] > threshold, average='micro')))
        
    elif PHASE == 'INFER':
        # txt_test_outputs, txt_test_targets = infer(test_loader, model, device='cuda', threshold=0.25)
        txt_test_outputs, txt_test_targets = infer(test_loader, model, device='cuda')

        gen_outputs = txt_test_outputs
        gen_targets = txt_test_targets
        # gen_outputs = txt_test_outputs[0]
        # gen_targets = txt_test_targets[0]
        
        out_file_ref = open('outputs/x_{}_{}_{}_{}_Ref.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        out_file_hyp = open('outputs/x_{}_{}_{}_{}_Hyp.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        out_file_lbl = open('outputs/x_{}_{}_{}_{}_Lbl.txt'.format(DATASET_NAME, MODEL_NAME, BACKBONE_NAME, COMMENT), 'w')
        
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