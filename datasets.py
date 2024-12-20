# --- Base packages ---
import os
import json
import pickle
import numpy as np
import pandas as pd

# --- PyTorch packages ---
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
# --- Helper packages ---
from random import shuffle
import sentencepiece as spm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from collections import defaultdict
from utils import *
# --- Datasets ---
class MIMIC(data.Dataset): # MIMIC-CXR Dataset
    def __init__(self, directory, input_size=(224,224), random_transform=True,
                view_pos=['AP'], max_views=2, sources=['image','history'], targets=['label'], 
                max_len=256, model_name='microsoft/BiomedVLP-CXR-BERT-specialized', train_stage=2):

        self.source_sections = ['INDICATION:', 'HISTORY:', 'CLINICAL HISTORY:', 'REASON FOR EXAM:', 'REASON FOR EXAMINATION:', 'CLINICAL INFORMATION:', 'CLINICAL INDICATION:', 'PATIENT HISTORY:']
        self.target_sections = ['FINDINGS:', 'IMPRESSION:']
        # 使用与 CXR-BERT 一致的 tokenizer
        # self.embedding_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True
        ).to(device)
        self.embedding_layer = self.embedding_model.bert.embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        self.sources = sources # Choose which section as input
        self.targets = targets # Choose which section as output
        self.max_views = max_views
        self.view_pos = view_pos
        self.max_len = max_len
        
        self.dir = directory
        self.input_size = input_size
        self.random_transform = random_transform
        self.train_stage = train_stage
        self.__input_data(binary_mode=True)
        
        if random_transform:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.1,0.1,0.1), 
                    transforms.RandomRotation(15, expand=True)]),
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.idx_pidsid)
    
    def __getitem__(self, idx):
        idx = self.idx_pidsid[idx] 
        info = self.img_captions[idx]
        
        # 准备所有可能需要的数据
        findings = info.get('FINDINGS:', '')
        impression = info.get('IMPRESSION:', '')
        
        output = {
            'findings': findings,
            'impression': impression,
            'label': self.img_labels[idx[:2]],
            'idx': idx,
            'gts': [findings, impression]  # 修正这里，添加正确的gts
        }
        
        # 收集history文本
        source_info = []
        for section, content in info.items():
            if section in self.source_sections:
                source_info.append(content)
        output['history'] = ' '.join(source_info)
        
        # 处理图像
        if 'image' in self.sources:
            img_file = os.path.join(self.dir, 'images', idx[0][:3], idx[0], idx[1], idx[2] + '.jpg')
            img = Image.open(img_file).convert('RGB')
            pos = self.img_positions[idx[2]]
            img = self.transform(img)
            vpos = self.dict_positions[pos]
            output['image'] = (img, vpos)

        return output
    
    def get_embeddings(self, text, max_len=None, device="cuda"):
        # Tokenize
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_len if max_len is not None else self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # 返回PyTorch张量
        )

        input_ids = encoded['input_ids'].to(device)
        token_type_ids = encoded['token_type_ids'].to(device)
        
        # 直接使用embedding层
        with torch.no_grad():
            embeddings = self.embedding_layer(
                input_ids=input_ids,
                token_type_ids=token_type_ids
            )
            embeddings = embeddings.squeeze(0)  # 移除batch维度
            
        return input_ids, embeddings
    
    def get_token_ids(self, text, max_len=None):
        """
        只获取token ids，不进行embedding转换
        Returns:
            token_ids: shape (seq_len,) 包含padding的token id序列
            attention_mask: shape (seq_len,) 表示哪些位置是真实token，哪些是padding
        """
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len if max_len is not None else self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return encoded['input_ids'].squeeze(0)

    def __get_reports_images(self, file_name='reports.json'):
        caption_file = json.load(open(os.path.join(self.dir, file_name), 'r'))
        img_captions = {}
        img_files = {}
        self.findings_token_distribution = defaultdict(int)
        self.impression_token_distribution = defaultdict(int)
        self.source_info_token_distribution = defaultdict(int)
        miss_cnt = 0
        for file_name, report in tqdm(caption_file.items()):
            k = file_name[-23:-4]
            p = file_name[-23:-20]
            pid,sid = k.split('/')
            try:
                # List all available images in each folder
                file_list = os.listdir(os.path.join(self.dir, 'images', p, pid, sid))

                if len(file_list):
                    for i, file in enumerate(file_list): 
                        img_files[(pid,sid,file[:-4])] = file
                        img_captions[(pid,sid,file[:-4])] = report    # Include FINDINGS and IMPRESSION
            
                        findings = report.get('FINDINGS:', '')
                        impression = report.get('IMPRESSION:', '')

                        source_info = []
                        for section, content in report.items():
                            if section in self.source_sections:
                                source_info.append(content)
                        source_info = ' '.join(source_info)

                        # 对 FINDINGS 和 IMPRESSION 进行编码
                        origin_encoded_findings = self.tokenizer.encode(findings, add_special_tokens=True)
                        origin_encoded_impression = self.tokenizer.encode(impression, add_special_tokens=True)
                        origin_encoded_source_info = self.tokenizer.encode(source_info, add_special_tokens=True)

                        # 更新分布字典
                        self.findings_token_distribution[len(origin_encoded_findings)] += 1
                        self.impression_token_distribution[len(origin_encoded_impression)] += 1
                        self.source_info_token_distribution[len(origin_encoded_source_info)] += 1
            except Exception as e:
                miss_cnt += 1
                pass

        plot_length_distribution(self.findings_token_distribution, "Findings Token Length Distribution")
        plot_length_distribution(self.impression_token_distribution, "Impression Token Length Distribution")
        plot_length_distribution(self.source_info_token_distribution, "History Token Length Distribution")
        print("无对应文件夹数量：", miss_cnt)
        return img_captions, img_files

    def __get_view_positions(self, file_name='mimic-cxr-2.0.0-metadata.csv'):
        txt_file = os.path.join(self.dir, file_name)
        data = pd.read_csv(txt_file, dtype=object)
        data = data.to_numpy().astype(str)
        return dict(zip(data[:,0].tolist(), data[:,4].tolist())), np.unique(data[:,4]).tolist()

    def __get_labels(self, binary_mode, file_name='mimic-cxr-2.0.0-chexpert.csv'):
        txt_file = os.path.join(self.dir, file_name)
        data = pd.read_csv(txt_file, dtype=object)

        label_names = list(data.columns.values[2:])
        data = data.to_numpy().astype(str)
        if binary_mode:
            data[data == '-1.0'] = "1" # 2 Not sure
            data[data ==  'nan'] = "0" # 3 Not mentioned
        else:
            data[data == '-1.0'] = "2" # 2 Not sure
            data[data ==  'nan'] = "3" # 3 Not mentioned
        
        img_labels = {}
        for i in range(len(data)):
            pid = 'p' + data[i,0].item()
            sid = 's' + data[i,1].item()
            labels = data[i,2:].astype(float)
            img_labels[(pid,sid)] = labels
        return img_labels, label_names

    def __get_nounphrase(self, top_k=100, file_name='count_nounphrase.json'):
        count_np = json.load(open(os.path.join(self.dir, file_name), 'r'))
        sorted_count_np = sorted([(k,v) for k,v in count_np.items()], key=lambda x: x[1], reverse=True)
        top_nounphrases = [k for k,v in sorted_count_np][:top_k]
        return top_nounphrases
           
    def __input_data(self, binary_mode=True):
        self.img_positions, self.list_positions = self.__get_view_positions()
        self.dict_positions = dict(zip(self.list_positions, range(len(self.list_positions))))

        img_info_checkpoint_file = "/home/chenlb/xray_report_generation/checkpoints/img_info.pkl"
        if os.path.exists(img_info_checkpoint_file):
            with open(img_info_checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                self.img_captions, self.img_files = checkpoint
                print(f"Loaded checkpoint from {img_info_checkpoint_file}.")
        else:
            print("Checkpoint file not found. Generating new data.")
            self.img_captions, self.img_files = self.__get_reports_images()
            checkpoint = (self.img_captions, self.img_files)
            with open(img_info_checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
                print("Checkpoint saved to file.")

        self.img_labels, self.list_diseases = self.__get_labels(binary_mode)
        self.dict_diseases = dict(zip(self.list_diseases, range(len(self.list_diseases))))
        self.idx_pidsid = list(self.img_captions.keys())
        self.top_np = self.__get_nounphrase()
            
    def __generate_splits(self, test_size=0.2, seed=0, file_name='mimic-cxr-2.0.0-chexpert.csv'):
        train_val_file = open(os.path.join(self.dir, 'train_val_list.txt'), 'w')
        test_file = open(os.path.join(self.dir, 'test_list.txt'), 'w')

        txt_file = os.path.join(self.dir, file_name)
        data = pd.read_csv(txt_file, dtype=object)
        data = data.to_numpy().astype(str)

        # 1 PID can have multiple SIDs
        pid_sid = {}
        for i in range(len(data)):
            pid = data[i,0].item()
            sid = data[i,1].item() 
            
            if pid in pid_sid:
                pid_sid[pid].append(sid)
            else:
                pid_sid[pid] = [sid]

        np.random.seed(seed)
        unique_pid = np.unique(data[:,0])        
        random_pid = np.random.permutation(unique_pid)

        pvt = int((1-test_size) * len(unique_pid))
        train_pid = random_pid[:pvt]
        test_pid = random_pid[pvt:]

        for pid in train_pid:
            for sid in pid_sid[pid]:
                if ('p'+pid,'s'+sid) in self.img_captions:
                    train_val_file.write('p' + pid + '/' + 's' + sid + '\n')
        
        for pid in test_pid:
            for sid in pid_sid[pid]:
                if ('p'+pid,'s'+sid) in self.img_captions:
                    test_file.write('p' + pid + '/' + 's' + sid + '\n')

    def get_subsets(self, pvt=0.9, seed=0, generate_splits=False, debug_mode=False, train_phase=True):
        if generate_splits:
            self.__generate_splits(seed=0)
            print('New splits generated')
        

        if self.train_stage == 1:
            train_files = np.loadtxt(os.path.join(self.dir, 'train_val_list_findings.txt'), dtype=str)
            test_files = np.loadtxt(os.path.join(self.dir, 'test_list_findings.txt'), dtype=str)
        else:
            train_files = np.loadtxt(os.path.join(self.dir, 'train_val_list_findings_impression.txt'), dtype=str)
            test_files = np.loadtxt(os.path.join(self.dir, 'test_list_findings_impression.txt'), dtype=str)
            
        train_files = np.array([f.split('/') for f in train_files])
        test_files = np.array([f.split('/') for f in test_files])
        
        np.random.seed(seed)
        indices = np.random.permutation(len(train_files))
        pivot = int(len(train_files) * pvt)
        train_indices = indices[:pivot]
        val_indices = indices[pivot:]

        train_dataset = MIMIC(self.dir, self.input_size, self.random_transform, 
                              self.view_pos, self.max_views, self.sources, self.targets, 
                              self.max_len)
        train_dataset.idx_pidsid = [(pid,sid,uuid) for pid,sid,uuid in train_files[train_indices]] if not debug_mode else [(pid,sid,uuid) for pid,sid,uuid in train_files[train_indices]][:10000]
        
        val_dataset = MIMIC(self.dir, self.input_size, False, 
                            self.view_pos, self.max_views, self.sources, self.targets, 
                            self.max_len)
        val_dataset.idx_pidsid = [(pid,sid,uuid) for pid,sid,uuid in train_files[val_indices]] if not debug_mode else [(pid,sid,uuid) for pid,sid,uuid in train_files[val_indices]][:1000]

        test_dataset = MIMIC(self.dir, self.input_size, False, 
                            self.view_pos, self.max_views, self.sources, self.targets, 
                            self.max_len)
        test_dataset.idx_pidsid = [(pid,sid,uuid) for pid,sid,uuid in test_files] if not debug_mode else [(pid,sid,uuid) for pid,sid,uuid in test_files][:1000]

        # Use only a subset to make the model run quickly
        if train_phase:
            subset_size = 100
        else:
            subset_size = 100#000
        
        val_idx = np.random.choice(len(val_dataset.idx_pidsid), size=min(subset_size, len(val_dataset.idx_pidsid)), replace=False)
        test_idx = np.random.choice(len(test_dataset.idx_pidsid), size=min(subset_size, len(test_dataset.idx_pidsid)), replace=False)
        
        train_dataset.idx_pidsid = train_dataset.idx_pidsid[:]
        val_dataset.idx_pidsid = [val_dataset.idx_pidsid[i] for i in val_idx]
        test_dataset.idx_pidsid = [test_dataset.idx_pidsid[i] for i in test_idx]
        
        return train_dataset, val_dataset, test_dataset
        # return train_dataset, None, None
