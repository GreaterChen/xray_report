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
class NIHCXR(data.Dataset): # Chest X-Ray 14 Dataset
    def __init__(self, directory, input_size=(512,512), random_transform=True):
        self.list_diseases = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        self.dict_diseases = dict(zip(self.list_diseases, range(len(self.list_diseases))))

        self.dir = directory
        self.input_size = input_size
        self.random_transform = random_transform
        self.__input_data()
        
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
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.dir + 'images/' + self.img_files[idx]).convert('RGB')
        return self.transform(img), self.img_labels[idx]

    def __input_data(self):
        txt_file = self.dir + 'Data_Entry_2017_v2020.csv'
        data = np.loadtxt(open(txt_file, "rb"), delimiter=",", skiprows=1, dtype=str)
        self.img_files = data[..., 0]
        self.img_labels = self.__one_hot_outer(data[..., 1])

    def __one_hot_inner(self, labels):
        labels = labels.split('|')
        indices = []

        for label in labels:
            if label in self.dict_diseases:
                indices.append(self.dict_diseases[label])
            else:
                # Filtering invalid labels
                index = np.argmax([label in disease for disease in self.list_diseases])
                indices.append(index.item())

        labels = np.zeros(len(self.list_diseases))
        labels[indices] = 1
        return labels

    def __one_hot_outer(self, labels):
        one_hot = []
        for i in range(labels.shape[0]):
            one_hot.append(self.__one_hot_inner(labels[i]))
        return np.array(one_hot)

    def get_subsets(self, pvt=0.9, seed=0):
        file_to_label = dict(zip(self.img_files, self.img_labels))

        train_files = np.loadtxt(self.dir + 'train_val_list.txt', dtype=str)
        train_labels = np.array([file_to_label[f] for f in train_files])

        test_files = np.loadtxt(self.dir + 'test_list.txt', dtype=str)
        test_labels = np.array([file_to_label[f] for f in test_files])

        np.random.seed(seed)
        indices = np.random.permutation(len(train_files))
        pivot = int(len(train_files) * pvt)
        train_indices = indices[:pivot]
        val_indices = indices[pivot:]

        train_dataset = NIHCXR(self.dir, input_size=self.input_size, random_transform=self.random_transform)
        train_dataset.img_files = train_files[train_indices]
        train_dataset.img_labels = train_labels[train_indices]

        val_dataset = NIHCXR(self.dir, input_size=self.input_size, random_transform=False)
        val_dataset.img_files = train_files[val_indices]
        val_dataset.img_labels = train_labels[val_indices]

        test_dataset = NIHCXR(self.dir, input_size=self.input_size, random_transform=False)
        test_dataset.img_files = test_files
        test_dataset.img_labels = test_labels

        return train_dataset, val_dataset, test_dataset

class MIMIC(data.Dataset): # MIMIC-CXR Dataset
    def __init__(self, directory, input_size=(224,224), random_transform=True,
                view_pos=['AP'], max_views=2, sources=['image','history'], targets=['label'], 
                max_len=256, model_name='microsoft/BiomedVLP-CXR-BERT-specialized', train_stage=2):

        self.source_sections = ['INDICATION:', 'HISTORY:', 'CLINICAL HISTORY:', 'REASON FOR EXAM:', 'REASON FOR EXAMINATION:', 'CLINICAL INFORMATION:', 'CLINICAL INDICATION:', 'PATIENT HISTORY:']
        self.target_sections = ['FINDINGS:', 'IMPRESSION:']
        # 使用与 CXR-BERT 一致的 tokenizer
        # self.embedding_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.embedding_model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
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
        return len(self.idx_pidsid[:50])
    
    def __getitem__(self, idx):
        idx = self.idx_pidsid[idx] 

        sources = []
        targets = []
        gts = []

        if 'image' in self.sources:
            img_file = os.path.join(self.dir, 'images', idx[0][:3] , idx[0], idx[1], idx[2] + '.jpg')
            img = Image.open(img_file).convert('RGB')
            pos = self.img_positions[idx[2]]
            img = self.transform(img) # (1,C,W,H) 
            vpos = self.dict_positions[pos]

        label = self.img_labels[idx[:2]]

        info = self.img_captions[idx]

        # 获取 FINDINGS 和 IMPRESSION
        findings = info.get('FINDINGS:', '')
        impression = info.get('IMPRESSION:', '')

        gts.append(findings)
        gts.append(impression)

        # 使用 CXR-BERT 对 FINDINGS 和 IMPRESSION 进行编码
        token_ids_findings, embeddings_findings = self.get_embeddings(findings, max_len=self.max_len)
        token_ids_impression, embeddings_impression = self.get_embeddings(impression, max_len=self.max_len)

        source_info = []
        for section, content in info.items():
            if section in self.source_sections:
                source_info.append(content)
        source_info = ' '.join(source_info)
        _, embeddings_source_info = self.get_embeddings(source_info, max_len=self.max_len)

        for i in range(len(self.sources)):
            if self.sources[i] == 'image':
                sources.append((img,vpos))
            elif self.sources[i] == 'findings':
                sources.append(embeddings_findings)
            elif self.sources[i] == 'impression':
                sources.append(embeddings_impression)
            elif self.sources[i] == 'history':
                sources.append(embeddings_source_info)

        for i in range(len(self.targets)):
            if self.targets[i] == 'findings':
                targets.append(token_ids_findings)
            elif self.targets[i] == 'impression':
                targets.append(token_ids_impression)
            elif self.targets[i] == 'label':
                targets.append(label)

        return sources if len(sources) > 1 else sources[0], targets if len(targets) > 1 else targets[0], idx, gts
    
    def get_embeddings(self, text, max_len=None):
        # Tokenize
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len if max_len is not None else self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # 返回PyTorch张量
        )
        
        # 直接使用embedding层
        with torch.no_grad():
            embeddings = self.embedding_layer(
                input_ids=encoded['input_ids'],
                token_type_ids=encoded['token_type_ids']
            )
            embeddings = embeddings.squeeze(0)  # 移除batch维度
            
        return encoded['input_ids'], embeddings
    
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
            subset_size = 1000
        else:
            subset_size = 100#000
        
        val_idx = np.random.choice(len(val_dataset.idx_pidsid), size=min(subset_size, len(val_dataset.idx_pidsid)), replace=False)
        test_idx = np.random.choice(len(test_dataset.idx_pidsid), size=min(subset_size, len(test_dataset.idx_pidsid)), replace=False)
        
        train_dataset.idx_pidsid = train_dataset.idx_pidsid[:]
        val_dataset.idx_pidsid = [val_dataset.idx_pidsid[i] for i in val_idx]
        test_dataset.idx_pidsid = [test_dataset.idx_pidsid[i] for i in test_idx]
        
        return train_dataset, val_dataset, test_dataset
        # return train_dataset, None, None

class NLMCXR(data.Dataset): # Open-I Dataset
    def __init__(self, directory, input_size=(256,256), random_transform=True,
                view_pos=['AP', 'PA', 'LATERAL'], max_views=2, sources=['image','history'], targets=['label'], 
                max_len=1000, vocab_file='nlmcxr_unigram_1000.model'):
        
        self.source_sections = ['INDICATION', 'COMPARISON']
        self.target_sections = ['FINDINGS']
        self.vocab = spm.SentencePieceProcessor(model_file=directory + vocab_file)
        self.vocab_file = vocab_file # Save it for subsets

        self.sources = sources # Choose which section as input
        self.targets = targets # Choose which section as output
        self.max_views = max_views
        self.view_pos = view_pos
        self.max_len = max_len

        self.dir = directory
        self.input_size = input_size
        self.random_transform = random_transform
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
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        sources, targets = [], []
        tmp_rep = self.captions[self.file_report[file_name]['image'][0] + '.png']
        
        # ------ Multiview Images ------
        if 'image' in self.sources:
            imgs, vpos = [], []
            images = self.file_report[file_name]['image']

            # Randomly select V images from each folder 
            new_orders = np.random.permutation(len(images))
            img_files = np.array(images)[new_orders].tolist()

            for i in range(min(self.max_views,len(img_files))):
                img_file = self.dir + 'images/' + img_files[i] + '.png'
                img = Image.open(img_file).convert('RGB')
                imgs.append(self.transform(img).unsqueeze(0)) # (1,C,W,H)
                vpos.append(1) # We do not know what view position of the image is, so just let it be 1
                
            # If the number of images is smaller than V, pad the tensor with dummy images
            cur_len = len(vpos)
            for i in range(cur_len, self.max_views):
                imgs.append(torch.zeros_like(imgs[0]))
                vpos.append(-1) # Empty mask
            
            imgs = torch.cat(imgs, dim=0) # (V,C,W,H)
            vpos = np.array(vpos, dtype=np.int64) # (V)

        # ------ Additional Information ------
        info = self.file_report[file_name]['report']
        
        source_info = []
        for section, content in info.items():
            if section in self.source_sections:
                source_info.append(content)
        source_info = ' '.join(source_info)
        
        encoded_source_info = [self.vocab.bos_id()] + self.vocab.encode(source_info) + [self.vocab.eos_id()]
        source_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        source_info[:min(len(encoded_source_info), self.max_len)] = encoded_source_info[:min(len(encoded_source_info), self.max_len)]

        target_info = []
        for section, content in info.items():
            if section in self.target_sections:
                target_info.append(content)
        # target_info = ' '.join(target_info)
        target_info = tmp_rep # This load the document from our previous AAAI paper (preprocessed documents)
        
        np_labels = np.zeros(len(self.top_np), dtype=float)
        for i in range(len(self.top_np)):
            if self.top_np[i] in target_info:
                np_labels[i] = 1
        
        encoded_target_info = [self.vocab.bos_id()] + self.vocab.encode(target_info) + [self.vocab.eos_id()]
        target_info = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        target_info[:min(len(encoded_target_info), self.max_len)] = encoded_target_info[:min(len(encoded_target_info), self.max_len)]

        for i in range(len(self.sources)):
            if self.sources[i] == 'image':
                sources.append((imgs,vpos))
            if self.sources[i] == 'history':
                sources.append(source_info)
            if self.sources[i] == 'label':
                sources.append(np.concatenate([np.array(self.file_labels[file_name]), np_labels]))
            if self.sources[i] == 'caption':
                sources.append(target_info)
            if self.sources[i] == 'caption_length':
                sources.append(min(len(encoded_target_info), self.max_len))
                
        for i in range(len(self.targets)):
            if self.targets[i] == 'label':
                targets.append(np.concatenate([np.array(self.file_labels[file_name]), np_labels]))
            if self.targets[i] == 'caption':
                targets.append(target_info)
            if self.targets[i] == 'caption_length':
                targets.append(min(len(encoded_target_info), self.max_len))
                
        return sources if len(sources) > 1 else sources[0], targets if len(targets) > 1 else targets[0]

    def __get_nounphrase(self, top_k=100, file_name='count_nounphrase.json'):
        count_np = json.load(open(self.dir + file_name, 'r'))
        sorted_count_np = sorted([(k,v) for k,v in count_np.items()], key=lambda x: x[1], reverse=True)
        top_nounphrases = [k for k,v in sorted_count_np][:top_k]
        return top_nounphrases

    def __input_data(self, binary_mode=True):
        self.__input_caption()
        self.__input_report()
        self.__input_label()
        self.__filter_inputs()
        self.top_np = self.__get_nounphrase()
        
    def __input_label(self):
        with open(self.dir + 'file2label.json') as f:
            labels = json.load(f)
        self.file_labels = labels
        
    def __input_caption(self):
        with open(self.dir + 'captions.json') as f:
            captions = json.load(f)
        self.captions = captions
        
    def __input_report(self):
        with open(self.dir + 'reports_ori.json') as f:
            reports = json.load(f)
        self.file_list = [k for k in reports.keys()]
        self.file_report = reports

    def __filter_inputs(self):
        filtered_file_report = {}
        for k, v in self.file_report.items():
            if (len(v['image']) > 0) and (('FINDINGS' in v['report']) and (v['report']['FINDINGS'] != '')): # or (('IMPRESSION' in v['report']) and (v['report']['IMPRESSION'] != ''))):
                filtered_file_report[k] = v
        self.file_report = filtered_file_report
        self.file_list = [k for k in self.file_report.keys()]

    def get_subsets(self, train_size=0.7, val_size=0.1, test_size=0.2, seed=0):
        np.random.seed(seed)
        indices = np.random.permutation(len(self.file_list))
        train_pvt = int(train_size * len(self.file_list))
        val_pvt = int((train_size + val_size) * len(self.file_list))
        train_indices = indices[:train_pvt]
        val_indices = indices[train_pvt:val_pvt]
        test_indices = indices[val_pvt:]

        master_file_list = np.array(self.file_list)

        train_dataset = NLMCXR(self.dir, self.input_size, self.random_transform, 
                              self.view_pos, self.max_views, self.sources, self.targets, self.max_len, self.vocab_file)
        train_dataset.file_list = master_file_list[train_indices].tolist()

        # Consider change random_transform to False for validation
        val_dataset = NLMCXR(self.dir, self.input_size, False, 
                            self.view_pos, self.max_views, self.sources, self.targets, self.max_len, self.vocab_file)
        val_dataset.file_list = master_file_list[val_indices].tolist()

        # Consider change random_transform to False for testing
        test_dataset = NLMCXR(self.dir, self.input_size, False, 
                             self.view_pos, self.max_views, self.sources, self.targets, self.max_len, self.vocab_file)
        test_dataset.file_list = master_file_list[test_indices].tolist()

        return train_dataset, val_dataset, test_dataset
    
class TextDataset(data.Dataset):
    def __init__(self, text_file, label_file, sources=['caption'], targets=['label'],
                 vocab_file='/home/hoang/Datasets/MIMIC/mimic_unigram_1000.model', max_len=1000):
        self.text_file = text_file
        self.label_file = label_file
        self.vocab = spm.SentencePieceProcessor(model_file=vocab_file)
        self.sources = sources # Choose which section as input
        self.targets = targets # Choose which section as output
        self.max_len = max_len
        self.__input_data()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        encoded_text = [self.vocab.bos_id()] + self.vocab.encode(self.lines[idx].strip()) + [self.vocab.eos_id()]
        text = np.ones(self.max_len, dtype=np.int64) * self.vocab.pad_id()
        text[:min(len(encoded_text), self.max_len)] = encoded_text[:min(len(encoded_text), self.max_len)]
        
        sources = []
        for i in range(len(self.sources)):
            if self.sources[i] == 'label':
                sources.append(self.labels[idx])
            if self.sources[i] == 'caption':
                sources.append(text)
            if self.sources[i] == 'caption_length':
                sources.append(min(len(encoded_text), self.max_len))
        
        targets = []
        for i in range(len(self.targets)):
            if self.targets[i] == 'label':
                targets.append(self.labels[idx])
            if self.targets[i] == 'caption':
                targets.append(text)
            if self.targets[i] == 'caption_length':
                targets.append(min(len(encoded_text), self.max_len))
                
        return sources if len(sources) > 1 else sources[0], targets if len(targets) > 1 else targets[0]
    
    def __input_data(self):
        data_file = open(self.text_file, 'r') 
        self.lines = data_file.readlines()
        self.labels = np.loadtxt(self.label_file, dtype='float')