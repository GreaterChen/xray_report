# --- Base packages ---
import os
import json
import pickle
import re
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
    # 类变量用于存储共享数据
    _shared_data = {
        'loaded': False,
        'annotation': None,
    }

    source_sections = ['INDICATION:', 'HISTORY:', 'CLINICAL HISTORY:', 'REASON FOR EXAM:', 'REASON FOR EXAMINATION:', 'CLINICAL INFORMATION:', 'CLINICAL INDICATION:', 'PATIENT HISTORY:']
    target_sections = ['FINDINGS:', 'IMPRESSION:']

    @classmethod
    def load_shared_data(cls, directory, binary_mode=True):
        """静态方法，用于加载所有数据集共享的数据"""
        if cls._shared_data['loaded']:
            return
        
        annotation_file = os.path.join(directory, 'mimic_annotation_promptmrg_new_mrgn.json')
        with open(annotation_file, 'r') as f:
            annotation_data = json.load(f)

        cls._shared_data['annotation'] = annotation_data
        cls._shared_data['loaded'] = True


    def __init__(self, directory, input_size=(224,224), random_transform=True, train_stage=2, tokenizer=None, mode='train', max_len=None, subset_size=None):
        
        self.load_shared_data(directory)

        self.tokenizer = tokenizer
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.sep_token_id  # BERT使用[SEP]作为EOS
        self.pad_token_id = self.tokenizer.pad_token_id

        self.sources = ['image', 'findings', 'impression', 'history']
        self.targets = ['findings', 'impression', 'label']
        
        self.dir = directory
        self.input_size = input_size
        self.random_transform = random_transform
        self.train_stage = train_stage
        self.mode = mode
        self.subset_size = subset_size
        self.max_len = max_len
        # 使用共享数据
        self.data = self._shared_data['annotation'][self.mode]

        if random_transform:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(input_size),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))])

    def __len__(self):
        if self.subset_size is not None:
            return self.subset_size
        else:
            return len(self.data)
    
    def __getitem__(self, idx):
        info = self.data[idx]

        impression = self.my_pre_caption(info['impression'])
        findings = self.my_pre_caption(info['findings'])
        history = self.my_pre_caption(info['history'])
        output = {
            'findings': findings,
            'impression': impression,
            'history': history,
            'label': info['labels'],
            'gts': [findings, impression]
        }

        
        # 处理图像
        img_file = os.path.join(self.dir, 'images', info['image_path'][0])
        img = Image.open(img_file).convert('RGB')
        img = self.transform(img)
        output['image'] = img

        return output
    
    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
                .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def my_pre_caption(self, caption, max_words=196):
        caption = self.clean_report_mimic_cxr(caption)
        #truncate caption
        caption_words = caption.split(' ')
        if len(caption_words)>max_words:
            caption = ' '.join(caption_words[:])
        return caption
    

    # def get_embeddings(self, text, max_len=None, device="cuda"):
    #     # Tokenize
    #     encoded = self.tokenizer(
    #         text,
    #         add_special_tokens=True,
    #         max_length=max_len if max_len is not None else self.max_len,
    #         padding='max_length',
    #         truncation=True,
    #         return_tensors='pt'  # 返回PyTorch张量
    #     )

    #     input_ids = encoded['input_ids'].to(device)
    #     token_type_ids = encoded['token_type_ids'].to(device)
        
    #     # 直接使用embedding层
    #     with torch.no_grad():
    #         embeddings = self.embedding_layer(
    #             input_ids=input_ids,
    #             token_type_ids=token_type_ids
    #         )
    #         embeddings = embeddings.squeeze(0)  # 移除batch维度
            
    #     return input_ids, embeddings
    
    # def get_token_ids(self, text, max_len=None):
    #     """
    #     只获取token ids，不进行embedding转换
    #     Returns:
    #         token_ids: shape (seq_len,) 包含padding的token id序列
    #     """
    #     encoded = self.tokenizer.encode_plus(
    #         text,
    #         add_special_tokens=True,
    #         max_length=max_len if max_len is not None else self.max_len,
    #         padding='max_length',
    #         truncation=True,
    #         return_tensors='pt'
    #     )

    #     return encoded['input_ids'].squeeze(0)

    # @staticmethod
    # def __get_reports_images(directory, file_name='reports.json'):
    #     caption_file = json.load(open(os.path.join(directory, file_name), 'r'))
    #     img_captions = {}
    #     img_files = {}
    #     findings_token_distribution = defaultdict(int)
    #     impression_token_distribution = defaultdict(int)
    #     source_info_token_distribution = defaultdict(int)
    #     miss_cnt = 0
    #     for file_name, report in tqdm(caption_file.items()):
    #         k = file_name[-23:-4]
    #         p = file_name[-23:-20]
    #         pid,sid = k.split('/')
    #         try:
    #             # List all available images in each folder
    #             file_list = os.listdir(os.path.join(directory, 'images', p, pid, sid))

    #             if len(file_list):
    #                 for i, file in enumerate(file_list): 
    #                     img_files[(pid,sid,file[:-4])] = file
    #                     img_captions[(pid,sid,file[:-4])] = report    # Include FINDINGS and IMPRESSION
            
    #                     findings = report.get('FINDINGS:', '')
    #                     impression = report.get('IMPRESSION:', '')

    #                     source_info = []
    #                     for section, content in report.items():
    #                         if section in MIMIC.source_sections:
    #                             source_info.append(content)
    #                     source_info = ' '.join(source_info)

    #         except Exception as e:
    #             miss_cnt += 1
    #             pass

    #     print("无对应文件夹数量：", miss_cnt)
    #     return img_captions, img_files

    # @staticmethod
    # def __get_view_positions(directory, file_name='mimic-cxr-2.0.0-metadata.csv'):
    #     txt_file = os.path.join(directory, file_name)
    #     data = pd.read_csv(txt_file, dtype=object)
    #     data = data.to_numpy().astype(str)
    #     return dict(zip(data[:,0].tolist(), data[:,4].tolist())), np.unique(data[:,4]).tolist()

    # @staticmethod
    # def __get_labels(directory, binary_mode, file_name='mimic-cxr-2.0.0-chexpert.csv'):
    #     txt_file = os.path.join(directory, file_name)
    #     data = pd.read_csv(txt_file, dtype=object)

    #     label_names = list(data.columns.values[2:])
    #     data = data.to_numpy().astype(str)
    #     if binary_mode:
    #         data[data == '-1.0'] = "1" # 2 Not sure
    #         data[data ==  'nan'] = "0" # 3 Not mentioned
    #     else:
    #         data[data == '-1.0'] = "2" # 2 Not sure
    #         data[data ==  'nan'] = "3" # 3 Not mentioned
        
    #     img_labels = {}
    #     for i in range(len(data)):
    #         pid = 'p' + data[i,0].item()
    #         sid = 's' + data[i,1].item()
    #         labels = data[i,2:].astype(float)
    #         img_labels[(pid,sid)] = labels
    #     return img_labels, label_names

    # @staticmethod
    # def __get_nounphrase(directory, top_k=100, file_name='count_nounphrase.json'):
    #     count_np = json.load(open(os.path.join(directory, file_name), 'r'))
    #     sorted_count_np = sorted([(k,v) for k,v in count_np.items()], key=lambda x: x[1], reverse=True)
    #     top_nounphrases = [k for k,v in sorted_count_np][:top_k]
    #     return top_nounphrases
           
    # def __input_data(self, binary_mode=True):
    #     self.img_positions, self.list_positions = self.__get_view_positions()
    #     self.dict_positions = dict(zip(self.list_positions, range(len(self.list_positions))))

    #     img_info_checkpoint_file = "/home/chenlb/xray_report_generation/checkpoints/img_info.pkl"
    #     if os.path.exists(img_info_checkpoint_file):
    #         with open(img_info_checkpoint_file, 'rb') as f:
    #             checkpoint = pickle.load(f)
    #             self.img_captions, self.img_files = checkpoint
    #             print(f"Loaded checkpoint from {img_info_checkpoint_file}.")
    #     else:
    #         print("Checkpoint file not found. Generating new data.")
    #         self.img_captions, self.img_files = self.__get_reports_images()
    #         checkpoint = (self.img_captions, self.img_files)
    #         with open(img_info_checkpoint_file, 'wb') as f:
    #             pickle.dump(checkpoint, f)
    #             print("Checkpoint saved to file.")

    #     self.img_labels, self.list_diseases = self.__get_labels(binary_mode)
    #     self.dict_diseases = dict(zip(self.list_diseases, range(len(self.list_diseases))))
    #     self.idx_pidsid = list(self.img_captions.keys())
    #     self.top_np = self.__get_nounphrase()
            
    # def __generate_splits(self, test_size=0.2, seed=0, file_name='mimic-cxr-2.0.0-chexpert.csv'):
    #     train_val_file = open(os.path.join(self.dir, 'train_val_list.txt'), 'w')
    #     test_file = open(os.path.join(self.dir, 'test_list.txt'), 'w')

    #     txt_file = os.path.join(self.dir, file_name)
    #     data = pd.read_csv(txt_file, dtype=object)
    #     data = data.to_numpy().astype(str)

    #     # 1 PID can have multiple SIDs
    #     pid_sid = {}
    #     for i in range(len(data)):
    #         pid = data[i,0].item()
    #         sid = data[i,1].item() 
            
    #         if pid in pid_sid:
    #             pid_sid[pid].append(sid)
    #         else:
    #             pid_sid[pid] = [sid]

    #     np.random.seed(seed)
    #     unique_pid = np.unique(data[:,0])        
    #     random_pid = np.random.permutation(unique_pid)

    #     pvt = int((1-test_size) * len(unique_pid))
    #     train_pid = random_pid[:pvt]
    #     test_pid = random_pid[pvt:]

    #     for pid in train_pid:
    #         for sid in pid_sid[pid]:
    #             if ('p'+pid,'s'+sid) in self.img_captions:
    #                 train_val_file.write('p' + pid + '/' + 's' + sid + '\n')
        
    #     for pid in test_pid:
    #         for sid in pid_sid[pid]:
    #             if ('p'+pid,'s'+sid) in self.img_captions:
    #                 test_file.write('p' + pid + '/' + 's' + sid + '\n')