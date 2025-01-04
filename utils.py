import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
from tqdm import tqdm

# ------ Helper Functions ------
def data_to_device(data, device='cpu'):
	if isinstance(data, torch.Tensor):
		data = data.to(device)
	elif isinstance(data, tuple):
		data = tuple(data_to_device(item,device) for item in data)
	elif isinstance(data, list):
		data = list(data_to_device(item,device) for item in data)
	elif isinstance(data, dict):
		data = dict((k,data_to_device(v,device)) for k,v in data.items())
	# else:
		# raise TypeError('Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.')
        
	return data

def data_concatenate(iterable_data, dim=0):
	data = iterable_data[0] # can be a list / tuple / dict / tensor
	if isinstance(data, torch.Tensor):
		return torch.cat([*iterable_data], dim=dim)
	elif isinstance(data, tuple):
		num_cols = len(data)
		num_rows = len(iterable_data)
		return_data = []
		for col in range(num_cols):
			data_col = []
			for row in range(num_rows):
				data_col.append(iterable_data[row][col])
			return_data.append(torch.cat([*data_col], dim=dim))
		return tuple(return_data)
	elif isinstance(data, list):
		num_cols = len(data)
		num_rows = len(iterable_data)
		return_data = []
		for col in range(num_cols):
			data_col = []
			for row in range(num_rows):
				data_col.append(iterable_data[row][col])
			return_data.append(torch.cat([*data_col], dim=dim))
		return list(return_data)
	elif isinstance(data, dict):
		num_cols = len(data)
		num_rows = len(iterable_data)
		return_data = []
		for col in data.keys():
			data_col = []
			for row in range(num_rows):
				data_col.append(iterable_data[row][col])
			return_data.append(torch.cat([*data_col], dim=dim))
		return dict((k,return_data[i]) for i,k in enumerate(data.keys()))
	else:
		raise TypeError('Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.')

def data_distributor(model, source):
	if isinstance(source, torch.Tensor):
		output = model(source)
	elif isinstance(source, tuple) or isinstance(source, list):
		output = model(*source)
	elif isinstance(source, dict):
		output = model(**source)
	else:
		raise TypeError('Unsupported DataType! Try List/Tuple!')
	return output
	
def args_to_kwargs(args, kwargs_list=None): # This function helps distribute input to corresponding arguments in Torch models
	if kwargs_list != None:
		if isinstance(args, dict): # Nothing to do here
			return args 
		else: # args is a list or tuple or single element
			if isinstance(args, torch.Tensor): # single element
				args = [args]
			assert len(args) == len(kwargs_list)
			return dict(zip(kwargs_list, args))
	else: # Nothing to do here
		return args
	
def prepare_batch_data(batch, data_loader, device):
    """准备批次数据，对整个batch进行tokenization
    
    Args:
        batch: 输入的批次数据
        data_loader: 数据加载器
        device: 计算设备
    
    Returns:
        source_data: 源数据字典
        target_data: 目标数据字典
    """
    # 处理图像数据
    if 'image' in batch:
        batch['image'] = data_to_device(batch['image'], device)
    
    # 对整个batch的文本进行tokenization
    text_fields = ['findings', 'impression', 'history']
    for field in text_fields:
        if field in batch:
            # 收集batch中的所有文本
            texts = batch[field]
            # 对整个batch进行tokenization
            encoded = data_loader.dataset.tokenizer(
                texts,
                max_length=data_loader.dataset.max_len,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ).to(device)  # [batch_size, max_len]
            encoded.input_ids[:,0] = data_loader.dataset.tokenizer.bos_token_id
            batch[field] = encoded
    
    # 将label移到device上
    if 'label' in batch:
        batch['label'] = data_to_device(batch['label'], device)
    
    # 组装source和target
    source = []
    target = []
    
    for src in data_loader.dataset.sources:
        if src == 'image':
            source.append(batch['image'])
        elif src in ['history', 'findings', 'impression']:
            source.append(batch[src])
            
    for tgt in data_loader.dataset.targets:
        if tgt == 'label':
            target.append(batch['label'])
        elif tgt in ['findings', 'impression']:
            target.append(batch[tgt]) 
            
    return source, target, None

# ------ Core Functions ------
def train(data_loader, model, optimizer, criterion, num_epochs, current_epoch, scheduler=None, train_stage=2, device='cpu', kw_src=None, kw_tgt=None, kw_out=None, scaler=None):
    model.train()
    running_loss = 0

    # 计算当前训练进度
    total_steps = len(data_loader) * num_epochs  # 总步数
    
    prog_bar = tqdm(data_loader)
    for i, batch in enumerate(prog_bar):
        # 准备批次数据
        source, target, _ = prepare_batch_data(batch, data_loader, device)
        
        # 转换为kwargs格式
        source = args_to_kwargs(source, kw_src)
        target = args_to_kwargs(target, kw_tgt)
        
        source['train_stage'] = train_stage
		
        source['idx'] = batch['idx']
        source['mode'] = 'train'
		
        scheduler.step(cur_epoch=current_epoch, cur_step=i)
        current_lr = optimizer.param_groups[0]['lr']

        # 剩余的训练逻辑保持不变
        if scaler != None:
            with torch.cuda.amp.autocast():
                output = data_distributor(model, source)
                output = args_to_kwargs(output, kw_out)
                if train_stage == 1:
                    loss = output['loss_lm']
                else:
                    loss, _ = criterion(output, target)
                # loss, _ = criterion(output, target)
				
                
            running_loss += loss.item()
            prog_bar.set_description(f'Loss: {running_loss/(i+1)} | LR: {current_lr}')
			
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

            # optimizer.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
        else:
            output = data_distributor(model, source)
            output = args_to_kwargs(output, kw_out)
            loss = criterion(output, target)

            running_loss += loss.item()
            prog_bar.set_description('Loss: {}'.format(running_loss/(i+1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()

    return running_loss / len(data_loader)

def test(data_loader, model, logger, mode='val', metric_ftns=None, train_stage=2, criterion=None, device='cpu', return_results=True, kw_src=None, kw_tgt=None, kw_out=None, select_outputs=[]):
    model.eval()
    running_loss = 0

    outputs = []
    targets = []

    findings_gts_list = []
    findings_preds_list = []
    impression_gts_list = []
    impression_preds_list = []
    report_gts_list = []
    report_preds_list = []

    with torch.no_grad():
        prog_bar = tqdm(data_loader)
        for i, batch in enumerate(prog_bar):
            # 收集ground truth
            findings_gts_list.extend([gt for gt in batch['gts'][0]])
            impression_gts_list.extend([gt for gt in batch['gts'][1]])

            # 准备批次数据
            source, target, _ = prepare_batch_data(batch, data_loader, device)
            
            # 转换为kwargs格式
            source = args_to_kwargs(source, kw_src)
            target = args_to_kwargs(target, kw_tgt)
            
            source['train_stage'] = train_stage
            source['idx'] = batch['idx']
            source['mode'] = mode
            # 模型推理
            output = data_distributor(model, source)
            output = args_to_kwargs(output, kw_out)

            # 收集预测结果
            findings_preds_list.extend([re for re in output['findings_text']])
            if train_stage == 2:
                impression_preds_list.extend([re for re in output['impression_text']])
            
            # 记录日志
            logger.info(f"findings_preds: {findings_preds_list[0]}")

            # 计算损失
            if criterion is not None:
                # loss, detailed_loss = criterion(output, target)
                loss = torch.tensor(0.0)
                running_loss += loss.item()
            prog_bar.set_description('Loss: {}'.format(running_loss/(i+1)))

        # 计算评估指标
        findings_met = metric_ftns(
            {i: [gt] for i, gt in enumerate(findings_gts_list)},
            {i: [re] for i, re in enumerate(findings_preds_list)}
        )

        # 如果需要返回结果
        if return_results:
            results = {
                'findings_gts': findings_gts_list,
                'findings_preds': findings_preds_list,
                'impression_gts': impression_gts_list,
                'impression_preds': impression_preds_list,
                'report_gts': report_gts_list,
                'report_preds': report_preds_list
            }
            return running_loss / len(data_loader), findings_met, results

    return running_loss / len(data_loader), findings_met


def save(path, model, optimizer=None, scheduler=None, epoch=-1, stats=None):
	torch.save({
		# --- Model Statistics ---
		'epoch': epoch,
		'stats': stats,
		# --- Model Parameters ---
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict() if optimizer != None else None,
		# 'scheduler_state_dict': scheduler.state_dict() if scheduler != None else None,
	}, path)

def load(path, model, optimizer=None, scheduler=None):
	checkpoint = torch.load(path)
	# --- Model Statistics ---
	epoch = checkpoint['epoch']
	stats = checkpoint['stats']
	# --- Model Parameters ---
	model.load_state_dict(checkpoint['model_state_dict'])
	if optimizer != None:
		try:
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		except: # Input optimizer doesn't fit the checkpoint one --> should be ignored
			print('Cannot load the optimizer')
	if scheduler != None:
		try:
			scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		except: # Input scheduler doesn't fit the checkpoint one --> should be ignored
			print('Cannot load the scheduler')
	return epoch, stats

def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def visual_parameters(modules, parameters):
	# 数据
	total_parameters = sum(parameters)

	# 计算占比
	percentages = [param / total_parameters * 100 for param in parameters]

	# 绘制饼状图
	plt.figure(figsize=(8, 8))
	plt.pie(percentages, labels=modules, autopct='%1.1f%%', startangle=140)
	plt.title("Parameter Distribution Among Modules")
	plt.savefig("/home/chenlb/xray_report_generation/results/parameter_distribution.png")


# 绘制直方图
def plot_length_distribution(distribution, title):
	# 准备数据
	lengths = list(distribution.keys())  # Token 长度
	counts = list(distribution.values())  # 出现次数

	# 绘制直方图
	plt.figure(figsize=(10, 6))
	plt.bar(lengths, counts)
	plt.title(title)
	plt.xlabel("Token Length")
	plt.ylabel("Count")
	
	# 计算合适的刻度间隔
	max_length = max(lengths)
	min_length = min(lengths)
	step = max(1, (max_length - min_length) // 5)  # 最多显示约10个刻度
	
	# 设置x轴刻度
	plt.xticks(range(min_length, max_length + 1, step))
	
	plt.tight_layout()
	plt.savefig(f"/home/chenlb/xray_report_generation/results/{title.replace(' ', '_')}.png")

def setup_logger(log_dir='logs'):
    """
    设置logger，同时输出到控制台和文件
    
    Args:
        log_dir: 日志文件存储目录
    Returns:
        logger: 配置好的logger对象
    """
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成日志文件名（使用当前时间）
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{current_time}.log')
    
    # 创建logger对象
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger