import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
	else:
		raise TypeError('Unsupported Datatype! Must be a Tensor/List/Tuple/Dict.')
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

# ------ Core Functions ------
def train(data_loader, model, optimizer, criterion, train_stage=2, scheduler=None, device='cpu', kw_src=None, kw_tgt=None, kw_out=None, scaler=None):
	model.train()
	running_loss = 0
 
	prog_bar = tqdm(data_loader)
	for i, (source, target, idx, gts) in enumerate(prog_bar):
		source = data_to_device(source, device)
		target = data_to_device(target, device)

		source = args_to_kwargs(source, kw_src)
		target = args_to_kwargs(target, kw_tgt)

		source['train_stage'] = train_stage
		source['idx'] = idx

		if scaler != None:
			with torch.cuda.amp.autocast():
				output = data_distributor(model, source)
				output = args_to_kwargs(output, kw_out)
				loss, detailed_loss = criterion(output, target)
				
			running_loss += loss.item()
			prog_bar.set_description('Loss: {}'.format(running_loss/(i+1)))

			if torch.isnan(loss) or loss.item() > 1000:
				print("error!")
				loss = criterion(output, target)

			# Back-propagate and update weights
			optimizer.zero_grad()
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			if scheduler != None:
				scheduler.step()
		else:
			output = data_distributor(model, source)
			output = args_to_kwargs(output, kw_out)
			loss = criterion(output, target)

			if torch.isnan(loss) or loss.item() > 1000:
				print("error!")
				loss = criterion(output, target)

			running_loss += loss.item()
			prog_bar.set_description('Loss: {}'.format(running_loss/(i+1)))

			# Back-propagate and update weights
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if scheduler != None:
				scheduler.step()

	return running_loss / len(data_loader)

def test(data_loader, model, mode='val', metric_ftns=None, train_stage=2, criterion=None, device='cpu', return_results=True, kw_src=None, kw_tgt=None, kw_out=None, select_outputs=[]):
	model.eval()
	running_loss = 0

	outputs = []
	targets = []

	with torch.no_grad():
		prog_bar = tqdm(data_loader)
		findings_gts_list = []
		findings_preds_list = []
		impression_gts_list = []
		impression_preds_list = []
		report_gts_list = []
		report_preds_list = []

		for i, (source, target, idx, gts) in enumerate(prog_bar):
			findings_gts_list.extend([gt for gt in gts[0]])
			impression_gts_list.extend([gt for gt in gts[1]])

			source = data_to_device(source, device)
			target = data_to_device(target, device)

			source = args_to_kwargs(source, kw_src)
			target = args_to_kwargs(target, kw_tgt)

			source['train_stage'] = train_stage
			source['idx'] = idx

			output = data_distributor(model, source)
			output = args_to_kwargs(output, kw_out)

			findings_preds_list.extend([re for re in output['findings_text']])
			if train_stage == 2:
				impression_preds_list.extend([re for re in output['impression_text']])

			if criterion != None:
				loss, detailed_loss = criterion(output, target)
				running_loss += loss.item()
			prog_bar.set_description('Loss: {}'.format(running_loss/(i+1)))


		findings_met = metric_ftns({i: [gt] for i, gt in enumerate(findings_gts_list)},
							{i: [re] for i, re in enumerate(findings_preds_list)})
		
		for k, v in findings_met.items():
			print(f'{mode}_{k}: {v}')

	return running_loss / len(data_loader)


def save(path, model, optimizer=None, scheduler=None, epoch=-1, stats=None):
	torch.save({
		# --- Model Statistics ---
		'epoch': epoch,
		'stats': stats,
		# --- Model Parameters ---
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict() if optimizer != None else None,
		'scheduler_state_dict': scheduler.state_dict() if scheduler != None else None,
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