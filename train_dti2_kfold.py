#标签平滑BCE计算
import argparse
import pickle
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score
try:
    from torch_ema import ExponentialMovingAverage
    HAS_EMA = True
except ImportError:
    HAS_EMA = False
    print("Warning: torch_ema not installed, EMA will be disabled")

# 导入自定义模块
# 确保当前目录在 sys.path 中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 添加 UnseenDDIs 到路径以导入 utils
unseen_ddi_path = os.path.join(current_dir, 'Classification', 'UnseenDDIs')
if os.path.exists(unseen_ddi_path):
    if unseen_ddi_path not in sys.path:
        sys.path.insert(0, unseen_ddi_path)
    print(f"Added {unseen_ddi_path} to sys.path")
else:
    print(f"Warning: {unseen_ddi_path} does not exist!")

from interaction_model import MultiPerspectiveDTI as DTIModel
from dataset_loader import get_dataloader
try:
    from utils import Mol_Tokenizer
except ImportError:
    # Fallback: try importing from Classification.UnseenDDIs.utils if package structure allows
    try:
        from Classification.UnseenDDIs.utils import Mol_Tokenizer
    except ImportError:
        print("Error: Could not import Mol_Tokenizer from utils. Please check path.")
        raise

import torch.nn.functional as F
import random


def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred_label = (y_pred_prob > threshold).astype(int)
    return {
        'acc': accuracy_score(y_true, y_pred_label),
        'auc': roc_auc_score(y_true, y_pred_prob),
        'auprc': average_precision_score(y_true, y_pred_prob),
        'f1': f1_score(y_true, y_pred_label),
        'precision': precision_score(y_true, y_pred_label, zero_division=0),
        'recall': recall_score(y_true, y_pred_label)
    }

def find_best_threshold(y_true, y_pred_prob, min_thresh=0.1, max_thresh=0.9, step=0.02):
    best_threshold = 0.5
    best_f1 = -1
    thresholds = np.arange(min_thresh, max_thresh + step/2, step)
    for t in thresholds:
        t_clipped = min(max(t, 0.0), 1.0)
        y_pred = (y_pred_prob > t_clipped).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t_clipped
    return best_threshold, best_f1

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=0, ema=None, grad_accum_steps=1):
    """
    Args:
        epoch: 当前训练轮数（从0开始）
        ema: EMA对象（可选）
        grad_accum_steps: 梯度累加步数（用于显存优化）
    """
    model.train()
    total_loss = 0
    
    all_labels = []
    all_probs = []
    
    # ========== 门控值累积变量 ==========
    geo_gate_values = []
    protein_gate_values = []
    # ========== 第一优先级：记录Gate分布（均值和方差）==========
    # 监控先行：只有看到数据从0.58逐步滑向0.95，才能坐实"后期坍塌"的猜想
    # 注意：Late Fusion 模式下 protein_gate 为 None（无残基级门控），仅监控 geo_gate
    geo_gate_means = []      # 每个batch的gate均值
    geo_gate_stds = []       # 每个batch的gate标准差
    protein_gate_means = []  # 每个batch的protein gate均值 (Late Fusion 时为空)
    protein_gate_stds = []   # 每个batch的protein gate标准差 (Late Fusion 时为空)
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    # ⚠️ 必须把 optimizer.zero_grad() 移到循环外面初始化
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # 移动数据到设备
        # Atom Inputs
        atom_feat = batch['atom_feat'].to(device)
        atom_adj = batch['atom_adj'].to(device)
        atom_dist = batch['atom_dist'].to(device)
        atom_match = batch['atom_match'].to(device)
        sum_atoms = batch['sum_atoms'].to(device)
        
        # Motif Inputs
        motif_seq = batch['motif_seq'].to(device)
        motif_adj = batch['motif_adj'].to(device)
        motif_dist = batch['motif_dist'].to(device)
        
        # ⚠️ 已移除：training-time token masking
        # 原因：会破坏 mask 对齐（motif_seq被修改但drug_mask已提前生成）
        # 导致 semantic mismatch：同一个位置在不同模块中身份不同
        # 当前已有足够的正则化：dropout, modality dropout, stochastic residual, EMA, weight_decay
        
        # Protein Inputs
        protein_feat = batch['protein_feat'].to(device)
        protein_mask = batch['protein_mask'].to(device)
        
        # 蛋白质3D特征（总是存在，但由has_valid_protein_3d指示有效性）
        contact_prob = batch.get('contact_prob', None)
        distance_matrix = batch.get('distance_matrix', None)
        has_valid_protein_3d = batch.get('has_valid_protein_3d', None)
        if contact_prob is not None:
            contact_prob = contact_prob.to(device)
        if distance_matrix is not None:
            distance_matrix = distance_matrix.to(device)
        if has_valid_protein_3d is not None:
            has_valid_protein_3d = has_valid_protein_3d.to(device)
        
        # 药物3D特征（总是存在，但由has_valid_drug_3d指示有效性）
        drug_contact_prob = batch.get('drug_contact_prob', None)
        drug_distance = batch.get('drug_distance', None)
        has_valid_drug_3d = batch.get('has_valid_drug_3d', None)
        if drug_contact_prob is not None:
            drug_contact_prob = drug_contact_prob.to(device)
        if drug_distance is not None:
            drug_distance = drug_distance.to(device)
        if has_valid_drug_3d is not None:
            has_valid_drug_3d = has_valid_drug_3d.to(device)
        
        labels = batch['label'].to(device)
        
        # 前向传播
        # 完美打通3D拓扑控制流：显式传入原子原始特征和匹配矩阵
        outputs, drug_global, prot_global, drug_final, protein_final, geo_gate, protein_gate = model(
            atom_feat, atom_adj, atom_dist, atom_match, sum_atoms,
            motif_seq, motif_adj, motif_dist,
            protein_feat, protein_mask,
            contact_prob=contact_prob,
            distance_matrix=distance_matrix,
            drug_contact_prob=drug_contact_prob,
            drug_distance=drug_distance,
            has_valid_protein_3d=has_valid_protein_3d,      # 新增：样本级蛋白质3D有效性标志
            has_valid_drug_3d=has_valid_drug_3d,            # 新增：样本级药物3D有效性标志
            atom_mask=batch['atom_mask'].to(device),         # 传入原子mask
            motif_mask=batch['drug_mask'].to(device),        # 传入drug mask（与模型forward参数名一致）
            drug_atom_raw=atom_feat,                         # 传入原始原子化学特征
            atom_match_matrix=atom_match                     # 传入原子到Motif拓扑归属矩阵
        )
        
        # ========== 🚀 修复 2：严格阻止张量广播，确保精准的一对一梯度 ==========
        # 将 outputs 和 labels 都降维到 (Batch,)，确保维度完全匹配
        loss = criterion(outputs.squeeze(-1), labels.float().squeeze(-1))
        
        # ========== 门控值监控：累积整个 epoch 的门控值 ==========
        # 第一优先级：记录Gate分布（均值和方差随Epoch变化）
        # geo_gate: (B, common_dim), protein_gate: (B, Np, 1) 或 None (Late Fusion)
        geo_gate_values.append(geo_gate.mean().item())
        if protein_gate is not None:
            protein_gate_values.append(protein_gate.mean().item())
            protein_gate_means.append(protein_gate.mean().item())
            protein_gate_stds.append(protein_gate.std().item())
        # 记录每个batch的均值和标准差，用于绘制gate分布轨迹
        geo_gate_means.append(geo_gate.mean().item())
        geo_gate_stds.append(geo_gate.std().item())
        
        # 2. 损失归一化：因为累加了步数，为了保持真实 batch size 的梯度尺度，需除以 accum_steps
        scaled_loss = loss / grad_accum_steps
        
        # 3. 梯度反向传播（累加在叶子节点上）
        scaled_loss.backward()
        
        # 4. 判断是否达到了累加步数，或者是否是最后一个 batch
        if ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(dataloader)):
            # 梯度裁剪（必须在 step 前执行）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新权重
            optimizer.step()
            
            # 清空累计的梯度，准备下一轮
            optimizer.zero_grad()
            
            # 更新 EMA
            if ema is not None:
                ema.update()
        
        # 记录真实的 loss 标量
        total_loss += loss.item()
        
        # 收集结果用于计算指标
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        lbls = labels.detach().cpu().numpy()
        
        all_probs.extend(probs)
        all_labels.extend(lbls)
        
        progress_bar.set_postfix({'loss': loss.item()})
        
    # 计算所有指标
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    
    # 训练阶段不寻找最佳阈值，直接用 0.5
    metrics = calculate_metrics(y_true, y_prob, threshold=0.5)
    metrics['loss'] = total_loss / len(dataloader)
    
    # ========== 打印门控值统计（整个 epoch 的平均值） ==========
    geo_gate_epoch_mean = np.mean(geo_gate_values)
    protein_gate_epoch_mean = np.mean(protein_gate_values) if protein_gate_values else 0.0
    # 第一优先级：计算epoch级别的gate均值和方差
    geo_gate_epoch_std = np.mean(geo_gate_stds)   # batch内方差的平均
    protein_gate_epoch_std = np.mean(protein_gate_stds) if protein_gate_stds else 0.0

    print(f"[门控监控] geo_gate: mean={geo_gate_epoch_mean:.4f}, std={geo_gate_epoch_std:.4f} | "
          f"protein_gate: {'N/A (Late Fusion)' if not protein_gate_values else f'mean={protein_gate_epoch_mean:.4f}, std={protein_gate_epoch_std:.4f}'}")
    print(f"[门控状态] geo_gate: {'饱和(>0.95)' if geo_gate_epoch_mean > 0.95 else '饱和(<0.05)' if geo_gate_epoch_mean < 0.05 else '正常'}, "
          f"protein_gate: {'Late Fusion (无残基级门控)' if not protein_gate_values else ('饱和(>0.95)' if protein_gate_epoch_mean > 0.95 else '饱和(<0.05)' if protein_gate_epoch_mean < 0.05 else '正常')}")

    # 将gate统计写入metrics，供外部记录轨迹
    metrics['geo_gate_mean'] = geo_gate_epoch_mean
    metrics['geo_gate_std'] = geo_gate_epoch_std
    metrics['protein_gate_mean'] = protein_gate_epoch_mean
    metrics['protein_gate_std'] = protein_gate_epoch_std
    
    return metrics

def evaluate(model, dataloader, criterion, device, desc="Evaluating", ema=None, threshold=None):
    """
    Args:
        ema: EMA对象（可选）
        threshold: 分类阈值（None表示在验证集上搜索最佳阈值）
    """
    model.eval()
    total_loss = 0
    
    all_labels = []
    all_probs = []
    
    # ========== 第一优先级：验证集也记录Gate分布 ==========
    val_geo_gate_values = []
    val_protein_gate_values = []
    
    # 使用EMA权重进行评估（如果启用），同时禁用梯度
    if ema is not None:
        context_manager = ema.average_parameters()
    else:
        context_manager = contextlib.nullcontext()
    
    with context_manager:
        with torch.no_grad():  # 关键修复：EMA 上下文中也禁用梯度，防止验证集吃显存
            for batch in tqdm(dataloader, desc=desc, leave=False):
                # Atom Inputs
                atom_feat = batch['atom_feat'].to(device)
                atom_adj = batch['atom_adj'].to(device)
                atom_dist = batch['atom_dist'].to(device)
                atom_match = batch['atom_match'].to(device)
                sum_atoms = batch['sum_atoms'].to(device)
                
                # Motif Inputs
                motif_seq = batch['motif_seq'].to(device)
                motif_adj = batch['motif_adj'].to(device)
                motif_dist = batch['motif_dist'].to(device)
                
                # Protein Inputs
                protein_feat = batch['protein_feat'].to(device)
                protein_mask = batch['protein_mask'].to(device)
                
                # 蛋白质3D特征（总是存在，但由has_valid_protein_3d指示有效性）
                contact_prob = batch.get('contact_prob', None)
                distance_matrix = batch.get('distance_matrix', None)
                has_valid_protein_3d = batch.get('has_valid_protein_3d', None)
                if contact_prob is not None:
                    contact_prob = contact_prob.to(device)
                if distance_matrix is not None:
                    distance_matrix = distance_matrix.to(device)
                if has_valid_protein_3d is not None:
                    has_valid_protein_3d = has_valid_protein_3d.to(device)
                
                # 药物3D特征（总是存在，但由has_valid_drug_3d指示有效性）
                drug_contact_prob = batch.get('drug_contact_prob', None)
                drug_distance = batch.get('drug_distance', None)
                has_valid_drug_3d = batch.get('has_valid_drug_3d', None)
                if drug_contact_prob is not None:
                    drug_contact_prob = drug_contact_prob.to(device)
                if drug_distance is not None:
                    drug_distance = drug_distance.to(device)
                if has_valid_drug_3d is not None:
                    has_valid_drug_3d = has_valid_drug_3d.to(device)
                
                labels = batch['label'].to(device)
                
                # 验证/测试时同样传入完整的拓扑控制流参数
                outputs, _, _, _, _, val_geo_gate, val_protein_gate = model(
                    atom_feat, atom_adj, atom_dist, atom_match, sum_atoms,
                    motif_seq, motif_adj, motif_dist,
                    protein_feat, protein_mask,
                    contact_prob=contact_prob,
                    distance_matrix=distance_matrix,
                    drug_contact_prob=drug_contact_prob,
                    drug_distance=drug_distance,
                    has_valid_protein_3d=has_valid_protein_3d,      # 新增：样本级蛋白质3D有效性标志
                    has_valid_drug_3d=has_valid_drug_3d,            # 新增：样本级药物3D有效性标志
                    atom_mask=batch['atom_mask'].to(device),         # 传入原子mask
                    motif_mask=batch['drug_mask'].to(device),        # 传入drug mask（与模型forward参数名一致）
                    drug_atom_raw=atom_feat,                         # 传入原始原子化学特征
                    atom_match_matrix=atom_match                     # 传入原子到Motif拓扑归属矩阵
                )
                
                # ========== 第一优先级：记录验证集Gate分布 ==========
                val_geo_gate_values.append(val_geo_gate.mean().item())
                if val_protein_gate is not None:
                    val_protein_gate_values.append(val_protein_gate.mean().item())
                
                # ========== 🚀 修复 2：严格阻止张量广播，确保精准的一对一梯度 ==========
                # 将 outputs 和 labels 都降维到 (Batch,)，确保维度完全匹配
                loss = criterion(outputs.squeeze(-1), labels.float().squeeze(-1))
                
                total_loss += loss.item()
                
                # 收集结果（使用 detach() 避免梯度问题）
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                lbls = labels.detach().cpu().numpy()
                
                all_probs.extend(probs)
                all_labels.extend(lbls)
            
    # 计算所有指标
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    
    if threshold is None:
        best_threshold, _ = find_best_threshold(y_true, y_prob)
    else:
        best_threshold = threshold
    
    metrics = calculate_metrics(y_true, y_prob, threshold=best_threshold)
    metrics['loss'] = total_loss / len(dataloader)
    metrics['best_threshold'] = best_threshold
    metrics['y_prob'] = y_prob
    metrics['y_true'] = y_true
    # ========== 第一优先级：验证集Gate分布统计 ==========
    metrics['geo_gate_mean'] = np.mean(val_geo_gate_values) if val_geo_gate_values else 0.0
    metrics['protein_gate_mean'] = np.mean(val_protein_gate_values) if val_protein_gate_values else 0.0
    
    return metrics

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, mode='max'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        
        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')

    def __call__(self, score, model, save_path, ema=None):
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.delta)
        else:
            improved = score > (self.best_score + self.delta)

        if improved:
            self.save_checkpoint(score, model, save_path, ema)
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, score, model, save_path, ema=None):
        if self.verbose:
            print(f'Validation metric improved ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        
        # 如果有EMA，保存EMA的shadow weights而不是raw weights
        if ema is not None:
            # 使用EMA的average_parameters上下文管理器获取平滑后的权重
            with ema.average_parameters():
                torch.save(model.state_dict(), save_path)
        else:
            torch.save(model.state_dict(), save_path)

def get_free_gpu():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    try:
        import subprocess
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_free = [int(x) for x in result.strip().split('\n')]
        
        gpu_id = np.argmax(memory_free)
        print(f"Auto-selecting GPU {gpu_id} with {memory_free[gpu_id]} MiB free memory.")
        return torch.device(f"cuda:{gpu_id}")
    except Exception as e:
        print(f"Error querying nvidia-smi: {e}. Defaulting to cuda:0")
        return torch.device("cuda:0")

class LabelSmoothingBCE(nn.Module):
    def __init__(self, epsilon=0.05, reduction='mean'):
        super(LabelSmoothingBCE, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets_smooth = targets * (1.0 - 2.0 * self.epsilon) + self.epsilon
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth, reduction=self.reduction)
        return bce_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

import matplotlib.pyplot as plt

def plot_metrics(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    
    # Plot AUC
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_auc'], label='Train AUC')
    plt.plot(epochs, history['val_auc'], label='Val AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('AUC')
    plt.legend()
    
    # Plot AUPRC
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_auprc'], label='Train AUPRC')
    plt.plot(epochs, history['val_auprc'], label='Val AUPRC')
    plt.xlabel('Epochs')
    plt.ylabel('AUPRC')
    plt.title('AUPRC')
    plt.legend()
    
    # Plot F1
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_f1'], label='Train F1')
    plt.plot(epochs, history['val_f1'], label='Val F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def run_fold(fold, args, device, HYPERPARAMS, drug_atom_dict, drug_motif_dict, protein_dict, motif_vocab_size, protein_3d_dict=None, drug_3d_dict=None):
    print(f"\n{'='*20} Running Fold {fold} {'='*20}")
    
    # 数据集路径
    dataset_dir = os.path.join(current_dir, 'dataset', args.dataset, args.split_type, f'fold_{fold}')
    train_csv = os.path.join(dataset_dir, 'train.csv')
    val_csv = os.path.join(dataset_dir, 'val.csv')
    test_csv = os.path.join(dataset_dir, 'test.csv')
    
    # 检查文件是否存在
    files_to_check = [train_csv, val_csv, test_csv]
    for f in files_to_check:
        if not os.path.exists(f):
            print(f"Error: File not found: {f}")
            return None
            
    # 1. 数据加载
    print("Initializing DataLoaders...")
    
    # 为每个 fold 设置不同的随机种子，确保各 fold 的随机增强不同
    # 使用 base_seed + fold 确保可复现性
    base_seed = 42
    fold_seed = base_seed + fold * 1000
    
    train_loader = get_dataloader(train_csv, drug_atom_dict, drug_motif_dict, protein_dict, 
                                  batch_size=HYPERPARAMS['batch_size'], shuffle=True, 
                                  protein_3d_dict=protein_3d_dict, drug_3d_dict=drug_3d_dict,
                                  seed=fold_seed)
    val_loader = get_dataloader(val_csv, drug_atom_dict, drug_motif_dict, protein_dict, 
                                batch_size=HYPERPARAMS['batch_size'], shuffle=False, 
                                protein_3d_dict=protein_3d_dict, drug_3d_dict=drug_3d_dict,
                                seed=fold_seed + 100)
    test_loader = get_dataloader(test_csv, drug_atom_dict, drug_motif_dict, protein_dict, 
                                 batch_size=HYPERPARAMS['batch_size'], shuffle=False, 
                                 protein_3d_dict=protein_3d_dict, drug_3d_dict=drug_3d_dict,
                                 seed=fold_seed + 200)
    
    print(f"Dataset Sizes: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    # 2. 模型初始化
    print("Initializing MultiPerspectiveDTI Model...")
    model = DTIModel(
        motif_vocab_size=motif_vocab_size,
        atom_dim=HYPERPARAMS['atom_dim'],
        motif_dim=HYPERPARAMS['motif_dim'],
        drug_num_heads=HYPERPARAMS['num_heads'],
        protein_esm_dim=HYPERPARAMS['protein_dim'],
        protein_hidden_dim=HYPERPARAMS['common_dim'],
        protein_out_dim=HYPERPARAMS['common_dim'],
        protein_gat_layers=HYPERPARAMS['num_layers'],
        protein_num_heads=4,
        contact_threshold=0.5,
        use_rbf=True,
        common_dim=HYPERPARAMS['common_dim'],
        num_interaction_blocks=HYPERPARAMS['num_layers'],
        interaction_heads=HYPERPARAMS['num_heads'],
        dropout=HYPERPARAMS['dropout'],
        split_type=args.split_type,
        warm_identity_drop=True
    ).to(device)
    
    # ========== 🚀 EMA decay 差异化配置 ==========
    # BiosNap：数据规模小，EMA decay 降低，避免验证时用半个 epoch 前的模型
    # BindingDB：训练步数多，EMA decay 高，长期平均更稳定
    is_biosnap = args.dataset.lower() == 'biosnap'
    ema_decay = 0.992 if is_biosnap else 0.999
    
    ema = None
    if HAS_EMA:
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        print(f"EMA enabled with decay={ema_decay}")
    else:
        print("EMA disabled (torch_ema not installed)")
    
    # ========== 🚀 Label Smoothing 差异化配置 ==========
    # BioSNAP：数据更干净，label noise 少，降低 epsilon 避免压低 AUPRC 上限
    # BindingDB：数据噪声大，保持较大 epsilon 防止过拟合
    label_smoothing_epsilon = 0.01 if is_biosnap else 0.05
    
    print(f"启用标签平滑 (epsilon={label_smoothing_epsilon}, 标签范围 {label_smoothing_epsilon}-{1-label_smoothing_epsilon})")
    criterion = LabelSmoothingBCE(epsilon=label_smoothing_epsilon)
    
    def create_optimizer_param_groups(model, lr, weight_decay):
        """
        AdamW 参数分组 + 非对称学习率：
        - decay on Linear weights
        - no_decay on norm/bias
        - protein_seq_proj 使用更低的学习率 (0.2×base_lr)
        
        防止正则误伤 LayerNorm 和 bias，保护模型表达能力
        非对称学习率：让ESM不要学太快，给Graph模块成长空间
        """
        # 基础参数组
        decay_params = []
        no_decay_params = []
        
        # protein_seq_proj 专用参数组（使用更低学习率）
        protein_seq_decay = []
        protein_seq_no_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'protein_seq_proj' in name:
                # protein_seq_proj 使用更低学习率
                if 'bias' in name or 'norm' in name or 'LayerNorm' in name:
                    protein_seq_no_decay.append(param)
                else:
                    protein_seq_decay.append(param)
            else:
                # 其他参数使用默认学习率
                if 'bias' in name or 'norm' in name or 'LayerNorm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        # 计算 protein_seq_proj 的低学习率
        protein_seq_lr = lr * 0.2  # 0.2× base_lr，让ESM不要学太快
        
        return [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay_params, 'weight_decay': 0.0, 'lr': lr},
            {'params': protein_seq_decay, 'weight_decay': weight_decay, 'lr': protein_seq_lr},
            {'params': protein_seq_no_decay, 'weight_decay': 0.0, 'lr': protein_seq_lr}
        ]
    
    optimizer = optim.AdamW(create_optimizer_param_groups(model, HYPERPARAMS['learning_rate'], HYPERPARAMS['weight_decay']))
    
    # ========== 🚀 Warmup + CosineAnnealing 组合调度器 ==========
    # 配合去掉 detach 后的梯度打通，前期用 Warmup 稳定梯度流
    # 避免训练初期梯度过大冲乱深层 Encoder
    warmup_epochs = 5  # 前5个epoch线性warmup
    cosine_epochs = HYPERPARAMS['epochs'] - warmup_epochs  # 剩余epoch做余弦退火
    
    # Warmup: 从 0 线性增长到初始学习率
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,  # 起始学习率 = lr * 0.01
        end_factor=1.0,     # 终止学习率 = lr
        total_iters=warmup_epochs
    )
    
    # Cosine: 平滑下降到 eta_min
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=1e-6
    )
    
    # 组合：先 Warmup，后 Cosine
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    print(f"Warmup+Cosine scheduler enabled: warmup_epochs={warmup_epochs}, cosine_epochs={cosine_epochs}, eta_min=1e-6")
    
    # 3. 训练循环
    # 注意：路径必须包含 dataset 名，否则不同数据集训练会互相覆盖 best_model.pth
    save_dir = os.path.join(current_dir, 'checkpoints', args.dataset, args.split_type, f'fold_{fold}')
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    # 初始化早停 (使用综合分数作为监控指标，模式为 max)
    early_stopping = EarlyStopping(patience=HYPERPARAMS['patience'], verbose=True, mode='max')
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc': [], 'val_auc': [],
        'train_auprc': [], 'val_auprc': [],
        'train_f1': [], 'val_f1': [],
        # ========== 第一优先级：记录Gate分布轨迹 ==========
        'geo_gate_mean': [], 'geo_gate_std': [],
        'protein_gate_mean': [], 'protein_gate_std': []
    }
    
    for epoch in range(HYPERPARAMS['epochs']):
        
        # 训练一个epoch（添加数据增强和梯度累加）
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch=epoch,
            ema=ema,
            grad_accum_steps=HYPERPARAMS['grad_accum_steps']
        )
        # 验证集评估（使用EMA权重进行验证）
        # 使用EMA权重进行验证，避免raw weights高频振荡导致的虚假提升
        val_metrics = evaluate(model, val_loader, criterion, device, desc="Validation", ema=ema)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{HYPERPARAMS['epochs']} | Fold {fold} | LR: {current_lr:.2e}")
        print(f"  Train | Loss: {train_metrics['loss']:.4f} | AUC: {train_metrics['auc']:.4f} | AUPRC: {train_metrics['auprc']:.4f} | F1: {train_metrics['f1']:.4f} | Acc: {train_metrics['acc']:.4f}")
        print(f"  Val   | Loss: {val_metrics['loss']:.4f}   | AUC: {val_metrics['auc']:.4f}   | AUPRC: {val_metrics['auprc']:.4f}   | F1: {val_metrics['f1']:.4f}   | Acc: {val_metrics['acc']:.4f}")
        print(f"          Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | Best Thr: {val_metrics['best_threshold']:.2f}")
        # ========== 第一优先级：打印验证集Gate分布（对比训练集）==========
        print(f"  Gate  | Train: geo={train_metrics.get('geo_gate_mean',0):.4f}±{train_metrics.get('geo_gate_std',0):.4f}, "
              f"prot={train_metrics.get('protein_gate_mean',0):.4f}±{train_metrics.get('protein_gate_std',0):.4f}")
        print(f"        | Val  : geo={val_metrics.get('geo_gate_mean',0):.4f}, prot={val_metrics.get('protein_gate_mean',0):.4f}")
        
        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_auc'].append(train_metrics['auc'])
        history['val_auc'].append(val_metrics['auc'])
        history['train_auprc'].append(train_metrics['auprc'])
        history['val_auprc'].append(val_metrics['auprc'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        # ========== 第一优先级：记录Gate分布轨迹 ==========
        history['geo_gate_mean'].append(train_metrics.get('geo_gate_mean', 0.0))
        history['geo_gate_std'].append(train_metrics.get('geo_gate_std', 0.0))
        history['protein_gate_mean'].append(train_metrics.get('protein_gate_mean', 0.0))
        history['protein_gate_std'].append(train_metrics.get('protein_gate_std', 0.0))
        # 验证集gate轨迹（用于对比训练集，检测是否过拟合导致的gate坍塌）
        history.setdefault('val_geo_gate_mean', []).append(val_metrics.get('geo_gate_mean', 0.0))
        history.setdefault('val_protein_gate_mean', []).append(val_metrics.get('protein_gate_mean', 0.0))
        
        # 早停检查 (基于AUPRC，使用EMA权重保存)
        early_stopping(val_metrics['auprc'], model, best_model_path, ema=ema)
        
        # 更新学习率调度器
        scheduler.step()
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    print(f"\nTraining Finished for Fold {fold}.")
    print(f"Best Val AUPRC: {early_stopping.best_score:.4f}")
    
    # 绘制曲线
    plot_metrics(history, os.path.join(save_dir, 'training_curves.png'))
    
    # 4. 测试集评估
    print(f"\nEvaluating Fold {fold} on Test Set...")
    
    # 加载模型权重：处理新增的层和维度不匹配
    checkpoint = torch.load(best_model_path)
    
    # 获取模型当前的state_dict
    model_state_dict = model.state_dict()
    
    # 创建一个过滤后的checkpoint，只保留形状匹配的参数
    filtered_checkpoint = {}
    size_mismatch_keys = []
    
    for key in checkpoint.keys():
        if key in model_state_dict:
            if checkpoint[key].shape == model_state_dict[key].shape:
                filtered_checkpoint[key] = checkpoint[key]
            else:
                size_mismatch_keys.append(key)
    
    # 检查checkpoint中缺少哪些模型参数
    missing_keys = [key for key in model_state_dict.keys() if key not in checkpoint]
    
    # 检查checkpoint中有哪些参数在模型中不存在
    unexpected_keys = [key for key in checkpoint.keys() if key not in model_state_dict]
    
    # 输出警告信息
    if size_mismatch_keys:
        print(f"Warning: {len(size_mismatch_keys)} keys have size mismatch. "
              f"First 10: {size_mismatch_keys[:10]}...")
    if missing_keys:
        print(f"Warning: Checkpoint missing {len(missing_keys)} keys. "
              f"First 10: {missing_keys[:10]}...")
    if unexpected_keys:
        print(f"Warning: Checkpoint has {len(unexpected_keys)} unexpected keys. "
              f"First 10: {unexpected_keys[:10]}...")
    
    # 使用过滤后的checkpoint加载（只包含形状匹配的参数）
    print("Loading model with filtered checkpoint (strict=False)...")
    model.load_state_dict(filtered_checkpoint, strict=False)
    
    # 手动初始化所有缺失的和维度不匹配的参数
    print("Initializing missing or size-mismatched parameters...")
    keys_to_init = set(missing_keys) | set(size_mismatch_keys)
    
    for key in keys_to_init:
        if key not in model_state_dict:
            continue  # 跳过意外的key
        
        param = model_state_dict[key]
        if 'weight' in key:
            if len(param.size()) >= 2:  # 线性层、卷积层等
                nn.init.xavier_uniform_(param)
            else:  # 其他权重
                nn.init.normal_(param, std=0.01)
        elif 'bias' in key:
            nn.init.zeros_(param)
        elif 'offsets' in key or 'widths' in key:
            # RBF参数保持默认初始化
            pass
        
        # 将初始化后的参数赋值给模型
        keys = key.split('.')
        obj = model
        for k in keys[:-1]:
            if k.isdigit():
                obj = obj[int(k)]
            else:
                obj = getattr(obj, k)
        setattr(obj, keys[-1], torch.nn.Parameter(param))
    
    print(f"Successfully initialized {len(keys_to_init)} parameters.")
    
    # 测试集评估：使用固定阈值0.5，使用EMA（保持与验证集一致）
    # EMA可以降低parameter oscillation，对BindingDB cold_start_drug更稳定
    test_metrics = evaluate(model, test_loader, criterion, device, desc="Testing", ema=ema, threshold=0.5)
    print("-" * 50)
    print(f"Test Metrics (Fold {fold}):")
    print(f"  Loss:      {test_metrics['loss']:.4f}")
    print(f"  Accuracy:  {test_metrics['acc']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  AUPRC:     {test_metrics['auprc']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  Threshold (Fixed): {test_metrics['best_threshold']:.2f}")
    print("-" * 50)
    
    return test_metrics

def main():
    set_seed(42) # 设置随机种子
    parser = argparse.ArgumentParser(description="Train DTI Model (5-Fold Cross Validation)")
    parser.add_argument("--dataset", type=str, default="BindingDB", choices=["BindingDB", "BiosNap"], help="Dataset name (BindingDB or BiosNap)")
    parser.add_argument("--split_type", type=str, default="warm_start", choices=["warm_start", "cold_start_drug", "cold_start_protein"], help="Dataset split type")
    parser.add_argument("--gpu", type=int, default=-1, help="Specific GPU ID to use (-1 for auto-select)")
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal", "label_smoothing"], help="Loss function to use")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    # 配置
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = get_free_gpu()
        
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current Device Name: {torch.cuda.get_device_name(device)}")
        
    print(f"Using device: {device}")
    print(f"Split Type: {args.split_type}")
    
    # ========== 动态分配显存安全策略 ==========
    # 根据数据集特性动态调整 batch size 和梯度累加步数
    # BindingDB：长尾序列多，容易出现超长蛋白导致 OOM
    # BioSNAP：序列经过人工清洗，长度集中，可以推满显存
    if args.dataset.lower() == 'bindingdb':
        # BindingDB：降低物理 Batch，开启 4 步梯度累加
        # 等效 batch_size = 8 * 4 = 32，但物理显存开销仅为 1/4
        actual_batch_size = 8
        grad_accum_steps = 4
    elif args.dataset.lower() == 'biosnap':
        # BioSNAP：序列集中，但为防止 distance_mlp OOM，降低物理 batch 并启用梯度累加
        # 等效 batch_size = 16 * 2 = 32，显存开销降低 50%
        actual_batch_size = 8
        grad_accum_steps = 4
    else:
        # 默认配置
        actual_batch_size = 16
        grad_accum_steps = 2
    
    print(f"[显存安全策略] Dataset: {args.dataset}")
    print(f"              物理 Batch Size: {actual_batch_size}")
    print(f"              梯度累加步数: {grad_accum_steps}")
    print(f"              等效 Batch Size: {actual_batch_size * grad_accum_steps}")
    
    # ========== 🚀 BiosNap vs BindingDB 差异化配置 ==========
    # BiosNap：数据更干净、规模小，需要适度降低正则化，避免 overkill
    # BindingDB：长尾序列多、噪声大，需要强正则化防止过拟合
    is_biosnap = args.dataset.lower() == 'biosnap'
    
    # 超参数配置（按 dataset + split_type 自适应）
    # 核心原则：
    # - warm_start: 抑制过拟合 → 小 dropout, 小 num_layers, 大 weight_decay
    # - cold_start: 增强泛化 → 适中 dropout, 大 num_layers, 中 weight_decay
    # - BiosNap: 适度降低正则化，避免 train loss 下不去
    # 注意：L2 regularization 比 dropout 更针对参数复杂度，attention 系统上更稳
    if args.split_type == 'warm_start':
        if is_biosnap:
            # ========== BiosNap warm_start：适度降低正则化 ==========
            HYPERPARAMS = {
                'batch_size': actual_batch_size,
                'grad_accum_steps': grad_accum_steps,
                'epochs': args.epochs, 
                'learning_rate': 1e-4,   # 调大学习率
                'dropout': 0.15,        # BiosNap：适度降低，避免 overkill
                'common_dim': 256,
                'num_heads': 4,
                'num_layers': 2,
                'weight_decay': 1e-5,   # 统一设置为 1e-5
                'atom_dim': 256,
                'motif_dim': 512,
                'protein_dim': 1280,
                'patience': 10         # BiosNap：更长耐心，避免早停
            }
        else:
            # ========== BindingDB warm_start：强正则化 ==========
            HYPERPARAMS = {
                'batch_size': actual_batch_size,
                'grad_accum_steps': grad_accum_steps,
                'epochs': args.epochs, 
                'learning_rate': 1e-4,   # 调大学习率
                'dropout': 0.1,         # BindingDB：抑制过拟合
                'common_dim': 256,
                'num_heads': 4,
                'num_layers': 2,
                'weight_decay': 1e-5,   # 统一设置为 1e-5
                'atom_dim': 256,
                'motif_dim': 512,
                'protein_dim': 1280,
                'patience': 10
            }
    else:
        # cold_start_drug / cold_start_protein 走这里的强正则化逻辑
        if is_biosnap:
            # ========== BiosNap cold_start：适度正则化，保留泛化能力 ==========
            HYPERPARAMS = {
                'batch_size': actual_batch_size,
                'grad_accum_steps': grad_accum_steps,
                'epochs': args.epochs, 
                'learning_rate': 1e-4,   # 调大学习率
                'dropout': 0.20,        # BiosNap cold：适度 dropout，避免 train AUROC 飙升 val 抖动
                'common_dim': 256,
                'num_heads': 4,
                'num_layers': 2,
                'weight_decay': 1e-5,   # 统一设置为 1e-5
                'atom_dim': 256,
                'motif_dim': 512,
                'protein_dim': 1280,
                'patience': 10         # BiosNap：更长耐心
            }
        else:
            # ========== BindingDB cold_start：强正则化逼迫 3D 学习 ==========
            HYPERPARAMS = {
                'batch_size': actual_batch_size,
                'grad_accum_steps': grad_accum_steps,
                'epochs': args.epochs, 
                'learning_rate': 1e-4,   # 调大学习率
                'dropout': 0.3,         # BindingDB cold：必须是 0.3！逼迫模型关注 3D 交互
                'common_dim': 256,
                'num_heads': 4,
                'num_layers': 2,
                'weight_decay': 1e-5,   # 统一设置为 1e-5
                'atom_dim': 256,
                'motif_dim': 512,
                'protein_dim': 1280,
                'patience': 10
            }
    
    print(f"[超参数配置] Dataset: {args.dataset}, Split: {args.split_type}")
    print(f"             dropout: {HYPERPARAMS['dropout']}, weight_decay: {HYPERPARAMS['weight_decay']}")
    print(f"             patience: {HYPERPARAMS['patience']}")
    
    # 特征路径
    # Use the new MATRIX files containing raw graph data
    if args.dataset == 'BiosNap':
        drug_atom_path = os.path.join(current_dir, 'drug', 'BiosNap', 'drug_atom_matrices.npy')
        drug_motif_path = os.path.join(current_dir, 'drug', 'BiosNap', 'drug_motif_matrices.npy')
        # BIOSNAP蛋白质特征路径 - 服务器路径
        protein_feat_path = '/nanxing/ESM_2/Protein/protein_features/biosnap_adapted_features.pkl'
    else:
        drug_atom_path = os.path.join(current_dir, 'drug', 'drug_atom_matrices.npy')
        drug_motif_path = os.path.join(current_dir, 'drug', 'drug_motif_matrices.npy')
        # BindingDB蛋白质特征路径 - 服务器路径
        protein_feat_path = '/nanxing/ESM_2/Protein/protein_features/bindingdb_adapted_features.pkl'
    
    # Tokenizer Path
    token_id_path = os.path.join(unseen_ddi_path, 'token_id.json')
    print(f"Loading Tokenizer from {token_id_path}...")
    tokenizer = Mol_Tokenizer(token_id_path)
    motif_vocab_size = tokenizer.get_vocab_size
    print(f"Motif Vocab Size: {motif_vocab_size}")

    # 预加载特征数据 (只加载一次，节省内存和IO)
    print("Loading features into memory...")
    drug_atom_dict = np.load(drug_atom_path, allow_pickle=True).item()
    drug_motif_dict = np.load(drug_motif_path, allow_pickle=True).item()
    with open(protein_feat_path, 'rb') as f:
        protein_dict = pickle.load(f)
    
    # ======= 修复：同时预加载蛋白质和药物的 3D 特征 =======
    # 使用 rebuild_protein_3d_dict.py 生成的映射文件（键为数据集内部编号）
    if args.dataset == 'BiosNap':
        protein_3d_path = '/nanxing/ESM_2/dataset/BiosNap/protein_3d_dict_by_id.pkl'
        drug_3d_path = '/nanxing/ESM_2/drug/BiosNap/drug_3d_dict.pkl'
    else:  # BindingDB
        protein_3d_path = '/nanxing/ESM_2/dataset/BindingDB/protein_3d_dict_by_bdb_id.pkl'
        drug_3d_path = '/nanxing/ESM_2/drug/BindingDB/drug_3d_dict.pkl'
    
    # 加载蛋白质3D特征（来自PDB提取）
    protein_3d_dict = None
    if os.path.exists(protein_3d_path):
        print(f"Loading 3D protein features from {protein_3d_path}...")
        with open(protein_3d_path, 'rb') as f:
            protein_3d_dict = pickle.load(f)
        print(f"Loaded 3D features for {len(protein_3d_dict)} proteins")
    else:
        print(f"Warning: 3D protein features not found at {protein_3d_path}")

    # 加载药物3D特征（来自RDKit生成）
    drug_3d_dict = None
    if os.path.exists(drug_3d_path):
        print(f"Loading 3D drug features from {drug_3d_path}...")
        with open(drug_3d_path, 'rb') as f:
            drug_3d_dict = pickle.load(f)
        print(f"Loaded 3D features for {len(drug_3d_dict)} drugs")
    else:
        print(f"Warning: 3D drug features not found at {drug_3d_path}")
    
    # ========== 调试打印：检查 3D 数据是否加载成功 ==========
    print(f"\n[数据集诊断] protein_3d_dict is None: {protein_3d_dict is None}")
    print(f"[数据集诊断] drug_3d_dict is None: {drug_3d_dict is None}")
    if protein_3d_dict is not None:
        print(f"[数据集诊断] protein_3d_dict 样本数: {len(protein_3d_dict)}")
    if drug_3d_dict is not None:
        print(f"[数据集诊断] drug_3d_dict 样本数: {len(drug_3d_dict)}")

    # 检查特征维度
    print("Checking feature dimensions...")
    sample_drug_id = list(drug_atom_dict.keys())[0]
    # Check Atom Dim (61 is fixed for atom features)
    # Check Motif Dim (defined by model config, not data)
    
    print("Hyperparameters:")
    for k, v in HYPERPARAMS.items():
        print(f"  {k}: {v}")
    
    # 5折交叉验证
    all_metrics = []
    for fold in range(5):
        metrics = run_fold(fold, args, device, HYPERPARAMS, drug_atom_dict, drug_motif_dict, protein_dict, motif_vocab_size, protein_3d_dict, drug_3d_dict)
        if metrics:
            all_metrics.append(metrics)
            
    # 汇总结果
    if all_metrics:
        print("\n" + "="*50)
        print("5-Fold Cross Validation Results Summary:")
        print("="*50)
        keys = ['auc', 'auprc', 'acc', 'f1', 'precision', 'recall']
        for key in keys:
            values = [m[key] for m in all_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key.upper():<10}: {mean_val:.4f} ± {std_val:.4f}")
        print("="*50)

if __name__ == "__main__":
    main()
