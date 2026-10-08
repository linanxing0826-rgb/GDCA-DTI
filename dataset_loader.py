import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import os
import random

class DTIDataset(Dataset):
    def __init__(self, csv_file, drug_atom_matrix_dict, drug_motif_matrix_dict, protein_dict, 
                 protein_3d_dict=None, drug_3d_dict=None):
        """
        Args:
            csv_file: Path to CSV with drug_id, protein_id, label OR a pandas DataFrame
            drug_atom_matrix_dict: Dictionary containing raw atom matrices
            drug_motif_matrix_dict: Dictionary containing raw motif matrices
            protein_dict: Dictionary containing protein features
            protein_3d_dict: Dictionary containing protein 3D features (contact_prob, protein_distance)
            drug_3d_dict: Dictionary containing drug 3D features (drug_contact_prob, drug_distance)
        """
        if isinstance(csv_file, str):
            print(f"Loading data from {csv_file}...")
            self.data = pd.read_csv(csv_file)
        elif isinstance(csv_file, pd.DataFrame):
            print(f"Loading data from DataFrame ({len(csv_file)} rows)...")
            self.data = csv_file.copy()
        else:
            raise ValueError(f"csv_file must be a CSV path string or a pandas DataFrame, got {type(csv_file)}")
        
        self.drug_atom_matrix_dict = drug_atom_matrix_dict
        self.drug_motif_matrix_dict = drug_motif_matrix_dict
        self.protein_dict = protein_dict
        self.protein_3d_dict = protein_3d_dict
        self.drug_3d_dict = drug_3d_dict  # 药物3D特征字典
            
        # Filter missing data
        initial_len = len(self.data)
        self.data = self.data[
            self.data['drug_id'].isin(self.drug_atom_matrix_dict.keys()) & 
            self.data['protein_id'].apply(lambda x: str(x) in self.protein_dict.keys())
        ].reset_index(drop=True)
        
        if initial_len != len(self.data):
            print(f"Filtered {initial_len - len(self.data)} samples due to missing features in {csv_file}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        drug_id = row['drug_id']
        protein_id = str(row['protein_id'])
        label = row['label']
        
        # 1. Drug Atom Features
        atom_data = self.drug_atom_matrix_dict[drug_id]
        atom_feat = atom_data['atom_feat']      # (N_atoms, 61)
        atom_adj = atom_data['atom_adj']        # (N_atoms, N_atoms) 0/1
        atom_dist = atom_data['atom_dist']      # (N_atoms, N_atoms) dist
        atom_match = atom_data['atom_match_matrix'] # (N_motifs, N_atoms) 0/1
        sum_atoms = atom_data['sum_atoms']      # (N_motifs, 1)
        
        # 2. Drug Motif Features
        motif_data = self.drug_motif_matrix_dict[drug_id]
        motif_seq = motif_data['motif_seq']     # (N_motifs,)
        motif_adj = motif_data['motif_adj']     # (N_motifs, N_motifs) 0/1
        motif_dist = motif_data['motif_dist']   # (N_motifs, N_motifs) dist
        
        # 3. Drug 3D Features (可选)
        drug_contact_prob = None
        drug_distance = None
        
        if self.drug_3d_dict is not None and drug_id in self.drug_3d_dict:
            drug_3d = self.drug_3d_dict[drug_id]
            if 'drug_contact_prob' in drug_3d:
                drug_contact_prob = drug_3d['drug_contact_prob']
            if 'drug_distance' in drug_3d:
                drug_distance = drug_3d['drug_distance']
        
        # 如果有药物3D特征，进行padding
        max_atom_len = atom_feat.shape[0]
        if drug_contact_prob is not None:
            real_len = drug_contact_prob.shape[0]
            n = min(real_len, max_atom_len)
            padded_contact = np.zeros((max_atom_len, max_atom_len), dtype=np.float32)
            padded_contact[:n, :n] = drug_contact_prob[:n, :n]
            drug_contact_prob = padded_contact
        
        if drug_distance is not None:
            real_len = drug_distance.shape[0]
            n = min(real_len, max_atom_len)
            padded_distance = np.full((max_atom_len, max_atom_len), 1e9, dtype=np.float32)
            padded_distance[:n, :n] = drug_distance[:n, :n]
            drug_distance = padded_distance
        
        # 4. Protein Features (N_residues, 1280)
        protein_feat = self.protein_dict[protein_id][1]
        max_len = protein_feat.shape[0]
        
        # 4. Protein 3D Features from PDB (优先使用3D提取器的结果)
        contact_prob = None
        protein_distance = None
        has_protein_3d = False  # 标记是否有有效的3D特征
        
        # ========== 🚀 核心修复：全类型兼容自适应查找器 ==========
        if self.protein_3d_dict is not None:
            target_key = None
            
            # 1. 尝试原始字符串匹配
            if protein_id in self.protein_3d_dict:
                target_key = protein_id
            # 2. 尝试剔除 Pandas 自动生成的 .0 浮点后缀进行字符串匹配
            elif protein_id.endswith('.0') and protein_id[:-2] in self.protein_3d_dict:
                target_key = protein_id[:-2]
            # 3. 尝试将字符串逆向还原为干净的 int 整数型进行键匹配
            else:
                try:
                    int_key = int(float(protein_id))
                    if int_key in self.protein_3d_dict:
                        target_key = int_key
                except:
                    pass
            
            # 如果成功定位到了兼容的键，实施安全特征提取
            if target_key is not None:
                protein_3d = self.protein_3d_dict[target_key]
                if 'contact_prob' in protein_3d:
                    contact_prob = protein_3d['contact_prob']
                if 'protein_distance' in protein_3d:
                    protein_distance = protein_3d['protein_distance']
                has_protein_3d = True
        
        # 如果没有3D特征，则尝试从protein_dict获取（旧格式）
        if contact_prob is None and len(self.protein_dict[protein_id]) > 2:
            try:
                contact_prob = self.protein_dict[protein_id][2]
                has_protein_3d = True
            except:
                pass
        
        if protein_distance is None and len(self.protein_dict[protein_id]) > 3:
            try:
                protein_distance = self.protein_dict[protein_id][3]
                has_protein_3d = True
            except:
                pass
        
        # ========== 🚀 优雅降级：当没有3D特征时，保持为None，不强制使用默认值 ==========
        # cold_start 场景下，测试集的蛋白质可能没有3D特征，此时优雅降级到仅使用2D特征
        if contact_prob is None or protein_distance is None:
            # 标记为没有3D特征，模型会自动降级到2D模式
            has_protein_3d = False
        else:
            # 对3D特征进行padding到max_len
            real_len = contact_prob.shape[0]
            if real_len != max_len:
                n = min(real_len, max_len)
                padded_contact = np.zeros((max_len, max_len), dtype=np.float32)
                padded_contact[:n, :n] = contact_prob[:n, :n]
                contact_prob = padded_contact
            
            real_len = protein_distance.shape[0]
            if real_len != max_len:
                n = min(real_len, max_len)
                padded_distance = np.full((max_len, max_len), 1e9, dtype=np.float32)
                padded_distance[:n, :n] = protein_distance[:n, :n]
                np.fill_diagonal(padded_distance, 0.0)
                protein_distance = padded_distance
        
        # Convert Adjacency to Bias: (1 - adj) * -1e9
        # 1 -> 0, 0 -> -1e9
        atom_adj_bias = (1.0 - atom_adj) * (-1e9)
        motif_adj_bias = (1.0 - motif_adj) * (-1e9)
        
        # 计算 motif mask（标记哪些位置是真实的motif，哪些是padding）
        motif_len = motif_seq.shape[0]
        motif_mask = torch.zeros(motif_len, dtype=torch.bool)
        motif_mask[:motif_len] = True  # 所有位置都是真实的（没有padding）
        
        result = {
            'atom_feat': torch.tensor(atom_feat, dtype=torch.float32),
            'atom_adj': torch.tensor(atom_adj_bias, dtype=torch.float32),
            'atom_dist': torch.tensor(atom_dist, dtype=torch.float32),
            'atom_match': torch.tensor(atom_match, dtype=torch.float32),
            'sum_atoms': torch.tensor(sum_atoms, dtype=torch.float32),
            
            'motif_seq': torch.tensor(motif_seq, dtype=torch.long),
            'motif_adj': torch.tensor(motif_adj_bias, dtype=torch.float32),
            'motif_dist': torch.tensor(motif_dist, dtype=torch.float32),
            'motif_mask': motif_mask,
            'drug_mask': motif_mask,  # 别名：用于模型forward中的drug_mask参数
            
            'protein_feat': torch.tensor(protein_feat, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }
        
        # 添加蛋白质3D特征（如果存在）
        if contact_prob is not None:
            result['contact_prob'] = torch.tensor(contact_prob, dtype=torch.float32)
        if protein_distance is not None:
            result['protein_distance'] = torch.tensor(protein_distance, dtype=torch.float32)
        
        # 添加药物3D特征（如果存在）
        if drug_contact_prob is not None:
            result['drug_contact_prob'] = torch.tensor(drug_contact_prob, dtype=torch.float32)
        if drug_distance is not None:
            result['drug_distance'] = torch.tensor(drug_distance, dtype=torch.float32)
        
        return result

def collate_fn(batch):
    """
    Custom batch processing for padding variable length sequences and matrices
    """
    # Extract lists
    atom_feats = [item['atom_feat'] for item in batch]
    atom_adjs = [item['atom_adj'] for item in batch]
    atom_dists = [item['atom_dist'] for item in batch]
    atom_matches = [item['atom_match'] for item in batch]
    sum_atoms_list = [item['sum_atoms'] for item in batch]
    
    motif_seqs = [item['motif_seq'] for item in batch]
    motif_adjs = [item['motif_adj'] for item in batch]
    motif_dists = [item['motif_dist'] for item in batch]
    
    protein_feats = [item['protein_feat'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # ========== 检查整个 batch 是否有 3D 特征 ==========
    # 蛋白质3D特征：使用 any() 检查整个批次，避免第一个样本没有就丢弃所有
    has_protein_3d = any('contact_prob' in item and item['contact_prob'] is not None for item in batch)
    # 药物3D特征：可能不存在
    has_drug_3d = any('drug_contact_prob' in item and item['drug_contact_prob'] is not None for item in batch)
    
    # ========== 调试打印：检查 collate_fn 中 3D 数据是否存在 ==========
    # 只在第一次调用时打印（通过检查全局计数器）
    if not hasattr(collate_fn, '_print_count'):
        collate_fn._print_count = 0
    collate_fn._print_count += 1
    if collate_fn._print_count <= 3:  # 只打印前3次
        print(f"\n[collate诊断 #{collate_fn._print_count}] batch[0] keys: {list(batch[0].keys())}")
        print(f"[collate诊断] has_protein_3d: {has_protein_3d}, has_drug_3d: {has_drug_3d}")
    
    # Max lengths
    max_atom_len = max([f.shape[0] for f in atom_feats])
    max_motif_len = max([f.shape[0] for f in motif_seqs])
    max_prot_len = max([f.shape[0] for f in protein_feats])
    
    batch_size = len(batch)
    
    # --- Initialize Padded Tensors ---
    
    # 1. Atom Features
    # atom_feat: (B, MaxA, 61)
    padded_atom_feat = torch.zeros(batch_size, max_atom_len, 61)
    
    # atom_adj: (B, MaxA, MaxA) - Pad with -1e9 (disconnected)
    padded_atom_adj = torch.full((batch_size, max_atom_len, max_atom_len), -1e9)
    
    # atom_dist: (B, MaxA, MaxA) - Pad with 1e9 (infinite distance)
    padded_atom_dist = torch.full((batch_size, max_atom_len, max_atom_len), 1e9)
    
    # atom_match: (B, MaxM, MaxA) - Pad with 0
    padded_atom_match = torch.zeros(batch_size, max_motif_len, max_atom_len)
    
    # sum_atoms: (B, MaxM, 1) - Pad with 0 (will need care to avoid div by 0 if used, but masked usually)
    # Actually sum_atoms is used in division. We should pad with 1 to avoid NaN, but the result will be masked out anyway.
    # Let's pad with 1.
    padded_sum_atoms = torch.ones(batch_size, max_motif_len, 1)
    
    # 2. Motif Features
    # motif_seq: (B, MaxM) - Pad with 0
    padded_motif_seq = torch.zeros(batch_size, max_motif_len, dtype=torch.long)
    
    # motif_adj: (B, MaxM, MaxM) - Pad with -1e9
    padded_motif_adj = torch.full((batch_size, max_motif_len, max_motif_len), -1e9)
    
    # motif_dist: (B, MaxM, MaxM) - Pad with 1e9
    padded_motif_dist = torch.full((batch_size, max_motif_len, max_motif_len), 1e9)
    
    # 3. Protein Features
    # protein_feat: (B, MaxP, 1280)
    padded_protein_feat = torch.zeros(batch_size, max_prot_len, 1280)
    
    # 4. Protein Contact Probability and Distance Matrices (样本级填充)
    # contact_prob: (B, MaxP, MaxP) - Pad with 0 (no contact)
    # protein_distance: (B, MaxP, MaxP) - Pad with 1e9 (infinite distance)
    padded_contact_prob = torch.zeros(batch_size, max_prot_len, max_prot_len)
    padded_protein_distance = torch.full((batch_size, max_prot_len, max_prot_len), 1e9)
    
    # 5. Drug Contact Probability and Distance Matrices (样本级填充)
    # drug_contact_prob: (B, MaxA, MaxA) - Pad with 0 (no contact)
    # drug_distance: (B, MaxA, MaxA) - Pad with 1e9 (infinite distance)
    padded_drug_contact_prob = torch.zeros(batch_size, max_atom_len, max_atom_len)
    padded_drug_distance = torch.full((batch_size, max_atom_len, max_atom_len), 1e9)
    
    # 6. 样本级3D特征有效性标志（关键修复：让模型知道哪些样本有真实的3D特征）
    # has_valid_protein_3d: (B,) - True表示该样本有有效的蛋白质3D特征
    # has_valid_drug_3d: (B,) - True表示该样本有有效的药物3D特征
    has_valid_protein_3d = torch.zeros(batch_size, dtype=torch.bool)
    has_valid_drug_3d = torch.zeros(batch_size, dtype=torch.bool)
    
    # Masks (True = Padding/Ignore)
    atom_mask = torch.ones(batch_size, max_atom_len, dtype=torch.bool)
    motif_mask = torch.ones(batch_size, max_motif_len, dtype=torch.bool)
    protein_mask = torch.ones(batch_size, max_prot_len, dtype=torch.bool)

    # --- Fill Data ---
    for i in range(batch_size):
        # Atom
        na = atom_feats[i].shape[0]
        padded_atom_feat[i, :na, :] = atom_feats[i]
        padded_atom_adj[i, :na, :na] = atom_adjs[i]
        padded_atom_dist[i, :na, :na] = atom_dists[i]
        atom_mask[i, :na] = False  # 有效位置设为 False

        # Motif
        nm = motif_seqs[i].shape[0]
        padded_motif_seq[i, :nm] = motif_seqs[i]
        padded_motif_adj[i, :nm, :nm] = motif_adjs[i]
        padded_motif_dist[i, :nm, :nm] = motif_dists[i]
        motif_mask[i, :nm] = False  # 有效位置设为 False

        # Cross (Motif x Atom)
        # Check dimensions: atom_match is (nm_local, na_local)
        nm_local, na_local = atom_matches[i].shape
        padded_atom_match[i, :nm_local, :na_local] = atom_matches[i]
        padded_sum_atoms[i, :nm_local, :] = sum_atoms_list[i]

        # Protein
        np_len = protein_feats[i].shape[0]
        padded_protein_feat[i, :np_len, :] = protein_feats[i]
        protein_mask[i, :np_len] = False
        
        # ========== 关键修复：样本级 3D 特征填充 ==========
        # 蛋白质 3D 特征：每个样本独立处理，有则用，没有则用默认值
        contact_prob = batch[i].get('contact_prob')
        protein_distance = batch[i].get('protein_distance')
        
        # 检查该样本是否有有效的3D特征
        has_protein_3d_for_sample = contact_prob is not None and protein_distance is not None
        
        if has_protein_3d_for_sample:
            contact_prob = torch.tensor(contact_prob, dtype=torch.float32) if not isinstance(contact_prob, torch.Tensor) else contact_prob.clone().detach().float()
            padded_contact_prob[i, :np_len, :np_len] = contact_prob
            
            protein_distance = torch.tensor(protein_distance, dtype=torch.float32) if not isinstance(protein_distance, torch.Tensor) else protein_distance.clone().detach().float()
            padded_protein_distance[i, :np_len, :np_len] = protein_distance
            
            # 标记该样本有有效的蛋白质3D特征
            has_valid_protein_3d[i] = True
        # 缺失时保持默认值（零矩阵和1e9）
        
        # 药物 3D 特征：每个样本独立处理
        drug_contact_prob = batch[i].get('drug_contact_prob')
        drug_distance = batch[i].get('drug_distance')
        
        # 检查该样本是否有有效的药物3D特征
        has_drug_3d_for_sample = drug_contact_prob is not None and drug_distance is not None
        
        if has_drug_3d_for_sample:
            drug_contact_prob = torch.tensor(drug_contact_prob, dtype=torch.float32) if not isinstance(drug_contact_prob, torch.Tensor) else drug_contact_prob.clone().detach().float()
            padded_drug_contact_prob[i, :na, :na] = drug_contact_prob
            
            drug_distance = torch.tensor(drug_distance, dtype=torch.float32) if not isinstance(drug_distance, torch.Tensor) else drug_distance.clone().detach().float()
            padded_drug_distance[i, :na, :na] = drug_distance
            
            # 标记该样本有有效的药物3D特征
            has_valid_drug_3d[i] = True
        # 缺失时保持默认值（零矩阵和1e9）

    result = {
        'atom_feat': padded_atom_feat,
        'atom_adj': padded_atom_adj,
        'atom_dist': padded_atom_dist,
        'atom_match': padded_atom_match,
        'sum_atoms': padded_sum_atoms,
        'atom_mask': atom_mask,  # 新增：原子级别mask

        'motif_seq': padded_motif_seq,
        'motif_adj': padded_motif_adj,
        'motif_dist': padded_motif_dist,
        'motif_mask': motif_mask,  # 新增：motif级别mask
        'drug_mask': motif_mask,   # 别名：用于模型forward中的drug_mask参数

        'protein_feat': padded_protein_feat,
        'protein_mask': protein_mask,

        'label': labels.unsqueeze(1),
        
        # 蛋白质3D特征（总是返回，由has_valid_protein_3d指示有效性）
        'contact_prob': padded_contact_prob,
        'distance_matrix': padded_protein_distance,
        
        # 药物3D特征（总是返回，由has_valid_drug_3d指示有效性）
        'drug_contact_prob': padded_drug_contact_prob,
        'drug_distance': padded_drug_distance,
        
        # 样本级3D特征有效性标志（关键修复）
        'has_valid_protein_3d': has_valid_protein_3d,
        'has_valid_drug_3d': has_valid_drug_3d
    }
    
    return result

def get_dataloader(csv_file, drug_atom_matrix_dict, drug_motif_matrix_dict, protein_dict, batch_size=32, shuffle=True, protein_3d_dict=None, drug_3d_dict=None, num_workers=0, seed=42):
    """
    Args:
        csv_file: Path to CSV with drug_id, protein_id, label
        drug_atom_matrix_dict: Dictionary containing raw atom matrices
        drug_motif_matrix_dict: Dictionary containing raw motif matrices
        protein_dict: Dictionary containing protein features
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        protein_3d_dict: Dictionary containing protein 3D features
        drug_3d_dict: Dictionary containing drug 3D features
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducibility (especially for multi-worker)
    """
    dataset = DTIDataset(csv_file, drug_atom_matrix_dict, drug_motif_matrix_dict, protein_dict, protein_3d_dict, drug_3d_dict)
    
    # ========== 多进程随机种子处理 ==========
    # 防止多个 worker 复制相同的 RNG state，导致各 worker 做相同的随机增强
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # 创建独立的随机数生成器
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # ========== 🚀 性能优化：针对 CPU-heavy collate 的 DataLoader 配置 ==========
    # num_workers=4: 充分利用多核CPU处理复杂的collate操作（atom graph、motif、contact map等）
    # pin_memory=True: 加快数据从CPU到GPU的传输
    # persistent_workers=True: 避免每个epoch重启worker，减少开销（仅当 num_workers > 0）
    # prefetch_factor=2: 每个worker预取2个batch，减少GPU等待（仅当 num_workers > 0）
    
    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'collate_fn': collate_fn,
        'num_workers': num_workers,
        'pin_memory': True,
        'worker_init_fn': seed_worker if num_workers > 0 else None,
        'generator': generator if shuffle else None
    }
    
    if num_workers > 0:
        dataloader_kwargs['persistent_workers'] = True
        dataloader_kwargs['prefetch_factor'] = 2
    
    return DataLoader(**dataloader_kwargs)
