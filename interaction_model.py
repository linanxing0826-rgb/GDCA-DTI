import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys

# Add current directory to path to find drug_encoder
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from drug_encoder import AtomEncoder, MotifEncoder


def safe_masked_pooling(tensor, mask, dim=1):
    """
    安全的多视角池化：使用Mask控制，避免Padding稀释特征

    Args:
        tensor: (B, N, D) - 输入特征
        mask: (B, N) - True for padding, False for valid
        dim: 池化维度

    Returns:
        pooled: (B, D) - 池化后的特征
    """
    if mask is None:
        return torch.mean(tensor, dim=dim)

    mask_float = (~mask).float().unsqueeze(-1)
    masked_tensor = tensor * mask_float
    count = mask_float.sum(dim=dim, keepdim=True)
    pooled = masked_tensor.sum(dim=dim) / (count.squeeze(-1) + 1e-9)
    return pooled


class BilinearGating(nn.Module):
    """
    双向注意力门控融合机制 (Bilinear Gating) - 简化版
    
    设计思路：
    1. 使用轻量级的双线性交互替代参数密集的nn.Bilinear
    2. 通过共享投影减少参数数量
    3. warm split 小数据场景下避免 memorize validation manifold
    
    公式：
    gate = sigmoid(Linear(drug_feat + protein_feat + drug_feat * protein_feat))
    fused = concat(drug_feat * gate, protein_feat * (1 - gate))
    
    Args:
        in_dim: 输入特征维度
        hidden_dim: 中间层维度（默认设为in_dim的一半以减少参数）
        split_type: 训练类型 ('warm_start', 'cold_start', 'cold_start_drug', 'cold_start_protein')
    """
    
    def __init__(self, in_dim, hidden_dim=None, split_type='warm_start'):
        super().__init__()
        
        # 保存 split_type 用于动态门控范围调整
        self.split_type = split_type
        
        # 简化：使用较小的hidden_dim减少参数
        if hidden_dim is None:
            hidden_dim = max(32, in_dim // 2)  # 默认使用in_dim的一半
        
        # 共享投影层：药物和蛋白质共享同一投影
        self.shared_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 门控预测层：使用简单的线性层替代参数密集的双线性层
        # 输入：药物投影 + 蛋白质投影 + 元素级乘积
        self.gate_pred = nn.Sequential(
            nn.Linear(hidden_dim * 3, in_dim),
            nn.Sigmoid()  # 直接输出门控权重，无需额外激活
        )
        
        # 轻量级输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim * 2),
            nn.LayerNorm(in_dim * 2),
            nn.GELU()
        )
    
    def forward(self, drug_feat, protein_feat):
        """
        Args:
            drug_feat: (B, D) - 药物全局特征
            protein_feat: (B, D) - 蛋白质全局特征
        
        Returns:
            fused_feat: (B, 2D) - 门控融合后的特征
            gate_weight: (B, in_dim) - 逐维度门控权重（可用于分析）
        """
        # 共享投影到中间维度
        drug_h = self.shared_proj(drug_feat)  # (B, hidden_dim)
        protein_h = self.shared_proj(protein_feat)  # (B, hidden_dim)
        
        # 计算门控权重：结合加性和乘性交互
        # 输入：药物投影 + 蛋白质投影 + 元素级乘积
        interaction_feat = torch.cat([
            drug_h, 
            protein_h, 
            drug_h * protein_h  # 元素级乘积捕捉交互
        ], dim=-1)  # (B, hidden_dim * 3)
        
        # 预测逐维度门控权重
        gate_weight = self.gate_pred(interaction_feat)  # Sigmoid 输出 [0, 1]
        
        # ========== 关键修复：防止门控塌陷与梯度截断 ==========
        # 弃用 torch.clamp（会截断梯度），改用线性映射限制范围
        if self.split_type in ['cold_start', 'cold_start_drug', 'cold_start_protein']:
            # 冷启动：强制双模态不可偏科，将 [0, 1] 映射到 [0.25, 0.75]
            # 确保即使模型想偷懒，弱势分支也能获得至少 25% 的权重和对应的梯度流
            gate_weight = 0.25 + 0.5 * gate_weight
        else:
            # 热启动：允许更高的置信度，但仍保留底线 [0.1, 0.9]
            gate_weight = 0.1 + 0.8 * gate_weight
        
        # 门控融合 - 逐维度自适应融合
        drug_gated = drug_feat * gate_weight  # (B, D) - 逐维度门控
        protein_gated = protein_feat * (1 - gate_weight)  # (B, D) - 逐维度门控
        
        # 拼接融合后的特征
        fused = torch.cat([drug_gated, protein_gated], dim=-1)  # (B, 2D)
        
        # 可选：进一步处理融合特征
        fused = self.out_proj(fused)  # (B, 2D)
        
        return fused, gate_weight


class FeatureProjector(nn.Module):
    """
    Feature Projector: Maps different feature dimensions to common dimension
    Now mainly for Protein and Drug Encoder outputs
    """
    def __init__(self, drug_dim=512, protein_dim=1280, common_dim=256, dropout=0.3):
        super().__init__()
        
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, drug_feat, protein_feat):
        # drug_feat: (Batch, N_motifs, drug_dim)
        # protein_feat: (Batch, N_residues, protein_dim)
        
        drug_h = self.drug_proj(drug_feat)       # -> (Batch, N_motifs, common_dim)
        protein_h = self.protein_proj(protein_feat) # -> (Batch, N_residues, common_dim)
        
        return drug_h, protein_h

class CrossAttention(nn.Module):
    """
    Cross Attention: Query from one modality, Key/Value from another
    """
    def __init__(self, d_model, num_heads=4, num_layers=1, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, query, key_value, key_padding_mask=None, return_attn_weights=False):
        all_attn_weights = []
        for layer in self.layers:
            query, attn_weight = layer(query, key_value, key_padding_mask)
            if return_attn_weights:
                all_attn_weights.append(attn_weight)
        if return_attn_weights:
            return query, all_attn_weights
        return query

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.3):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key_value, key_padding_mask=None):
        # Attention
        # 使用 key_padding_mask 正确 mask key/value 的 padding
        attn_out, attn_weight = self.mha(query, key_value, key_value, key_padding_mask=key_padding_mask,
                                          need_weights=True, average_attn_weights=False)
        query = self.norm(query + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(query)
        query = self.norm2(query + self.dropout(ffn_out))
        
        return query, attn_weight

class DTIModel(nn.Module):
    """
    End-to-End DTI Model with Integrated Drug Encoders
    """
    def __init__(self, 
                 motif_vocab_size=None,
                 atom_dim=256, 
                 motif_dim=512, 
                 protein_dim=1280, 
                 common_dim=256, 
                 num_heads=4, 
                 num_layers=2,
                 dropout=0.3):
        super().__init__()
        
        # Default vocab size if not provided (e.g. for legacy loading)
        if motif_vocab_size is None:
            # Try to infer or set a safe default
            # In UnseenDDIs token_id.json, size is usually around 12335
            # We can default to a large number or require it
            motif_vocab_size = 13000 
            print(f"Warning: motif_vocab_size not provided, defaulting to {motif_vocab_size}")

        # 1. Drug Encoders
        self.atom_encoder = AtomEncoder(
            num_layers=2,
            d_model=atom_dim,
            num_heads=num_heads,
            dff=motif_dim, # Output dim matches motif_dim
            rate=dropout
        )
        
        self.motif_encoder = MotifEncoder(
            num_layers=num_layers,
            input_vocab_size=motif_vocab_size,
            d_model=motif_dim,
            num_heads=num_heads,
            dff=motif_dim,
            rate=dropout
        )
        
        # 2. Projectors
        # Drug encoders output motif_dim (512), Protein is 1280
        # Project both to common_dim (256)
        self.projector = FeatureProjector(drug_dim=motif_dim, protein_dim=protein_dim, common_dim=common_dim, dropout=dropout)
        
        # 3. Interaction Layers
        # Protein attends to Drug (Motifs)
        # Note: We treat the drug as a sequence of motifs (including global token)
        self.prot_drug_attn = CrossAttention(common_dim, num_heads, num_layers, dropout)
        
        # Drug attends to Protein (Bi-directional)
        self.drug_prot_attn = CrossAttention(common_dim, num_heads, num_layers, dropout)
        
        # 4. Prediction Layer
        # Inputs: 
        # - Drug Global (from Motif Encoder, projected)
        # - Protein Global (Pooled)
        # - Protein Context (from Cross Attn, Pooled)
        # - Drug Context (from Cross Attn, Pooled) [NEW]
        # Enhanced Classifier: More layers, wider hidden dims
        self.classifier = nn.Sequential(
            nn.Linear(common_dim * 4, 2048),
            nn.LayerNorm(2048), # Added LayerNorm
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024), # Added LayerNorm
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),  # Added LayerNorm
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 1)
        )

    def forward(self, 
                atom_feat, atom_adj, atom_dist, atom_match, sum_atoms,
                motif_seq, motif_adj, motif_dist,
                protein_feat, protein_mask=None):
        """
        Args:
            atom_*: Atom level raw inputs
            motif_*: Motif level raw inputs
            protein_feat: (B, N_res, 1280)
            protein_mask: (B, N_res) True for padding
        """
        
        # 1. Run Drug Encoders
        # Atom Encoder -> (B, N_motifs, motif_dim)
        atom_emb = self.atom_encoder(atom_feat, atom_adj, atom_dist, atom_match, sum_atoms)
        
        # Motif Encoder -> (B, N_motifs+1, motif_dim)
        # Note: motif_seq includes global token at 0, so N_motifs+1
        # atom_emb corresponds to N_motifs (indices 1..N)
        motif_emb = self.motif_encoder(motif_seq, atom_level_features=atom_emb, adjoin_matrix=motif_adj, dist_matrix=motif_dist)
        
        # 2. Projection
        # motif_emb: (B, Nm+1, D_drug) -> (B, Nm+1, D_common)
        # protein_feat: (B, Np, D_prot) -> (B, Np, D_common)
        drug_h, protein_h = self.projector(motif_emb, protein_feat)
        
        # 3. Cross Attention
        # Protein (Query) attends to Drug (Key/Value)
        # We use the whole drug sequence (Global + Motifs)
        prot_context = self.prot_drug_attn(protein_h, drug_h) # (B, Np, D_common)
        
        # Drug (Query) attends to Protein (Key/Value) [NEW]
        # Note: We should ideally mask protein padding here, but for simplicity we skip mask 
        # as the attention mechanism usually handles values robustly or we rely on pooling.
        # If strict masking is needed, we need to adapt CrossAttention to accept key_padding_mask.
        drug_context = self.drug_prot_attn(drug_h, protein_h) # (B, Nm+1, D_common)
        
        # 4. Pooling
        # Drug Global: Index 0 of drug_h
        drug_global = drug_h[:, 0, :] # (B, D_common)
        
        # Drug Context Global: Index 0 of drug_context (Global token attended to protein)
        drug_context_global = drug_context[:, 0, :] # (B, D_common)
        
        # Protein Pooling
        # 使用统一的safe_masked_pooling处理蛋白质池化
        prot_global = safe_masked_pooling(protein_h, protein_mask)
        prot_context_global = safe_masked_pooling(prot_context, protein_mask)
            
        # 5. Prediction
        combined = torch.cat([drug_global, prot_global, prot_context_global, drug_context_global], dim=-1)
        output = self.classifier(combined)
        
        return output, drug_global, prot_global, prot_context_global, drug_context_global

class DTIIInteractionBlock(nn.Module):
    """
    单个交互模块：CNN局部提炼 + 双向交叉注意力 + 残差连接 + 接触概率偏置
    
    设计思路：
    - CNN层：提取局部关键模式，降低计算成本
    - 双向交叉注意力：同时从药物和靶点两个视角学习相互作用
    - 残差连接：保留原始信息，避免梯度消失
    - 接触概率偏置：将蛋白质接触概率作为跨模态注意力的结构先验
    """
    
    def __init__(self, hidden_dim, num_heads=4, cnn_kernel_size=3, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 1. CNN局部提炼模块
        # 动态计算padding，确保不同kernel_size都能正确工作
        padding = cnn_kernel_size // 2
        
        # ========== 🚀 药物CNN：硬编码 kernel_size=1 ==========
        # 因为药物 Motif 是无序图节点，绝不能用滑动窗口混合非真实的相邻节点
        # kernel_size=1 是逐点网络（Point-wise FeedForward），不做任何空间混合
        self.drug_cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 蛋白质CNN - 保持原样，使用传入的 cnn_kernel_size(3) 和 padding
        # 蛋白质是 1D 有序氨基酸序列，用 kernel_size=3 提取相邻残基局部特征是合理的
        self.protein_cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=cnn_kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=cnn_kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. 双向交叉注意力
        # 药物→蛋白质注意力（药物关注蛋白质的哪些残基）
        self.drug_to_protein_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 蛋白质→药物注意力（蛋白质关注药物的哪些motif）
        self.protein_to_drug_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # 3. 归一化层
        self.norm_drug1 = nn.LayerNorm(hidden_dim)
        self.norm_drug2 = nn.LayerNorm(hidden_dim)
        self.norm_protein1 = nn.LayerNorm(hidden_dim)
        self.norm_protein2 = nn.LayerNorm(hidden_dim)
        
        # 4. 前馈网络
        self.drug_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.protein_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # ========== 新增：Geometry-aware cross attention 组件 ==========
        # 学习型距离偏置编码器，用于跨模态注意力（蛋白质侧，high=20Å）
        self.protein_distance_rbf = DistanceRBFEncoder(num_rbf=16, num_heads=1, low=0.0, high=20.0)
        
        # 可学习的深度温度参数：允许模型自动校正深度评分的符号方向
        # 使用raw参数配合softplus确保温度始终为正，避免物理意义翻转
        self.depth_temperature_raw = nn.Parameter(torch.tensor(0.0))
        
        # 药物3D距离编码器（小分子药物）
        # 【修改点】：将 high=6.0 修改为 high=15.0，覆盖绝大多数药物分子的宏观尺寸
        self.drug_distance_rbf = DistanceRBFEncoder(num_rbf=16, num_heads=num_heads, low=0.0, high=15.0)
        
        # 残基深度感知MLP：从距离矩阵学习残基的深度特征
        # 保持精简哲学：保留两层非线性，但砍掉LayerNorm缩减维度
        self.protein_depth_mlp = nn.Sequential(
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
        # ========== 关键修复：可学习的结合先验网络 ==========
        # 问题：硬编码 1.0 - contact_prob.mean() 存在理论风险
        # binding pocket 往往半埋藏，很多 active site 并不是低 contact
        # 解决方案：让模型自己学习接触概率到结合先验的映射
        self.binding_prior_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # ========== 可学习的 prior scale（防止 bias 爆炸）==========
        # 初始化为 1.0，让模型自己学习最优的 prior 强度
        # 使用 clamp 确保范围在 [0.1, 2.0]，避免极端值
        self.prior_scale = nn.Parameter(torch.tensor(1.0))
        
        # ========== 可学习的注意力惩罚强度 ==========
        # 初始化为 3.0 (对应 exp(-3) ≈ 0.05)，让模型自己寻找最优的惩罚力度
        # 使用 softplus 保证其物理意义（大于 0）
        self.prot_penalty_raw = nn.Parameter(torch.tensor(3.0))
        self.drug_penalty_raw = nn.Parameter(torch.tensor(3.0))
        
        # 药物原子重要性感知MLP：从距离矩阵学习药物原子的重要性
        # 保持精简哲学：保留两层非线性，但砍掉LayerNorm缩减维度
        self.drug_atom_mlp = nn.Sequential(
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
        
        # ========== 新增：原子到Motif的注意力映射模块 ==========
        # 用于将原子级重要性感知识别到Motif级，解决跨模态3D非对称问题
        self.atom_to_motif_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=1, dropout=dropout, batch_first=True
        )
        
        # 原子特征投影层：将RBF特征(16维)投影到hidden_dim
        self.atom_proj = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 原子化学特征投影层：将原始61维原子特征投影到hidden_dim
        self.atom_proj_for_chem = nn.Sequential(
            nn.Linear(61, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, drug_h, protein_h, protein_mask=None, drug_mask=None, contact_prob=None, distance_matrix=None, drug_distance_matrix=None, drug_atom_raw=None, atom_match=None, protein_rbf_feat=None):
        """
        Args:
            drug_h: (B, N_drug, D) - 药物特征（Motif级别）
            protein_h: (B, N_prot, D) - 蛋白质特征
            protein_mask: (B, N_prot) - 蛋白质padding mask
            drug_mask: (B, N_drug) - 药物padding mask（motif padding）
            contact_prob: (B, N_prot, N_prot) - 蛋白质接触概率矩阵（可选）
            distance_matrix: (B, N_prot, N_prot) - 蛋白质残基距离矩阵（可选，用于几何感知）
            drug_distance_matrix: (B, N_atom, N_atom) - 药物原子距离矩阵（可选，用于几何感知）
            drug_atom_raw: (B, N_atom, 61) - 原始药物原子特征（携带化学性质，用于原子到Motif映射）
            atom_match: (B, N_motif, N_atom) - 原子到Motif的拓扑归属矩阵（用于约束原子到Motif的映射）
            protein_rbf_feat: (B, N_prot, N_prot, num_rbf) - 预计算的蛋白质RBF特征（避免重复计算）
        
        Returns:
            drug_out: 交互后的药物特征
            protein_out: 交互后的蛋白质特征
        """
        # 保存残差连接的输入
        drug_residual = drug_h
        protein_residual = protein_h
        
        # 1. CNN局部提炼
        # Conv1d期望输入: (B, D, N)，所以需要转置
        # 🚀 关键修复：保护CLS token不被CNN污染
        # 分离CLS和其余tokens，仅对tokens应用CNN
        drug_cls = drug_h[:, 0:1, :]  # (B, 1, D)
        drug_tokens = drug_h[:, 1:, :]  # (B, N_drug-1, D)
        drug_cnn_input = drug_tokens.transpose(1, 2)  # (B, D, N_drug-1)
        drug_tokens_cnn = self.drug_cnn(drug_cnn_input).transpose(1, 2)  # (B, N_drug-1, D)
        drug_cnn_out = torch.cat([drug_cls, drug_tokens_cnn], dim=1)  # (B, N_drug, D)
        
        protein_cls = protein_h[:, 0:1, :]  # (B, 1, D)
        protein_tokens = protein_h[:, 1:, :]  # (B, N_prot-1, D)
        protein_cnn_input = protein_tokens.transpose(1, 2)  # (B, D, N_prot-1)
        protein_tokens_cnn = self.protein_cnn(protein_cnn_input).transpose(1, 2)  # (B, N_prot-1, D)
        protein_cnn_out = torch.cat([protein_cls, protein_tokens_cnn], dim=1)  # (B, N_prot, D)
        
        # 关键修复：CNN后强制mask padding区域，防止padding污染真实特征
        # Conv1d不认识padding mask，需要手动处理
        if drug_mask is not None:
            # 安全检查：确保mask长度与CNN输出匹配
            if drug_mask.size(1) == drug_cnn_out.size(1):
                drug_cnn_out = drug_cnn_out.masked_fill(
                    drug_mask.unsqueeze(-1),
                    0.0
                )
        
        if protein_mask is not None:
            # 安全检查：确保mask长度与CNN输出匹配
            if protein_mask.size(1) == protein_cnn_out.size(1):
                protein_cnn_out = protein_cnn_out.masked_fill(
                    protein_mask.unsqueeze(-1),
                    0.0
                )
        
        # 2. 构建残基暴露先验（用于药物→蛋白质交叉注意力）
        # 优先使用几何信息（distance_matrix），fallback到拓扑信息（contact_prob）
        protein_cross_attn_bias = None
        
        if distance_matrix is not None:
            # ========== 🚀 关键优化：复用 RBF 特征，避免重复计算 ==========
            # 如果提供了预计算的 protein_rbf_feat，直接使用；否则回退到 chunked 计算
            if protein_rbf_feat is not None:
                # 🚀 修复 Padding 稀释问题：使用 mask 进行加权平均
                # 避免短蛋白特征被 padding 位置稀释
                if protein_mask is not None:
                    # valid_mask: (B, N) -> (B, 1, N, 1)
                    valid_mask = ~protein_mask.unsqueeze(1).unsqueeze(-1)
                    # 只对有效位置求和
                    rbf_sum = (protein_rbf_feat * valid_mask.float()).sum(dim=2)
                    # 有效长度（至少为1避免除以0）
                    true_len = valid_mask.sum(dim=2).clamp(min=1.0)
                    avg_dist_feat = rbf_sum / true_len  # (B, N, num_rbf)
                else:
                    avg_dist_feat = protein_rbf_feat.mean(dim=2)  # (B, N, num_rbf)
            else:
                # Fallback：使用 chunked RBF 计算（兼容旧代码）
                rbf_centers = self.protein_distance_rbf.offsets
                rbf_widths = self.protein_distance_rbf.widths
                
                B, N_prot, _ = distance_matrix.shape
                chunk_size = 128  # 2080Ti 11GB 显存安全值
                
                avg_dist_feat_list = []
                for start in range(0, N_prot, chunk_size):
                    end = min(start + chunk_size, N_prot)
                    dist_chunk = distance_matrix[:, start:end, :]
                    
                    with torch.cuda.amp.autocast(enabled=True):
                        diff_chunk = dist_chunk.unsqueeze(-1) - rbf_centers.view(1, 1, 1, -1)
                        rbf_chunk = torch.exp(-(diff_chunk / (rbf_widths + 1e-6)) ** 2)
                        
                        # 🚀 修复 Padding 稀释问题：使用 mask 进行加权平均
                        if protein_mask is not None:
                            valid_mask_chunk = ~protein_mask.unsqueeze(1).unsqueeze(-1)[:, start:end, :, :]
                            rbf_sum_chunk = (rbf_chunk * valid_mask_chunk.float()).sum(dim=2)
                            true_len_chunk = valid_mask_chunk.sum(dim=2).clamp(min=1.0)
                            avg_chunk = rbf_sum_chunk / true_len_chunk
                        else:
                            avg_chunk = rbf_chunk.mean(dim=2)
                    
                    avg_dist_feat_list.append(avg_chunk)
                    del diff_chunk, rbf_chunk  # 立即释放临时变量
                
                avg_dist_feat = torch.cat(avg_dist_feat_list, dim=1)  # (B, N, num_rbf)
            
            # 通过 MLP 学习残基深度评分
            depth_score = self.protein_depth_mlp(avg_dist_feat).squeeze(-1)  # (B, N_prot)
            
            # 将深度评分转换为注意力偏置：表面残基（低深度）惩罚为0，内部残基（高深度）最高惩罚-3.0
            # 严格遵循PyTorch MultiheadAttention的数学规范：attn_mask必须为负值或零
            # 使用可学习的温度参数，允许模型自动校正深度评分的符号方向
            # 通过softplus确保温度始终为正，避免物理意义翻转
            depth_temperature = F.softplus(self.depth_temperature_raw) + 1e-6
            surface_prob = torch.sigmoid(-depth_score * depth_temperature)
            # 转换为负向惩罚项：表面残基惩罚为0，内部残基惩罚由模型学习
            prot_penalty = F.softplus(self.prot_penalty_raw)
            protein_cross_attn_bias = ((surface_prob - 1.0) * prot_penalty).unsqueeze(1).expand(-1, drug_cnn_out.size(1), -1)  # (B, N_drug, N_protein)
            # 扩展到 (B * num_heads, N_drug, N_protein) 以匹配MultiheadAttention的期望形状
            # 使用repeat_interleave确保Batch与Head顺序与PyTorch内部完全对齐
            protein_cross_attn_bias = protein_cross_attn_bias.repeat_interleave(self.num_heads, dim=0)
            
        elif contact_prob is not None:
            # Fallback：仅使用接触概率（拓扑信息）
            # ========== 关键修复：使用可学习的结合先验网络 ==========
            # 问题：硬编码 1.0 - contact_prob.mean() 存在理论风险
            # binding pocket 往往半埋藏，很多 active site 并不是低 contact
            # 解决方案：让模型自己学习接触概率到结合先验的映射
            
            # 🚀 修复 Padding 稀释问题：使用 mask 进行加权平均
            if protein_mask is not None:
                # protein_mask: (B, N_prot), True=padding, False=valid
                # valid_mask: (B, N_prot), True=valid
                valid_mask = ~protein_mask
                # 构建成对有效掩码：(B, N_prot, N_prot)
                valid_pair_mask = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
                # 只对有效位置求和
                valid_contact = contact_prob * valid_pair_mask.float()
                # 每个残基的有效邻居数（至少为1避免除以0）
                true_len = valid_pair_mask.sum(dim=-1).clamp(min=1.0)
                # 加权平均
                contact_mean = valid_contact.sum(dim=-1) / true_len  # (B, N_prot)
            else:
                contact_mean = contact_prob.mean(dim=-1)  # (B, N_prot) - 每个残基的平均接触概率
            
            # 使用可学习的prior net替代硬编码的 1-contact
            # 输入：接触概率的均值
            # 输出：学习到的结合先验分数
            residue_logits = self.binding_prior_net(contact_mean.unsqueeze(-1)).squeeze(-1)  # (B, N_prot)
            
            # ========== 🚀 关键修复：Bounded Attention Bias ==========
            # 使用 tanh 限制范围在 [-1, 1]，再乘以可学习的 scale
            # scale 被 clamp 在 [0.1, 2.0]，避免极端值
            prior_scale = torch.clamp(self.prior_scale, 0.1, 2.0)
            residue_bias = prior_scale * torch.tanh(residue_logits)  # (B, N_prot) - 范围 [-scale, scale]
            
            protein_cross_attn_bias = residue_bias.unsqueeze(1).expand(-1, drug_cnn_out.size(1), -1)  # (B, N_drug, N_protein)
            protein_cross_attn_bias = protein_cross_attn_bias.repeat_interleave(self.num_heads, dim=0)
            
            expected_shape = (protein_cross_attn_bias.size(0) // self.num_heads * self.num_heads, 
                             drug_cnn_out.size(1), 
                             contact_prob.size(-1))
            assert protein_cross_attn_bias.shape == expected_shape, \
                f"protein_cross_attn_bias shape {protein_cross_attn_bias.shape} != expected {expected_shape}"
            
            if protein_mask is not None:
                # 扩展 mask 到 (B * num_heads, 1, N_protein)
                padding_mask = protein_mask.repeat_interleave(self.num_heads, dim=0).unsqueeze(1)  # (B*H, 1, Np)
                protein_cross_attn_bias = protein_cross_attn_bias.masked_fill(padding_mask, -1e9)
        
        # 3. 构建药物原子重要性先验（用于蛋白质→药物交叉注意力）
        # 修复跨模态3D非对称问题：药物3D信息也应该进入跨模态注意力
        drug_cross_attn_bias = None
        if drug_distance_matrix is not None:
            # ========== Geometry-aware 药物原子重要性特征 (Chunked + AMP 版本) ==========
            # 使用 RBF 编码药物距离矩阵
            rbf_centers = self.drug_distance_rbf.offsets
            rbf_widths = self.drug_distance_rbf.widths
            
            # ========== 🚀 关键修复：Chunked RBF 计算防止 OOM ==========
            # 问题：(B, N_atom, N_atom, 16) 在长分子上会爆炸
            # 方案：沿 dim=1 分块计算，数学等价但显存可控
            B, N_atom, _ = drug_distance_matrix.shape
            chunk_size = 128  # 2080Ti 11GB 显存安全值
            
            # 🚀 修复 1.1：从拓扑矩阵精准提取真实的原子 Mask
            if atom_match is not None:
                # atom_match 形状为 (B, N_motif, N_atom)
                # 只要原子属于至少一个 motif，它就是真实的有效原子
                valid_atom_mask = (atom_match.sum(dim=1) > 0)  # (B, N_atom)
            else:
                valid_atom_mask = torch.ones(B, N_atom, dtype=torch.bool, device=drug_distance_matrix.device)
            
            avg_atom_dist_feat_list = []
            for start in range(0, N_atom, chunk_size):
                end = min(start + chunk_size, N_atom)
                dist_chunk = drug_distance_matrix[:, start:end, :]
                
                with torch.cuda.amp.autocast(enabled=True):
                    diff_chunk = dist_chunk.unsqueeze(-1) - rbf_centers.view(1, 1, 1, -1)
                    drug_rbf_chunk = torch.exp(-(diff_chunk / (rbf_widths + 1e-6)) ** 2)
                    
                    # 🚀 修复：使用真实的原子 Mask 屏蔽无效邻居，防止 3D 信号被 Padding 稀释
                    valid_neighbor_mask = valid_atom_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N_atom, 1)
                    valid_rbf = drug_rbf_chunk * valid_neighbor_mask.float()
                    
                    # 真实的邻居数量
                    true_len = valid_atom_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1.0)  # (B, 1, 1)
                    avg_chunk = valid_rbf.sum(dim=2) / true_len
                
                avg_atom_dist_feat_list.append(avg_chunk)
                del diff_chunk, drug_rbf_chunk  # 立即释放临时变量
            
            avg_atom_dist_feat = torch.cat(avg_atom_dist_feat_list, dim=1)  # (B, N_atom, num_rbf)
            
            # 通过 MLP 学习药物原子重要性评分（例如：表面原子更重要）
            atom_importance = self.drug_atom_mlp(avg_atom_dist_feat).squeeze(-1)  # (B, N_atom)
            
            # 将原子重要性转换为概率分布
            atom_prob = torch.sigmoid(atom_importance * 2.0)
            atom_prob = torch.clamp(atom_prob, min=1e-6, max=1.0)
            
            # ========== 完美修复：拓扑约束下的原子到Motif空间映射 ==========
            # 原子只能映射到自己隶属的Motif上，避免无约束的全局映射导致几何信息混乱
            # atom_feat_proj 包含原子的空间几何与化学身份
            
            # 将原子级RBF特征投影到hidden_dim，用于注意力计算
            atom_feat_proj = self.atom_proj(avg_atom_dist_feat)  # (B, N_atom, hidden_dim)
            
            # 修复：让query携带原子的化学语义，实现真正的物理-化学联动
            # 如果提供了原始原子特征，将其投影到hidden_dim并与几何特征融合
            if drug_atom_raw is not None:
                # 将原始61维原子特征投影到hidden_dim
                atom_chem_proj = self.atom_proj_for_chem(drug_atom_raw)  # (B, N_atom, hidden_dim)
                atom_feat_proj = atom_feat_proj + atom_chem_proj
            
            # 1. 计算原子与Motif的原始相似度Logits
            # query: (B, N_atom, D), key: (B, N_motif, D) -> logits: (B, N_atom, N_motif)
            match_logits = torch.bmm(atom_feat_proj, drug_cnn_out[:, 1:, :].transpose(1, 2))
            match_logits = match_logits / math.sqrt(self.hidden_dim)
            
            # 2. 注入原有的拓扑匹配先验 (atom_match 形状为 [B, N_motif, N_atom])
            # 转置为 (B, N_atom, N_motif)，1代表属于该Motif，0代表不属于
            if atom_match is not None:
                topology_mask = atom_match.transpose(1, 2)  # (B, N_atom, N_motif)
                # 确保拓扑掩码与match_logits维度匹配（处理数据预处理时可能的长度不一致）
                if topology_mask.size(-1) != match_logits.size(-1):
                    # 截断或填充到match_logits的motif数量
                    target_motif_num = match_logits.size(-1)
                    if topology_mask.size(-1) > target_motif_num:
                        topology_mask = topology_mask[:, :, :target_motif_num]
                    else:
                        # 用全1填充（表示允许匹配）
                        pad_size = target_motif_num - topology_mask.size(-1)
                        topology_mask = torch.cat(
                            [topology_mask, 
                             torch.ones(topology_mask.size(0), topology_mask.size(1), pad_size, 
                                       device=topology_mask.device)], 
                            dim=-1
                        )
                # 强迫不属于该官能团的原子相似度直接归零(-1e9)，斩断空间错配噪声
                match_logits = match_logits.masked_fill(topology_mask == 0, -1e9)
            
            # 🚀 修复 1.2：在 Softmax 之前，利用真实的原子 Mask 彻底斩断假原子的污染
            if atom_match is not None:
                invalid_atom_mask = ~valid_atom_mask  # (B, N_atom)
                # 强行将 padding 原子的相似度打入冷宫 (-1e9)
                match_logits = match_logits.masked_fill(invalid_atom_mask.unsqueeze(-1), -1e9)
            
            # 3. 在Motif维度做Softmax，得到纯净的原子-Motif空间映射权重
            atom_to_motif_weights = F.softmax(match_logits, dim=-1)  # (B, N_atom, N_motif)
            
            # Softmax 之后，将 padding 原子的分配权重强制归 0
            if atom_match is not None:
                atom_to_motif_weights = atom_to_motif_weights * valid_atom_mask.unsqueeze(-1).float()
            
            # 4. 聚合原子级重要性到Motif级
            motif_importance = torch.bmm(
                atom_prob.unsqueeze(1),  # (B, 1, N_atom)
                atom_to_motif_weights    # (B, N_atom, N_motif)
            ).squeeze(1)  # (B, N_motif)
            
            # 将Motif重要性转换为注意力偏置
            # 废除危险的对数运算，避免梯度爆炸和NaN问题
            # 统一对齐蛋白质侧的Soft-Masking设计：重要Motif惩罚为0，不重要的最高惩罚-3.0
            # 药物侧：使用可学习的惩罚强度
            drug_penalty = F.softplus(self.drug_penalty_raw)
            motif_logits = (motif_importance - 1.0) * drug_penalty  # (B, N_motif)
            
            # 在第0位手动补一个0.0（代表对Global Token不加任何空间惩罚）
            # 使其完美扩充为 (B, N_motif + 1)，与包含Global Token的drug_cnn_out对齐
            batch_size = motif_logits.size(0)
            global_token_bias = torch.zeros(batch_size, 1, device=motif_logits.device)  # (B, 1)
            motif_logits_with_global = torch.cat([global_token_bias, motif_logits], dim=1)  # (B, N_motif + 1)
            
            # ========== 关键修复：纠正跨模态注意力偏置的物理维度语义 ==========
            # 药物Motif偏置的形状为 (B, N_motif + 1)
            # 我们需要在行（Protein维）上进行unsqueeze(1)，在列（Drug维）上保持特异性
            # 正确形状：(B, N_protein, N_motif + 1) 且每行具有异质性
            
            # ⚠️ 关键点：为了让Protein的每个残基关注Drug的不同Motif时受到不同的惩罚，
            # 必须使用unsqueeze(1)并沿Protein轴进行repeat，此时列与列之间是变化的。
            # 确保传递给MHA的掩码在列方向（Key维）具备特异性：
            drug_cross_attn_bias = motif_logits_with_global.unsqueeze(1).repeat(1, protein_cnn_out.size(1), 1)  # (B, N_protein, N_motif + 1)
            
            # 扩展到 (B * num_heads, N_protein, N_motif + 1) 以匹配MultiheadAttention的期望形状
            # 使用repeat_interleave确保Batch与Head顺序与PyTorch内部完全对齐
            drug_cross_attn_bias = drug_cross_attn_bias.repeat_interleave(self.num_heads, dim=0)  # (B * num_heads, N_protein, N_motif + 1)
        
        # 4. 双向交叉注意力
        # ==================== 统一掩码类型，消除 PyTorch 警告 ====================
        
        # 1. 药物→蛋白质注意力：Query(Drug), Key/Value(Protein)
        # 原始的 protein_mask 是 Bool，如果存在，将其转化为 Float 偏置 (0.0 或 -1e9)
        if protein_mask is not None:
            # protein_mask: (B, N_prot) -> 扩展到多头形状 (B * num_heads, N_drug, N_prot)
            # True 的地方代表 padding，赋予 -1e9 惩罚；False 的地方代表有效，赋予 0.0
            prot_pad_bias = protein_mask.float().masked_fill(protein_mask, -1e9).unsqueeze(1)  # (B, 1, N_prot)
            prot_pad_bias = prot_pad_bias.expand(-1, drug_cnn_out.size(1), -1)  # (B, N_drug, N_prot)
            # 广播到多头
            prot_pad_bias = prot_pad_bias.repeat_interleave(self.num_heads, dim=0)  # (B * num_heads, N_drug, N_prot)
            
            # 将 padding 偏置直接加进 3D 几何偏置 protein_cross_attn_bias 中
            if protein_cross_attn_bias is not None:
                protein_cross_attn_bias = protein_cross_attn_bias + prot_pad_bias
            else:
                protein_cross_attn_bias = prot_pad_bias

        # 2. 蛋白质→药物注意力：Query(Protein), Key/Value(Drug)
        # 原始的 drug_mask 是 Bool，将其转化为 Float 偏置
        if drug_mask is not None:
            # True 的地方赋予 -1e9，False 的地方赋予 0.0
            drug_pad_bias = drug_mask.float().masked_fill(drug_mask, -1e9).unsqueeze(1)  # (B, 1, N_drug)
            drug_pad_bias = drug_pad_bias.expand(-1, protein_cnn_out.size(1), -1)  # (B, N_prot, N_drug)
            # 广播到多头
            drug_pad_bias = drug_pad_bias.repeat_interleave(self.num_heads, dim=0)  # (B * num_heads, N_prot, N_drug)
            
            # 将 padding 偏置直接加进药物 3D 偏置 drug_cross_attn_bias 中
            if drug_cross_attn_bias is not None:
                drug_cross_attn_bias = drug_cross_attn_bias + drug_pad_bias
            else:
                drug_cross_attn_bias = drug_pad_bias

        # ==================== 调用 MHA 时，将 key_padding_mask 设为 None ====================
        # 因为我们已经把 padding 信息以 Float 形式无损融合进了各向异性的 attn_mask 中！
        
        # 药物→蛋白质（药物关注蛋白质）
        # 使用残基暴露先验作为注意力偏置，强迫药物关注表面暴露的结合位点
        drug_attended, drug_to_prot_attn_weights = self.drug_to_protein_attn(
            query=drug_cnn_out, 
            key=protein_cnn_out, 
            value=protein_cnn_out,
            key_padding_mask=None,             # ◄── 改为 None，消灭类型冲突
            attn_mask=protein_cross_attn_bias  # ◄── 已经完美融合了3D与Padding的纯Float偏置
        )
        
        # 蛋白质→药物（蛋白质关注药物）
        # 使用药物原子重要性先验作为注意力偏置，修复跨模态3D非对称
        protein_attended, prot_to_drug_attn_weights = self.protein_to_drug_attn(
            query=protein_cnn_out,
            key=drug_cnn_out,
            value=drug_cnn_out,
            key_padding_mask=None,          # ◄── 改为 None，消灭类型冲突
            attn_mask=drug_cross_attn_bias  # ◄── 已经完美融合了3D与Padding的纯Float偏置
        )
        
        # 5. 残差连接 + FFN
        # 药物侧残差
        drug_out = self.norm_drug1(drug_residual + self.dropout(drug_attended))
        drug_out = self.norm_drug2(drug_out + self.dropout(self.drug_ffn(drug_out)))
        
        # 蛋白质侧残差
        protein_out = self.norm_protein1(protein_residual + self.dropout(protein_attended))
        protein_out = self.norm_protein2(protein_out + self.dropout(self.protein_ffn(protein_out)))
        
        return drug_out, protein_out, drug_to_prot_attn_weights, prot_to_drug_attn_weights


class DTIMultiScaleInteraction(nn.Module):
    """
    多尺度DTI交互模型 - 堆叠3个DTIIInteractionBlock
    
    设计思路：
    - 第一层：捕捉局部精细模式（如氨基酸侧链相互作用）
    - 第二层：提炼中等范围的交互特征（如motif-level结合）
    - 第三层：整合全局交互信息（整体结合亲和力预测）
    
    完全对应您提到的三层结构设计
    """
    
    def __init__(self, hidden_dim=256, num_heads=4, num_blocks=3, cnn_kernel_size=3, dropout=0.3):#（冷启动设置为2，热启动3？）
        super().__init__()
        
        self.blocks = nn.ModuleList([
            DTIIInteractionBlock(hidden_dim, num_heads, cnn_kernel_size, dropout)
            for _ in range(num_blocks)
        ])
        
        # 可选：输出层的额外投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, drug_feat, protein_feat, protein_mask=None, drug_mask=None, contact_prob=None, distance_matrix=None, drug_distance_matrix=None, drug_atom_raw=None, atom_match=None, protein_rbf_feat=None, return_attn_weights=False):
        """
        Args:
            drug_feat: (B, N_drug, D_drug) - 药物特征（Motif级别）
            protein_feat: (B, N_prot, D_prot) - 蛋白质特征
            protein_mask: (B, N_prot) - 蛋白质padding mask
            drug_mask: (B, N_drug) - 药物padding mask（motif padding）
            contact_prob: (B, N_prot, N_prot) - 蛋白质接触概率矩阵（可选）
            distance_matrix: (B, N_prot, N_prot) - 蛋白质残基距离矩阵（可选，用于几何感知）
            drug_distance_matrix: (B, N_atom, N_atom) - 药物原子距离矩阵（可选，用于几何感知）
            drug_atom_raw: (B, N_atom, 61) - 原始药物原子特征（携带化学性质）
            atom_match: (B, N_motif, N_atom) - 原子到Motif的拓扑归属矩阵（用于约束原子到Motif的映射）
            protein_rbf_feat: (B, N_prot, N_prot, num_rbf) - 预计算的蛋白质RBF特征（避免重复计算）
            return_attn_weights: 是否返回attention权重（用于分析）
        
        Returns:
            drug_final: 多尺度交互后的药物特征 (B, N_drug, hidden_dim)
            protein_final: 多尺度交互后的蛋白质特征 (B, N_prot, hidden_dim)
            intermediate_features: 各层输出的特征列表（用于分析）
            attn_weights: 各层的attention权重（仅当return_attn_weights=True时返回）
        """
        drug_h = drug_feat
        protein_h = protein_feat
        intermediate_features = []
        attn_weights = []
        
        # 堆叠多个Block，逐步提炼从局部到全局的特征
        for i, block in enumerate(self.blocks):
            drug_h, protein_h, drug_to_prot_attn, prot_to_drug_attn = block(drug_h, protein_h, protein_mask, drug_mask, contact_prob, distance_matrix, drug_distance_matrix, drug_atom_raw, atom_match, protein_rbf_feat)
            intermediate_features.append((drug_h, protein_h))
            if return_attn_weights:
                attn_weights.append({
                    'drug_to_protein': drug_to_prot_attn,
                    'protein_to_drug': prot_to_drug_attn
                })
        
        # 最后的输出投影
        drug_final = self.output_proj(drug_h)
        protein_final = self.output_proj(protein_h)
        
        if return_attn_weights:
            return drug_final, protein_final, intermediate_features, attn_weights
        return drug_final, protein_final, intermediate_features


class DistanceRBFEncoder(nn.Module):
    def __init__(self, num_rbf=16, num_heads=4, low=0.0, high=20.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.num_heads = num_heads
        self.high = high
        
        self.register_buffer("offsets", torch.linspace(low, high, num_rbf))
        # 增加安全分母防护
        self.register_buffer("widths", torch.tensor((high - low) / num_rbf if num_rbf > 1 else 1.0))
        
        # 🚀 恢复微型非线性：极度节省显存，同时完美拟合复杂的物理能量距离势能面
        self.distance_proj = nn.Sequential(
            nn.Linear(num_rbf, 16),
            nn.ReLU(inplace=True),  # 零显存激活
            nn.Linear(16, num_heads),
            nn.Tanh()  # 稳定输出在 [-1, 1] 防止注意力坍塌
        )
        
        # 🚀 可学习门控，初始化为 0.3，提供适度初始先验
        # 赋予它初始的"发言权"，但最终决定权交给反向传播
        self.struct_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, dist_matrix, mask=None, return_rbf_feat=False):
        """
        Args:
            dist_matrix: (B, N, N) - 距离矩阵
            mask: (B, N) 或 (B, N, N) - padding mask（可选）
            return_rbf_feat: 是否返回 RBF 特征（用于避免重复计算）
        """
        if dist_matrix is None:
            if return_rbf_feat:
                return None, None
            return None
        
        offsets = self.offsets
        widths = self.widths
        
        B, N, _ = dist_matrix.shape
        
        # 🚀 修复 mask 处理逻辑，确保形状正确
        mask_3d = None
        mask_2d = None
        if mask is not None:
            if mask.dim() == 2:
                mask_2d = mask
                mask_3d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            else:
                mask_3d = mask.bool()
                mask_2d = mask.any(dim=-1)
        
        # 🚀 关键优化：chunk_size = 128（2080Ti 11GB 安全值）
        chunk_size = 128
        bias_chunk_list = []
        rbf_chunk_list = [] if return_rbf_feat else None
        
        # 🚀 分块计算 RBF 和投影，控制峰值显存
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            
            # 取出当前块: (B, chunk, N)
            chunk_dist = dist_matrix[:, start:end, :]
            
            # Mask 填充
            if mask_3d is not None:
                chunk_dist = chunk_dist.masked_fill(~mask_3d[:, start:end, :], 999.0)
                
            # 软截断
            chunk_dist = torch.where(chunk_dist < 0, torch.exp(chunk_dist) - 1, chunk_dist)
            chunk_dist = torch.where(chunk_dist > self.high, 
                                     self.high + torch.log1p(chunk_dist - self.high), 
                                     chunk_dist)
            
            # RBF 计算 (B, chunk, N, num_rbf)
            diff = chunk_dist.unsqueeze(-1) - offsets
            denom = widths + 1e-6
            
            pow_diff = (diff / denom) ** 2
            pow_diff = pow_diff / (1 + pow_diff / 50.0)
            
            rbf_chunk = torch.exp(-pow_diff)  # (B, chunk, N, num_rbf)
            
            # 🚀 单线性层投影 + Tanh + 可学习门控
            bias_chunk = self.distance_proj(rbf_chunk)  # (B, chunk, N, num_heads)
            bias_chunk = bias_chunk * self.struct_scale  # 可学习缩放
            bias_chunk_list.append(bias_chunk)
            
            if return_rbf_feat:
                rbf_chunk_list.append(rbf_chunk)
        
        # 拼接回完整矩阵
        bias = torch.cat(bias_chunk_list, dim=1)  # (B, N, N, num_heads)
        bias = bias.permute(0, 3, 1, 2)           # (B, num_heads, N, N)
        
        # 最终的 Padding 掩码
        if mask_2d is not None:
            # bias shape: (B, num_heads, N, N)
            # mask_2d shape: (B, N) -> (B, 1, 1, N) for broadcasting
            bias = bias.masked_fill(~mask_2d.unsqueeze(1).unsqueeze(2).bool(), -1e9)
        
        if return_rbf_feat:
            rbf_feat = torch.cat(rbf_chunk_list, dim=1)
            return bias, rbf_feat
        else:
            return bias


class FeatureFusion(nn.Module):
    """
    多模态特征级联融合
    将额外嵌入矩阵与ESM-2嵌入在特征维度拼接，通过线性层投影回目标维度
    """
    def __init__(self, esm_dim, extra_dim, hidden_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(esm_dim + extra_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, esm_emb, extra_emb=None):
        # esm_emb: (B, N, esm_dim) - ESM-2嵌入
        # extra_emb: (B, N, extra_dim) - 额外嵌入矩阵（可选）
        
        if extra_emb is not None:
            # 特征级联融合
            fused = torch.cat([esm_emb, extra_emb], dim=-1)
            return self.fusion(fused)
        else:
            # 没有额外嵌入时，安全处理 padding
            pad_dim = self.fusion[0].in_features - esm_emb.shape[-1]
            if pad_dim > 0:
                zeros = torch.zeros(
                    *esm_emb.shape[:-1],
                    pad_dim,
                    device=esm_emb.device
                )
                fused = torch.cat([esm_emb, zeros], dim=-1)
            else:
                fused = esm_emb
            return self.fusion(fused)


class GATLayer(nn.Module):
    """
    优化后的图注意力层：
    1. 消除5D张量，显存占用降低 80% 以上，支持长蛋白质序列。
    2. 完美融入连续接触概率作为 Attention Bias。
    3. 支持3D空间结构注意力偏置（通过RBF处理距离矩阵）。
    4. 支持直接传入 contact_prob 和 distance_matrix 计算3D-aware attention bias。
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.3, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat
        
        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        # 将原有的 a (1, H, 2*D) 拆分为 src 和 dst 两个一维向量，避免显存爆炸
        # 形状改为 (1, 1, num_heads, out_dim)，使其在蛋白质长度维度 N 上保持为 1 以便正确广播
        self.a_src = nn.Parameter(torch.randn(1, 1, num_heads, out_dim))
        self.a_dst = nn.Parameter(torch.randn(1, 1, num_heads, out_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # 残差连接和层归一化（标准做法）
        output_dim = out_dim * num_heads if concat else out_dim
        if in_dim != output_dim:
            self.res_proj = nn.Linear(in_dim, output_dim)
        else:
            self.res_proj = nn.Identity()
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x, adj, struct_bias=None, contact_prob=None, distance_matrix=None):
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.num_heads, self.out_dim)
        
        attn_src = (h * self.a_src).sum(dim=-1).transpose(1, 2)
        attn_dst = (h * self.a_dst).sum(dim=-1).transpose(1, 2)
        
        e = self.leaky_relu(attn_src.unsqueeze(-1) + attn_dst.unsqueeze(-2))
        
        # 添加结构偏置（RBF距离编码 - 学习型几何偏置）
        # 注意：struct_bias 已经包含了学习型距离偏置，由DistanceRBFEncoder计算
        if struct_bias is not None:
            e = e + struct_bias
        
        # 添加接触概率偏置（拓扑信息）
        if contact_prob is not None:
            # 接触图作为attention prior：空间邻近残基更容易attention
            # 系数2.0控制接触概率的影响强度
            contact_bias = 2.0 * contact_prob.unsqueeze(1)  # (B, 1, N, N)
            e = e + contact_bias
        
        # 注意：distance_matrix不再在这里处理
        # 距离信息已经通过struct_bias（学习型RBF编码）传入
        
        # 添加邻接掩码（只使用 adj 作为掩码，不重复加权）
        if adj is not None:
            adj_expand = adj.unsqueeze(1)  # (B, 1, N, N)
            mask = (adj_expand <= 1e-6)
            e = e.masked_fill(mask, -1e9)
        
        # 不使用 clamp，避免截断梯度流，让模型能够自我校正异常值
        attn = F.softmax(e, dim=-1)
            
        attn = self.dropout(attn)
        
        h_trans = h.transpose(1, 2)
        output = torch.matmul(attn, h_trans).transpose(1, 2)
        
        # 残差连接（标准做法）
        res = self.res_proj(x)
        
        if self.concat:
            output_flat = output.flatten(2)
            return self.norm(output_flat + res)
        else:
            output_mean = output.mean(dim=2)
            return self.norm(output_mean + res)


class GATEncoder(nn.Module):
    """
    多层GAT编码器 - 用于蛋白质接触图的编码
    GATLayer参数: (in_dim, head_dim, num_heads, dropout, concat)
    - head_dim: 每个注意力头的维度
    - 输出维度: head_dim * num_heads (concat=True) 或 head_dim (concat=False)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, num_heads=4, dropout=0.3):
        super().__init__()
        
        # hidden_dim = head_dim * num_heads
        # out_dim = head_dim (for final layer with concat=False)
        head_dim = hidden_dim // num_heads
        
        self.layers = nn.ModuleList()
        # 第一层: in_dim -> hidden_dim (concat=True, 输出 = head_dim * num_heads)
        self.layers.append(GATLayer(in_dim, head_dim, num_heads, dropout))
        
        for _ in range(num_layers - 2):
            # 中间层: hidden_dim -> hidden_dim
            self.layers.append(GATLayer(hidden_dim, head_dim, num_heads, dropout))
        
        if num_layers > 1:
            # 最后一层: hidden_dim -> out_dim (concat=False, 输出 = head_dim)
            self.layers.append(GATLayer(hidden_dim, out_dim, num_heads, dropout, concat=False))
        else:
            # 单层: in_dim -> out_dim
            self.layers.append(GATLayer(in_dim, out_dim, num_heads, dropout, concat=False))
        
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj, struct_bias=None, contact_prob=None, distance_matrix=None):
        h = x
        all_hidden = []
        
        # 收集所有层的输出，用于 jump knowledge
        for layer in self.layers:
            h = layer(h, adj, struct_bias, contact_prob, distance_matrix)
            h = self.dropout(h)
            all_hidden.append(h)
        
        # Jump Knowledge: 对所有层的输出取平均，缓解 over-smoothing
        # 尤其对于深层模型（如2层GAT + 3层interaction block = 5层message passing）
        if len(all_hidden) > 1:
            h = torch.mean(torch.stack(all_hidden), dim=0)
        else:
            h = all_hidden[0]
        
        return self.norm(h)


class ProteinContactGATEncoder(nn.Module):
    """
    蛋白质接触图GAT编码器
    - 输入：ESM-2残基嵌入(1280维)、接触概率矩阵、距离矩阵（可选）、额外嵌入（可选）
    - 残基为节点，接触概率作为边权重
    - 用多层GAT编码蛋白质内部的残基依赖
    - 支持RBF距离编码 → 3D空间结构注意力偏置
    - 支持多模态特征级联融合
    """
    
    def __init__(self, esm_dim=1280, extra_dim=0, hidden_dim=256, out_dim=256, 
                 num_layers=2, num_heads=4, dropout=0.3, use_rbf=True):
        super().__init__()
        
        # 多模态特征融合（ESM + 额外嵌入）
        if extra_dim > 0:
            self.feature_fusion = FeatureFusion(esm_dim, extra_dim, hidden_dim)
        else:
            self.feature_fusion = None
            self.esm_proj = nn.Sequential(
                nn.Linear(esm_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # RBF距离编码器（将距离矩阵转换为结构注意力偏置）
        self.use_rbf = use_rbf
        if use_rbf:
            self.rbf_encoder = DistanceRBFEncoder(num_rbf=16, num_heads=num_heads, low=0.0, high=20.0)
        
        # 保存 num_heads 供后续使用
        self.num_heads = num_heads
        
        self.gat_encoder = GATEncoder(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.global_pool = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU()
        )
    
    def forward(self, esm_emb, contact_prob, mask=None, distance_matrix=None, extra_emb=None, contact_bias=None, return_rbf_feat=False, has_valid_3d=None):
        """
        Args:
            esm_emb: (B, N_res, 1280) - ESM-2残基嵌入
            contact_prob: (B, N_res, N_res) - 残基接触概率矩阵
            mask: (B, N_res) - padding mask
            distance_matrix: (B, N_res, N_res) - 残基间距离矩阵（可选，用于RBF编码）
            extra_emb: (B, N_res, extra_dim) - 额外嵌入矩阵（可选，用于多模态融合）
            contact_bias: (B, N_res, N_res) - 接触概率映射的注意力偏置（[-inf, 0]）
            return_rbf_feat: 是否返回 RBF 特征（用于避免重复计算）
            has_valid_3d: (B,) - 样本级3D有效性标志（True表示该样本有有效3D特征）
        Returns:
            residue_emb: (B, N_res, out_dim) - 残基嵌入
            global_emb: (B, out_dim) - 全局嵌入
            rbf_feat: (B, N_res, N_res, num_rbf) - RBF 特征（仅当 return_rbf_feat=True 时返回）
        """
        # ⚠️ 关键修复：在特征变换前强制清零padding区域，防止噪声扩散
        # Padding位置的补零经过Linear变换后加上bias会变成非零噪声
        if mask is not None:
            esm_emb = esm_emb.masked_fill(mask.unsqueeze(-1), 0.0)
        
        # 多模态特征融合
        if self.feature_fusion is not None:
            h = self.feature_fusion(esm_emb, extra_emb)
        else:
            h = self.esm_proj(esm_emb)
        
        # 构建邻接矩阵（保留连续接触概率）
        adj = contact_prob
        if mask is not None:
            mask_expand = mask.unsqueeze(1) | mask.unsqueeze(2)
            adj = adj.masked_fill(mask_expand, 0.0)
        
        # ========== 🚀 关键修复：防止垃圾3D偏置污染注意力矩阵 ==========
        # 问题：当distance_matrix=1e9（无3D样本）时，RBF≈0，但Linear(bias=True)会产生constant bias
        # 导致所有无3D样本获得相同的假几何先验，诱导伪相关（spurious correlation）
        # 方案：直接mask掉无效样本的struct_bias，而不是在上游抹零
        struct_bias = None
        rbf_feat = None
        if self.use_rbf and distance_matrix is not None:
            # 传递mask给RBF编码器（内部自动处理padding污染问题）
            valid_mask = None if mask is None else ~mask
            if return_rbf_feat:
                struct_bias, rbf_feat = self.rbf_encoder(distance_matrix, mask=valid_mask, return_rbf_feat=True)
            else:
                struct_bias = self.rbf_encoder(distance_matrix, mask=valid_mask)
            
            # ✅ 核心修复：对无效3D样本的struct_bias置零
            # 避免：W·0 + b = constant bias 导致的伪相关
            if has_valid_3d is not None:
                # has_valid_3d: (B,) → 扩展为 (B, 1, 1, 1) 以便广播到 struct_bias
                valid_mask_3d = has_valid_3d.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                struct_bias = torch.where(valid_mask_3d, struct_bias, torch.zeros_like(struct_bias))
        
        # 合并接触概率偏置和结构偏置
        # contact_bias: 接触概率映射的注意力偏置（[-inf, 0]）
        # struct_bias: RBF距离编码的结构偏置
        combined_bias = None
        if struct_bias is not None and contact_bias is not None:
            # contact_bias 需要扩展到多头
            contact_bias_expanded = contact_bias.unsqueeze(1).repeat(1, struct_bias.size(1), 1, 1)
            combined_bias = struct_bias + contact_bias_expanded
        elif struct_bias is not None:
            combined_bias = struct_bias
        elif contact_bias is not None:
            # 如果只有 contact_bias，扩展到多头
            combined_bias = contact_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # GAT编码（支持结构注意力偏置和接触概率偏置）
        residue_emb = self.gat_encoder(h, adj, combined_bias)
        
        # 全局池化：使用safe_masked_pooling统一处理
        global_emb = safe_masked_pooling(residue_emb, mask)
        global_emb = self.global_pool(global_emb)
        
        if return_rbf_feat:
            return residue_emb, global_emb, rbf_feat
        else:
            return residue_emb, global_emb


class MultiPerspectiveDTI(nn.Module):
    """
    多视角多模态DTI模型 - 整合3D结构信息的完整架构：
    
    视角设计：
    ┌─────────────────────────────────────────────────────────────────┐
    │  1D ↔ 1D (序列内部交互)                                         │
    │    - 药物：Motif序列自注意力（序列级建模）                          │
    │    - 蛋白质：ESM-2序列特征（序列级建模）                           │
    ├─────────────────────────────────────────────────────────────────┤
    │  2D ↔ 2D (图结构内部交互)                                       │
    │    - 药物：Atom-level图注意力（原子间关系）                        │
    │    - 蛋白质：Contact图GAT（残基接触关系+3D空间距离）                │
    ├─────────────────────────────────────────────────────────────────┤
    │  3D ↔ 3D (空间结构建模)                                         │
    │    - 蛋白质：RBF距离编码（连续空间位置建模）                       │
    │    - 接触概率作为注意力偏置（3D-aware Attention）                 │
    │    - 残基暴露先验（表面残基优先关注）                             │
    ├─────────────────────────────────────────────────────────────────┤
    │  跨模态交互（多视角融合）                                        │
    │    - 药物：Motif序列 ↔ Atom图特征融合                            │
    │    - 蛋白质：ESM序列 ↔ Contact图特征融合（自适应门控）             │
    │    - 药物 ↔ 蛋白质：双向交叉注意力（3D结构感知）                   │
    ├─────────────────────────────────────────────────────────────────┤
    │  融合层：Bilinear Gating自适应融合 + 多层分类器                    │
    └─────────────────────────────────────────────────────────────────┘
    
    核心特性：
    1. 优雅降级：部分蛋白质缺少3D结构时自动回退到纯序列建模
    2. 自适应门控：动态平衡ESM序列特征与Contact图特征的融合比例
    3. 冷热启动分离：热启动保留残差连接，冷启动强制学习交互特征
    4. 结构感知注意力：利用接触概率和距离矩阵增强跨模态交互
    """
    
    def __init__(self,
                 # 药物编码器参数
                 motif_vocab_size=None,
                 atom_dim=256,
                 motif_dim=512,
                 drug_num_heads=4,
                 
                 # 蛋白质参数
                 protein_esm_dim=1280,
                 protein_hidden_dim=256,
                 protein_out_dim=256,
                 protein_gat_layers=2,
                 protein_num_heads=4,
                 contact_threshold=0.5,
                 use_rbf=True,
                 
                 # 交互参数
                 common_dim=256,
                 num_interaction_blocks=3,
                 interaction_heads=4,
                 dropout=0.3,
                 
                 # 热启动过拟合控制参数
                 split_type='cold_start',
                 warm_identity_drop=True):
        super().__init__()
        
        if motif_vocab_size is None:
            motif_vocab_size = 13000
        
        self.contact_threshold = contact_threshold
        self.common_dim = common_dim  # 保存公共维度，供forward使用
        self.split_type = split_type
        self.warm_identity_drop = warm_identity_drop
        self.static_drop_prob = 0.2
        
        # ========== 1. 药物编码模块 ==========
        # 1.1 2D图编码 - Atom图注意力（原子级结构）
        self.atom_encoder = AtomEncoder(
            num_layers=2,
            d_model=atom_dim,
            num_heads=drug_num_heads,
            dff=motif_dim,
            rate=dropout
        )
        
        # 1.2 1D序列编码 - Motif序列Transformer
        self.motif_encoder = MotifEncoder(
            num_layers=2,
            input_vocab_size=motif_vocab_size,
            d_model=motif_dim,
            num_heads=drug_num_heads,
            dff=motif_dim,
            rate=dropout
        )
        
        # 1.3 药物1D↔2D融合（序列↔结构交互）
        self.drug_perspective_fusion = nn.Sequential(
            nn.Linear(motif_dim * 2, common_dim),  # 输出维度改为 common_dim，与蛋白质侧一致
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 可学习的Motif全局特征融合门控：让模型自动决定CLS token和池化特征的融合比例
        self.motif_global_gate = nn.Sequential(
            nn.Linear(motif_dim * 2, motif_dim),
            nn.LayerNorm(motif_dim),
            nn.GELU(),
            nn.Linear(motif_dim, 1),
            nn.Sigmoid()
        )
        
        self.drug_proj = nn.Sequential(
            nn.Linear(motif_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ========== 关键修复：可学习的Atom CLS Token ==========
        # 解决atom branch的global token是手工统计量（mean pooling）的问题
        # 让interaction自己学global，比pooling强很多
        self.atom_cls = nn.Parameter(torch.randn(1, 1, common_dim))
        nn.init.xavier_uniform_(self.atom_cls)
        
        # ========== 关键修复：Atom特征投影层 ==========
        # 将atom_emb投影到common_dim，用于独立的2D交互
        self.atom_proj = nn.Sequential(
            nn.Linear(motif_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ========== 关键修复：Final Interaction融合层 ==========
        # 融合1D和2D交互结果，实现真正的multi-view interaction
        # 关键修复：删掉LayerNorm，避免有3D和无3D时的分布漂移
        # 确保有3D和无3D时都走同一条网络，classifier看到的是同一manifold
        # 关键设计：Late Normalization，避免把 interaction magnitude 洗平
        # 1D + 2D chemistry complementarity 会被削弱如果LN放太前
        self.final_drug_fusion = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.final_protein_fusion = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ========== 关键修复：Late Normalization ==========
        # 在 classifier 前添加 LayerNorm，而不是在 fusion 中
        # 这样可以保留 interaction magnitude，同时稳定特征尺度
        # 特别适合 DTI 的 OOD scaffold generalization 场景
        self.drug_interaction_norm = nn.LayerNorm(common_dim)
        self.protein_interaction_norm = nn.LayerNorm(common_dim)
        
        # ========== 2. 蛋白质编码模块 ==========
        # 2.1 1D序列编码 - ESM特征投影
        self.protein_seq_proj = nn.Sequential(
            nn.Linear(protein_esm_dim, protein_hidden_dim),
            nn.LayerNorm(protein_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2.2 2D图编码 - 接触图GAT
        self.protein_contact_encoder = ProteinContactGATEncoder(
            esm_dim=protein_esm_dim,
            extra_dim=0,
            hidden_dim=protein_hidden_dim,
            out_dim=protein_out_dim,
            num_layers=protein_gat_layers,
            num_heads=protein_num_heads,
            dropout=dropout,
            use_rbf=use_rbf
        )
        
        # 2.3 蛋白质1D↔2D融合（序列↔结构交互）
        self.protein_perspective_fusion = nn.Sequential(
            nn.Linear(protein_out_dim * 2, common_dim),  # 输出维度改为 common_dim，与药物侧一致
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_out_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 新增：protein_fused 投影层，用于将全局融合特征扩展后加回主路径
        self.protein_fused_proj = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 新增：自适应门控融合层（防止全局ESM信号淹没图拓扑特征）
        # 使用可学习的门控权重，确保两模态特征协同发挥作用
        # ========== 第二优先级：Gate Temperature (T=2.0) ==========
        # 移除Sigmoid，改为在forward中手动应用温度缩放
        # T=2.0给Sigmoid踩刹车，强迫模型在整个训练周期内为3D模块保留梯度生命线
        # gate = sigmoid(logits / T), T>1使输出更接近0.5，避免早期坍塌到0或1
        self.protein_gate = nn.Sequential(
            nn.Linear(common_dim * 2, common_dim),
            nn.LayerNorm(common_dim),
            nn.GELU(),
            nn.Linear(common_dim, 1)
            # Sigmoid移除，改为forward中带温度的sigmoid
        )
        # 门控温度：T=2.0，给Sigmoid踩刹车
        # T>1 → 输出更平滑，避免gate早期饱和到0或1
        # 这为3D模块保留了一条"梯度生命线"
        self.gate_temperature = 2.0
        
        # ========== 🚀 修复：防止伪3D训练 shortcut ==========
        # 可学习的 null embedding，用于没有3D结构的样本
        # 替代直接置零，防止模型学到 "zero = special signal" shortcut
        self.no_structure_token = nn.Embedding(1, common_dim)
        nn.init.xavier_uniform_(self.no_structure_token.weight)
        
        # ========== 3. 跨模态交互模块（药物↔蛋白质） ==========
        # 修复：1D和2D分支使用独立的交互网络，避免梯度冲突
        # 1D branch（Motif ↔ ESM）：偏向序列上下文和motif语义
        # 2D branch（Atom Graph ↔ Contact Graph）：偏向拓扑结构和局部几何
        # 关键设计：避免参数翻倍导致的scaffold memorization
        # 1D branch：2 blocks（处理序列语义）
        # 2D branch：1 block（已有AtomEncoder和ProteinGraphEncoder的message passing）
        self.interaction_1d = DTIMultiScaleInteraction(
            hidden_dim=common_dim,
            num_heads=interaction_heads,
            num_blocks=2,  # 固定2 blocks，避免过深
            dropout=dropout
        )
        
        self.interaction_2d = DTIMultiScaleInteraction(
            hidden_dim=common_dim,
            num_heads=interaction_heads,
            num_blocks=1,  # 固定1 block，避免过平滑
            dropout=dropout
        )
        
        # ========== 4. 特征对齐投影 ==========
        # 注意：bilinear_gate 和 protein_proj_to_common 已删除（dead module）
        
        # ========== Residual Projector：对齐静态特征与交互特征的分布 ==========
        # 解决 drug_fused/protein_fused 与 drug_interaction_global/protein_interaction_global 的分布不匹配问题
        # 交互特征经过 CNN+attention+LayerNorm+FFN，而静态特征没有，导致 scale mismatch
        self.static_proj = nn.Sequential(
            nn.Linear(common_dim, common_dim),
            nn.LayerNorm(common_dim)
        )
        
        # ========== 高阶交互特征归一化 ==========
        # 解决 d*p (重尾分布) 和 |d-p| (均值偏移) 导致的梯度失衡问题
        # 分类头第一层看到的特征方差差异巨大，梯度会偏向高方差特征
        # 对乘性/绝对差特征单独做 LayerNorm，稳定分布
        self.mul_norm = nn.LayerNorm(common_dim)
        self.abs_norm = nn.LayerNorm(common_dim)
        
        # ========== 最终特征统一 LayerNorm（关键修复） ==========
        # 虽然 d_mul 和 d_abs 做了单独的 Norm，但 d 和 p 没有
        # 导致分类器看到的特征方差差异巨大：d/p 方差≈4-5，d_mul/d_abs 方差≈1
        # 统一做 LayerNorm 可以平衡各特征的重要性
        self.final_feature_norm = nn.LayerNorm(common_dim * 4)
        
        # ========== 5. 分类器（恢复三层结构，降低 dropout） ==========
        # 输入维度：4*common_dim（[d, p, d*p, |d-p|]）
        # 高阶交互特征需要足够的容量来学习非线性决策边界
        # 使用较低的 dropout 避免信息丢失过多
        self.classifier = nn.Sequential(
            nn.Linear(common_dim * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, 1)
        )
        
        # ========== 6. 3D几何置信度门控 ==========
        # 根据药物和蛋白质的3D信息强度，动态调节残差融合
        # 升级：从简单的 mean(contact_prob) 升级到 MLP(global_3d_embedding)
        # 输入: drug_3d_embedding + protein_3d_embedding
        # 修复：输出逐通道的3D置信度（common_dim），而非单一标量
        self.geo_gate_mlp = nn.Sequential(
            nn.Linear(common_dim * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, common_dim),
            nn.Sigmoid()
        )

        # ========== 7. 1D序列置信度门控 ==========
        # 自适应测量1D序列特征(ESM/Motif)的置信度
        # 当3D结构退化时，1D gate应上升以补偿
        self.drug_1d_gate_mlp = nn.Sequential(
            nn.Linear(common_dim * 2, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, common_dim),
            nn.Sigmoid()
        )
        self.protein_1d_gate_mlp = nn.Sequential(
            nn.Linear(common_dim * 2, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, common_dim),
            nn.Sigmoid()
        )

        # ========== 8. 模态融合权重门控 ==========
        # 学习1D vs 2D/3D的整体融合权重
        # 用于分析模型在不同结构条件下的模态依赖
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(common_dim * 4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                atom_feat, atom_adj, atom_dist, atom_match, sum_atoms,
                motif_seq, motif_adj, motif_dist,
                protein_feat, protein_mask=None,
                contact_prob=None, distance_matrix=None,
                drug_contact_prob=None, drug_distance=None,
                has_valid_protein_3d=None, has_valid_drug_3d=None,
                atom_mask=None, motif_mask=None,
                drug_atom_raw=None, atom_match_matrix=None,
                return_intermediate=False):
        """
        多视角前向传播：
        1. 药物：Atom图(2D) + Motif序列(1D) → 融合
        2. 蛋白质：ESM序列(1D) + Contact图(2D) → 融合
        3. 跨模态交互：药物↔蛋白质双向注意力
        4. 融合所有信息做预测

        Args:
            atom_mask: (B, N_atoms) True for padding
            motif_mask: (B, N_motifs) True for padding
            drug_contact_prob: (B, N_atoms, N_atoms) 药物接触概率矩阵（可选）
            drug_distance: (B, N_atoms, N_atoms) 药物3D距离矩阵（可选）
            drug_atom_raw: (B, N_atom, 61) 原始药物原子特征（携带化学性质，用于原子到Motif映射）
            atom_match_matrix: (B, N_motif, N_atom) 原子到Motif的拓扑归属矩阵（用于约束原子到Motif的映射）
        """
        B = atom_feat.size(0)

        # ========== 1. 药物多视角编码 ==========
        # 1.1 2D视角：Atom图编码（加入药物3D几何偏置）
        atom_emb = self.atom_encoder(
            atom_feat, atom_adj, atom_dist, atom_match, sum_atoms,
            drug_contact_prob=drug_contact_prob, drug_distance=drug_distance
        )  # (B, Nm, motif_dim)

        # 1.2 1D视角：Motif序列编码
        motif_emb = self.motif_encoder(motif_seq, atom_level_features=atom_emb,
                                       adjoin_matrix=motif_adj, dist_matrix=motif_dist)  # (B, Nm+1, motif_dim)

        # ========== 1.3 1D↔2D融合：序列级与结构级特征融合 ==========
        # ========== Topology-Aware Masking：基于物理拓扑的原子有效性判断 ==========
        # 避免使用特征值启发式 (sum(abs(atom_emb)) == 0)，因为：
        # 1. atom_emb 经过 GELU + LayerNorm + dropout 后，真实特征可能接近 0
        # 2. 冷启动时 motif 数量少，一个误判会导致严重的 representation drift
        # 
        # 正确做法：基于 atom_match_matrix（原子到Motif的拓扑归属矩阵）生成 mask
        # valid_atom = (atom_match_matrix.sum(-1) > 0) 表示原子至少属于一个 motif
        if atom_match_matrix is not None:
            # 基于拓扑归属矩阵判断motif有效性：至少包含一个原子的motif是有效的
            # atom_match_matrix: (B, N_motifs, N_atoms)
            # sum(dim=-1): (B, N_motifs) - 每个motif包含多少个原子
            valid_motif = (atom_match_matrix.sum(dim=-1) > 0)  # (B, N_motifs) - True 表示有效motif
            atom_mask_for_pooling = ~valid_motif  # (B, N_motifs) - True 表示 padding
        elif motif_mask is not None:
            # 降级方案：使用 motif_mask 裁剪后的形状（排除 CLS token）
            atom_mask_for_pooling = motif_mask[:, 1:]
        else:
            atom_mask_for_pooling = None

        # 使用拓扑真值生成的 mask，完美避免 false padding 问题
        atom_rep = safe_masked_pooling(atom_emb, atom_mask_for_pooling)  # 原子级全局特征（修复padding泄漏）
        
        # 修复Padding泄漏：采用可学习门控的双流融合策略
        motif_valid_features = motif_emb[:, 1:, :]  # 提取真实的Motif向量（排除CLS token）
        motif_valid_mask = motif_mask[:, 1:] if motif_mask is not None else None
        # 计算不含Padding污染的真实分子语义平均池化
        motif_pooled = safe_masked_pooling(motif_valid_features, motif_valid_mask)
        
        # 使用可学习门控自动决定CLS token和池化特征的融合比例
        cls_feat = motif_emb[:, 0, :]
        gate_input = torch.cat([cls_feat, motif_pooled], dim=-1)
        gate = self.motif_global_gate(gate_input)  # (B, 1)
        # gate=1: 更信CLS; gate=0: 更信pooling
        motif_global_clean = gate * cls_feat + (1 - gate) * motif_pooled
        
        drug_fused = self.drug_perspective_fusion(torch.cat([atom_rep, motif_global_clean], dim=-1))  # (B, motif_dim)
        
        # 投影到公共维度
        drug_h = self.drug_proj(motif_emb)  # (B, Nm+1, common_dim)
        
        # ========== 2. 蛋白质多视角编码 ==========
        # 2.1 1D视角：ESM序列特征
        protein_seq_h = self.protein_seq_proj(protein_feat)  # (B, Np, hidden_dim)
        
        # 初始化 RBF 特征（用于复用，避免重复计算）
        protein_rbf_feat = None
        
        # 2.2 2D视角：接触图GAT编码（优雅降级策略）
        if contact_prob is None:
            # 【优雅降级】：没有接触图/3D结构时，两条通路都使用纯序列特征
            protein_seq_h_proj = self.protein_proj(protein_seq_h)  # (B, Np, common_dim)
            protein_graph_h = protein_seq_h_proj  # 无3D时回退：两条通路共享序列特征
            protein_fused = safe_masked_pooling(protein_seq_h_proj, protein_mask)
        else:
            # 有3D结构时，激活完整的多层GAT拓扑融合逻辑
            # ========== 🚀 关键优化：复用 RBF 特征，避免重复计算 ==========
            # 问题：distance_matrix 在 protein_contact_encoder 和 interaction block 中各计算一次 RBF
            # 方案：在 protein_contact_encoder 中返回 rbf_feat，传递给 interaction block 复用
            # ✅ 同时传递 has_valid_protein_3d，用于mask掉无效3D样本的struct_bias
            protein_graph_h, _, protein_rbf_feat = self.protein_contact_encoder(
                protein_feat, contact_prob, protein_mask, distance_matrix, 
                return_rbf_feat=True, has_valid_3d=has_valid_protein_3d
            )  # (B, Np, out_dim), _, (B, Np, Np, num_rbf)
            
            # 投影到公共维度
            protein_graph_h = self.protein_proj(protein_graph_h)  # (B, Np, common_dim)
            
            # ========== 🚀 Late Fusion：保留两条独立通路 ==========
            # ESM 通路：protein_seq_h_proj — 完整的残基级序列特征
            # 3D 通路：protein_graph_h — 完整的残基级图结构+3D几何特征
            # 不再进行残基级融合！避免 ESM 与 3D 的特征空间污染
            # 两条通路各自走自己的 interaction block，在分类器前拼接全球特征
            protein_seq_h_proj = self.protein_proj(protein_seq_h)  # (B, Np, common_dim)
            
            # ========== 🚀 关键优化：轻量ESM Dropout让Graph模块被迫成长 ==========
            # 训练时：ESM偶尔失明，Graph被迫工作
            # 验证时：ESM恢复，两边都能利用
            # ========== 第三优先级：ESM Modality Dropout (p=0.15) ==========
            # 15%的概率恰到好处：既不会摧毁ESM的流形空间，又能时不时把3D模块"踹上主舞台"锻炼
            # 这是一种极佳的对抗训练
            protein_seq_h_proj = F.dropout(
                protein_seq_h_proj,
                p=0.15,  # 15%概率失活ESM，迫使Graph模块成长
                training=self.training
            )
            
            # ========== 🚀 关键修复：3D优雅降级 ==========
            # 无3D样本时，3D通路回退为 ESM sequence + missing signal
            # 保持分布一致，避免模型学到 "zero = special signal" shortcut
            if has_valid_protein_3d is not None and has_valid_protein_3d.any():
                valid_mask = has_valid_protein_3d.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
                no_struct = self.no_structure_token.weight[0].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
                fallback = protein_seq_h_proj + no_struct  # 序列先验 + 缺失信号
                protein_graph_h = torch.where(valid_mask, protein_graph_h, fallback)
            else:
                no_struct = self.no_structure_token.weight[0].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
                protein_graph_h = protein_seq_h_proj + no_struct
            # 注意：这里不再计算 fused protein_h! 两条通路保持独立直到分类器
        
        # ========== 2.3 1D↔2D融合（在fallback之后，确保global与node对齐）==========
        # ✅ 重要：protein_graph_global 从处理后的 protein_graph_h 重新pool
        # 避免：global feature 和 node feature distribution mismatch
        protein_graph_global = safe_masked_pooling(protein_graph_h, protein_mask)
        protein_seq_global = safe_masked_pooling(protein_seq_h, protein_mask)
        protein_fused = self.protein_perspective_fusion(
            torch.cat([protein_seq_global, protein_graph_global], dim=-1)
        )  # (B, common_dim)
        
        # ========== 3. 跨模态交互（药物↔蛋白质双向注意力） ==========
        # ========== 关键修复：真正的Multi-View Interaction ==========
        # 问题：当前架构只使用1D interaction结果（drug_h, protein_h）
        # 导致2D branch（atom_interact, protein_graph_interact）被边缘化
        # 解决方案：分别进行1D和2D交互，然后融合
        
        # ========== 3.1 准备1D交互输入 ==========
        # drug_h: (B, Nm+1, common_dim) - Motif序列（1D视角）
        # protein_seq_h: (B, Np, hidden_dim) - ESM序列（1D视角）
        
        # ========== 3.2 准备2D交互输入 ==========
        # Atom分支：添加可学习的CLS token
        atom_h_proj = self.atom_proj(atom_emb)  # (B, Nm, common_dim)
        atom_cls = self.atom_cls.expand(atom_h_proj.size(0), -1, -1)  # (B, 1, common_dim)
        atom_h_with_global = torch.cat([atom_cls, atom_h_proj], dim=1)  # (B, Nm+1, common_dim)
        
        # ========== 3.3 1D交互流（Motif ↔ ESM 序列级交互）==========
        # 【修改点】：彻底删除激进的 35% token-level dropout，保留完整的序列语义
        drug_h_1d = drug_h
        protein_seq_h_1d = protein_seq_h
        
        # ========== 🚀 P0 修复：Padding Mask 对齐 ==========
        # motif_mask 形状是 (B, Nm)，不包含 CLS token
        # drug_h_1d 形状是 (B, Nm+1)，包含 CLS token
        # 必须确保 mask 长度与特征长度一致
        drug_mask_1d = None
        if motif_mask is not None:
            # Fail-Fast 策略：只允许两种合法情况
            # Case A: motif_mask 不含 CLS，长度为 Nm
            # Case B: motif_mask 已含 CLS，长度为 Nm+1
            if motif_mask.size(1) == drug_h_1d.size(1) - 1:
                # Case A: 补 CLS
                global_mask = torch.zeros(motif_mask.size(0), 1, dtype=torch.bool, device=motif_mask.device)
                drug_mask_1d = torch.cat([global_mask, motif_mask], dim=1)
            elif motif_mask.size(1) == drug_h_1d.size(1):
                # Case B: 已含 CLS，直接使用
                drug_mask_1d = motif_mask
            else:
                # Case C: 不合法，立即失败
                raise ValueError(
                    f"motif_mask length {motif_mask.size(1)} must be either "
                    f"{drug_h_1d.size(1)-1} (no CLS) or {drug_h_1d.size(1)} (with CLS), "
                    f"but drug_h_1d has length {drug_h_1d.size(1)}"
                )
        
        # ========== Sanity Check：确保最终 mask 与特征长度一致 ==========
        if drug_mask_1d is not None:
            assert drug_mask_1d.shape[1] == drug_h_1d.shape[1], \
                f"drug_mask_1d length {drug_mask_1d.shape[1]} must match drug_h_1d length {drug_h_1d.shape[1]}"
            
        if return_intermediate:
            drug_final_1d, protein_final_1d, _, attn_1d = self.interaction_1d(
                drug_feat=drug_h_1d, protein_feat=protein_seq_h_1d,
                protein_mask=protein_mask, drug_mask=drug_mask_1d,
                contact_prob=None, distance_matrix=None,
                drug_distance_matrix=None, drug_atom_raw=None, atom_match=None,
                return_attn_weights=True
            )
        else:
            drug_final_1d, protein_final_1d, _ = self.interaction_1d(
                drug_feat=drug_h_1d, protein_feat=protein_seq_h_1d,
                protein_mask=protein_mask, drug_mask=drug_mask_1d,
                contact_prob=None, distance_matrix=None,
                drug_distance_matrix=None, drug_atom_raw=None, atom_match=None
            )
        
        # ========== 3.4 🚀 2D交互流（Atom 图 ↔ Contact 图拓扑交互）==========
        # 【修改点】：同样删除 2D 视角的 token-level dropout，保护分子拓扑与 3D 几何特征不被撕裂
        atom_h_2d = atom_h_with_global
        # 🚀 关键修复：使用纯 Graph+3D 特征（而非融合特征）进行2D交互
        # Late Fusion 策略：ESM 和 Graph+3D 作为独立通路，互不干扰
        protein_graph_h_2d = protein_graph_h
        
        # 🚀 关键修复：drug_mask 必须与 drug_feat 的实际长度匹配
        # atom_h_with_global 是 motif 级别的特征（Nm+1），所以应该用 motif_mask
        if motif_mask is not None:
            global_mask = torch.zeros(motif_mask.size(0), 1, dtype=torch.bool, device=motif_mask.device)
            drug_mask_2d = torch.cat([global_mask, motif_mask], dim=1)  # (B, Nm+1) - 与 atom_h_2d 匹配
        else:
            drug_mask_2d = None
            
        # 🚀 样本级 3D 矩阵安全性检查：防止空矩阵引发的零距离 RBF 爆炸
        # 仍然保留这个检查，但对2D网络本身的运行不再构成条件限制
        drug_distance_safe = drug_distance
        distance_matrix_safe = distance_matrix
        if drug_distance is not None and distance_matrix is not None:
            drug_3d_valid = drug_distance.abs().sum(dim=(1, 2)) > 1e-4
            prot_3d_valid = distance_matrix.abs().sum(dim=(1, 2)) > 1e-4
            sample_3d_valid = drug_3d_valid & prot_3d_valid
            
            if not sample_3d_valid.all():
                invalid_mask = ~sample_3d_valid
                drug_distance_safe = drug_distance.clone()
                distance_matrix_safe = distance_matrix.clone()
                drug_distance_safe[invalid_mask] = 999.0
                distance_matrix_safe[invalid_mask] = 999.0
        
        # 核心算子通电：当数据源中 3D 矩阵传入为 None 时，底层注意力机制会自动退化为纯 2D 拓扑交互
        if return_intermediate:
            drug_final_2d, protein_final_2d, _, attn_2d = self.interaction_2d(
                drug_feat=atom_h_2d,
                protein_feat=protein_graph_h_2d,
                protein_mask=protein_mask,
                drug_mask=drug_mask_2d,
                contact_prob=contact_prob,
                distance_matrix=distance_matrix_safe,
                drug_distance_matrix=drug_distance_safe,
                drug_atom_raw=drug_atom_raw,
                atom_match=atom_match_matrix,
                protein_rbf_feat=protein_rbf_feat,
                return_attn_weights=True
            )
        else:
            drug_final_2d, protein_final_2d, _ = self.interaction_2d(
                drug_feat=atom_h_2d,
                protein_feat=protein_graph_h_2d,
                protein_mask=protein_mask,
                drug_mask=drug_mask_2d,
                contact_prob=contact_prob,
                distance_matrix=distance_matrix_safe,
                drug_distance_matrix=drug_distance_safe,
                drug_atom_raw=drug_atom_raw,
                atom_match=atom_match_matrix,
                protein_rbf_feat=protein_rbf_feat  # 传递预计算的 RBF 特征，避免重复计算
            )
        
        # ========== 3.5 🚀 终极晚期特征融合（1D 与 2D 并行对齐）==========
        drug_1d_global = drug_final_1d[:, 0, :]
        protein_1d_global = safe_masked_pooling(protein_final_1d, protein_mask)
        
        drug_2d_global = drug_final_2d[:, 0, :]
        protein_2d_global = safe_masked_pooling(protein_final_2d, protein_mask)
        
        # ========== 3.5 🚀 终极晚期特征融合（非对称特征 Dropout）==========
        # 核心策略：稳定地削弱 1D dominance，保护弱势的 2D/3D 分支
        # - 1D 分支：warm_start=0.1, cold_start=0.15（序列特征太强，需要压制）
        # - 2D/3D 分支：不使用 dropout（信号已经很弱，避免 signal starvation）
        if self.training:
            # 根据训练类型设置不同的 1D dropout，将 cold_start_protein 加入强正则
            dropout_1d = 0.15 if self.split_type in ['cold_start', 'cold_start_drug', 'cold_start_protein'] else 0.1
            
            # ESM 和 Motif 序列特征很强，施加适度的 Dropout
            drug_1d_global = F.dropout(drug_1d_global, p=dropout_1d, training=self.training)
            protein_1d_global = F.dropout(protein_1d_global, p=dropout_1d, training=self.training)
            
            # 2D/3D 分支已修复，添加适度 dropout 防止过拟合
            drug_2d_global = F.dropout(drug_2d_global, p=0.15, training=self.training)
            protein_2d_global = F.dropout(protein_2d_global, p=0.15, training=self.training)
        
        # 贯彻你的"退回到 1D+2D"核心思想：取消条件分支，所有样本共享同一融合矩阵空间
        # 极其平滑地处理了 BindingDB 样本间 3D 信息不均衡的世纪难题
        drug_interaction_global = self.final_drug_fusion(
            torch.cat([drug_1d_global, drug_2d_global], dim=-1)
        )
        protein_interaction_global = self.final_protein_fusion(
            torch.cat([protein_1d_global, protein_2d_global], dim=-1)
        )
        
        # ========== 关键修复：Late Normalization ==========
        # 在 classifier 前添加 LayerNorm，保留 interaction magnitude
        # 避免 1D + 2D chemistry complementarity 被 LN 洗平
        drug_interaction_global = self.drug_interaction_norm(drug_interaction_global)
        protein_interaction_global = self.protein_interaction_norm(protein_interaction_global)
        
        # ========== 计算几何置信度（升级：使用全局3D嵌入而非简单mean）==========
        # 从3D嵌入计算几何结构的置信度
        # drug_fused 和 protein_fused 已经包含了3D信息（通过atom_encoder中的distance_bias）
        
        # 药物3D嵌入：使用药物融合特征（已包含3D几何信息）
        drug_geo_emb = drug_fused  # (B, common_dim)
        
        # 蛋白质3D嵌入：使用蛋白质融合特征（已包含3D几何信息）
        protein_geo_emb = protein_fused  # (B, common_dim)
        
        # ========== 🚀 修复：Geo Gate 不再置零，保留药物/蛋白质身份信息 ==========
        # 关键原则：没有3D ≠ 没有化学身份
        # 不同药物（如Aspirin和Gefitinib）即使都没有3D，也应该有不同的geo_emb
        # 让网络自己学习：哪些representation pattern意味着unreliable geometry
        # 而不是强行置零导致所有无3D样本看起来都一样
        
        # 保留原始的 fused feature，不做任何masked_fill
        # drug_geo_emb = drug_fused （已在上方定义）
        # protein_geo_emb = protein_fused （已在上方定义）
        
        # 注意：不需要对geo_emb做任何masking，geo_gate_mlp会自动学习
        # 不同样本的geometry confidence estimation
        
        # 生成几何置信度门控：MLP([drug_geo_emb, protein_geo_emb])
        geo_input = torch.cat([drug_geo_emb, protein_geo_emb], dim=-1)
        geo_gate = self.geo_gate_mlp(geo_input)

        # ========== 新增：1D序列置信度门控 ==========
        # drug_1d_gate: 衡量1D药物特征(Motif)的置信度
        drug_1d_gate_input = torch.cat([drug_1d_global, drug_fused], dim=-1)
        drug_1d_gate = self.drug_1d_gate_mlp(drug_1d_gate_input)

        # protein_1d_gate: 衡量1D蛋白质特征(ESM)的置信度
        protein_1d_gate_input = torch.cat([protein_1d_global, protein_fused], dim=-1)
        protein_1d_gate = self.protein_1d_gate_mlp(protein_1d_gate_input)

        # ========== 新增：模态融合权重门控 ==========
        # 学习1D vs 2D/3D的整体融合权重
        fusion_input = torch.cat([drug_1d_global, protein_1d_global, drug_2d_global, protein_2d_global], dim=-1)
        fusion_weight = self.fusion_gate_mlp(fusion_input)

        # ========== Residual Projector：对齐静态特征与交互特征的分布 ==========
        # 交互特征经过 CNN+attention+LayerNorm+FFN，而静态特征没有，导致 scale mismatch
        # 使用投影层将静态特征对齐到交互特征的分布空间
        drug_fused_proj = self.static_proj(drug_fused)
        protein_fused_proj = self.static_proj(protein_fused)
        
        # 根据 split_type 动态调整残差融合强度
        if self.split_type == 'warm_start':
            # ========== warm_start：保留适度 identity，使用小权重 + detach ==========
            # warm split 下训练见过类似药物，适度 identity 有价值
            static_alpha = 0.15
            
            # 移除 detach，允许梯度回传到静态特征
            drug_static_residual = static_alpha * geo_gate * drug_fused_proj
            protein_static_residual = static_alpha * geo_gate * protein_fused_proj
            
            drug_combined = drug_interaction_global + drug_static_residual
            protein_combined = protein_interaction_global + protein_static_residual
        else:
            if self.split_type == 'cold_start_protein':
                # ========== cold_start_protein：蛋白质冷启动，药物是热的 ==========
                # 药物侧大多见过，权重略高，同样开启 Dropout 防御
                cold_drug_alpha = 0.08
                drug_res_dropped = F.dropout(drug_fused_proj, p=0.2, training=self.training)
                drug_identity_residual = cold_drug_alpha * geo_gate * drug_res_dropped
                drug_combined = drug_interaction_global + drug_identity_residual
                
                # 蛋白质侧是冷的，微残差 (Micro-Residual) + 随机残差 (Stochastic Residual)
                # 1. 降低权重到 0.05：极大地削弱锚点信号，不给分类器产生依赖的机会
                # 2. 引入 F.dropout(p=0.2)：让残差传过来的特征随机缺失，迫使分类器必须去交叉验证 interaction 网络传来的主特征
                cold_protein_alpha = 0.05
                protein_res_dropped = F.dropout(protein_fused_proj, p=0.2, training=self.training)
                protein_static_residual = cold_protein_alpha * geo_gate * protein_res_dropped
                protein_combined = protein_interaction_global + protein_static_residual
            else:
                # ========== cold_start_drug：药物冷启动，蛋白质是热的 ==========
                # 防御过拟合的终极形态：
                # 1. 降低权重到 0.05：极大地削弱锚点信号，不给分类器产生依赖的机会。
                # 2. 引入 F.dropout(p=0.2)：让残差传过来的特征随机缺失，迫使分类器必须去交叉验证 interaction 网络传来的主特征。
                
                cold_drug_alpha = 0.05  # 微残差，提供方向但不主导
                drug_res_dropped = F.dropout(drug_fused_proj, p=0.2, training=self.training)
                drug_identity_residual = cold_drug_alpha * geo_gate * drug_res_dropped
                drug_combined = drug_interaction_global + drug_identity_residual
                
                # 蛋白质侧大多见过，权重略高，同样开启 Dropout 防御
                cold_protein_alpha = 0.08
                protein_res_dropped = F.dropout(protein_fused_proj, p=0.2, training=self.training)
                protein_static_residual = cold_protein_alpha * geo_gate * protein_res_dropped
                protein_combined = protein_interaction_global + protein_static_residual
        
        # ========== 🚀 修复 3：移除没有梯度的无效前向传播 ==========
        # 既然发现直接使用 concat[d, p, d*p, |d-p|] 效果更好，坚决切断无效计算
        # _, _ = self.bilinear_gate(drug_combined, protein_combined)  # 删除无效计算
        
        # ========== 添加 [d,p,d*p,|d-p|] 特征（增强交互归纳偏置）==========
        # 从原始的 drug_combined 和 protein_combined 计算，保持语义纯净
        # 避免从 bilinear_gate 输出中拆分（已被非线性混合）
        d_pure = drug_combined
        p_pure = protein_combined
        d_mul = self.mul_norm(d_pure * p_pure)  # 归一化重尾分布
        d_abs = self.abs_norm(torch.abs(d_pure - p_pure))  # 归一化均值偏移
        
        # 经典稳定组合：[d, p, d*p, |d-p|]
        # 不拼入 gate_combined，避免 redundancy 和 overfit
        combined = torch.cat([d_pure, p_pure, d_mul, d_abs], dim=-1)  # (B, 4*common_dim)
        
        # ========== 统一 LayerNorm（关键修复） ==========
        # 平衡各特征的重要性，避免分类器优先利用高方差特征
        combined = self.final_feature_norm(combined)
        
        output = self.classifier(combined)
        
        if return_intermediate:
            return {
                'prediction': output,
                'drug_fused': drug_fused,
                'protein_fused': protein_fused,
                'drug_interaction_global': drug_interaction_global,
                'protein_interaction_global': protein_interaction_global,
                'geo_gate': geo_gate,
                'drug_h': drug_h,
                'protein_seq_h': protein_seq_h,
                'protein_graph_h': protein_graph_h,
                'atom_emb': atom_emb,
                'motif_emb': motif_emb,
                'drug_final_1d': drug_final_1d,
                'protein_final_1d': protein_final_1d,
                'drug_final_2d': drug_final_2d,
                'protein_final_2d': protein_final_2d,
                'cross_attention_1d': attn_1d,
                'cross_attention_2d': attn_2d,
                'geo_gate': geo_gate,
                'drug_1d_gate': drug_1d_gate,
                'protein_1d_gate': protein_1d_gate,
                'fusion_weight': fusion_weight,
            }
        
        # 返回融合后的交互结果作为 drug_final 和 protein_final
        # drug_interaction_global 和 protein_interaction_global 是融合了1D和2D交互的全局特征
        # 同时返回门控值用于监控训练过程
        # gate_weight 已弃用（Late Fusion 不再使用残基级门控融合）
        return output, drug_fused, protein_fused, drug_interaction_global, protein_interaction_global, geo_gate, None

#
