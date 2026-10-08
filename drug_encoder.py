import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def rescale_distance_matrix(w):
    constant_value = 1.0
    return (constant_value + math.exp(constant_value)) / (constant_value + torch.exp(constant_value - w))


class DistanceRBFEncoder(nn.Module):
    """药物3D距离RBF编码器 - 将连续距离转换为注意力偏置"""
    def __init__(self, num_rbf=16, num_heads=4, low=0.0, high=6.0):  # 药物范围改为6Å
        super().__init__()
        self.num_rbf = num_rbf
        self.num_heads = num_heads
        self.high = high
        
        self.register_buffer("offsets", torch.linspace(low, high, num_rbf))
        self.register_buffer("widths", torch.tensor((high - low) / num_rbf if num_rbf > 1 else 1.0))
        
        # 学习型距离偏置MLP
        self.distance_mlp = nn.Sequential(
            nn.Linear(num_rbf, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, num_heads),
            nn.Tanh()
        )

    def forward(self, dist_matrix, mask=None):
        """
        Args:
            dist_matrix: (B, N, N) - 距离矩阵
            mask: (B, N) 或 (B, N, N) - padding mask（可选）
        """
        if dist_matrix is None:
            return None
        
        # ========== 修复设备不匹配问题 ==========
        # 将所有相关tensor和模块移动到与dist_matrix相同的设备
        device = dist_matrix.device
        if mask is not None:
            mask = mask.to(device)
        offsets = self.offsets.to(device)
        widths = self.widths.to(device)
        # 将MLP模块移动到正确设备
        self.distance_mlp = self.distance_mlp.to(device)
        
        # ========== 修复Padding污染问题 ==========
        if mask is not None:
            # mask: (B, N) -> (B, N, N)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            # 将padding位置的距离设为很大的值（表示无穷远）
            dist_matrix = dist_matrix.masked_fill(~mask.bool(), 999.0)
        
        # 软截断
        dist_matrix = torch.where(dist_matrix < 0, torch.exp(dist_matrix) - 1, dist_matrix)
        dist_matrix = torch.where(dist_matrix > self.high, 
                                  self.high + torch.log1p(dist_matrix - self.high), 
                                  dist_matrix)
        
        diff = dist_matrix.unsqueeze(-1) - offsets
        denom = widths + 1e-6
        
        pow_diff = (diff / denom) ** 2
        pow_diff = pow_diff / (1 + pow_diff / 50.0)
        
        rbf = torch.exp(-pow_diff)
        
        # 学习型偏置: (B, N, N, num_heads)
        bias = self.distance_mlp(rbf).permute(0, 3, 1, 2)
        
        # 再次mask bias（确保padding位置不会产生强交互）
        if mask is not None:
            bias = bias.masked_fill(~mask.unsqueeze(1).bool(), -1e9)
        
        return bias

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask=None, adjoin_matrix=None, dist_matrix=None, 
                drug_contact_prob=None, drug_distance_bias=None):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # scaled_dot_product_attention
        if dist_matrix is not None:
            matmul_qk = torch.matmul(q, k.transpose(-2, -1))
            dist_w = rescale_distance_matrix(dist_matrix)
            dist_w = dist_w.unsqueeze(1) # Add head dim
            dk = torch.tensor(k.size(-1), dtype=torch.float32).to(q.device)
            scaled_attention_logits = (matmul_qk * dist_w) / torch.sqrt(dk)
        else:
            matmul_qk = torch.matmul(q, k.transpose(-2, -1))
            dk = torch.tensor(k.size(-1), dtype=torch.float32).to(q.device)
            scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        if mask is not None:
             # mask: (batch, 1, 1, seq_len)
            scaled_attention_logits += (mask * -1e9)
        
        if adjoin_matrix is not None:
            # adjoin_matrix: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
            scaled_attention_logits += adjoin_matrix.unsqueeze(1)
        
        # ========== 添加药物3D几何偏置 ==========
        # 1. 接触概率偏置（拓扑信息）
        if drug_contact_prob is not None:
            # drug_contact_prob: (B, N, N) -> (B, 1, N, N)
            contact_bias = 1.5 * drug_contact_prob.unsqueeze(1)
            scaled_attention_logits = scaled_attention_logits + contact_bias
        
        # 2. RBF距离编码偏置（几何信息）
        if drug_distance_bias is not None:
            # drug_distance_bias: (B, num_heads, N, N)
            scaled_attention_logits = scaled_attention_logits + drug_distance_bias

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v) # (batch, num_heads, seq_len, depth)
        
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.d_model)
        output = self.dense(output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, dff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        return self.linear2(gelu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model // 2, num_heads)
        self.mha2 = MultiHeadAttention(d_model // 2, num_heads)
        self.ffn = FeedForward(d_model, dff)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask=None, adjoin_matrix=None, dist_matrix=None, 
                drug_contact_prob=None, drug_distance_bias=None):
        # x split
        x1, x2 = torch.chunk(x, 2, dim=-1)
        
        # local attention: 使用药物3D拓扑偏置
        x_l, _ = self.mha1(x1, x1, x1, mask=mask, adjoin_matrix=adjoin_matrix, 
                           drug_contact_prob=drug_contact_prob, drug_distance_bias=drug_distance_bias)
        # global attention: 不使用drug 3D bias（全局特征）
        x_g, _ = self.mha2(x2, x2, x2, mask=mask, adjoin_matrix=None, dist_matrix=dist_matrix)
        
        attn_output = torch.cat([x_l, x_g], dim=-1)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class AtomEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(AtomEncoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.embedding = nn.Linear(61, d_model) # Atom features dim=61
        self.dropout = nn.Dropout(rate)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ])
        self.global_embedding = nn.Linear(d_model, dff)  # Projection to dff (512) to match checkpoint
        
        # 药物3D距离RBF编码器（high=6.0适合小分子药物）
        self.drug_distance_rbf = DistanceRBFEncoder(num_rbf=16, num_heads=num_heads, low=0.0, high=6.0)

    def forward(self, x, adjoin_matrix, dist_matrix, atom_match_matrix, sum_atoms,
                drug_contact_prob=None, drug_distance=None):
        """
        Args:
            x: (batch, n_atoms, 61) - 原子特征
            adjoin_matrix: (batch, n_atoms, n_atoms) - 邻接矩阵
            dist_matrix: (batch, n_atoms, n_atoms) - 距离矩阵
            atom_match_matrix: (batch, n_motifs, n_atoms) - 原子到motif的匹配矩阵
            sum_atoms: (batch, n_motifs, 1) - 每个motif的原子数量
            drug_contact_prob: (batch, n_atoms, n_atoms) - 药物接触概率矩阵（可选）
            drug_distance: (batch, n_atoms, n_atoms) - 药物3D距离矩阵（可选）
        """
        # 构建原子级mask（用于解决padding污染问题）
        atom_mask = (torch.sum(x, dim=-1) != 0).bool()  # (batch, n_atoms)
        
        # 计算药物3D几何偏置（在atom层处理，解决维度断层问题）
        drug_distance_bias = None
        if drug_distance is not None:
            # 将padding位置的距离设为无穷大，避免污染
            drug_distance_bias = self.drug_distance_rbf(drug_distance, mask=atom_mask)
        
        # 构建attention mask
        mask = (torch.sum(x, dim=-1) == 0).float() # (batch, n_atoms)
        mask = mask.unsqueeze(1).unsqueeze(2) # (batch, 1, 1, n_atoms)
        
        x = F.relu(self.embedding(x))
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask=mask, adjoin_matrix=adjoin_matrix, dist_matrix=dist_matrix,
                     drug_contact_prob=drug_contact_prob, drug_distance_bias=drug_distance_bias)
            
        # Aggregate to motif level
        # atom_match_matrix: (batch, n_motifs, n_atoms)
        # x: (batch, n_atoms, d_model)
        # Output: (batch, n_motifs, d_model)
        x = torch.matmul(atom_match_matrix, x)
        x = x / (sum_atoms + 1e-9) # Avoid div by zero
        
        # Project to match motif encoder dimension (for motif model) or keep as-is
        target_dim = self.global_embedding.weight.shape[0]
        if target_dim != self.d_model:
            x = self.global_embedding(x)
        
        return x

class MotifEncoder(nn.Module):
    def __init__(self, num_layers, input_vocab_size, d_model, num_heads, dff, rate=0.1):
        super(MotifEncoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.dropout = nn.Dropout(rate)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ])

    def forward(self, x, atom_level_features=None, adjoin_matrix=None, dist_matrix=None):
        # x: (batch, seq_len)
        mask = (x == 0).float()
        mask = mask.unsqueeze(1).unsqueeze(2)
        
        x = self.embedding(x)
        x = x * math.sqrt(self.d_model)
        x = self.dropout(x)
        
        if atom_level_features is not None:
            # atom_level_features: (batch, n_motifs, d_model)
            # x: (batch, n_motifs+1, d_model) -> split to global + motifs
            
            # Check dimensions compatibility
            # x has (N_motifs + 1) tokens (Global + Motifs)
            # atom_level_features has N_motifs tokens
            
            x_global = x[:, 0:1, :] # (batch, 1, d_model)
            x_motifs = x[:, 1:, :]  # (batch, n_motifs, d_model)
            
            # Ensure dimensions match before addition
            if x_motifs.shape[1] != atom_level_features.shape[1]:
                # This might happen if padding was handled differently
                # Truncate or Pad to match x_motifs (which comes from motif_seq)
                min_len = min(x_motifs.shape[1], atom_level_features.shape[1])
                x_motifs = x_motifs[:, :min_len, :]
                atom_level_features = atom_level_features[:, :min_len, :]
                
                # If x_motifs was truncated, we need to adjust x_global and concatenation
                # But wait, x comes from embedding(motif_seq). motif_seq length is ground truth.
                # atom_level_features comes from matmul(atom_match, atom_emb).
                # atom_match shape is (batch, n_motifs, n_atoms).
                # So atom_level_features SHOULD match n_motifs.
                
                # Debug print if mismatch persists
                # print(f"Shape mismatch: x_motifs {x_motifs.shape}, atom_feats {atom_level_features.shape}")
                
            x_motifs = x_motifs + atom_level_features
            x = torch.cat([x_global, x_motifs], dim=1)
            
        for layer in self.layers:
            x = layer(x, mask=mask, adjoin_matrix=adjoin_matrix, dist_matrix=dist_matrix)
            
        return x
