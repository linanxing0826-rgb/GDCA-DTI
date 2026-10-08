"""
蛋白质3D结构特征提取器
从PDB文件提取Cα坐标、距离矩阵和接触图
"""

import os
import pickle
import numpy as np
from Bio.PDB import PDBParser


def extract_ca_coords(pdb_file):
    """从PDB文件提取Cα原子坐标"""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_file)
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
        return np.array(coords) if coords else None
    except Exception as e:
        print(f"Error parsing {pdb_file}: {e}")
        return None


def calculate_distance_matrix(coords):
    """计算残基间欧氏距离矩阵（向量化实现，快几十倍）"""
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    return dist_matrix.astype(np.float32)


def calculate_soft_contact(dist_matrix, cutoff=8.0, temperature=1.5):
    """Soft Contact Map（使用sigmoid，更符合物理连续性）"""
    # 使用完全数值稳定的sigmoid实现
    x = (cutoff - dist_matrix) / temperature
    
    # 创建结果数组
    result = np.zeros_like(dist_matrix, dtype=np.float32)
    
    # 情况1: x >= 0
    mask_pos = x >= 0
    result[mask_pos] = 1.0 / (1.0 + np.exp(-np.minimum(x[mask_pos], 709.0)))  # exp(709) 是 float32 的上限
    
    # 情况2: x < 0
    mask_neg = x < 0
    exp_x = np.exp(np.maximum(x[mask_neg], -709.0))  # 防止下溢
    result[mask_neg] = exp_x / (1.0 + exp_x)
    
    return result


def compute_relative_position(n):
    """计算相对位置编码（用于Transformer attention bias）"""
    idx = np.arange(n)
    rel_pos = np.abs(idx[:, None] - idx[None, :])
    return rel_pos.astype(np.float32)


def main():
    # 配置 - 使用本地路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_dir = os.path.join(current_dir, '..', 'Protein', 'BindingDB', 'matched_3d_files_BDB')
    output_file = os.path.join(current_dir, 'protein_features', 'bindingdb_protein_3d_features.pkl')
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 遍历所有PDB文件（支持.pdb和.txt扩展名）
    features_dict = {}
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb') or f.endswith('.txt')]
    
    print(f"Found {len(pdb_files)} PDB files")
    
    for idx, pdb_file in enumerate(pdb_files):
        # 去除.pdb或.txt后缀
        if pdb_file.endswith('.pdb'):
            protein_id = pdb_file[:-4]
        elif pdb_file.endswith('.txt'):
            protein_id = pdb_file[:-4]
        else:
            protein_id = os.path.splitext(pdb_file)[0]
        pdb_path = os.path.join(pdb_dir, pdb_file)
        
        coords = extract_ca_coords(pdb_path)
        if coords is not None and len(coords) > 0:
            # 1. 距离矩阵（向量化）
            dist_matrix = calculate_distance_matrix(coords)
            
            # 2. Soft Contact Map（由CA距离转成，代替原来的伪接触图）
            soft_contact = calculate_soft_contact(dist_matrix)
            
            features_dict[protein_id] = {
                'contact_prob': soft_contact,       # 由CA距离转成的soft contact，代替原来的伪接触图
                'protein_distance': dist_matrix     # 真实3D距离矩阵，用于模型内部的RBF/结构偏置
            }
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(pdb_files)} proteins")
    
    # 保存特征文件
    with open(output_file, 'wb') as f:
        pickle.dump(features_dict, f)
    
    print(f"\nSuccessfully processed {len(features_dict)} proteins")
    print(f"Features saved to: {output_file}")
    print(f"\nFeature structure:")
    print("  {protein_id}: {")
    print("    'contact_prob': numpy array (N_residues, N_residues) - soft contact map from CA distances")
    print("    'protein_distance': numpy array (N_residues, N_residues) - 3D distance matrix")
    print("  }")


if __name__ == '__main__':
    main()