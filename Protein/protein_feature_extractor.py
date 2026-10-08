"""
蛋白质特征提取器 - 使用ESM-2预训练模型提取蛋白质特征
包括蛋白质级特征向量和氨基酸级嵌入矩阵
"""

import torch
import numpy as np
import pickle
import os
import csv
import subprocess
from transformers import AutoTokenizer, EsmModel
from config import get_config, validate_config


def get_free_gpu():
    """
    Selects the GPU with the most free memory.
    Returns:
        torch.device: The selected device (cuda:X or cpu).
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    try:
        # Use nvidia-smi to get memory usage
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        # Parse output
        memory_free = [int(x) for x in result.strip().split('\n')]
        
        # Select GPU with max free memory
        gpu_id = np.argmax(memory_free)
        print(f"Auto-selecting GPU {gpu_id} with {memory_free[gpu_id]} MiB free memory.")
        return torch.device(f"cuda:{gpu_id}")
    except Exception as e:
        print(f"Error querying nvidia-smi: {e}. Defaulting to cuda:0")
        return torch.device("cuda:0")


class ProteinFeatureExtractor:
    """蛋白质特征提取器类"""
    
    def __init__(self, config=None):
        """
        初始化特征提取器
        
        Args:
            config: 配置参数，如果为None则使用默认配置
        """
        if config is None:
            config = get_config()
        
        validate_config(config)
        self.config = config
        
        # Auto-select GPU if configured to use cuda or auto
        if config["device"] == "cpu":
             self.device = torch.device("cpu")
        else:
             self.device = get_free_gpu()
        
        # 加载ESM-2模型和tokenizer
        self.model_name = config["model_name"]
        # check if it is a local path
        if os.path.exists(self.model_name):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = EsmModel.from_pretrained(self.model_name, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = EsmModel.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"ESM-2模型加载完成: {self.model_name}")
        print(f"设备: {self.device}")
    
    def embed_sequence(self, sequence):
        """
        对单个蛋白质序列进行嵌入
        
        Args:
            sequence: 蛋白质氨基酸序列字符串
            
        Returns:
            tuple: (embedding_matrix, protein_vector)
            - embedding_matrix: L×1024维的氨基酸级嵌入矩阵
            - protein_vector: 1024维的蛋白质级特征向量
        """
        # 1. 序列预处理和tokenization
        inputs = self.tokenizer(
            sequence, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.config["max_sequence_length"]
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 2. 前向传播获取嵌入
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 3. 提取最后一层的隐藏状态作为嵌入
        # outputs.last_hidden_state形状: [batch_size, seq_len, hidden_size]
        embeddings = outputs.last_hidden_state
        
        # 4. 去除特殊token的嵌入（CLS和SEP）
        # 获取实际序列长度（去除特殊token）
        attention_mask = inputs['attention_mask']
        seq_len = attention_mask.sum().item() - 2  # 减去CLS和SEP
        
        # 提取实际氨基酸的嵌入（去除CLS和SEP）
        amino_acid_embeddings = embeddings[0, 1:seq_len+1, :]  # [L, 1024]
        
        # 5. 转换为numpy数组
        embedding_matrix = amino_acid_embeddings.cpu().numpy()
        
        # 6. 蛋白质级特征向量（平均池化）
        protein_vector = np.mean(embedding_matrix, axis=0)
        
        return embedding_matrix, protein_vector
    
    def extract_features_from_file(self, input_file, output_dir=None, group_by_type=True):
        """
        从蛋白质序列文件中提取特征
        
        Args:
            input_file: 输入蛋白质序列文件路径
            output_dir: 输出目录，如果为None则使用配置中的输出目录
            group_by_type: 是否按特征类型分组保存（True），还是整合到一个文件（False）
                          True: 所有蛋白质的同一类型特征保存在一起
                          False: 所有特征整合到一个pkl文件
                          
        Returns:
            dict: 特征字典，键为蛋白质ID，值为[vector, matrix]元组
        """
        if output_dir is None:
            output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # 检查特征文件是否已存在
        if group_by_type:
            expected_files = [
                os.path.join(output_dir, f"{base_name}_vectors.npy"),
                os.path.join(output_dir, f"{base_name}_matrices.npz"),
                os.path.join(output_dir, f"{base_name}_adjacency.npz"),
                os.path.join(output_dir, f"{base_name}_distance.npz"),
                os.path.join(output_dir, f"{base_name}_protein_ids.pkl")
            ]
            all_exists = all(os.path.exists(f) for f in expected_files)
            
            if all_exists:
                print(f"✓ 特征文件已存在，跳过提取")
                return self._load_grouped_features(output_dir, base_name)
        else:
            output_file = os.path.join(output_dir, f"{base_name}_features.pkl")
            if os.path.exists(output_file):
                print(f"✓ 特征文件已存在: {output_file}")
                with open(output_file, 'rb') as f:
                    return pickle.load(f)
        
        # 读取蛋白质序列文件
        proteins = self._read_protein_file(input_file)
        
        # 按类型分组存储特征
        protein_ids = []
        vectors = []           # 特征向量列表
        matrices = {}          # 嵌入矩阵字典 {protein_id: matrix}
        adjacency_mats = {}    # 邻接矩阵字典
        distance_mats = {}     # 距离矩阵字典
        
        print(f"开始提取{len(proteins)}个蛋白质的特征...")
        
        for i, (protein_id, sequence) in enumerate(proteins.items()):
            try:
                print(f"处理蛋白质 {i+1}/{len(proteins)}: {protein_id}")
                
                # 提取特征
                embedding_matrix, protein_vector = self.embed_sequence(sequence)
                
                # 计算接触图/邻接矩阵
                seq_len = len(sequence)
                adjacency_matrix = self.calculate_adjacency_matrix(seq_len)
                
                # 计算距离矩阵
                distance_matrix = self.calculate_distance_matrix(seq_len)
                
                # 存储特征
                protein_ids.append(protein_id)
                vectors.append(protein_vector)
                matrices[protein_id] = embedding_matrix
                adjacency_mats[protein_id] = adjacency_matrix
                distance_mats[protein_id] = distance_matrix
                
                print(f"  - 序列长度: {len(sequence)}")
                print(f"  - 嵌入矩阵形状: {embedding_matrix.shape}")
                print(f"  - 特征向量形状: {protein_vector.shape}")
                print(f"  - 邻接矩阵形状: {adjacency_matrix.shape}")
                
            except Exception as e:
                print(f"处理蛋白质 {protein_id} 时出错: {e}")
                continue
        
        # 保存特征
        if group_by_type:
            self._save_grouped_features(
                output_dir, base_name, 
                protein_ids, vectors, matrices, adjacency_mats, distance_mats
            )
        else:
            features_dict = {pid: [vectors[i], matrices[pid]] for i, pid in enumerate(protein_ids)}
            output_file = os.path.join(output_dir, f"{base_name}_features.pkl")
            with open(output_file, 'wb') as f:
                pickle.dump(features_dict, f)
            print(f"特征提取完成，保存到: {output_file}")
        
        print(f"成功处理 {len(protein_ids)} 个蛋白质")
        
        # 返回传统格式的特征字典
        return {pid: [vectors[i], matrices[pid]] for i, pid in enumerate(protein_ids)}
    
    def _save_grouped_features(self, output_dir, base_name, protein_ids, vectors, matrices, adjacency_mats, distance_mats):
        """
        将特征按类型分组保存到单独文件
        
        Args:
            output_dir: 输出目录
            base_name: 基础文件名
            protein_ids: 蛋白质ID列表
            vectors: 特征向量列表
            matrices: 嵌入矩阵字典
            adjacency_mats: 邻接矩阵字典
            distance_mats: 距离矩阵字典
        """
        # 保存蛋白质ID列表
        with open(os.path.join(output_dir, f"{base_name}_protein_ids.pkl"), 'wb') as f:
            pickle.dump(protein_ids, f)
        
        # 保存特征向量 (N, 1024)
        vectors_array = np.array(vectors)
        np.save(os.path.join(output_dir, f"{base_name}_vectors.npy"), vectors_array)
        
        # 保存嵌入矩阵（变长，使用npz存储）
        np.savez(os.path.join(output_dir, f"{base_name}_matrices.npz"), **matrices)
        
        # 保存邻接矩阵（接触图）
        np.savez(os.path.join(output_dir, f"{base_name}_adjacency.npz"), **adjacency_mats)
        
        # 保存距离矩阵
        np.savez(os.path.join(output_dir, f"{base_name}_distance.npz"), **distance_mats)
        
        print(f"\n特征按类型分组保存完成:")
        print(f"  - {base_name}_protein_ids.pkl  (蛋白质ID列表)")
        print(f"  - {base_name}_vectors.npy     (特征向量: {vectors_array.shape})")
        print(f"  - {base_name}_matrices.npz    (嵌入矩阵: {len(matrices)}个)")
        print(f"  - {base_name}_adjacency.npz   (邻接矩阵/接触图: {len(adjacency_mats)}个)")
        print(f"  - {base_name}_distance.npz    (距离矩阵: {len(distance_mats)}个)")
    
    def _load_grouped_features(self, output_dir, base_name):
        """
        加载按类型分组保存的特征
        
        Args:
            output_dir: 输出目录
            base_name: 基础文件名
            
        Returns:
            dict: 特征字典
        """
        with open(os.path.join(output_dir, f"{base_name}_protein_ids.pkl"), 'rb') as f:
            protein_ids = pickle.load(f)
        
        vectors = np.load(os.path.join(output_dir, f"{base_name}_vectors.npy"))
        matrices = dict(np.load(os.path.join(output_dir, f"{base_name}_matrices.npz")))
        
        print(f"成功加载 {len(protein_ids)} 个蛋白质特征")
        return {pid: [vectors[i], matrices[pid]] for i, pid in enumerate(protein_ids)}
    
    def _read_protein_file(self, file_path):
        """
        读取蛋白质序列文件 (支持CSV或TXT)
        
        Args:
            file_path: 文件路径
            
        Returns:
            dict: 蛋白质ID到序列的映射
        """
        proteins = {}
        
        if file_path.endswith('.csv'):
            print(f"读取CSV文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'protein_id' in row and 'sequence' in row:
                        p_id = row['protein_id']
                        seq = row['sequence']
                        if p_id not in proteins:
                            proteins[p_id] = seq
        else:
            print(f"读取TXT文件: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    if line.startswith("Protein:"):
                        # 每行格式: "Protein:ID 氨基酸序列"
                        parts = line.split("Protein:")
                        if len(parts) >= 2:
                            # 提取蛋白质ID和序列
                            id_and_seq = parts[1].strip()
                            # 第一个空格前的是ID，后面的是序列
                            if ' ' in id_and_seq:
                                protein_id, sequence = id_and_seq.split(' ', 1)
                                proteins[protein_id] = sequence
                            else:
                                # 如果没有空格，可能是格式问题
                                protein_id = id_and_seq
                                proteins[protein_id] = ""
        
        print(f"成功读取 {len(proteins)} 个唯一蛋白质序列")
        return proteins
    
    def calculate_adjacency_matrix(self, sequence_length):
        """
        计算邻接矩阵
        
        Args:
            sequence_length: 序列长度
            
        Returns:
            np.ndarray: 邻接矩阵
        """
        window_size = self.config["window_size"]
        adjacency_matrix = np.zeros((sequence_length, sequence_length))
        
        for i in range(sequence_length):
            start = max(0, i - window_size // 2)
            end = min(sequence_length, i + window_size // 2 + 1)
            
            for j in range(start, end):
                if i != j:
                    adjacency_matrix[i, j] = 1
        
        return adjacency_matrix
    
    def calculate_distance_matrix(self, sequence_length):
        """
        计算距离矩阵
        
        Args:
            sequence_length: 序列长度
            
        Returns:
            np.ndarray: 距离矩阵
        """
        distance_matrix = np.zeros((sequence_length, sequence_length))
        
        for i in range(sequence_length):
            for j in range(sequence_length):
                distance_matrix[i, j] = abs(i - j)
        
        # 归一化
        if sequence_length > 1:
            distance_matrix = distance_matrix / (sequence_length - 1)
        
        return distance_matrix


def main():
    """主函数 - 仅进行ESM-2特征提取"""
    print("=== ESM-2蛋白质特征提取器 ===")
    
    # 添加命令行参数支持
    import argparse
    parser = argparse.ArgumentParser(description="ESM-2 Protein Feature Extractor")
    parser.add_argument("--dataset", type=str, default="BiosNap", 
                        choices=["BindingDB", "BiosNap"],
                        help="选择要提取特征的数据集")
    parser.add_argument("--input_file", type=str, default=None,
                        help="手动指定输入文件路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="手动指定输出目录")
    args = parser.parse_args()
    
    # 配置文件 - 使用本地模型路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "esm2_t33_650M_UR50D")
    
    # 如果本地模型不存在，尝试使用配置中的默认名称
    if not os.path.exists(model_path):
        print(f"Warning: Local model not found at {model_path}, using default model name.")
        model_name = "esm2_t33_650M_UR50D"
    else:
        model_name = model_path
        
    config = get_config(model_name=model_name)
    
    # 输入文件路径
    input_file = None
    user_protein_path = "/nanxing/ESM_2/Protein/protein_features"
    
    # 根据选择的数据集构建路径
    if args.input_file:
        # 如果用户手动指定了输入文件
        input_file = args.input_file
    else:
        # 根据数据集类型构建路径
        dataset_name = args.dataset
        print(f"选择的数据集: {dataset_name}")
        
        possible_paths = []
        
        # 添加指定数据集的路径（优先搜索数据集特定目录）
        if dataset_name == "BindingDB":
            possible_paths.extend([
                os.path.join(user_protein_path, "BindingDB", "bindingdb_adapted.csv"),
                os.path.join(user_protein_path, "BindingDB", "BindingDB_adapted.csv"),
                os.path.join(user_protein_path, "bindingdb_adapted.csv"),
                os.path.join(user_protein_path, "BindingDB_adapted.csv"),
                os.path.join("/nanxing/ESM_2/Protein/BindingDB", "bindingdb_adapted.csv"),
                os.path.join("/nanxing/ESM_2/Protein/BindingDB", "BindingDB_adapted.csv"),
                os.path.join(current_dir, "BindingDB", "bindingdb_adapted.csv"),
                os.path.join(current_dir, "BindingDB", "BindingDB_adapted.csv"),
            ])
        else:  # BiosNap
            possible_paths.extend([
                os.path.join(user_protein_path, "BiosNap", "BiosNap_adapted.csv"),
                os.path.join(user_protein_path, "BiosNap", "biosnap_adapted.csv"),
                os.path.join(user_protein_path, "BiosNap_adapted.csv"),
                os.path.join(user_protein_path, "biosnap_adapted.csv"),
                os.path.join("/nanxing/ESM_2/Protein/BiosNap", "BiosNap_adapted.csv"),
                os.path.join(current_dir, "BiosNap", "BiosNap_adapted.csv"),
            ])
            
        # 添加通用路径作为备用
        possible_paths.append(os.path.join(current_dir, "protein_list.txt"))
        
        for path in possible_paths:
            if os.path.exists(path):
                input_file = path
                break
    
    if input_file is None:
        print(f"Error: No input file found for dataset '{args.dataset}'. Tried paths:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease ensure one of these files exists, or use --input_file to specify manually.")
        return
        
    print(f"输入文件: {input_file}")
    print(f"模型: {model_name}")
    
    # 创建特征提取器并提取特征
    extractor = ProteinFeatureExtractor(config)
    
    # 设置输出目录（优先使用用户指定的，否则使用数据集特定的目录）
    output_dir = args.output_dir
    if output_dir is None:
        # 使用用户指定的输出路径
        output_dir = os.path.join(user_protein_path, f"{dataset_name}_features")
    
    print(f"输出目录: {output_dir}")
    features_dict = extractor.extract_features_from_file(input_file, output_dir=output_dir)
    
    # 打印特征统计信息
    if features_dict and len(features_dict) > 0:
        print("\n特征统计信息:")
        print(f"总蛋白质数量: {len(features_dict)}")
        for protein_id, (vector, matrix) in list(features_dict.items())[:3]:
            print(f"\n蛋白质 {protein_id}:")
            print(f"  - 特征向量维度: {vector.shape}")
            print(f"  - 嵌入矩阵形状: {matrix.shape}")
    
    print("\n✓ ESM-2特征提取完成！")


if __name__ == "__main__":
    main()
