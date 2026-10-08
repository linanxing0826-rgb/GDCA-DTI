import warnings
warnings.warn('ignore')
import numpy as np
from rdkit.Chem import MolFromSmiles,MolToSmiles
from features import (atom_features,bond_features,
                        one_of_k_encoding, one_of_k_encoding_unk)

degrees = [0, 1, 2, 3, 4, 5]   #定义键可能度数
class MolGraph(object):
    def __init__(self):
        self.nodes = {} # 按节点类型组织节点字典
    def new_node(self, ntype, features=None, rdkit_ix=None):   #创建新节点添加到图中
        new_node = Node(ntype, features, rdkit_ix)   #实例化，添加到node字典
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):  #合并子图到当前图
        old_nodes = self.nodes      #当前图字典节点
        new_nodes = subgraph.nodes   #待合并的节点

        #遍历当前图和子图的所有节点类型
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            #把新图的节点列表拼接到当前图类型节点列表
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))# gain node

    def sort_nodes_by_degree(self, ntype):     #根据节点的键度数分类
        nodes_by_degree = {i : [] for i in degrees}       #初始化，覆盖分子可能成键度数分类

        for node in self.nodes[ntype]:  #遍历该类型所有节点，根据度数分组
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)  #按照度数加入对应列表

        new_nodes = []       #重新组合节点列表，升序排序，更新self.nodes，按照预定义的度数遍历
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]             #当前度数所有节点
            self.nodes[(ntype, degree)] = cur_nodes    # 新增键（类型+度数），存储该组节点（方便后续快速调用）
            new_nodes.extend(cur_nodes)         #拼接
        self.nodes[ntype] = new_nodes            #覆盖原节点列表，完成排序

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]]) # 列表推导式：遍历该类型所有节点，取每个节点的features属性，再转为numpy数组

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]

class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    def __init__(self,ntype,features,rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix
    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)
    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]

def graph_from_smiles_tuple(smiles_tuple):
    graph_list = [graph_from_smiles(s) for s in smiles_tuple]
    big_graph = MolGraph()
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)
    # This sorting allows an efficient (but brittle!) indexing later on.
    big_graph.sort_nodes_by_degree('atom')
    return big_graph
def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = MolFromSmiles(smiles)
    mol = MolFromSmiles(MolToSmiles(mol)) 
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node
    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))
    
    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph

def array_rep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    graph = graph_from_smiles(smiles)
    molgraph = MolGraph()
    molgraph.add_subgraph(graph)
    molgraph.sort_nodes_by_degree('atom')
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only. 
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep

