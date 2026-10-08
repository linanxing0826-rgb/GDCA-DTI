import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

def load_vocabulary(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    id_to_smiles = {v: k for k, v in vocab.items()}
    return id_to_smiles

def smiles_to_chemical_name(smiles):
    special_tokens = {'<pad>': 'Padding', '<unk>': 'Unknown token', '<mask>': 'Mask token', '<global>': 'Global token'}
    if smiles in special_tokens:
        return special_tokens[smiles]
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmarts(smiles)
        if mol is None:
            return f"Unparseable: {smiles[:30]}..."
        
        functional_groups = [
            (Chem.MolFromSmarts('c1ccccc1'), 'Benzene ring'),
            (Chem.MolFromSmarts('C(=O)N'), 'Amide'),
            (Chem.MolFromSmarts('O=C-O'), 'Carboxylic acid'),
            (Chem.MolFromSmarts('C(=O)'), 'Ketone'),
            (Chem.MolFromSmarts('C-O-C'), 'Ether'),
            (Chem.MolFromSmarts('[C]-O'), 'Alcohol'),
            (Chem.MolFromSmarts('c1ccccc1O'), 'Phenol'),
            (Chem.MolFromSmarts('C#N'), 'Nitrile'),
            (Chem.MolFromSmarts('C=N'), 'Imine'),
            (Chem.MolFromSmarts('C-S-C'), 'Thioether'),
            (Chem.MolFromSmarts('S(=O)(=O)'), 'Sulfone'),
            (Chem.MolFromSmarts('P(=O)(O)'), 'Phosphate'),
            (Chem.MolFromSmarts('N-C(=O)-N'), 'Urea'),
            (Chem.MolFromSmarts('c1ncnc[nH]1'), 'Imidazole'),
            (Chem.MolFromSmarts('c1ccncc1'), 'Pyridine'),
            (Chem.MolFromSmarts('c1cncnc1'), 'Pyrimidine'),
            (Chem.MolFromSmarts('c1nc[nH]c2ccccc12'), 'Indole'),
            (Chem.MolFromSmarts('c1cc2c(n1)cccc2'), 'Quinoline'),
            (Chem.MolFromSmarts('c1ccsc1'), 'Thiophene'),
            (Chem.MolFromSmarts('c1ccoc1'), 'Furan'),
            (Chem.MolFromSmarts('c1nncc1'), 'Pyrazole'),
            (Chem.MolFromSmarts('c1nccn1'), 'Pyrazine'),
            (Chem.MolFromSmarts('c1cc[nH]c1'), 'Pyrrole'),
            (Chem.MolFromSmarts('c1ccccc1-c1ccccc1'), 'Biphenyl'),
            (Chem.MolFromSmarts('N(=O)=O'), 'Nitro'),
            (Chem.MolFromSmarts('C-N-C'), 'Tertiary amine'),
            (Chem.MolFromSmarts('C-N'), 'Amine'),
            (Chem.MolFromSmarts('C-O-C(=O)'), 'Ester'),
            (Chem.MolFromSmarts('C(=O)-N-C(=O)'), 'Imide'),
            (Chem.MolFromSmarts('C-S(=O)(=O)-N'), 'Sulfonamide'),
            (Chem.MolFromSmarts('[nH]'), 'Pyrrole nitrogen'),
            (Chem.MolFromSmarts('[n+]'), 'Pyridinium'),
            (Chem.MolFromSmarts('[se]'), 'Selenium'),
            (Chem.MolFromSmarts('[Si]'), 'Silicon'),
            (Chem.MolFromSmarts('[P]'), 'Phosphorus'),
            (Chem.MolFromSmarts('Br'), 'Bromine'),
            (Chem.MolFromSmarts('Cl'), 'Chlorine'),
            (Chem.MolFromSmarts('F'), 'Fluorine'),
            (Chem.MolFromSmarts('I'), 'Iodine'),
            (Chem.MolFromSmarts('C=C'), 'Alkene'),
            (Chem.MolFromSmarts('C#C'), 'Alkyne'),
            (Chem.MolFromSmarts('C1CCCC1'), 'Cyclopentane'),
            (Chem.MolFromSmarts('C1CCCCC1'), 'Cyclohexane'),
            (Chem.MolFromSmarts('C1CCCCC1'), 'Cyclohexane'),
        ]
        
        matches = []
        for pattern, name in functional_groups:
            if pattern and mol.HasSubstructMatch(pattern):
                matches.append(name)
        
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        ring_sizes = set()
        for ring in ring_info.AtomRings():
            ring_sizes.add(len(ring))
        
        num_atoms = mol.GetNumAtoms()
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        
        if len(matches) > 0:
            ring_desc = []
            if num_rings > 0:
                ring_desc.append(f"{num_rings} ring(s)")
                if ring_sizes:
                    ring_desc.append(f"ring sizes: {','.join(map(str, sorted(ring_sizes)))}")
            desc_parts = matches[:4]
            if ring_desc:
                desc_parts.append('; '.join(ring_desc))
            return ', '.join(desc_parts)
        
        if num_rings == 1:
            if 6 in ring_sizes:
                return "Six-membered ring"
            elif 5 in ring_sizes:
                return "Five-membered ring"
            else:
                return f"{list(ring_sizes)[0]}-membered ring"
        elif num_rings > 1:
            return f"Polycyclic structure ({num_rings} rings)"
        
        if num_atoms <= 3:
            atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(num_atoms)]
            return f"Small molecule ({', '.join(atom_symbols)})"
        elif num_atoms <= 6:
            return f"Aliphatic chain ({num_atoms} atoms)"
        else:
            return f"Complex structure ({num_atoms} atoms, {num_rings} rings)"
            
    except Exception as e:
        return f"Error: {str(e)[:30]}..."

def map_motif_numbers(input_csv, vocab_path, output_csv):
    id_to_smiles = load_vocabulary(vocab_path)
    df = pd.read_csv(input_csv)
    
    motif_info = []
    for idx, row in df.iterrows():
        if row['Token_Type'] == 'Motif':
            motif_index = int(row['Drug_Index'])
            if motif_index in id_to_smiles:
                smiles = id_to_smiles[motif_index]
                chem_name = smiles_to_chemical_name(smiles)
                df.loc[idx, 'Motif_SMILES'] = smiles
                df.loc[idx, 'Chemical_Name'] = chem_name
                df.loc[idx, 'Drug_Name'] = f"{row['Drug_Name']} ({chem_name})"
                motif_info.append({
                    'Motif_Index': motif_index,
                    'Motif_Name': row['Drug_Name'],
                    'SMILES': smiles,
                    'Chemical_Name': chem_name
                })
            else:
                df.loc[idx, 'Motif_SMILES'] = 'Unknown'
                df.loc[idx, 'Chemical_Name'] = 'Index not found in vocabulary'
    
    df.to_csv(output_csv, index=False)
    
    motif_summary = pd.DataFrame(motif_info).drop_duplicates()
    summary_path = output_csv.replace('.csv', '_motif_summary.csv')
    motif_summary.to_csv(summary_path, index=False)
    
    return df, motif_summary

if __name__ == '__main__':
    input_csv = r'e:\shuoshishiyan\drug-Protein interaction\MeTDDI-main\MeTDDI-main\code\5-19代码备份\case_study_results\crossmodal_pairs.csv'
    vocab_path = r'e:\shuoshishiyan\drug-Protein interaction\MeTDDI-main\MeTDDI-main\code\Classification\Unseendrugs\token_id.json'
    output_csv = r'e:\shuoshishiyan\drug-Protein interaction\MeTDDI-main\MeTDDI-main\code\5-19代码备份\case_study_results\mapped_crossmodal_pairs.csv'
    summary_path = output_csv.replace('.csv', '_motif_summary.csv')
    
    df, summary = map_motif_numbers(input_csv, vocab_path, output_csv)
    
    print("Motif Mapping Summary:")
    print("=" * 80)
    print(summary.to_string(index=False))
    print(f"\nMapped file saved to: {output_csv}")
    print(f"Summary file saved to: {summary_path}")