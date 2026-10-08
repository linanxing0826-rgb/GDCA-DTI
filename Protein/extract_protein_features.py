import os
import sys
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_esm_features(protein_dict_path, output_path, model_name='facebook/esm2_t33_650M_UR50D', max_len=1024):
    print(f"Loading protein dict from {protein_dict_path}")
    with open(protein_dict_path, 'rb') as f:
        protein_dict = pickle.load(f)
    
    print(f"Loaded {len(protein_dict)} proteins")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading ESM model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    protein_features = {}
    
    for idx, (protein_id, (protein_name, protein_seq)) in enumerate(protein_dict.items()):
        if idx % 100 == 0:
            print(f"Processing protein {idx}/{len(protein_dict)}")
        
        seq_len = len(protein_seq)
        
        if seq_len > max_len:
            protein_seq = protein_seq[:max_len]
        
        try:
            with torch.no_grad():
                inputs = tokenizer(protein_seq, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                
                embeddings = outputs.last_hidden_state
                
                seq_len_actual = min(seq_len, max_len)
                embeddings = embeddings[0, 1:seq_len_actual+1, :].cpu().numpy()
                
                protein_features[protein_id] = (protein_name, embeddings)
        
        except Exception as e:
            print(f"Error processing protein {protein_id}: {e}")
            continue
    
    print(f"Saving features to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(protein_features, f)
    
    print(f"Done! Saved features for {len(protein_features)} proteins")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract ESM features for proteins')
    parser.add_argument('--input', type=str, required=True, help='Input protein_dict.pkl path')
    parser.add_argument('--output', type=str, required=True, help='Output features path')
    parser.add_argument('--model', type=str, default='facebook/esm2_t33_650M_UR50D', help='ESM model name')
    parser.add_argument('--max_len', type=int, default=1024, help='Max sequence length')
    
    args = parser.parse_args()
    
    extract_esm_features(args.input, args.output, args.model, args.max_len)


if __name__ == '__main__':
    main()