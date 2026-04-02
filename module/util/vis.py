import pickle
import numpy as np
import os
import torch

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        raise TypeError("Input must be a PyTorch tensor or a NumPy array.")

class EvalResultCache:
    def __init__(self, cache_path, visualize=False):
        self.cache_path = cache_path
        self.visualize = visualize
        self.cache_dict = {}
        self.current_save_file = None
        os.makedirs(self.cache_path, exist_ok=True)
        print(f"Embedding cache initialized at {self.cache_path}")

    
    def write_cache(self, utt_ids, scores, labels, lengths, embeddings, emb_labels, emb_lengths):
        if len(utt_ids) != len(scores) or len(utt_ids) != len(labels) or len(utt_ids) != len(lengths):
            raise ValueError("Length of utt_ids, embeddings, labels, and lengths must match.")

        scores = to_numpy(scores)
        labels = to_numpy(labels)
        lengths = to_numpy(lengths)
        embeddings = to_numpy(embeddings) if self.visualize else embeddings
        emb_labels = to_numpy(emb_labels) if self.visualize else emb_labels
        emb_lengths = to_numpy(emb_lengths) if self.visualize else emb_lengths
        for i, utt_id in enumerate(utt_ids):
            if self.visualize:
                L = emb_lengths[i]
                item = {
                    'embedding': embeddings[i, :L, :],
                    'label': emb_labels[i, :L],
                }
            else:
                L = lengths[i]
                item = {
                    'score': scores[i, :L],
                    'label': labels[i, :L]
                }

            self.cache_dict[utt_id] = item

    def save_cache(self):
        save_file = self.current_save_file
        if save_file is None:
            save_file = os.path.join(self.cache_path, 'eval_cache.pkl')
        print(f"Packing and saving to .pkl file: {save_file}")
        # save the cache_dict to a .pkl file
        with open(save_file, 'wb') as f:
            pickle.dump(self.cache_dict, f)

        file_size = os.path.getsize(save_file) / (1024 * 1024)
        print(f"Cache saved successfully. Size: {file_size:.2f} MB")
        self.cache_dict.clear()

    def set_cache_file(self, dataset_key, dataloader_idx):
        if dataset_key is None:
            self.current_save_file = os.path.join(self.cache_path, f'eval_cache_dataloader{dataloader_idx}.pkl')
        else:
            self.current_save_file = os.path.join(self.cache_path, f'eval_cache_{dataset_key}.pkl')
        print(f"Set current cache file to: {self.current_save_file}")

    def close(self):
        self.save_cache()