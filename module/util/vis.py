import h5py
import numpy as np
import os

class EmbeddingCache:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.h5_file = h5py.File(self.cache_path, 'w')
        print(f"Embedding cache initialized at {self.cache_path}")
    
    def write_batch(self, utt_ids, scores, embeddings, labels, lengths):
        if len(utt_ids) != len(embeddings) or len(utt_ids) != len(labels) or len(utt_ids) != len(lengths):
            raise ValueError("Length of utt_ids, embeddings, labels, and lengths must match.")
        for utt_id, score, embedding, label, length in zip(utt_ids, scores, embeddings, labels, lengths):
            if utt_id in self.h5_file:
                print(f"Warning: {utt_id} already exists in cache. Overwriting.")
            group = self.h5_file.create_group(utt_id)
            group.create_dataset('score', data=score[:length], compression='gzip')
            group.create_dataset('embedding', data=embedding[:length], compression='gzip')
            group.create_dataset('label', data=label[:length], compression='gzip')
    
    def close(self):
        if self.h5_file:
            file_size = os.path.getsize(self.cache_path) / (1024 * 1024)
            print(f"Closing embedding cache. Size: {file_size:.2f} MB")
            self.h5_file.close()
            self.h5_file = None
            print(f"Embedding cache closed at {self.cache_path}")


def plot_diff_ori_recon(ori, recon, diff, uttids, seglabs, lengths):
    import matplotlib.pyplot as plt
    for src, tgt, dff, uttid, seglab, length in zip(ori, recon, diff, uttids, seglabs, lengths):
        src, tgt, dff = src.cpu().numpy().T, tgt.cpu().numpy().T, dff.cpu().numpy().T
        seglab = seglab.cpu().numpy()
        length = length.cpu().item()
        plt.figure(figsize=(6, 6))
        plt.subplot(221)
        plt.imshow(src[:, :length], aspect='auto')
        plt.subplot(222)
        plt.imshow(tgt[:, :length], aspect='auto')
        plt.subplot(223)
        plt.imshow(dff[:, :length], aspect='auto')
        plt.subplot(224)
        plt.plot(seglab[:length])
        plt.xlim(0, length)
        plt.savefig(f'temp/{uttid}.png')
        plt.close()