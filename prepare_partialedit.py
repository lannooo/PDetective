import os

import numpy as np
import math
import shutil
import csv
import tqdm

def convert_to_frame_labels(parts, spoofing_type:str='CAP', frame_duration=0.02):
    """
    convert text labels to frame-level labels
    :param label_str: original csv, e.g., "E1/p237/p237_256_edited_partial_16k.wav,1.442,1.824,2.249,2.732,3.1"
    :param frame_duration: duration of each frame (seconds), default 0.02s
    :return: numpy array (int)
    """
    wav_path = parts[0]
    model_name, speaker_id, wav_name = wav_path.split('/')
    utterance_id = f"{spoofing_type}_{model_name}_{wav_name.split('_edited_')[0]}"
    segpoints = parts[1:-1]
    assert len(segpoints) % 2 == 0, "Segment points should be in pairs of start and end times."
    total_duration = float(parts[-1])
    
    # 2. parse the segment points into a list of (start, end, label) tuples
    segments = []
    for i in range(0, len(segpoints), 2):
        start, end = float(segpoints[i]), float(segpoints[i+1])
        segments.append((start, end, 0))
    
    # 3. total number of frames
    num_frames = int(round(total_duration / frame_duration))
    
    # 4. init to ones first 1 -> genuine, 0 -> fake
    frame_labels = np.ones(num_frames, dtype=np.int32)
    
    # 5. fill in the frame labels based on segments
    for start, end, label in segments:
        start_idx = math.floor(start / frame_duration)
        end_idx = math.ceil(end / frame_duration)
        
        frame_labels[start_idx:end_idx] = label
        
    return utterance_id, frame_labels.astype(np.int32)

if __name__ == "__main__":

    seg_label_file = "data/PartialEdit/PartialEdit_E1E2_seglab_0.02.npy"
    csv_label_file = "data/PartialEdit/PartialEdit_E1E2.csv"
    wav_dir = "data/PartialEdit/wav"
    # read csv file
    with open(csv_label_file, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
    
    # prepare a global wav directory
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    
    label_dict = {}
    for line in tqdm.tqdm(lines, desc="Processing"):
        # CaP type
        ori_wav_path = line[0]
        uttid, frame_labels = convert_to_frame_labels(line, spoofing_type='CAP')
        shutil.copyfile(f"{ori_wav_path}", f"{wav_dir}/{uttid}.wav")
        print(f"{uttid}, {frame_labels.shape}, {frame_labels}")
        label_dict[uttid] = frame_labels

        # Edi type
        model_tag = ori_wav_path.split('/')[0]
        edi_wav_path = line[0].replace(model_tag, f"{model_tag}-Codec").replace("edited_partial", "edited")
        uttid, frame_labels = convert_to_frame_labels(line, spoofing_type='EDI')
        shutil.copyfile(f"{edi_wav_path}", f"{wav_dir}/{uttid}.wav")
        print(f"{uttid}, {frame_labels.shape}, {frame_labels}")
        label_dict[uttid] = frame_labels

    np.save(seg_label_file, label_dict)