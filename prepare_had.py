import numpy as np
import math

def convert_to_frame_labels(label_str, frame_duration=0.02):
    """
    Convert text labels to frame-level labels
    :param label_str: Original string, e.g., "HAD_test_00000003 0.00-1.43-T/1.43-2.02-F/2.02-4.03-T 0"
    :param frame_duration: Duration of each frame (seconds), default 0.02s
    :return: numpy array (int)
    """
    # 1. Split the string
    parts = label_str.strip().split(' ')
    if len(parts) != 3:
        raise ValueError("Input string format is incorrect, it should contain at least the file name and segment information")
    
    utterance_id = parts[0]
    segments_str = parts[1]
    utterance_label = int(parts[2])
    # segments_str = "0.00-1.43-T/1.43-2.02-F/2.02-4.03-T"
    
    # 2. Parse all segments and determine the total duration
    segments = []
    max_time = 0.0
    for seg in segments_str.split('/'):
        start, end, tag = seg.split('-')
        start, end = float(start), float(end)
        # mapping labels：T -> 1 (genuine), F -> 0 (fake)
        label = 1 if tag == 'T' else 0
        segments.append((start, end, label))
        max_time = max(max_time, end)
    
    # 3. Calculate the total number of frames (using round to avoid floating point precision issues like 0.9999)
    num_frames = int(round(max_time / frame_duration))
    
    # 4. Initialize an array of ones, 1 -> genuine, 0 -> fake
    frame_labels = np.ones(num_frames, dtype=np.int32)
    
    # 5. Fill in the frame labels based on segments
    for start, end, label in segments:
        if label == 1:
            continue  # Genuine segments are already 1, no need to modify
        # Calculate the start and end frame indices
        start_idx = math.floor(start / frame_duration)
        end_idx = math.ceil(end / frame_duration)
        
        # Set the labels for this segment
        # Note: Python slicing is left-inclusive, right-exclusive [start_idx:end_idx]
        frame_labels[start_idx:end_idx] = label
        
    return utterance_id, frame_labels.astype(np.int32), utterance_label


if __name__ == "__main__":
    seg_label_file = "data/HAD/HAD_test_seglab_0.02.npy"
    txt_label_file = "data/HAD/HAD_test/HAD_test_label.txt"
    with open(txt_label_file, 'r') as f:
        lines = f.readlines()
    label_dict = {}
    for line in lines:
        if not line.strip():
            continue
        uttid, frame_labels, utt_label = convert_to_frame_labels(line)
        print(f"{uttid}, {utt_label} {frame_labels.shape}")
        label_dict[uttid] = frame_labels
    np.save(seg_label_file, label_dict)