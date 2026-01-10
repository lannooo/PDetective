import numpy as np
import math


def convert_to_frame_labels(utt_id, duration, utt_label, segment_labels) -> np.ndarray:
    # in resolution of 20 ms, sampling rate 16kHz
    parts = utt_id.split('-')
    total_frames = int(duration / 0.02)  # total frames in the utterance
    # for genuine utterances, the utt_label is 'bonafide'
    if utt_label == 'bonafide':
        return np.ones(int(total_frames), dtype=np.int64)  # all frames are genuine

    # for entirely spoofing utterances, the utt_label is 'spoof' and the third part of utt_id is 'full'
    if utt_label == 'spoof' and parts[2] == 'full':
        return np.zeros(int(total_frames), dtype=np.int64)
    
    # for partially spoofing utterances, the utt_label is 'spoof' and the third part of utt_id is 'partial'
    # each segment label is in the format of <start>-<end>-<label>, e.g., "0.5-1.0-spoof"
    # convert them into frame labels (an numpy array of 0 or 1, 0 for spoof and 1 for genuine), 
    frame_labels = np.ones(int(total_frames), dtype=np.int64)  # start with all frames as genuine
    for segment in segment_labels:
        start, end, label = segment.split('-')
        if label != 'spoof':
            continue
        start = float(start)
        end = float(end)
        start_frame = math.floor(start / 0.02)
        end_frame = math.ceil(end / 0.02)
        # set the frames in the range [start_frame, end_frame) to 0
        frame_labels[start_frame:end_frame] = 0
    return frame_labels

def read_label_file(label_file):
    label_dict = {}
    with open(label_file, 'r') as f:
        # dev-clean_1462_170145_000020_000002 0.9600 bonafide 0.0000-0.9600-bonafide
        # dev01-yourtts-full_3000_15664_000010_000001 10.9930 spoof 0.0000-10.9930-spoof
        # dev01-yourtts-partial-cf_8842_304647_000034_000001 1.9740 spoof 0.0000-0.8510-bonafide 0.8510-1.1910-spoof 1.1910-1.9740-bonafide
        for line in f:
            parts = line.strip().split()
            utt_id = parts[0]
            duration = float(parts[1])
            utt_label = parts[2]
            segment_labels = parts[3:] if len(parts) > 3 else []
            # convert segment labels to frame labels
            frame_labels = convert_to_frame_labels(utt_id, duration, utt_label, segment_labels)
            # store in dictionary
            label_dict[utt_id] = frame_labels
    return label_dict

# generate frame level labels according to the label text file

label_file_a = "data/LLamaPartialSpoof/label_R01TTS.0.a.txt"
label_file_b = "data/LLamaPartialSpoof/label_R01TTS.0.b.txt"
frame_labels_a = read_label_file(label_file_a)
frame_labels_b = read_label_file(label_file_b)
# merge the two dictionaries
frame_labels_a.update(frame_labels_b)

# save into one npy file
target_label_file = "./data/llamaps_eval_seglab_0.02.npy"
np.save(target_label_file, frame_labels_a)