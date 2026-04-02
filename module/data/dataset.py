import os
import math
import random
import numpy as np
import torch
import soundfile as sf

import torch.nn.functional as F
import lightning as L
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from module.util.utils import find_files, get_boundary_label
from module.aug.augment import GriffinLimAugmentor, WaveGlowAugmentor, DiffWaveAugmentor, WaveAugmentor

class BaseDataset(Dataset):
    def __init__(
        self,
        name = None,
        root = None,
        label_root=None,
        samplerate = None,
        label_resolution=None,
        subset = None,
        input_maxlength = None,
        input_minlength = None,
        duration_lst = None,
        split_segment_ratio = 1.8,
        input_query = None,
        filter_rules = None,
        label_maxlength = None,
        pad_mode = 'label',
        add_label = 'boundary',
        sampling_config=None,
        enable_augment=False,
        augment_args=None,
        device='cpu'
        ) -> None:
        super().__init__()
        self.device = device
        self.name = name
        self.root = root
        self.subset = subset
        self.pad_mode = pad_mode
        self.add_label = add_label
        self.label_root = label_root
        self.samplerate = samplerate
        # input settings
        self.input_query = input_query
        self.label_resolution = label_resolution
        self.label_maxlength = label_maxlength
        self.input_maxlength = input_maxlength
        self.input_minlength = input_minlength
        self.duration_lst = duration_lst
        self.split_segment_ratio = split_segment_ratio

        self.filter_rules = filter_rules
        self.sampling_config = sampling_config
        # data augmentation settings
        self.enable_augment = enable_augment
        self.augment_configs = {} if not augment_args else {arg["name"]: arg for arg in augment_args}
        # the real augmentor dict built from the augment_configs, which will be applied selectively by random choice
        self.augmentors = {}

        if isinstance(root, str):
            sample_list = sorted(find_files(root, query=self.input_query))
        elif isinstance(root, list):
            sample_list = []
            for r in root:
                sample_list += sorted(find_files(r, query=self.input_query))  
        else:
            raise AttributeError(f'{subset} root is not a list or str, {root}')
        
        # filter by minlength, discard short samples
        sample_length = self.load_audio_length_info(self.duration_lst, sample_list)
        if input_minlength is not None:
            idxs = [idx for idx in range(len(sample_list)) if sample_length[idx] > input_minlength]  
            if len(sample_list) != len(idxs):
                print(
                    "some files are filtered by audio length threshold "
                    f"({len(sample_list)} -> {len(idxs)})."
                )
            sample_list = [sample_list[idx] for idx in idxs]
            sample_length = [sample_length[idx] for idx in idxs]
        self.length_list = sample_length

        # assert the number of files
        assert len(sample_list) != 0, f"Not found any sample files in {root}."
        assert self.label_resolution in [0.02, 0.04, 0.08, 0.16], f"label resolution {self.label_resolution} is not supported"
        self.scale = int(self.label_resolution // 0.02) # e.g., 0.02s -> 0.16s, scale = 8

        # load frame-level labels
        self.labels = self.label_load_fn()

        # filting audio samples
        self.sample_list = self.sample_filter(sample_list)
        # Utt_id: SPOOF_MODEL_SEQ
        self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in self.sample_list]

        # split into pieces for evaluation if necessary
        if subset == 'eval' and pad_mode == 'label' and label_maxlength is not None:
            # original utterance are split into fix-length segments for evaluation
            # FLT_CAP_00001 -> FLT_CAP_00001#0,1600
            self.split_pieces()

        self.domain_to_id = self.build_domain_map()
        print(f"[{self.subset}] domain to id map: {self.domain_to_id}")
        self.unknown_domain_id = len(self.domain_to_id)

        self.post_statistics()
    
    def filter_spoof_types(self, sample_list):
        # print(self.filter_rules)
        rule_indices = sorted(list(self.filter_rules.keys()))
        rule_configs = [self.filter_rules[idx].split(",") for idx in rule_indices]
        new_sample_list = []
        for sample in sample_list:
            utt_segs = self.get_rule_segs(sample)
            flags = []
            for rules, tag in zip(rule_configs, utt_segs):   # automatically match the minimum number of rules and tags
                if (len(rules) == 1 and rules[0] == '*') or tag in rules:
                    flags.append(True)
                else:
                    flags.append(False)
            if all(flags):  # pass filtering
                new_sample_list.append(sample)
        print(f"[{self.subset}] {len(sample_list)} samples filtered to {len(new_sample_list)} samples, rules: {self.filter_rules}")
        return new_sample_list
    
    def sample_filter(self, sample_list):
        # filter out samples not in labels
        old_n = len(sample_list)
        sample_list = [f for f in sample_list if os.path.splitext(os.path.basename(f))[0] in self.labels]
        new_n = len(sample_list)

        if old_n != new_n:
            print(f"[{self.subset}] Filter out samples not in label metadata: {old_n} => {new_n}")


        if self.filter_rules is not None:
            sample_list = self.filter_spoof_types(sample_list)
        else:
            print(f"[{self.subset}] No spoofing filter applied for <{self.subset}> dataset")
        # sampling random or fixed size of samples as the configuration
        if self.sampling_config is not None:
            mode = self.sampling_config.get('mode', 'random')
            ratio = self.sampling_config.get('ratio', 1.0)
            old_n = len(sample_list)
            if mode == 'random':
                random.seed(42)
                sample_list = random.sample(sample_list, int(len(sample_list) * ratio))
            elif mode == 'fix':
                random.seed(42)
                random.shuffle(sample_list)  # keep everytime the same order
                if isinstance(ratio, list) or isinstance(ratio, tuple):
                    if len(ratio) != 2:
                        raise ValueError(f"Fix sampling ratio should be a list of two values, got {ratio}")
                    sample_list = sample_list[int(len(sample_list) * ratio[0]):int(len(sample_list) * ratio[1])]
                else:
                    if ratio < 0:   # e.g., -0.2 means the last 20% of samples
                        sample_list = sample_list[int(len(sample_list) * ratio):]
                    else:
                        sample_list = sample_list[:int(len(sample_list) * ratio)]
            else:
                raise ValueError(f"Unknown sampling mode: {mode}")
            new_n = len(sample_list)
            print(f"[{self.subset}] Filter by sampling config: {old_n} => {new_n}")
        return sample_list
    

    def split_pieces(self):
        print("[Info] Splitting long audio samples into segments for evaluation...")
        # e.g., max_label_length = 25, resolution = 0.16s -> 4s
        # e.g., max_label_length = 200, resolution = 0.02s -> 4s
        tol_ratio = self.split_segment_ratio
        length_per_frame = int(self.samplerate * self.label_resolution)
        dur_per_segment = self.label_maxlength * self.label_resolution  # in seconds, e.g., 4.0s
        length_per_segment = int(self.samplerate * dur_per_segment)
        print(f"[Info] segment duration: {dur_per_segment}s, {length_per_segment} frames per segment")

        new_labels, new_sample_list, new_utt_ids, new_length_list = {}, [], [], []
        for uttid, wavpath, wavlength in tqdm(zip(self.utt_ids, self.sample_list, self.length_list), desc="Splitting"):
            # if there is no enougn frames, no need to split
            if math.floor(wavlength / length_per_frame) < int(self.label_maxlength * tol_ratio):
                new_uttid = f"{uttid}#0,{wavlength}"
                new_utt_ids.append(new_uttid)
                new_sample_list.append(wavpath)
                new_length_list.append(wavlength)
                new_labels[new_uttid] = self.labels[uttid]
            else: # longer than the fixed length (such as 4 seconds), split is necessary
                n_segments = math.ceil(wavlength / length_per_segment)
                for seg_idx in range(n_segments):
                    start_frame = seg_idx * length_per_segment
                    end_frame = min((seg_idx + 1) * length_per_segment, wavlength)
                    new_uttid = f"{uttid}#{start_frame},{end_frame}"
                    new_utt_ids.append(new_uttid)
                    new_sample_list.append(wavpath)
                    new_length_list.append(end_frame - start_frame)
                    start_label = seg_idx * self.label_maxlength
                    end_label = min(start_label + self.label_maxlength, len(self.labels[uttid]))
                    new_labels[new_uttid] = self.labels[uttid][start_label:end_label]
        print(f"[Info] total {len(self.utt_ids)} utterances split into {len(new_utt_ids)} segments.")
        self.utt_ids = new_utt_ids
        self.sample_list = new_sample_list
        self.length_list = new_length_list
        self.labels = new_labels

    def input_load_fn(self, path, utt_id):
        audio, sr = sf.read(path)
        # TODO: check samplerate and resample if necessary
        assert self.samplerate == sr, f"Sample rate mismatch: {self.samplerate} != {sr}"
        if utt_id.count("#") == 1: # get segment info
            start, end = utt_id.split("#")[1].split(",")  # get start,end
            start, end = int(start), int(end)
            audio = audio[start:end]
        return audio
    
    
    def load_audio_length_info(self, dur_lst_file, filenames:list):
        if dur_lst_file is None:
            # directly compute from audio files
            length_list = []
            for filepath in filenames:
                info = sf.info(filepath)
                length_list.append(info.frames)
            return length_list
        if not os.path.exists(dur_lst_file):
            # TODO load at the first time
            raise FileNotFoundError(f"Duration list file not found: {dur_lst_file}")
        
        dur_dict = {}
        with open(dur_lst_file, 'r') as f:
            for line in f:
                items = line.strip().split()
                dur_dict[items[0]] = int(items[1])
        length_list = [dur_dict[os.path.basename(f)] for f in filenames]
        return length_list


    def get_boundary_label(self, seg_label, boundry_type):
        if boundry_type == 'boundary':
            boundary_label = get_boundary_label(seg_label)
            boundary_label = torch.FloatTensor(boundary_label)
        else:
            raise ValueError(f"Unknown boundary type: {boundry_type}")
        n_boundary_label = len(boundary_label)
        return boundary_label, n_boundary_label
    

    def _cut_or_pad_signal(self, sig, expect_length):
        if len(sig) >= expect_length:
            sig = sig[:expect_length]
        else:
            sig = F.pad(sig, (0, expect_length - len(sig)), mode='constant', value=0)
        return sig

    def pad_and_cut_input(self, sig, seglab, n_seglab, label_length:int):
        unit_length = int(self.samplerate * 0.02)
        n_label = n_seglab

        sig = self._cut_or_pad_signal(sig, n_label * unit_length)
        
        # pad the label to fixed length
        if n_label < label_length:
            sig = F.pad(sig, (0, label_length * unit_length - len(sig)), mode='constant', value=0)
            seglab = F.pad(seglab, (0, label_length - n_label), mode='constant', value=0)

        elif n_label > label_length:
            offset = np.random.randint(0, n_label - label_length)  # add some randomness
            sig = sig[int(offset*unit_length):int(offset*unit_length+label_length*unit_length)]
            seglab = seglab[offset:offset+label_length]
            n_seglab = label_length
        
        return sig, seglab, n_seglab

    def label_load_fn(self):
        raise NotImplementedError
    
    def augment(self, sig, utt_id):
        raise NotImplementedError

    def post_statistics(self):
        raise NotImplementedError
    
    def get_rule_segs(self, samplename):
        raise NotImplementedError

    def normalize_utt_name(self, samplename):
        utt_name = os.path.splitext(os.path.basename(samplename))[0]
        return utt_name.split('#', 1)[0]

    def get_domain_key(self, utt_id):
        utt_segs = self.get_rule_segs(utt_id)
        if len(utt_segs) <= 1:
            return self.name
        return utt_segs[1]

    def get_known_domain_keys(self):
        if self.filter_rules:
            domain_rule_idx = 1
            if domain_rule_idx not in self.filter_rules:
                return sorted({self.get_domain_key(utt_id) for utt_id in self.utt_ids})
            rule_keys = [key.strip() for key in self.filter_rules[domain_rule_idx].split(',')]
            rule_keys = [key for key in rule_keys if key and key != '*']
            if rule_keys:
                return sorted(set(rule_keys))
        return sorted({self.get_domain_key(utt_id) for utt_id in self.utt_ids})

    def build_domain_map(self):
        domain_keys = self.get_known_domain_keys()
        return {domain_key: idx for idx, domain_key in enumerate(domain_keys)}

    def set_domain_map(self, domain_to_id):
        self.domain_to_id = dict(domain_to_id)
        print(f"[{self.subset}] set domain to id map: {self.domain_to_id}")
        self.unknown_domain_id = len(self.domain_to_id)

    def get_domain_id(self, utt_id):
        return self.domain_to_id.get(self.get_domain_key(utt_id), self.unknown_domain_id)

    def __getitem__(self, index):
        utt_id = self.utt_ids[index] # SPOOF_MODEL_SEQ#start,end
        input = self.input_load_fn(self.sample_list[index], utt_id)
        input = torch.FloatTensor(input)

        # augment audio signals with RIR, noise, vocoder, etc. (configured in augment_pipelines)
        if self.enable_augment:
            input = self.augment(input, utt_id)
            assert input is not None, f"Augmentation failed for {utt_id}, return None"
            assert len(input) > 0, f"Augmentation resulted in empty signal for {utt_id}"
        
        # the original frame-level labels in 0.02s/frame resolution
        ori_frame_label = self.labels[utt_id]
        ori_frame_label = torch.LongTensor(ori_frame_label)
        ori_frame_length = len(ori_frame_label)

        # !!! Downsampled label length, e.g., 0.02s -> 0.16s (8 frames for each group)
        frame_length = math.floor(ori_frame_length / self.scale)

        if self.pad_mode == 'label':   # only for training & validate
            if frame_length > self.label_maxlength:
                frame_length = self.label_maxlength
            # pad audio to certain length, if necessary pad/trim signal to align with label
            input, ori_frame_label, ori_frame_length = \
                self.pad_and_cut_input(input, ori_frame_label, ori_frame_length, 
                                       self.label_maxlength * self.scale)   # fix to max label length
        else:    # for test maybe, keep the actual length
            input, ori_frame_label, ori_frame_length = \
                self.pad_and_cut_input(input, ori_frame_label, ori_frame_length,
                                       frame_length * self.scale)   # fix to actual label length

        # !!! obtain the actual frame labels by grouping <scale> frames
        if self.scale > 1:
            group_frame_label = ori_frame_label.view(-1, self.scale)
            frame_label = torch.all(group_frame_label, dim=1).long()   # downsampled label, e.g., in 0.16s resolution
        else:
            frame_label = ori_frame_label

        # add boundary information
        if self.add_label:
            boundary_label, boundary_length = self.get_boundary_label(frame_label, self.add_label)
        else:
            boundary_label, boundary_length = None, None

        # add utt-level labels, obtained from frame-level labels
        # if all frames are 1, then utt-label is 1 (bonafide), else 0 (spoofing)
        utt_label = torch.tensor(1 if torch.sum(frame_label) == frame_length else 0).long()

        return {
            "utt_id": utt_id,
            "domain_key": self.get_domain_key(utt_id),
            "domain_id": self.get_domain_id(utt_id),
            "sig": input,
            "utt_label": utt_label,
            "frame_label": frame_label,
            "frame_length": frame_length,
            "boundary_label": boundary_label,
            "boundary_length": boundary_length,
            "original_frame_label": ori_frame_label,
            "original_frame_length": ori_frame_length
        }
    
    def __len__(self):
        return len(self.sample_list)


def convert_config_to_augmentor_pipeline(configs, instance:BaseDataset):
    augmentors = []
    for cfg in configs:
        name, p = cfg['name'], cfg['p'] # inner probability check for each augmentor in the pipeline
        if p <= 0:
            continue
        if name == 'griffin_lim':
            freq_distort_scale = cfg.get('freq_scale', 0.0)
            augmentors.append((GriffinLimAugmentor(n_fft=320, hop_length=160, freq_distortion=freq_distort_scale), p))
        elif name == 'waveglow':
            cache_key = cfg.get("cache_key", None)
            cache_key = None if cache_key is None else f"{instance.name}/{instance.subset}/{cache_key}"
            augmentors.append((WaveGlowAugmentor(device=instance.device, cache_key=cache_key), p))
        elif name == 'diffwave':
            cache_key = cfg.get("cache_key", None)
            cache_key = None if cache_key is None else f"{instance.name}/{instance.subset}/{cache_key}"
            augmentors.append((DiffWaveAugmentor(device=instance.device, cache_key=cache_key), p))
        elif name == 'gaussian_noise':
            extra_args = cfg.get("snr", None)
            augmentors.append((WaveAugmentor(device=instance.device, transform_type='gaussian', extra_args=extra_args), p))
        elif name == 'background_noise':
            extra_args = cfg.get("snr", None)
            augmentors.append((WaveAugmentor(device=instance.device, transform_type='background', extra_args=extra_args), p))
        elif name == 'reverb':
            augmentors.append((WaveAugmentor(device=instance.device, transform_type='reverb'), p))
        elif name == 'mp3comp':
            augmentors.append((WaveAugmentor(device=instance.device, transform_type='mp3'), p))
        elif name == 'pitch':
            augmentors.append((WaveAugmentor(device=instance.device, transform_type='pitch'), p))
        elif name == 'filter':
            extra_args = [cfg.get("filter_type", None), cfg.get("cutoff", None)]
            augmentors.append((WaveAugmentor(device=instance.device, transform_type='filter', extra_args=extra_args), p))
        elif name == 'clip':
            augmentors.append((WaveAugmentor(device=instance.device, transform_type='clip'), p))
        elif name == 'time_mask':
            augmentors.append((WaveAugmentor(device=instance.device, transform_type='time_mask'), p))
        else:
            raise ValueError(f"Unknown augmentor name: {name}")
    return augmentors

class PartialSpoofDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment_choices = []  # [(name, p), ...] for random selection
        for cfg_name, cfgs in self.augment_configs.items():
            # outter probability check for the whole pipeline
            if cfgs['p'] <= 0: continue

            if cfg_name.startswith("pipeline-"):
                # A augment pipeline is a combination of multiple augmentors, e.g., "pipeline-1": {"p": 0.5, "augs": ["gaussian_noise", "reverb"]},
                # which means randomly apply gaussian noise and reverb sequentially with 0.5 probability
                aug_configs, aug_p = cfgs['augs'], cfgs['p']
                pipe_augmentors = convert_config_to_augmentor_pipeline(aug_configs, self)
                if len(pipe_augmentors) > 0:
                    self.augmentors[cfg_name] = pipe_augmentors
                    self.augment_choices.append((cfg_name, aug_p))
            else:  
                # a single augmentor, e.g., "gaussian_noise": {"p": 0.5, "snr": [20, 30]}
                temp_augmentor = convert_config_to_augmentor_pipeline([cfgs], self)  # convert single augmentor config to pipeline format
                if len(temp_augmentor) > 0:
                    aug_ins, aug_p = temp_augmentor[0]
                    self.augmentors[cfg_name] = aug_ins
                    self.augment_choices.append((cfg_name, aug_p))
        # No augment choice
        if len(self.augment_choices) > 0:
            sum_p = sum([t[1] for t in self.augment_choices])
            if sum_p < 1.0:
                self.augment_choices.append((None, 1.0-sum_p))
    
    def label_load_fn(self):
        labels = np.load(self.label_root, allow_pickle=True).item()
        labels = {k:v.astype(int) for k, v in labels.items()}
        return labels

    def post_statistics(self):
        # analyze the prefix of utt_id
        sp_set = set([utt_id.split('_')[0] for utt_id in self.utt_ids])
        md_set = set([utt_id.split('_')[1] for utt_id in self.utt_ids])
        print(f"[{self.subset}] Total: {len(self.utt_ids)}, "
              f"spoofing types: {sp_set}, models: {md_set}")
        if self.enable_augment:
            print(f"[{self.subset}] Augmentation pipelines: {self.augment_configs}")
        else:
            print(f"[{self.subset}] No augmentation applied.")
    
    def get_rule_segs(self, samplename):
        # for partial spoof dataset, utt_id is like <SpoofingType>_<Model_Type>_<Seq>
        utt_id = self.normalize_utt_name(samplename)
        utt_segs = utt_id.split('_')[:2]   # only consider the first two segments
        return utt_segs
        
    def augment(self, sig, utt_id):
        if not self.augment_choices:  # None or empty list
            return sig
        # random select a augment option from self.augment_choices given their probabilities
        keys, weights = zip(*self.augment_choices)
        selected_option = random.choices(keys, weights=weights, k=1)[0]

        if selected_option is None:
            return sig
        elif selected_option in self.augmentors:
            augmentor = self.augmentors[selected_option]
            if isinstance(augmentor, list):  # a pipeline of augmentors
                for i, (aug_ins, aug_p) in enumerate(augmentor):
                    if random.random() < aug_p:  # check probability for each augmentor in the pipeline
                        # print(f"[Augment] applying {selected_option} - step {i+1}/{len(augmentor)}: {type(aug_ins).__name__} with p={aug_p}")
                        sig = aug_ins.transform(sig, utt_id)
                return sig
            else: # a single augmentor
                # print(f"[Augment] applying {selected_option}: {type(augmentor).__name__} with p={dict(self.augment_choices)[selected_option]}")
                return augmentor.transform(sig, utt_id)
            
        else:
            raise ValueError(f"Unknown augmentation key: {selected_option}")


class LLamaPartialSpoofDataset(PartialSpoofDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_rule_segs(self, samplename):
        # for LLama-partialspoof dataset, utt_id is like:
        # <prefix>-<model/clean>[-<full/partial>][-<oa/cf/cp>]_<sequence>
        utt_id = self.normalize_utt_name(samplename)
        utt_segs = utt_id.split('_', 1)[0].split('-')
        return utt_segs
    
    def post_statistics(self):
        # for LLamaPS
        utt_id_segs = [utt_id.split('_', 1)[0].split('-') for utt_id in self.utt_ids]
        md_set = set([item[1] for item in utt_id_segs if len(item) > 1])  # model/clean
        sp_set = set([item[2] for item in utt_id_segs if len(item) > 2])  # partial/full
        ov_set = set([item[3] for item in utt_id_segs if len(item) > 3])  # oa/cf/cp

        print(f"[{self.subset}] Total: {len(self.utt_ids)}, "
              f"spoofing types: {sp_set}, models: {md_set}, overlap: {ov_set}")
        if self.enable_augment:
            print(f"[{self.subset}] Augmentation pipelines: {self.augment_configs}")
        else:
            print(f"[{self.subset}] No augmentation applied.")

class DataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.shared_domain_to_id = None

    def _collect_domain_map(self, datasets):
        domain_keys = sorted({
            domain_key
            for dataset in datasets
            for domain_key in dataset.get_known_domain_keys()
        })
        return {domain_key: idx for idx, domain_key in enumerate(domain_keys)}

    def _apply_domain_map(self, datasets, domain_to_id):
        for dataset in datasets:
            dataset.set_domain_map(domain_to_id)

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.get_dataset('train')
            self.shared_domain_to_id = self._collect_domain_map(self.train_dataset)
            self._apply_domain_map(self.train_dataset, self.shared_domain_to_id)
            self.validate_dataset = self.get_dataset('dev')
            self._apply_domain_map(self.validate_dataset, self.shared_domain_to_id)
        if stage == 'test' or stage is None:
            if self.shared_domain_to_id is None and hasattr(self.cfg.datasets, "train_set"):
                train_datasets = self.get_dataset('train')
                self.shared_domain_to_id = self._collect_domain_map(train_datasets)
                self.test_dataset = self.get_dataset('eval')
                self._apply_domain_map(self.test_dataset, self.shared_domain_to_id)
            elif self.shared_domain_to_id is not None:
                self.test_dataset = self.get_dataset('eval')
                self._apply_domain_map(self.test_dataset, self.shared_domain_to_id)
            else:
                self.test_dataset = self.get_dataset('eval')
                self.shared_domain_to_id = self._collect_domain_map(self.test_dataset)
                self._apply_domain_map(self.test_dataset, self.shared_domain_to_id)

    def train_dataloader(self):
        dataset = self.train_dataset[0] if len(self.train_dataset) == 1 else ConcatDataset(self.train_dataset)
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, 
                          num_workers=self.cfg.num_workers, persistent_workers=True, pin_memory=True)
    
    def val_dataloader(self):
        dataset = self.validate_dataset[0] if len(self.validate_dataset) == 1 else ConcatDataset(self.validate_dataset)
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False, 
                          num_workers=self.cfg.num_workers, persistent_workers=True, pin_memory=True)
    
    def test_dataloader(self):
        batch_eval = self.cfg.batch_size if self.cfg.fast_eval else 1
        dataloaders = [DataLoader(dataset, batch_size=batch_eval, shuffle=False, 
                                  num_workers=self.cfg.num_workers, 
                                  persistent_workers=True, pin_memory=True)
                          for dataset in self.test_dataset]
        self.test_dataset_names = [ds.name for ds in self.test_dataset]
        return dataloaders
    
    def get_dataset(self, subset):
        pad_mode = 'label' if (subset != 'eval' or self.cfg.fast_eval) else None
        # sampling_ratio = getattr(self.cfg, f'{subset}_ratio', None)
        dataset_cfgs = self.cfg.datasets[f"{subset}_set"]
        split_segment_ratio = getattr(self.cfg, "segment_split_ratio", 1.8)
        datasets = []
        for dcfg in dataset_cfgs:
            if dcfg['name'].startswith('llama'):
                dataset_cls = LLamaPartialSpoofDataset
            else:
                dataset_cls = PartialSpoofDataset
            dataset = dataset_cls(
                name = dcfg['name'],
                label_root = dcfg['label_root'],
                label_resolution = self.cfg.resolution,
                label_maxlength = self.cfg.label_maxlength, # for 4 seconds, might be 200 frames with 0.02s resolution
                root = dcfg['data_root'],
                samplerate = self.cfg.samplerate,
                subset = subset,
                input_maxlength = self.cfg.input_maxlength,
                input_minlength = self.cfg.input_minlength,
                duration_lst = dcfg.get('dur_lst', None),
                split_segment_ratio = split_segment_ratio,
                input_query = '*.wav',
                filter_rules = dcfg['filters'],
                pad_mode = pad_mode, # if subset != 'eval' else None,
                sampling_config = dcfg.get('sampling', {'mode': 'random', 'ratio': 1.0}),  # e.g., {'mode': 'fix', 'ratio': 0.8}
                enable_augment = dcfg["augment"],
                augment_args = self.cfg.augmentors,
                device = f'cuda:{self.cfg.gpu[0]}',
            )
            datasets.append(dataset)
        return datasets