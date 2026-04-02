import os
import torch
import soundfile as sf
import torchaudio.transforms as T
import module.aug.meldataset as bigvgan_preprocess
from diffwave.inference import predict as diffwave_predict
from module.config import config

CACHE_BASE_DIR = config.get("path.cache")
DIFFWAVE_CKPT = config.get("model.diffwave_ckpt")
BACKGROUND_NOISE_PATH = config.get("path.background_noise")
REVERB_IR_PATH = config.get("path.reverb_ir")

class BaseAugmentor:
    def __init__(self, device='cpu', cache_key=None):
        self.device = device
        self.cache_key = cache_key
        self.cache_dir = None
    
    def is_cache_enabled(self):
        return self.cache_key is not None and self.cache_dir is not None
    
    def build_cache(self):
        if self.cache_key is None:
            print("Augmentor: cache_key is None, disable cache")
            return
        cache_dir = os.path.join(CACHE_BASE_DIR, self.cache_key)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

    def save_cache(self, sig, utt_id):
        if isinstance(sig, torch.Tensor):
            sig = sig.cpu().numpy()
        if len(sig.shape) > 1:
            raise ValueError("Signal must be 1D array")
        cache_sig_path = os.path.join(self.cache_dir, f"{utt_id}.wav")
        # use soundfile to save wav file
        sf.write(cache_sig_path, sig, 16000)

    def load_cache(self, utt_id) -> torch.Tensor:
        cache_sig_path = os.path.join(self.cache_dir, f"{utt_id}.wav")
        if not os.path.exists(cache_sig_path):
            return None
        sig, _ = sf.read(cache_sig_path)
        return torch.tensor(sig, dtype=torch.float32)

    def _call_transform(self, sig, **kwargs):
        raise NotImplementedError("Subclasses should implement this method")
    
    def transform(self, sig, utt_id=None, **kwargs):
        cache_enabled = self.is_cache_enabled() and utt_id is not None
        # try to load from cache
        if cache_enabled:
            cached_sig = self.load_cache(utt_id)
            if cached_sig is not None:
                # print(f"Loaded cached signal for utt_id: {utt_id}")
                return cached_sig 
        # not cached or cache disabled, process the signal
        if not isinstance(sig, torch.Tensor):
            sig = torch.tensor(sig, dtype=torch.float32)
        # print(f"Processing signal for utt_id: {utt_id}")
        transformed_sig = self._call_transform(sig, **kwargs)
        # try to save to cache
        if cache_enabled:
            self.save_cache(transformed_sig, utt_id)
        return transformed_sig


class WaveGlow_Singleton:
    _instance = None
    @classmethod
    def get_instance(cls, device):
        if cls._instance is None:
            waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
            waveglow = waveglow.remove_weightnorm(waveglow)
            waveglow = waveglow.to(device)
            waveglow = waveglow.eval()
            waveglow.share_memory()
            cls._instance = waveglow
        return cls._instance


class WaveGlowAugmentor(BaseAugmentor):
    def __init__(self, device='cpu', cache_key=None):
        super().__init__(device, cache_key)
        self.build_cache()
        self.build_pipelines()

    def build_pipelines(self):
        self.model = WaveGlow_Singleton.get_instance(self.device)
        self.upsample = T.Resample(orig_freq=16000, new_freq=22050).to(self.device)
        self.downsample = T.Resample(orig_freq=22050, new_freq=16000).to(self.device)
    
    def mel_func(self, sig):
        return bigvgan_preprocess.mel_spectrogram(
            sig, n_fft=1024, num_mels=80, sampling_rate=22050, 
            hop_size=256, win_size=1024, fmin=0, fmax=8000)

    def _call_transform(self, sig, **kwargs):
        sig = sig.to(self.device).unsqueeze(0)  # Add batch dimension
        sig = self.upsample(sig)
        mel = self.mel_func(sig)
        with torch.no_grad():
            sig_inv = self.model.infer(mel)
        sig_inv = self.downsample(sig_inv)
        sig_inv = sig_inv.squeeze(0)  # Remove batch dimension
        return sig_inv.cpu()

class DiffWaveAugmentor(BaseAugmentor):
    def __init__(self, device='cpu', cache_key=None):
        super().__init__(device, cache_key)
        self.build_cache()
        self.build_pipelines()

    def build_pipelines(self):
        self.upsample = T.Resample(orig_freq=16000, new_freq=22050).to(self.device)
        self.downsample = T.Resample(orig_freq=22050, new_freq=16000).to(self.device)
        self.melspec_func = T.MelSpectrogram(
            sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256,
            f_min=20.0, f_max=22050 / 2.0, n_mels=80, power=1.0, normalized=True, center=True).to(self.device)

    def _call_transform(self, sig, **kwargs):
        sig = sig.to(self.device).unsqueeze(0)  # Add batch dimension
        sig = self.upsample(sig)
        spectrogram = self.melspec_func(sig)
        spectrogram = 20 * torch.log10(torch.clamp(spectrogram, min=1e-5)) - 20
        spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
        sig_inv, _ = diffwave_predict(spectrogram, DIFFWAVE_CKPT, 
                                      device=torch.device(self.device), fast_sampling=True)
        sig_inv = self.downsample(sig_inv)
        sig_inv = sig_inv.squeeze(0)  # Remove batch dimension
        return sig_inv.cpu()


class GriffinLimAugmentor(BaseAugmentor):
    def __init__(self, device='cpu', cache_key=None, n_fft=320, hop_length=None, freq_distortion=None):
        super().__init__(device, cache_key)
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.freq_distortion = freq_distortion
        self.build_pipelines()

    def build_pipelines(self):
        self.spectrogram_pipe = T.Spectrogram(n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length, power=2.0)
        self.griffin_pipe = T.GriffinLim(n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length, power=2.0)

    def _call_transform(self, sig, **kwargs):
        stft = self.spectrogram_pipe(sig)
        # spectral augmentation, randomly increase / decrease the magnitude of the spectrogram on specific frequency bands
        if self.freq_distortion is not None:
            # stft = stft * (1 + torch.randn_like(stft) * 0.1)
            stft = stft * (1.0 + torch.randn((stft.shape[0], 1)) * float(self.freq_distortion))
            # clamp the stft to avoid numerical issues
            stft = torch.clamp(stft, min=1e-12)
        return self.griffin_pipe(stft)
    
class WaveAugmentor(BaseAugmentor):
    def __init__(self, device='cpu', cache_key=None, transform_type=None, extra_args=None):
        super().__init__(device, cache_key)
        self.transform_type = transform_type
        self.extra_args = extra_args

        self.build_pipelines()

    def build_pipelines(self):
        if self.transform_type == 'gaussian':
            from audiomentations import AddGaussianSNR
            min_snr_db, max_snr_db = self.extra_args if self.extra_args is not None else (5, 40)
            self.transform_func = AddGaussianSNR(min_snr_db=min_snr_db, max_snr_db=max_snr_db, p=1.0)
        elif self.transform_type == 'background':
            from audiomentations import AddBackgroundNoise, PolarityInversion
            min_snr_db, max_snr_db = self.extra_args if self.extra_args is not None else (3, 30)
            self.transform_func = AddBackgroundNoise(sounds_path=BACKGROUND_NOISE_PATH, min_snr_db=min_snr_db, max_snr_db=max_snr_db, noise_transform=PolarityInversion(), p=1.0)
        elif self.transform_type == 'reverb':
            from audiomentations import ApplyImpulseResponse
            self.transform_func = ApplyImpulseResponse(ir_path=REVERB_IR_PATH, p=1.0)
        elif self.transform_type == 'mp3':
            from audiomentations import Mp3Compression
            self.transform_func = Mp3Compression(min_bitrate=16, max_bitrate=96, backend="fast-mp3-augment", preserve_delay=False, p=1.0)
        elif self.transform_type == 'pitch':
            from audiomentations import PitchShift
            self.transform_func = PitchShift(min_semitones=-5.0, max_semitones=5.0, p=1.0)
        elif self.transform_type == 'filter':
            from audiomentations import LowPassFilter, HighPassFilter, BandPassFilter
            filter_type = self.extra_args[0] if self.extra_args is not None else 'lowpass'
            min_cutoff, max_cutoff = self.extra_args[1] if self.extra_args is not None else (150, 7500)
            if filter_type == 'lowpass':
                self.transform_func = LowPassFilter(max_cutoff_freq=max_cutoff, p=1.0)
            elif filter_type == 'highpass':
                self.transform_func = HighPassFilter(min_cutoff_freq=min_cutoff, p=1.0)
            elif filter_type == 'bandpass':
                self.transform_func = BandPassFilter(min_center_freq=min_cutoff, max_center_freq=max_cutoff, p=1.0)
            else:
                raise ValueError(f"Unsupported filter type: {filter_type}")
        elif self.transform_type == 'clip':
            from audiomentations import ClippingDistortion
            self.transform_func = ClippingDistortion(0, 15, p=1.0)
        elif self.transform_type == 'time_mask':
            from audiomentations import TimeMask
            self.transform_func = TimeMask(min_band_part=0.05, max_band_part=0.15, p=1.0)
        else:
            raise ValueError(f"Unsupported transform_type: {self.transform_type}")
    
    def _call_transform(self, sig, **kwargs):
        if isinstance(sig, torch.Tensor):
            sig = sig.cpu().numpy()
        sig = self.transform_func(samples=sig, sample_rate=16000)
        return torch.tensor(sig, dtype=torch.float32)