import os
import torch
import soundfile as sf
import torchaudio.transforms as T
import module.aug.meldataset as bigvgan_preprocess
from diffwave.inference import predict as diffwave_predict

CACHE_BASE_DIR = "./cache"
DIFFWAVE_CKPT = "./diffwave-ljspeech-22kHz-1000578.pt"

class BaseReconAugmentor:
    def __init__(self, device='cpu', cache_key=None):
        self.device = device
        self.cache_key = cache_key
        self.cache_dir = None
    
    def is_cache_enabled(self):
        return self.cache_key is not None and self.cache_dir is not None
    
    def build_cache(self):
        if self.cache_key is None:
            print("Reconstruction-based Augmentor: cache_key is None, disable cache")
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

    def _call_transform(self, sig):
        raise NotImplementedError("Subclasses should implement this method")
    
    def transform(self, sig, utt_id=None):
        cache_enabled = self.is_cache_enabled() and utt_id is not None
        # try to load from cache
        if cache_enabled:
            cached_sig = self.load_cache(utt_id)
            if cached_sig is not None:
                return cached_sig 
        # not cached or cache disabled, process the signal
        if not isinstance(sig, torch.Tensor):
            sig = torch.tensor(sig, dtype=torch.float32)
        transformed_sig = self._call_transform(sig)
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


class WaveGlowAugmentor(BaseReconAugmentor):
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

    def _call_transform(self, sig):
        sig = sig.to(self.device).unsqueeze(0)  # Add batch dimension
        sig = self.upsample(sig)
        mel = self.mel_func(sig)
        with torch.no_grad():
            sig_inv = self.model.infer(mel)
        sig_inv = self.downsample(sig_inv)
        sig_inv = sig_inv.squeeze(0)  # Remove batch dimension
        return sig_inv.cpu()

class DiffWaveAugmentor(BaseReconAugmentor):
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

    def _call_transform(self, sig):
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


class GriffinLimAugmentor(BaseReconAugmentor):
    def __init__(self, device='cpu', cache_key=None, n_fft=320, hop_length=None, freq_distortion=None):
        super().__init__(device, cache_key)
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.freq_distortion = freq_distortion
        self.build_pipelines()

    def build_pipelines(self):
        self.spectrogram_pipe = T.Spectrogram(n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length, power=2.0)
        self.griffin_pipe = T.GriffinLim(n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length, power=2.0)
        self.noise_pipe = T.AddNoise()

    def griffin_lim(self, sig):
        if not isinstance(sig, torch.Tensor):
            sig = torch.tensor(sig, dtype=torch.float32)
        stft = self.spectrogram_pipe(sig)
        # spectral augmentation, randomly increase / decrease the magnitude of the spectrogram on specific frequency bands
        if self.freq_distortion is not None:
            # stft = stft * (1 + torch.randn_like(stft) * 0.1)
            stft = stft * (1.0 + torch.randn((stft.shape[0], 1)) * float(self.freq_distortion))
            # clamp the stft to avoid numerical issues
            stft = torch.clamp(stft, min=1e-12)
        return self.griffin_pipe(stft)
    
    def gaussian_noise(self, sig, snr=15.0):
        if not isinstance(sig, torch.Tensor):
            sig = torch.tensor(sig, dtype=torch.float32)
        gaussian_noise = torch.randn_like(sig)
        return self.noise_pipe(sig, gaussian_noise, snr=torch.tensor(snr))
