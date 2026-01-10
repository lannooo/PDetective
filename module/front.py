import torch
import torch.nn as nn
import torch.nn.functional as F
# Loading HuggingFace Models Locally, uncomment them if loading from local
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ['HF_DATASETS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import Wav2Vec2Model, WavLMModel
from peft import LoraConfig, get_peft_model

from module.nn.pooling import SelfWeightedPooling


def freeze_automatically(model, force_freeze_all=False):
    from transformers.modeling_utils import PreTrainedModel

    def freeze(weights):
        for param in weights:
            param.requires_grad = False

    if isinstance(model, PreTrainedModel):
        if not force_freeze_all and hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
            print("Transformers Model > Freeze the feature encoder")
        else:
            freeze(model.parameters())
            print("Transformers Model > Freeze the entire model")
    elif isinstance(model, torch.nn.Module):
        print("Other Model > Freeze the entire module")
        freeze(model.parameters())
    


class Frontend(nn.Module):
    def __init__(self, 
                 padding_ms=5, 
                 normalize=False, 
                 sampling_rate=16000, 
                 max_frames=200):
        super(Frontend, self).__init__()
        self.padding_ms = padding_ms
        self.normalize = normalize
        self.sampling_rate = sampling_rate
        self.max_frames = max_frames
    
    def forward(self, x):
        out = self.preprocess(x)
        out = self.extract(out)
        out = self.postprocess(out)
        return out
    
    def preprocess(self, x):
        # padding at the head and tail of the input
        pad_length = int(self.sampling_rate * self.padding_ms / 1000)
        x = F.pad(x, (0, pad_length), mode='constant', value=0)   # (B, T)
        if self.normalize:
            x = F.layer_norm(x, x.size()[1:])
        return x
    
    def postprocess(self, x):
        # weighted pooling in child classes
        return x
    
    def extract(self, x):
        raise NotImplementedError("extract method is not implemented")
        
    @staticmethod
    def load_front_layer(ssl_name:str, ssl_ckpt:str, ssl_out_size:int, **kwargs):
        if ssl_name.lower() == "wavlm":
            extractor = WavLMModel.from_pretrained(ssl_ckpt)
            return Transformer_Frontend(extractor, ssl_out_size, **kwargs)
        if ssl_name.lower() == 'wav2vec2':
            extractor = Wav2Vec2Model.from_pretrained(ssl_ckpt)
            return Transformer_Frontend(extractor, ssl_out_size, **kwargs)
        raise ValueError(f"Unknown ssl_name: {ssl_name}")


class SSL_Out:
    TYPE_LAST_HIDDEN_STATE = "hidden_state"
    TYPE_CUSTOM_HIDDEN_STATES = "hidden_states-"

class Transformer_Frontend(Frontend):
    def __init__(self, 
                 model,
                 out_dim,
                 output_type=SSL_Out.TYPE_LAST_HIDDEN_STATE,
                 pool_frame_num=1,
                 pool_head_num=1,
                 normalize_input=True, 
                 padding_ms=5,  # only pad 5 ms, because the Wav2Vec2/WavLM/Hubert models have a 25ms frame length & 20ms hop length
                 max_frames=200,
                 **kwargs):
        super(Transformer_Frontend, self).__init__(
            padding_ms=padding_ms, 
            normalize=normalize_input, 
            max_frames=max_frames)
        self.extractor = model
        self.output_type = output_type
        self.out_dim = out_dim * pool_head_num
        self.pool_nframe = pool_frame_num
        self.pool_nhead = pool_head_num
        self.weighted_pool = SelfWeightedPooling(out_dim, num_head=pool_head_num, mean_only=True) if self.pool_nframe > 1 else None
        # Support finetuning the SSL model with the following modes:
        # - fix: no finetune, all parameters are freezed
        # - lora: use LoRA to finetune the model
        # - default: fix feature extractor layer, while transformer layers can be finetuned
        # - all: all parameters are trainable
        sft_mode = kwargs.get("finetune_mode", "default")
        sft_rank = kwargs.get("lora_rank", 8)
        sft_alpha = kwargs.get("lora_alpha", 16)
        sft_target_modules = kwargs.get("lora_target_modules", ["q_proj", "v_proj"])
        if sft_mode == "fix":
            freeze_automatically(self.extractor, force_freeze_all=True)
        elif sft_mode == "default":
            freeze_automatically(self.extractor, force_freeze_all=False)
        elif sft_mode == "lora":
            lora_config = LoraConfig(
                r=sft_rank,  # Rank of the update matrices
                lora_alpha=sft_alpha, # Scaling factor for the LoRA update
                target_modules=sft_target_modules, # Modules to apply LoRA (Q/V proj has better results)
                lora_dropout=0.05, # Dropout probability of the LoRA layers
                bias="none", # Whether to train bias parameters
            )
            self.extractor = get_peft_model(self.extractor, lora_config)
            self.extractor.print_trainable_parameters()
    
    def postprocess(self, x):
        if self.pool_nframe == 1: # no need to pool at all
            return x
        B, _, D = x.size()
        x = x.reshape(-1, self.pool_nframe, D)              # (B, F, D) => (B*F/p, p, D)
        x = self.weighted_pool(x)                           # (B*F/p, p, D) => (B*F/p, D*nhead)
        assert x.size(1) == D * self.pool_nhead, \
            f"Expected {D * self.pool_nhead}, got {x.size(1)}"
        x = x.reshape(B, -1, D * self.pool_nhead)           # (B*F/p, D*nhead) => (B, F/p, D*nhead) ~ (B, F', D')
        return x

    def extract(self, x):
        output_hidden_states = self.output_type.startswith(SSL_Out.TYPE_CUSTOM_HIDDEN_STATES)
        out = self.extractor(x, output_hidden_states=output_hidden_states)

        if self.output_type == SSL_Out.TYPE_LAST_HIDDEN_STATE:
            return out.last_hidden_state
        
        elif self.output_type.startswith(SSL_Out.TYPE_CUSTOM_HIDDEN_STATES):
            state_type = self.output_type.split("-")
            if state_type[1] == '*': # return all
                return out.hidden_states
            else:
                states_ids = state_type[1].split(",")
                states_ids = [int(i) for i in states_ids]
                outputs = [out.hidden_states[i] for i in states_ids]
                return outputs
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")
