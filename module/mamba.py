import torch
import torch.nn as nn

from mambapy.mamba import Mamba, MambaConfig
from mambapy.jamba import Jamba, JambaLMConfig
from module.front import Frontend

class MambaDetector(nn.Module):
    def __init__(self, args, config):
        super(MambaDetector, self).__init__()
        self.config = config
        self.args = args

        pool_frames = int(args.resolution // 0.02)
        assert pool_frames >= 1, "The pooling frames must be greater than 1"
        
        self.ssl_layer = Frontend.load_front_layer(
            ssl_name=config.ssl.name,
            ssl_ckpt=config.ssl.ckpt,
            ssl_out_size=config.ssl.hidden_size,
            output_type=config.ssl.out_type,
            pool_frame_num=pool_frames,
            pool_head_num=config.ssl.pool_head,
            max_frames=args.label_maxlength,
            finetune_mode=config.ssl.finetune_mode,
            # lora_rank=config.ssl.lora_rank,
            # lora_alpha=config.ssl.lora_alpha,
            # lora_target_modules=config.ssl.lora_target_modules
        )
        in_dim = self.ssl_layer.out_dim
        assert in_dim % 32 == 0, "The input dimension must be multiple of 32"

        self.classifier = MambaClassifier(
            input_dim=in_dim,
            hidden_dim=config.classifier.hidden_size,
            n_layers=config.classifier.n_layers,
            moe=config.classifier.enable_moe
        )

    def forward_x(self, batch):
        x = batch["sig"]
        bdy_pred, frame_pred, reg_loss = self.forward(x)
        return {
            "utt_id": batch['utt_id'],
            "frame_pred": frame_pred,
            "frame_target": batch['frame_label'],
            "frame_length": batch['frame_length'],
            "boundary_pred": bdy_pred,
            "boundary_target": batch['boundary_label'],
            "boundary_length": batch['boundary_length'],
            "reg_loss": reg_loss
        }

    def forward(self, x):
        x = self.ssl_layer(x)
        cls_pred, bdr_pred, reg_loss = self.classifier(x)
        return bdr_pred, cls_pred, reg_loss


class MambaClassifier(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 n_layers=2,
                 cls_num=2,
                 moe=True):
        super(MambaClassifier, self).__init__()
        
        # define projection
        self.input_proj = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_dim, hidden_dim)
        )

        # Define Mamba backbone as classifier
        if moe:
            self.mamba_encoder = Jamba(JambaLMConfig(d_model=hidden_dim, n_layers=n_layers, mlp_size=hidden_dim*4))
        else:
            self.mamba_encoder = Mamba(MambaConfig(d_model=hidden_dim, n_layers=n_layers))
        
        # Define task heads
        self.frame_cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, cls_num)
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_proj(x)
        if isinstance(self.mamba_encoder, Jamba):
            x, router_logits = self.mamba_encoder(x)
            if self.training:
                reg_loss = _load_balancing_loss(router_logits[1])
            else:
                reg_loss = torch.tensor(0.0, device=x.device)
        else:
            x = self.mamba_encoder(x)
            reg_loss = torch.tensor(0.0, device=x.device)
        frame_logits = self.frame_cls_head(x)  # (B, L, 2)
        boundary_logits = self.boundary_head(x).squeeze(-1)   # (B, L)
        return frame_logits, boundary_logits, reg_loss


def _load_balancing_loss(router_logits):
    # balance loss for MoE load balancing
    router_probs = torch.softmax(router_logits, dim=-1)
    load_per_expert = router_probs.sum(dim=1)
    avg_load = load_per_expert.mean()
    load_variance = ((load_per_expert - avg_load) ** 2).sum()
    return load_variance