import torch
import torch.nn as nn
import torch.nn.functional as F

from mambapy.mamba import Mamba, MambaConfig
from module.front import Frontend
from module.nn.pooling import AttentionPooling, DownsampPooling

class PartialDetector(nn.Module):
    def __init__(self, args, config):
        super(PartialDetector, self).__init__()
        self.config = config
        self.args = args

        pool_frames = int(args.resolution // 0.02)
        assert pool_frames >= 1, "The pooling frames must be greater than 1"
        
        self.ssl_layer = Frontend.load_front_layer(
            ssl_name=config.ssl.name,
            ssl_ckpt=config.ssl.ckpt,
            ssl_out_size=config.ssl.hidden_size,
            output_type=config.ssl.out_type,
            pool_frame_num=1, # do not downsample here
            pool_head_num=1,  # pool head num
            max_frames=args.label_maxlength,
            finetune_mode=config.ssl.finetune_mode,
            lora_rank=config.ssl.lora_rank,
            lora_alpha=config.ssl.lora_alpha,
            lora_target_modules=config.ssl.lora_target_modules
        )
        in_dim = self.ssl_layer.out_dim
        assert in_dim % 32 == 0, "The input dimension must be multiple of 32"
        self.classifier = DualHeadClassifier(
            input_dim=in_dim,
            proj_dim=config.classifier.proj_size,
            hidden_dim=config.classifier.hidden_size,
            n_layers=config.classifier.n_layers,
            topk=config.classifier.topk,
            compression_mode=config.classifier.compression_mode,
            pool_frames=pool_frames,
            pool_type=config.classifier.pool
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


class DualHeadClassifier(nn.Module):
    def __init__(self, 
                 input_dim, 
                 proj_dim,
                 hidden_dim, 
                 n_layers=2,
                 cls_num=2,
                 topk=5,
                 compression_mode='svd',
                 pool_frames=1,
                 pool_head=1,
                 pool_type='attention'):
        super(DualHeadClassifier, self).__init__()
        self.topk = topk
        self.compression_mode = compression_mode
        self.pool_frames = pool_frames
        self.pool_nhead = pool_head
        self.pool_type = pool_type

        # define projection: input_dim -> expanded hidden_dim -> hidden_dim
        self.feed_forward = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, hidden_dim),
        )

        if self.compression_mode == 'ae':
            self.ae_encoder = AutoEncoder(input_dim, ae_dim=self.topk)
            self.ae_fc = nn.Sequential(
                nn.Linear(self.ae_encoder.ae_dim, hidden_dim),
                nn.GELU()
            )

        # Define Mamba backbone as classifier
        self.mamba_encoder = Mamba(MambaConfig(d_model=hidden_dim, n_layers=n_layers))
        
        # Pooling
        if pool_type == 'attention':
            self.pooling = AttentionPooling(hidden_dim, num_head=1, pool_frames=pool_frames) if pool_frames > 1 else None
        elif pool_type in ['avg', 'max', 'adaptive_avg', 'adaptive_max']:
            self.pooling = DownsampPooling(mode=pool_type, reduction_factor=pool_frames) if pool_frames > 1 else None
        else:
            raise ValueError(f"Unsupported pooling type: {pool_type}")
        
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

    def decompose(self, X, k=5):
        """
        X: (B, T, D)
        """
        B, T, D = X.shape
        device = X.device
        dtype = X.dtype
        
        mu = X.mean(dim=1, keepdim=True)     # (B, 1, D)
        Xc = X - mu                           # (B, T, D)
        Xc = torch.clamp(Xc, min=-1e12, max=1e12)

        def perform_svd(matrix):
            U, S, Vh = torch.linalg.svd(matrix.to(torch.float64), full_matrices=False)
            return U.to(dtype), S.to(dtype), Vh.to(dtype)
    
        try:
            U, S, Vh = perform_svd(Xc)
        except RuntimeError as e:
            # In case of SVD not converging, add small noise
            print("Warning: SVD did not converge, adding small noise to input.", e)
            try:
                eps = 1e-6
                noise = torch.eye(T, D, device=device).unsqueeze(0) * eps
                U, S, Vh = perform_svd(Xc + noise)
            except RuntimeError as e2:
                print("Error: SVD failed twice. Using fallback.", e2)
                X_low = X.detach() 
                X_high = torch.zeros_like(X)
                return X_low, X_high, (None, None, None)
        
        if torch.isnan(S).any():
            print("Error! NaN encountered in SVD decomposition.")
            return X.detach(), torch.zeros_like(X), (U, S, Vh)

        U_k = U[:, :, :k]                     # (B, T, k)
        S_k = torch.diag_embed(S[:, :k])      # (B, k, k)
        Vh_k = Vh[:, :k, :]                   # (B, k, D)

        X_low = U_k @ S_k @ Vh_k + mu         # (B, T, D)
        X_high = X - X_low

        return X_low, X_high, (U, S, Vh)

    def forward(self, x):
        compression = self.compression_mode
        reg_loss = torch.tensor(0.0, device=x.device)
        if compression == 'svd':
            try:
                x_low, _, _ = self.decompose(x, k=self.topk)
            except Exception as e:
                print("[Error] SVD decomposition failed. Using original features.", e)
                x_low = torch.zeros_like(x) # discard this data
            x = x_low
        elif compression == 'ae':
            z, _, reg_loss_ae = self.ae_encoder(x.detach())
            reg_loss += reg_loss_ae

        # Feed forward to expand the dimension and enhance representation
        if compression == 'ae':
            x = self.feed_forward(x) + self.ae_fc(z)
        else:
            x = self.feed_forward(x)

        h = self.mamba_encoder(x)
            
        if self.pool_type == 'attention':
            h = self.pooling(h)
        elif self.pool_type.startswith('adaptive'):
            h = self.pooling(h, output_length=h.size(1)//self.pool_frames)
        else:
            h = self.pooling(h, output_length=None)
        
        frame_logits = self.frame_cls_head(h)  # (B, L, 2)
        boundary_logits = self.boundary_head(h).squeeze(-1)   # (B, L)
        return frame_logits, boundary_logits, reg_loss


class AutoEncoder(nn.Module):
    
    def __init__(self, input_dim, ae_dim=128,
                 ae_alpha=[1.0, 1.0, 0.0001]):
        super(AutoEncoder, self).__init__()
        self.alphas = ae_alpha
        self.ae_dim = ae_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim //2),
            nn.GELU(),
            nn.Linear(self.input_dim//2, self.input_dim//4),
            nn.GELU(),
            nn.Linear(self.input_dim//4, self.ae_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.ae_dim, self.input_dim//4),
            nn.GELU(),
            nn.Linear(self.input_dim//4, self.input_dim//2),
            nn.GELU(),
            nn.Linear(self.input_dim//2, self.input_dim),
            # nn.Tanh()  # Assuming input is normalized between -1 and 1
        )

    def loss(self, x, x_hat, z, alpha, monitor=None):
        recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    
        cos_sim_loss = 1 - F.cosine_similarity(x, x_hat, dim=-1).mean()
        
        sparse_reg = torch.norm(z, p=1)

        if monitor is not None:
            sparsity_ratio = (torch.abs(z) < 1e-5).float().mean()
            monitor("sparsity", sparsity_ratio.item(), on_step=True)
        return recon_loss * alpha[0] + cos_sim_loss * alpha[1] + sparse_reg * alpha[2]

    def forward(self, x:torch.Tensor, x_ref:torch.Tensor=None): 
        # x shape of (B, T, D)
        if x_ref is None:
            x_ref = x.detach()
        z = self.encoder(x)
        x_hat = self.decoder(z)
        reg_loss = self.loss(x_ref, x_hat, z, alpha=self.alphas)
        return z, x_hat, reg_loss