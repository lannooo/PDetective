import torch
import torch.nn as nn

from module.front import Frontend

from module.nn.gmlp import gMLP
from module.nn.p2sgrad import P2SActivationLayer


class MultiResoModel(torch.nn.Module): # Wav2Vec2 reso is 20ms per frame
    def __init__(self, args, config):
        super(MultiResoModel, self).__init__()
        self.config = config
        self.args = args

        self.reso_units = config.classifier.units
        self.num_scales  = len(config.classifier.units)
        self.include_utt = config.classifier.include_utt
        pool_frames = int(args.resolution // 0.02)
        assert pool_frames >= 1, "pool_frames should be greater than or equal to 1"
        
        self.ssl_layer = Frontend.load_front_layer(
            ssl_name=config.ssl.name,
            ssl_ckpt=config.ssl.ckpt,
            ssl_out_size=config.ssl.hidden_size,
            output_type=config.ssl.out_type,
            pool_frame_num=1,   # keep the original resolution, i.e., 0.02s/frame
            max_frames=config.classifier.n_frames,
            finetune_mode=config.ssl.finetune_mode,
        )

        max_seq_len = config.classifier.n_frames
        self.ssl_layer_weights = nn.Parameter(torch.ones(30))

        ds_blocks = []
        cl_blocks = []
        ac_blocks = []
        hdim = config.ssl.hidden_size
        for i in range(self.num_scales):
            hdim = config.ssl.hidden_size // pow(2, i)
            edim = hdim // 2
            seq_len = max_seq_len // pow(2, i)
            # Downsampling
            if i == 0:
                ds_blocks.append(nn.Identity())
            else:
                ds_blocks.append(
                    nn.Sequential(
                        nn.MaxPool2d([2,2], [2,2]),
                        nn.Linear(hdim, hdim)
                    )
                )
            # Classifying
            cl_blocks.append(gMLP(hdim, edim, seq_len, gmlp_layers=5))
            # Activation Function
            ac_blocks.append(P2SActivationLayer(edim, 2))

        self.ds_blocks = nn.ModuleList(ds_blocks)
        self.cl_blocks = nn.ModuleList(cl_blocks)
        self.ac_blocks = nn.ModuleList(ac_blocks)

        self.cl_utt = None
        self.ac_utt = None
        if self.include_utt:
            self.cl_utt = nn.Sequential(
                nn.Dropout(0.7),
                gMLP(hdim, edim, seq_len, gmlp_layers=5, flag_pool="mean")
            )
            self.ac_utt = P2SActivationLayer(edim, 2)


    def extract_ssl_feat(self, x):
        outs = self.ssl_layer(x)
        nlayers = len(outs)
        norm_weights = nn.functional.softmax(self.ssl_layer_weights[:nlayers], dim=-1)
        feat = torch.stack(outs, dim=-1)
        feat = feat * norm_weights
        feat = torch.sum(feat, dim=-1)
        return feat

    def forward(self, x):
        logits = []
        hidd = self.extract_ssl_feat(x)

        for i in range(self.num_scales):
            ds_block, cl_block, ac_block = self.ds_blocks[i], self.cl_blocks[i], self.ac_blocks[i]
            hidd  = ds_block(hidd)  # hidden: B x T x D
            B, T, D = hidd.size()
            logit = ac_block(torch.flatten(cl_block(hidd), start_dim=0, end_dim=1))
            # view shape back to (B, T, 2)
            logit = logit.view(B, T, 2)
            logits.append(logit)

        if self.include_utt:
            logit = self.ac_utt(self.cl_utt(hidd))
            logits.append(logit)

        return logits
    
    def forward_x(self, batch):
        x = batch['sig']
        frm_labels = batch['frame_label']    # 0.02s reso.
        frm_lengths = batch['frame_length']  # 0.02s reso.
        # predictions under multiple resolutions (0.02/0.04/0.08/0.16/utt)
        logits = self.forward(x)
        multi_frm_labels = [frm_labels]
        multi_frm_lengths = [frm_lengths]
        for unit in self.reso_units[1:]:
            scale = int(unit // 0.02)
            agg_labels, agg_lengths = aggregate_labels(
                frm_labels, frm_lengths, scale=scale
            )
            multi_frm_labels.append(agg_labels)
            multi_frm_lengths.append(agg_lengths)
        
        utt_preds, utt_labels, utt_lengths = None, None, None
        if self.include_utt:
            utt_labels, utt_lengths = utterance_labels(
                frm_labels, frm_lengths
            )
            utt_preds = logits[-1]
            logits = logits[:-1] # remove the last utt-level prediction
        return {
            "utt_id": batch['utt_id'],
            "frame_pred": logits,
            "frame_target": multi_frm_labels,
            "frame_length": multi_frm_lengths,
            "utt_pred": utt_preds,
            "utt_target": utt_labels,
            "utt_length": utt_lengths
        }

    def get_num_scales(self):
        if self.cl_utt is None:
            return self.num_scales
        return self.num_scales + 1


def aggregate_labels(
    frm_labels: torch.Tensor,   # (B, L)
    frm_length: torch.Tensor,   # (B,)
    scale: int                  # 1, 2, 4, 8
):
    """
    Aggregate frame-level labels to coarser resolution.
    """
    B, L = frm_labels.shape
    device = frm_labels.device

    max_out_len = (L + scale - 1) // scale
    out_labels = torch.zeros(B, max_out_len, device=device, dtype=torch.int32)
    out_lengths = torch.zeros(B, device=device, dtype=torch.int32)
    for b in range(B):
        valid_len = frm_length[b].item()
        num_chunks = (valid_len + scale - 1) // scale
        out_lengths[b] = num_chunks
        for i in range(num_chunks):
            start = i * scale
            end = min(start + scale, valid_len)
            out_labels[b, i] = (torch.sum(frm_labels[b, start:end]) == (end - start)).to(torch.int32)

    return out_labels, out_lengths

def utterance_labels(
    frm_labels: torch.Tensor,  # (B, L)
    frm_length: torch.Tensor   # (B,)
):
    B = frm_labels.size(0)
    utt_labels = torch.zeros(B, device=frm_labels.device, dtype=torch.int32)

    for b in range(B):
        utt_labels[b] = (torch.sum(frm_labels[b, :frm_length[b]]) == frm_length[b]).to(torch.int32)

    return utt_labels, torch.ones(B, device=frm_labels.device, dtype=torch.int32)

