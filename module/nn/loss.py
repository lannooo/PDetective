import torch
import torch.nn as nn
import torch.nn.functional as F

class PassThroughLoss(nn.Module):
    """
    A loss function that does nothing, used for debugging or placeholder.
    """
    def __init__(self, ):
        super(PassThroughLoss, self).__init__()

    def forward(self, pred, target, mask=None):
        # Simply return zero loss
        return torch.tensor(0.0, device=pred.device)
    

class EmbeddingLoss(nn.Module):
    def __init__(self):
        super(EmbeddingLoss, self).__init__()
        self.th_similar_min = 0.9
        self.th_different_max = 0.1

    def cosine_similarity(self, x1, x2, eps=1e-8):
        '''
        pair-wise cosine distance
        x1: [M, D]
        x2: [N, D]
        similarity: [M, N]
        '''
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        similarity = torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
        return similarity

    def forward(self, embeddings, frame_labels, pad_mask):
        '''
        embeddings: [B, C, F, scale]
        frame labels: [B, F]
        pad_mask: [B, F]
        '''
        loss_batch = torch.tensor(0.0, dtype=torch.float32, device=embeddings.device)
        # scale: ratio of label sequence to the embedding sequence
        num_batch = embeddings.size()[0]
        emb_size = embeddings.size()[1]
        for i_batch, mask in enumerate(pad_mask):
            embedding = embeddings[i_batch]   # [32, 50, 8]

            valid_labels = frame_labels[i_batch, mask.bool()]  # obtain the true length before zero padding
            real_mask = torch.where(valid_labels == 1)  # real frame position
            fake_mask = torch.where(valid_labels == 0)  # fake frame position
            real_embeddings_list = [torch.empty([emb_size, 0], device=embedding.device)]
            fake_embeddings_list = [torch.empty([emb_size, 0], device=embedding.device)]
            for i in real_mask[0]:
                real_embeddings_list.append(embedding[:, i])
            for i in fake_mask[0]:
                fake_embeddings_list.append(embedding[:, i])
            real_embeddings = torch.cat(real_embeddings_list, dim=1)  # concat all real embedding frames
            fake_embeddings = torch.cat(fake_embeddings_list, dim=1)  # concat all fake embedding frames

            r_embedding = real_embeddings.t()  # [M, D]
            f_embedding = fake_embeddings.t()
            if real_embeddings.size()[1] == 0 or fake_embeddings.size()[1] == 0:  # if no real/fake frames, skip
                continue
            
            # all fake embedings should be similar
            sim_f2f = self.cosine_similarity(f_embedding, f_embedding)
            sim_f2f_hard = torch.min(sim_f2f, dim=1)[0]
            zero = torch.zeros_like(sim_f2f_hard)
            loss_f2f = torch.max(self.th_similar_min - sim_f2f_hard, zero)
            loss_f2f = loss_f2f.mean()

            # all real embeddings should be similar
            sim_r2r = self.cosine_similarity(r_embedding, r_embedding)
            sim_r2r_hard = torch.min(sim_r2r, dim=1)[0]
            zero = torch.zeros_like(sim_r2r_hard)
            loss_r2r = torch.max(self.th_similar_min - sim_r2r_hard, zero)
            loss_r2r = loss_r2r.mean()

            # fake embeddings should be different with real embeddings
            sim_f2r = self.cosine_similarity(f_embedding, r_embedding)
            # f2r
            sim_f2r_hard = torch.max(sim_f2r, dim=1)[0]
            zero = torch.zeros_like(sim_f2r_hard)
            loss_f2r = torch.max(sim_f2r_hard - self.th_different_max, zero)
            loss_f2r = loss_f2r.mean()
            # r2f
            sim_r2f_hard = torch.max(sim_f2r, dim=0)[0]
            zero = torch.zeros_like(sim_r2f_hard)
            loss_r2f = torch.max(sim_r2f_hard - self.th_different_max, zero)
            loss_r2f = loss_r2f.mean()

            loss_batch = loss_batch + loss_f2f + loss_r2r + loss_f2r + loss_r2f

        loss_batch = loss_batch / num_batch
        return loss_batch


class CRLoss(nn.Module):
    """
    CRLoss (Modified from ICASSP 2024 Xie et al.)
    Fixed several critical issues in the original implementation:
    1. Fixed a critical bug where all-real/all-fake samples were skipped after computing the loss.
    2. Added pad_mask parameter to support correct truncation of variable-length sequences.
    3. Replaced cumbersome torch.max(..., zero) with F.relu.
    4. Optimized cosine similarity computation to prevent NaN due to division by zero.
    """
    def __init__(self):
        super(CRLoss, self).__init__()
        self.th_similar_min = 0.9
        self.th_different_max = 0.1

    def forward(self, embeddings, frame_labels, pad_mask=None):
        """
        args:
            embeddings: [Batch, Time, Dim]
            frame_labels: [Batch, Time] (1 for Real, 0 for Fake)
            pad_mask: [Batch, Time] (1 for valid, 0 for padding)
        """
        loss_batch = 0.0
        num_batch = embeddings.size(0)
        valid_batch_count = 0

        for ibat in range(num_batch):
            # 1. Use pad_mask to extract valid frames for the current utterance
            if pad_mask is not None:
                valid_indices = torch.where(pad_mask[ibat] == 1)[0]
                if len(valid_indices) == 0:
                    continue  # Edge case: all frames are padded
                emb = embeddings[ibat, valid_indices]  # [Valid_T, Dim]
                lbl = frame_labels[ibat, valid_indices]       # [Valid_T]
            else:
                emb = embeddings[ibat]
                lbl = frame_labels[ibat]

            # 2. Split real and fake frames
            real_indices = torch.where(lbl == 1)[0]
            fake_indices = torch.where(lbl == 0)[0]
            
            R_emb = emb[real_indices]  # [Num_R, Dim]
            F_emb = emb[fake_indices]  # [Num_F, Dim]

            loss_ibat = 0.0

            # 3. L2 normalize valid frames for cosine similarity computation
            if R_emb.size(0) > 0:
                R_emb_norm = F.normalize(R_emb, p=2, dim=1)
            if F_emb.size(0) > 0:
                F_emb_norm = F.normalize(F_emb, p=2, dim=1)

            # === Compute positive similarity (Positive Mining) ===
            # R2R: All real frames should be similar
            if R_emb.size(0) > 0:
                sim_r2r = torch.mm(R_emb_norm, R_emb_norm.t())
                sim_r2r_hard = torch.min(sim_r2r, dim=1)[0]
                loss_r2r = F.relu(self.th_similar_min - sim_r2r_hard).mean()
                loss_ibat += loss_r2r

            # F2F: All fake frames should be similar
            if F_emb.size(0) > 0:
                sim_f2f = torch.mm(F_emb_norm, F_emb_norm.t())
                sim_f2f_hard = torch.min(sim_f2f, dim=1)[0]
                loss_f2f = F.relu(self.th_similar_min - sim_f2f_hard).mean()
                loss_ibat += loss_f2f

            # === Compute negative similarity (Negative Mining) ===
            # Only compute negative loss when both real and fake frames exist (partially fake)
            if R_emb.size(0) > 0 and F_emb.size(0) > 0:
                # Cross similarity matrix [Num_F, Num_R]
                sim_cross = torch.mm(F_emb_norm, R_emb_norm.t())

                # F2R: For each fake frame, find the most similar real frame
                sim_f2r_hard = torch.max(sim_cross, dim=1)[0]
                loss_f2r = F.relu(sim_f2r_hard - self.th_different_max).mean()

                # R2F: For each real frame, find the most similar fake frame
                sim_r2f_hard = torch.max(sim_cross, dim=0)[0]
                loss_r2f = F.relu(sim_r2f_hard - self.th_different_max).mean()

                loss_ibat += (loss_f2r + loss_r2f)

            loss_batch += loss_ibat
            valid_batch_count += 1

        # Avoid division by 0
        loss_batch = loss_batch / max(valid_batch_count, 1)
        return loss_batch


class TripletContrastiveLoss(nn.Module):
    """
    Frame-wise Triplet Loss based on Euclidean distance.

    For each anchor frame, select the hardest positive (farthest) from the same label,
    and the hardest negative (closest) from different labels, then compute:
        max(0, margin + ||ea - ep||_2 - ||ea - en||_2)
    """
    def __init__(self, margin=0.1, eps=1e-8):
        super(TripletContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, embeddings, frame_labels, pad_mask=None):
        """
        Parameters:
            embeddings: [Batch, Time, Dim]
            frame_labels: [Batch, Time] (1 for Real, 0 for Fake)
            pad_mask: [Batch, Time] (1 for valid, 0 for padding)
        """
        loss_batch = torch.tensor(0.0, device=embeddings.device)
        valid_batch_count = 0

        for ibat in range(embeddings.size(0)):
            if pad_mask is not None:
                valid_mask = pad_mask[ibat] == 1
                if not torch.any(valid_mask):
                    continue
                emb = embeddings[ibat][valid_mask]
                lbl = frame_labels[ibat][valid_mask]
            else:
                emb = embeddings[ibat]
                lbl = frame_labels[ibat]

            if emb.size(0) < 2:
                continue

            dist_mat = torch.cdist(emb, emb, p=2)
            same_label = lbl.unsqueeze(0) == lbl.unsqueeze(1)
            diff_label = ~same_label

            same_label.fill_diagonal_(False)

            valid_anchor = same_label.any(dim=1) & diff_label.any(dim=1)
            if not torch.any(valid_anchor):
                continue

            hardest_positive = dist_mat.masked_fill(~same_label, float('-inf')).max(dim=1).values
            hardest_negative = dist_mat.masked_fill(~diff_label, float('inf')).min(dim=1).values

            triplet_loss = F.relu(self.margin + hardest_positive - hardest_negative)
            loss_batch = loss_batch + triplet_loss[valid_anchor].mean()
            valid_batch_count += 1

        if valid_batch_count == 0:
            return embeddings.sum() * 0.0
        return loss_batch / valid_batch_count


class TemporalContrastiveLoss(nn.Module):
    """
    Boundary-aware Temporal Contrastive Loss, supporting two computation modes:
    1. penalty: standard TCL, using cosine similarity to separately constrain intra-class smoothness and inter-class abrupt changes.
    2. triplet: For each anchor, only the previous and next frames are used to construct a local triplet loss based on Euclidean distance.
    """
    def __init__(self, pos_margin=0.8, neg_margin=0.2, smooth_weight=1.0, boundary_weight=2.0,
                 mode='penalty', triplet_margin=0.1, eps=1e-8):
        super(TemporalContrastiveLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.smooth_weight = smooth_weight
        self.boundary_weight = boundary_weight
        self.mode = mode
        self.triplet_margin = triplet_margin
        self.eps = eps

        if self.mode not in {'penalty', 'triplet'}:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def _forward_penalty(self, embeddings, frame_labels, pad_mask=None):
        # 1. Extract adjacent frame pairs
        emb_t = embeddings[:, :-1, :]   # [B, T-1, Dim]
        emb_t1 = embeddings[:, 1:, :]   # [B, T-1, Dim]
        lbl_t = frame_labels[:, :-1]    # [B, T-1]
        lbl_t1 = frame_labels[:, 1:]    # [B, T-1]

        # 2. Compute cosine similarity [-1, 1]
        sim_adjacent = F.cosine_similarity(emb_t, emb_t1, dim=-1)

        # 3. Distinguish between same class and boundary
        is_same = (lbl_t == lbl_t1).float()
        is_boundary = (lbl_t != lbl_t1).float()

        # 4. Handle Padding Mask
        if pad_mask is not None:
            valid_transition = (pad_mask[:, :-1] == 1).float() * (pad_mask[:, 1:] == 1).float()
            is_same = is_same * valid_transition
            is_boundary = is_boundary * valid_transition

        # 5. Intra-class smoothness loss
        num_same = is_same.sum()
        if num_same > 0:
            loss_smooth = (is_same * F.relu(self.pos_margin - sim_adjacent)).sum() / num_same.clamp_min(self.eps)
        else:
            loss_smooth = embeddings.sum() * 0.0

        # 6. Boundary abruptness loss
        num_boundary = is_boundary.sum()
        if num_boundary > 0:
            loss_boundary = (is_boundary * F.relu(sim_adjacent - self.neg_margin)).sum() / num_boundary.clamp_min(self.eps)
        else:
            loss_boundary = embeddings.sum() * 0.0

        return (self.smooth_weight * loss_smooth) + (self.boundary_weight * loss_boundary)

    def _forward_triplet(self, embeddings, frame_labels, pad_mask=None):
        loss_batch = embeddings.sum() * 0.0
        valid_batch_count = 0

        for ibat in range(embeddings.size(0)):
            if pad_mask is not None:
                valid_mask = pad_mask[ibat] == 1
                if not torch.any(valid_mask):
                    continue
                emb = embeddings[ibat][valid_mask]
                lbl = frame_labels[ibat][valid_mask]
            else:
                emb = embeddings[ibat]
                lbl = frame_labels[ibat]

            if emb.size(0) < 3:
                continue

            anchor = emb[1:-1]
            prev_emb = emb[:-2]
            next_emb = emb[2:]

            anchor_lbl = lbl[1:-1]
            prev_lbl = lbl[:-2]
            next_lbl = lbl[2:]

            prev_dist = torch.norm(anchor - prev_emb, p=2, dim=-1)
            next_dist = torch.norm(anchor - next_emb, p=2, dim=-1)

            prev_pos = anchor_lbl == prev_lbl
            next_pos = anchor_lbl == next_lbl
            prev_neg = ~prev_pos
            next_neg = ~next_pos

            has_positive = prev_pos | next_pos
            has_negative = prev_neg | next_neg
            valid_anchor = has_positive & has_negative
            if not torch.any(valid_anchor):
                continue

            positive_candidates = torch.stack([
                prev_dist.masked_fill(~prev_pos, float('-inf')),
                next_dist.masked_fill(~next_pos, float('-inf')),
            ], dim=1)
            negative_candidates = torch.stack([
                prev_dist.masked_fill(~prev_neg, float('inf')),
                next_dist.masked_fill(~next_neg, float('inf')),
            ], dim=1)

            hardest_positive = positive_candidates.max(dim=1).values
            hardest_negative = negative_candidates.min(dim=1).values
            triplet_loss = F.relu(self.triplet_margin + hardest_positive - hardest_negative)

            loss_batch = loss_batch + triplet_loss[valid_anchor].mean()
            valid_batch_count += 1

        if valid_batch_count == 0:
            return embeddings.sum() * 0.0
        return loss_batch / valid_batch_count
    
    def forward(self, embeddings, frame_labels, pad_mask=None):
        """
        Parameters:
            embeddings: [Batch, Time, Dim] - Features output by the projection head
            frame_labels: [Batch, Time] - Frame-level labels (e.g., 1 for true, 0 for false)
            pad_mask: [Batch, Time] - Mask (1 for valid frames, 0 for padding frames)
        Returns:
            total_loss: Scalar Tensor
        """
        if self.mode == 'triplet':
            return self._forward_triplet(embeddings, frame_labels, pad_mask)
        return self._forward_penalty(embeddings, frame_labels, pad_mask)

class MaskCrossEnrtopyLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskCrossEnrtopyLoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction='none',)

    def forward(self, pred, target, mask=None):
        loss = self.ce(pred, target)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.sum() / loss.numel()
        return loss
    

class MaskMSELoss(nn.Module):
    def __init__(self):
        super(MaskMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')

    def forward(self, pred, target, mask=None):
        loss = self.mse(pred, target)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.sum() / loss.numel()
        return loss


class WeightedMSELoss(nn.Module):
    def __init__(self, positive_weight=1.0, negative_weight=1.0, positive_value=1.0, eps=1e-8):
        super(WeightedMSELoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.positive_value = positive_value
        self.eps = eps

    def forward(self, pred, target, mask=None):
        loss = self.mse(pred, target)

        if mask is not None:
            loss = loss * mask

        positive_index = (target == self.positive_value).float()
        negative_index = (target != self.positive_value).float()
        if mask is not None:
            positive_index = positive_index * mask
            negative_index = negative_index * mask

        positive_count = positive_index.sum()
        negative_count = negative_index.sum()

        has_positive = positive_count > 0
        has_negative = negative_count > 0

        if has_positive and has_negative:
            positive_loss = (loss * positive_index).sum() / positive_count.clamp_min(self.eps)
            negative_loss = (loss * negative_index).sum() / negative_count.clamp_min(self.eps)
            return (self.positive_weight * positive_loss) + (self.negative_weight * negative_loss)

        valid_count = positive_count + negative_count
        if valid_count <= 0:
            return loss.sum() * 0.0

        return loss.sum() / valid_count.clamp_min(self.eps)


class MaskBCELoss(nn.Module):
    def __init__(self):
        super(MaskBCELoss, self).__init__()
        self.bce = torch.nn.BCELoss(reduction='none')

    def forward(self, pred, target, mask=None):
        loss = self.bce(pred, target)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.sum() / loss.numel()
        return loss


class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
    
    def forward(self, pred, target, mask=None):
        loss = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        if mask is not None:
            loss = loss * mask
        positive_index = (target == 1.0).float()
        negative_index = (target < 1.0).float()
        pos_cnt = int(positive_index.sum())
        neg_cnt = int(negative_index.sum())
        pos_loss = loss * positive_index
        neg_loss = loss * negative_index
        pos_weight = neg_cnt / (pos_cnt + neg_cnt)
        neg_weight = pos_cnt / (pos_cnt + neg_cnt)
        balance_loss = pos_weight * pos_loss.sum() + neg_weight * neg_loss.sum()
        return balance_loss


class BalanceBCELoss(nn.Module):
    def __init__(self, negative_ratio=5.0, eps=1e-8):
        super(BalanceBCELoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred, target, mask=None):
        # print(pred.shape, target.shape, mask.shape)
        
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7) # prevent log(0)
        loss = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        loss = loss * mask
        positive_index = (target == 1.0).float()
        negative_index = (target < 1.0).float()
        positive_count = int(positive_index.sum())
        negative_count = min(int(negative_index.sum()), int(max(positive_count, 1) * self.negative_ratio))
        positive_loss = loss * positive_index
        negative_loss = loss * negative_index
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        # balance_loss = (positive_loss.sum() + negative_loss.sum()) /\
        #     (positive_count + negative_count + self.eps)
        total = positive_count + negative_count
        if total == 0:
            print(f"WARNING! total count is zero! {positive_count}/{negative_count}/{positive_index}/{negative_index}")
            return 0.0
        balance_loss = positive_loss.sum() * (negative_count / total) + negative_loss.sum() * (positive_count / total)

        return balance_loss


class P2SGradLoss(nn.Module):
    def __init__(self):
        super(P2SGradLoss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')

    def forward(self, pred, target, mask=None):
        # target (batchsize, 1)
        target = target.long()
        
        # filling in the one-hot target
        # target_onehot (batchsize, 2)
        print(target.shape, pred.shape)
        with torch.no_grad():
            target_onehot = torch.zeros_like(pred)
            if target.dim() == 1:
                B = target.shape[0]
                target_onehot.scatter_(dim=1, index=target.data.view(B, 1), value=1)
            else:
                B, L = target.shape
                target_onehot.scatter_(dim=2, index=target.data.view(B, L, 1), value=1)

        # MSE between \cos\theta and one-hot vectors
        loss = self.mse(pred, target_onehot)
        
        if mask is not None:
            # mask shape should be broadcastable with loss
            # if mask is (batchsize,) and loss is (batchsize, 2), expand mask
            if mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1)  # (batchsize, 1)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.sum() / loss.numel()
        
        return loss


def make_loss_fn(cfg, key):
    loss_name = cfg['name']
    loss_weight = cfg['weight']
    loss_args = cfg['args'] if 'args' in cfg else {}
    loss_type = cfg['type']
    if loss_type == 'mask_ce':     # for frame-level loss
        loss_cls = MaskCrossEnrtopyLoss
    elif loss_type == 'mask_bce':  # for frame-level loss
        loss_cls = MaskBCELoss
    elif loss_type == 'mask_mse':  # for frame-level loss
        loss_cls = MaskMSELoss
    elif loss_type == 'weighted_mse':  # for imbalanced frame-level regression/classification loss
        loss_cls = WeightedMSELoss
    elif loss_type == 'p2s':       # for frame-level loss
        loss_cls = P2SGradLoss
    elif loss_type == 'balance':   # for boundary loss
        loss_cls = BalanceBCELoss
    elif loss_type == 'embedding':  # for embedding loss
        loss_cls = EmbeddingLoss
    elif loss_type == 'tcl':   # for embedding temporal contrastive loss
        loss_cls = TemporalContrastiveLoss
    elif loss_type == 'crl':
        loss_cls = CRLoss
    elif loss_type == 'triplet':
        loss_cls = TripletContrastiveLoss
    elif loss_type == 'ce':   # for utterance-level loss
        loss_cls = nn.CrossEntropyLoss
    elif loss_type == 'direct': 
        loss_cls = PassThroughLoss
    else:
        raise ValueError(f"Unknown loss type: {loss_type} in loss config: {key}")
    loss = loss_cls(**loss_args)
    return loss_name, loss, loss_weight
