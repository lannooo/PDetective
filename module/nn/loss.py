import torch
import torch.nn as nn

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

    def forward(self, embeddings, frame_length, frame_labels):
        '''
        embeddings: [B, C, F, scale]
        label length: [B, ]
        frame labels: [B, F]
        '''
        loss_batch = torch.tensor(0.0, dtype=torch.float32, device=embeddings.device)
        # scale: ratio of label sequence to the embedding sequence
        num_batch = embeddings.size()[0]
        emb_size = embeddings.size()[1]
        for i_batch, il in enumerate(frame_length):
            embedding = embeddings[i_batch]   # [32, 50, 8]
            ori_length = int(il)

            valid_labels = frame_labels[i_batch, :ori_length]  # obtain the true length before zero padding
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

class PassThroughLoss(nn.Module):
    """
    A loss function that does nothing, used for debugging or placeholder.
    """
    def __init__(self, ):
        super(PassThroughLoss, self).__init__()

    def forward(self, pred, target, mask=None):
        # Simply return zero loss
        return torch.tensor(0.0, device=pred.device)

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
    elif loss_type == 'p2s':       # for frame-level loss
        loss_cls = P2SGradLoss
    elif loss_type == 'balance':   # for boundary loss
        loss_cls = BalanceBCELoss
    elif loss_type == 'embedding':  # for embedding loss
        loss_cls = EmbeddingLoss
    elif loss_type == 'ce':   # for utterance-level loss
        loss_cls = nn.CrossEntropyLoss
    elif loss_type == 'direct': 
        loss_cls = PassThroughLoss
    else:
        raise ValueError(f"Unknown loss type: {loss_type} in loss config: {key}")
    loss = loss_cls(**loss_args)
    return loss_name, loss, loss_weight