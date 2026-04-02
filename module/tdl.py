import torch
import torch.nn as nn
import torch.nn.functional as F

from module.front import Frontend


class AttentionConv1d(nn.Module):
    def __init__(self, kernel_size, out_channels):
        super(AttentionConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def calculate_similarity(self, embedding, embedding_neighbor):
        similarity = self.cosine_similarity(embedding, embedding_neighbor)
        similarity = torch.unsqueeze(similarity, dim=1)
        return similarity

    def cal_local_attenttion(self, embedding, feature, kernel_size):
        embedding_l = torch.zeros_like(embedding)
        embedding_l[:, :, 1:] = embedding[:, :, :-1]
        similarity_l = self.calculate_similarity(embedding, embedding_l)
        similarity_c = self.calculate_similarity(embedding, embedding)
        embedding_r = torch.zeros_like(embedding)
        embedding_r[:, :, :-1] = embedding[:, :, 1:]
        similarity_r = self.calculate_similarity(embedding, embedding_r)
        similarity = torch.cat([similarity_l, similarity_c, similarity_r], dim=1)  
        # expand for D times
        batch, channel, temporal_length = feature.size()
        similarity_tile = torch.zeros(batch, kernel_size * channel, temporal_length).type_as(feature)
        similarity_tile[:, :channel * 1, :] = similarity[:, :1, :]
        similarity_tile[:, channel * 1:channel * 2, :] = similarity[:, 1:2, :]
        similarity_tile[:, channel * 2:, :] = similarity[:, 2:, :]
        return similarity_tile

    def forward(self, feature, embedding, weight):
        batch, channel, temporal_length = feature.size()
        inp = torch.unsqueeze(feature, dim=3)
        w = torch.unsqueeze(weight, dim=3)

        unfold = nn.Unfold(kernel_size=(self.kernel_size, 1), stride=1, padding=[1, 0])
        inp_unf = unfold(inp)
        # local attention
        attention = self.cal_local_attenttion(embedding, feature, kernel_size=self.kernel_size)
        inp_weight = inp_unf * attention
        inp_unf_t = inp_weight.transpose(1, 2)
        w_t = w.view(w.size(0), -1).t()
        results = torch.matmul(inp_unf_t, w_t)
        out_unf = results.transpose(1, 2)
        out = out_unf.view(batch, self.out_channels, temporal_length)
        return out


class BaseModule(nn.Module):
    def __init__(self, in_dim, p_drop=0.7):
        super(BaseModule, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_1_att = AttentionConv1d(kernel_size=3, out_channels=in_dim)
        self.conv_2 = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2_att = AttentionConv1d(kernel_size=3, out_channels=in_dim)
        self.lrelu = nn.LeakyReLU()
        self.drop_out = nn.Dropout(p_drop)
    
    def forward(self, x, embedding):
        feat1 = self.lrelu(self.conv_1_att(x, embedding, self.conv_1.weight))
        feat2 = self.lrelu(self.conv_2_att(feat1, embedding, self.conv_2.weight))
        feature = self.drop_out(feat2)
        return feat1, feature

class ClassifierModule(nn.Module):
    def __init__(self, in_dim, out_channel=2, downsample_scale=4, fix_frames=50):
        super(ClassifierModule, self).__init__()
        self.fix_frames = fix_frames
        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        if fix_frames is not None:
            flat_dim = int(out_channel * fix_frames * downsample_scale)
            self.fc = nn.Linear(flat_dim, fix_frames)   # global mapping
        else:
            self.fc = nn.Conv1d(in_channels=out_channel, out_channels=1, 
                                kernel_size=downsample_scale,
                                stride=downsample_scale, padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)      # bs, out_channel, L or fix_frames*scale
        if self.fix_frames is not None:
            x = torch.flatten(x, 1)     # bs, out_channel*fix_frames*scale
            x = self.fc(x)              # bs, fix_frames
        else:
            x = self.fc(x)              # bs, 1, L
            x = torch.squeeze(x, dim=1) # bs, L
        out = self.sigmoid(x)  # bs, L
        return out


class EmbeddingModule(nn.Module):
    def __init__(self, in_dim, out_dim=32, hidden_dim=512):
        super(EmbeddingModule, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU()
    def forward(self, x):
        out = self.lrelu(self.conv_1(x))
        out = self.conv_2(out)
        embedding = F.normalize(out, p=2, dim=1)
        return embedding
    

class TDL(nn.Module):
    def __init__(self, args, config):
        super(TDL, self).__init__()
        self.config = config
        self.args = args

        pool_frames = int(args.resolution // 0.02)  # scale for downsampling
        self.scale = pool_frames
        assert pool_frames >= 1, "pool_frames should be greater than or equal to 1"
        
        self.ssl_layer = Frontend.load_front_layer(
            ssl_name=config.ssl.name,
            ssl_ckpt=config.ssl.ckpt,
            ssl_out_size=config.ssl.hidden_size,
            output_type=config.ssl.out_type,
            pool_frame_num=1,   # keep the original resolution, i.e., 0.02s/frame
            pool_head_num=1,
            max_frames=args.label_maxlength * pool_frames,
            finetune_mode=config.ssl.finetune_mode,   # freeze all the parameters of the SSL model
        )
        
        in_dim = self.ssl_layer.out_dim

        self.embedding_module = EmbeddingModule(in_dim, out_dim=config.classifier.emb_dim)
        self.base_module = BaseModule(in_dim)
        self.classifier_module = \
            ClassifierModule(in_dim, out_channel=config.classifier.out_channel, 
                             downsample_scale=self.scale,
                             fix_frames=None if config.classifier.dynamic else config.classifier.nframes)


    def forward(self, x):
        x = self.ssl_layer(x)   # (B, F, D)
        x = x.transpose(1, 2)
        embedding = self.embedding_module(x)
        feature1, feature2 = self.base_module(x, embedding)
        logits = self.classifier_module(feature2)
        # reshape embedding from (B, emb_dim, ori_frames) to (B, emb_dim, ori_frames/p, p)
        nB, nC, nF = embedding.size()
        embedding = embedding.view(nB, nC, int(nF/self.scale), self.scale)
        return embedding, logits

    def forward_x(self, batch):
        x = batch['sig']
        embedding, frame_pred = self.forward(x)
        # print(embedding.shape, frame_pred.shape)
        return {
            "utt_id": batch['utt_id'],
            "frame_pred": frame_pred,
            "frame_target": batch['frame_label'],
            "frame_length": batch['frame_length'],
            "frame_emb": embedding,
            "emb_frame_target": None,
            "emb_frame_length": None
        }

if __name__ == "__main__":
    model = TDL()
    input = torch.randn((8, 200, 1024))   # (B, F, D), 0.02 reso. 200 frames
    embeddings, features = model(input)  
    print(embeddings.shape)   # (8, 32, 50, 4)
    print(features.shape)     # ! (8, 50), 0.08 reso. 50 frames

