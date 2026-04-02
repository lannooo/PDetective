import torch
import torch.nn as nn

from module.front import Frontend
from module.nn.gap import MessageControlGraphAttentionLayer
from module.nn.attention import SelfAttentionLayer

class BAM(nn.Module):
    def __init__(self, args, config):
        super(BAM, self).__init__()
        self.config = config
        self.args = args

        pool_frames = int(args.resolution // 0.02)   # scale for pre-downsample, e.g., 0.02s/frame -> 0.16s/frame
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
        )
        in_dim = self.ssl_layer.out_dim
        assert in_dim % 32 == 0, "The input dimension must be multiple of 32"
        # if in_dim % 32 != 0:   # Error for BAM/ResNet module if not multiple of 32
        #     in_dim = in_dim + (32 - in_dim % 32)
        self.classifier = Classifier(
            in_dim=in_dim,
            out_dim=config.classifier.hidden_size,
            resnet_hidden_size=config.classifier.resnet_channels,
            attn_head_num=config.classifier.attn_head,
            gap_layer_num=config.classifier.gap_layers,
            cut_gradient=config.classifier.cut_gradient
        )
    
    def forward_x(self, batch):
        x = batch['sig']
        bdy_pred, frame_pred, embedding = self.forward(x, return_embedding=True)
        return {
            "utt_id": batch['utt_id'],
            "frame_pred": frame_pred,
            "frame_target": batch['frame_label'],
            "frame_length": batch['frame_length'],
            "frame_emb": embedding,
            "emb_frame_target": batch['frame_label'],
            "emb_frame_length": batch['frame_length'],
            "boundary_pred": bdy_pred,
            "boundary_target": batch['boundary_label'],
            "boundary_length": batch['boundary_length'],
        }

    def forward(self, x, return_embedding=False, ssl_perturb=None, perturb_level=0.01):
        x = self.ssl_layer(x)     # output x shape (B, F, D)
        
        if return_embedding:
            # return the embedding from ssl / final linear layer
            prob_pred, cls_pred, embedding = self.classifier(x, return_embedding=True)
        else:
            prob_pred, cls_pred = self.classifier(x)
            embedding = None
        # print(prob_pred.shape, cls_pred.shape)  # (B, F), (B, F, 2)
        return prob_pred, cls_pred, embedding

class Classifier(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 resnet_hidden_size=32, 
                 cls_num=2, 
                 attn_head_num=1, 
                 gap_layer_num=2,
                 cut_gradient=False):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cut_gradient = cut_gradient
        self.att_layer = SelfAttentionLayer(in_dim=out_dim, out_dim=out_dim, head_num=attn_head_num)
        self.resnet_layer = ResnetLayer(in_channel=resnet_hidden_size, in_dim=out_dim, out_dim=out_dim)
        self.proj = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.selu = nn.SELU()
        
        self.prediction_layer = nn.Sequential(
            nn.Linear(in_features=2*out_dim, out_features=1),
            nn.Sigmoid()
        )
        self.gap_layers = nn.ModuleList([
            MessageControlGraphAttentionLayer(in_dim=out_dim, out_dim=out_dim, head_num=attn_head_num)
            for _ in range(gap_layer_num)
        ])
        # Prediction the classification results of e.g., bonafide, spoof, non-speech, etc.
        self.fusion_proj = nn.Linear(in_features=2*out_dim, out_features=out_dim)
        self.out_layer = nn.Linear(in_features=2*out_dim, out_features=cls_num)


    def forward(self, x, return_embedding=False):
        x = self.proj(x)
        x_inter = self.att_layer(x)
        x_intra = self.resnet_layer(x)
        h1 = torch.cat([x_inter, x_intra], dim=-1)
        prob_pred = self.prediction_layer(h1).squeeze(-1)
        prob_message = torch.where(prob_pred.detach() > 0.5, 1, 0)
        # add the message control layers from BAM's implementation but not provide label supervision
        h2 = x
        for layer in self.gap_layers:
            h2 = layer(h2, prob_message)
        if self.cut_gradient:
            h1 = h1.detach()
        h1_ = self.selu(self.fusion_proj(h1))
        embedding = torch.cat([h2, h1_], dim=-1)
        cls_pred = self.out_layer(embedding)
        if return_embedding:
            return prob_pred, cls_pred, embedding
        return prob_pred, cls_pred



class ResnetLayer(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_channel, in_dim, out_dim):
        super(ResnetLayer, self).__init__()
        self.resnet = ResNet1D(BottleNeck,[2,2,2,2], in_channel=in_channel)
        self.proj_layer = nn.Linear(in_features=in_dim*32, out_features=out_dim)

    def forward(self,x):
        m_batch, T, D = x.size()
        out = self.resnet(x.reshape(-1, 1, D))
        out = self.proj_layer(out.view(m_batch, T, -1))
        return out


class BottleNeck(nn.Module):
    expansion = 2
    def __init__(self, in_channel, channel, stride=1, downsample=None):
        super().__init__()

        self.conv1=nn.Conv1d(in_channel, channel, kernel_size=1, stride=stride, bias=False)
        self.bn1=nn.BatchNorm1d(channel)

        self.conv2=nn.Conv1d(channel, channel, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn2=nn.BatchNorm1d(channel)

        self.conv3=nn.Conv1d(channel, channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3=nn.BatchNorm1d(channel*self.expansion)

        self.relu=nn.ReLU(False)

        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        residual=x

        out=self.relu(self.bn1(self.conv1(x)))   #bs, c,  h, w
        out=self.relu(self.bn2(self.conv2(out))) #bs, c,  h, w
        out=self.relu(self.bn3(self.conv3(out))) #bs, 4c, h, w

        if(self.downsample != None):
            residual=self.downsample(residual)

        out = out + residual
        return self.relu(out)

class ResNet1D(nn.Module):
    def __init__(self,block=BottleNeck,layers=[2,2,2,2],in_channel=32):
        super().__init__()
        self.in_channel=in_channel
        ### stem layer
        self.conv1=nn.Conv1d(1, in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1=nn.BatchNorm1d(in_channel)
        self.relu=nn.ReLU(False)
        self.maxpool=nn.MaxPool1d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        ### main layer
        self.layer1=self._make_layer(block, 64,  layers[0])
        self.layer2=self._make_layer(block, 128, layers[1], stride=2)
        self.layer3=self._make_layer(block, 256, layers[2], stride=2)
        self.layer4=self._make_layer(block, 512, layers[3], stride=2)

    def forward(self,x):
        # x shape (B*T, 1, D)
        ##stem layer
        out=self.relu(self.bn1(self.conv1(x))) #bs,112,112,64
        out=self.maxpool(out) #bs,56,56,64

        ##layers:
        out=self.layer1(out) #bs,56,56,64*4
        out=self.layer2(out) #bs,28,28,128*4
        out=self.layer3(out) #bs,14,14,256*4
        out=self.layer4(out) #bs,7,7,512*4

        return out

    def _make_layer(self, block, channel, blocks, stride=1):
        if(stride != 1 or self.in_channel != channel*block.expansion):
            self.downsample=nn.Conv1d(self.in_channel,channel*block.expansion,stride=stride,kernel_size=1,bias=False)
        layers=[]
        layers.append(block(self.in_channel ,channel, downsample=self.downsample, stride=stride))
        self.in_channel=channel*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)