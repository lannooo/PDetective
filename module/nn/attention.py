import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    """ attention among ssl encoder output features."""
    def __init__(self, in_dim, out_dim, head_num):
        super(SelfAttentionLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.head_num = head_num

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, head_num)

        # project
        self.proj_with_att = nn.Linear(head_num * in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # activate
        self.act = nn.SELU(inplace=True)

    def forward(self,x):
        att_map = self._derive_att_map(x)
        att_map = F.softmax(att_map, dim=-2)
        x = self._project(x, att_map)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _project(self, x, att_map):
        x1 = torch.matmul(att_map.permute(0,3,1,2).contiguous(), x.unsqueeze(1).expand(-1,self.head_num,-1,-1))
        x1 = self.proj_with_att(x1.permute(0,2,3,1).flatten(start_dim=2))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        return att_map

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror