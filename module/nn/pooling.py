import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfWeightedPooling(nn.Module):

    """ SelfWeightedPooling module
    Inspired by
    https://github.com/joaomonteirof/e2e_antispoofing/blob/master/model.py
    To avoid confusion, I will call it self weighted pooling
    
    l_selfpool = SelfWeightedPooling(5, 1, False)
    with torch.no_grad():
        input_data = torch.rand([3, 10, 5])
        output_data = l_selfpool(input_data)
    """
    def __init__(self, feature_dim, num_head=1, mean_only=False):
        """ SelfWeightedPooling(feature_dim, num_head=1, mean_only=False)
        Attention-based pooling
        
        input (batchsize, length, feature_dim) ->
        output 
           (batchsize, feature_dim * num_head), when mean_only=True
           (batchsize, feature_dim * num_head * 2), when mean_only=False
        
        args
        ----
          feature_dim: dimension of input tensor
          num_head: number of heads of attention
          mean_only: whether compute mean or mean with std
                     False: output will be (batchsize, feature_dim*2)
                     True: output will be (batchsize, feature_dim)
        """
        super(SelfWeightedPooling, self).__init__()

        self.feature_dim = feature_dim
        self.mean_only = mean_only
        self.noise_std = 1e-5
        self.num_head = num_head

        # transformation matrix (num_head, feature_dim)
        self.mm_weights = nn.Parameter(torch.Tensor(feature_dim, num_head), requires_grad=True)
        nn.init.kaiming_uniform_(self.mm_weights)
        return
    
    def _forward(self, inputs):
        """ output, attention = forward(inputs)
        inputs
        ------
          inputs: tensor, shape (batchsize, length, feature_dim)
        
        output
        ------
          output: tensor
           (batchsize, feature_dim * num_head), when mean_only=True
           (batchsize, feature_dim * num_head * 2), when mean_only=False
          attention: tensor, shape (batchsize, length, num_head)
        """        
        # batch size
        batch_size = inputs.size(0)
        # feature dimension
        feat_dim = inputs.size(2)
        
        # input is (batch, legth, feature_dim)
        # change mm_weights to (batchsize, feature_dim, num_head)
        # weights will be in shape (batchsize, length, num_head)
        weights = torch.bmm(inputs, self.mm_weights.unsqueeze(0).repeat(batch_size, 1, 1))
        
        # attention (batchsize, length, num_head)
        attentions = nn.functional.softmax(torch.tanh(weights),dim=1)        
        
        # apply attention weight to input vectors
        if self.num_head == 1:
            # We can use the mode below to compute self.num_head too
            # But there is numerical difference.
            #  original implementation in github
            
            # elmentwise multiplication
            # weighted input vector: (batchsize, length, feature_dim)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            # weights_mat = (batch * length, feat_dim, num_head)
            #    inputs.view(-1, feat_dim, 1), zl, error
            #    RuntimeError: view size is not compatible with input tensor's size and stride 
            #    (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            weighted = torch.bmm(
                inputs.reshape(-1, feat_dim, 1), 
                attentions.view(-1, 1, self.num_head))
            
            # weights_mat = (batch, length, feat_dim * num_head)
            weighted = weighted.view(batch_size, -1, feat_dim * self.num_head)
            
        # pooling
        if self.mean_only:
            # only output the mean vector
            representations = weighted.sum(1)
        else:
            # output the mean and std vector
            noise = self.noise_std * torch.randn(
                weighted.size(), dtype=weighted.dtype, device=weighted.device)

            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)
            # concatenate mean and std
            representations = torch.cat((avg_repr,std_repr),1)
        # done
        return representations, attentions
    
    def forward(self, inputs):
        """ output = forward(inputs)
        inputs
        ------
          inputs: tensor, shape (batchsize, length, feature_dim)
        
        output
        ------
          output: tensor
           (batchsize, feature_dim * num_head), when mean_only=True
           (batchsize, feature_dim * num_head * 2), when mean_only=False
        """
        output, _ = self._forward(inputs)
        return output

    def debug(self, inputs):
        return self._forward(inputs)
    

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim, num_head, pool_frames):
        super(AttentionPooling, self).__init__()
        self.pool_frames = pool_frames
        self.pool_nhead = num_head
        self.pooling_layer = SelfWeightedPooling(hidden_dim, num_head=num_head, mean_only=True) 

    def forward(self, x):
        # x: (B, L*pool_frames, D) -> (B, L, D)
        B, L, D = x.size()
        x = x.reshape(-1, self.pool_frames, D)      # (B, F, D) => (B*F/p, p, D)
        x = self.pooling_layer(x)                   # (B*F/p, p, D) => (B*F/p, D*nhead)
        assert x.size(1) == D * self.pool_nhead, f"Expected {D * self.pool_nhead}, got {x.size(1)}"
        x = x.reshape(B, -1, D * self.pool_nhead)   # (B*F/p, D*nhead) => (B, F/p, D*nhead) ~ (B, F', D')
        x = x.contiguous()
        return x
    
class DownsampPooling(nn.Module):
    def __init__(self, mode, reduction_factor=4):
        super(DownsampPooling, self).__init__()
        assert mode in ['avg', 'max', 'adaptive_avg', 'adaptive_max'], f"Unsupported pooling mode: {mode}"
        self.mode = mode
        self.reduction_factor = reduction_factor

    def forward(self, x:torch.Tensor, output_length=None):
        # x: (B, L, D) -> (B, L//factor, D)
        B, L, D = x.size()
        x = x.transpose(1, 2) # (B, D, L)
        if self.mode == 'avg':
            x = F.avg_pool1d(x, kernel_size=self.reduction_factor, stride=self.reduction_factor)
        elif self.mode == 'max':
            x = F.max_pool1d(x, kernel_size=self.reduction_factor, stride=self.reduction_factor)
        elif self.mode == 'adaptive_avg':
            x = F.adaptive_avg_pool1d(x, output_length)
        elif self.mode == 'adaptive_max':
            x = F.adaptive_max_pool1d(x, output_length)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.mode}")
        x = x.transpose(1, 2) # (B, L//factor, D)
        return x
    

class ConvPooling(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int = None,
                 kernel_size: int = 4,
                 stride: int = 4,
                 use_batch_norm: bool = True,
                 activation: str = 'gelu'):
        super(ConvPooling, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim if output_dim is not None else input_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=not use_batch_norm
        )
        self.batch_norm = nn.BatchNorm1d(self.output_dim) if use_batch_norm else None
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
    def forward(self, x: torch.Tensor):
        # B, L, D = x.size()
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv(x)       # (B, D_out, L_out)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # (B, L_out, D_out)
        return x
