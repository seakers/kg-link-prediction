import torch
from torch import nn
from torch.nn.parameter import Parameter, Tensor
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

from numba import cuda
class MultiheadAttentionRelationBias(nn.Module):

    def __init__(self,
                 d_emb,
                 n_heads,
                 relation_bias,
                 dropout,
                 bias=False,
                 d_k = None,
                 d_v = None,
                 device = None,
                 dtype = None) -> None:
        
        if d_emb <= 0 or n_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={d_emb} and num_heads={n_heads} instead"
            )
        
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_emb = d_emb
        self.d_k = d_k if d_k is not None else d_emb
        self.d_v = d_v if d_v is not None else d_emb
        self.relation_bias = relation_bias

        self.n_heads = n_heads
        self.dropout = dropout
        self.d_head = d_emb // n_heads
        assert( self.d_head * n_heads == self.d_emb), "d_emb must be divisible by n_heads"

        if not (self.d_k == self.d_emb and self.d_v == self.d_emb):

            self.q_proj_weight = Parameter(torch.empty((d_emb, n_heads*d_k), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((d_emb, n_heads*d_k), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((d_emb, n_heads*d_v), **factory_kwargs))
        
        else:
            self.in_proj_weight = Parameter(torch.empty((3* d_emb, d_emb), **factory_kwargs))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3*d_emb, **factory_kwargs))

        self.out_proj_weight = Parameter(torch.empty((n_heads*d_v, d_emb), **factory_kwargs))
        #self.out_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(d_emb, d_emb, bias=bias, **factory_kwargs)


        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        # if self.bias_k is not None:
        #     xavier_normal_(self.bias_k)
        # if self.bias_v is not None:
        #     xavier_normal_(self.bias_v)

    def forward(self,
                query,
                key,
                value
                ) -> torch.Tensor:
        query = cuda.as_cuda_array(query)                            ### Might need to name variables differently to
        key = cuda.as_cuda_array(key)                                ### not overalap with other variables in the class
        value = cuda.as_cuda_array(value)
        cuda_q_proj_weight = cuda.as_cuda_array(self.q_proj_weight)
        cuda_k_proj_weight = cuda.as_cuda_array(self.k_proj_weight)
        cuda_k_proj_weight = cuda.as_cuda_array(self.v_proj_weight)
        cuda_relation_bias = cuda.as_cuda_array(self.relation_bias)

        Q = self.project_parallel(query,cuda_q_proj_weight)
        K_t = self.transpose(self.project_parallel(key, cuda_k_proj_weight))
        QK_t = self.add_parallel(self.project_parallel(Q,K) , cuda_relation_bias)
        
        V = self.transpose(self.project_parallel(cuda_k_proj_weight,value))
        attention_scores = self.project_parallel(self.softmax(QK_t) , V)

        """
        NEED TO MAKE ATTENTION SCORES CPU AVAILABLE BEFORE RETURNING
        """
        return attention_scores
        
    
    @cuda.jit
    def project_parallel(mat1,
                        mat2):
        
        # mat1 is a 1xm matrix and mat2 is a mxn matrix

        row = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
        col = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y
        
        return mat1 @ mat2[row]

    @cuda.jit  
    def transpose(mat):
        pass

    @cuda.jit  
    def add_parallel(vec1,
                     vec2):
        pass

    @cuda.jit
    def softmax(mat,
                dim):
        pass