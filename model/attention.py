import torch

"""
    Reference: Fast Transformer Decoding: One Write-Head is All You Need
    Reference: Attention is all you need
    these functions are just references. The actual code used is in model.py
"""

def DotProductAttention(q, K, V):
    """ 
    Regular DPA.

    Params:
    m: Length of the input
    k: key dimension
    v: value dimension
    d: embedding dimension (d_model)
    h: number of heads

    Args:
    q: a vector with shape [k]
    K: a matrix with shape [m, k]
    V: a matrix with shape [m, v]

    Returns:
    y: a vector with shape [v]
    """

    logits = torch.einsum('k, mk -> m', K, q)
    key_weights = torch.nn.Softmax(logits)
    return torch.einsum('mv, m -> v', key_weights, V)


def MultiHeadAttentionIncremental(
    x,
    K_prev,
    V_prev,
    P_q,
    P_k,
    P_v,
    P_o):

    """ 
    Incremental MHA for decoding. x comes from the encoder, therefore has 2 dims instead of 3.
    this is the incremental version: K and V are concatenated and fed to the next step.
    The concatenate step isn't mentioned in the original paper, but it makes sense:
    each decoder step should have access to all previous steps, up to the first one 

    Params:
    m: length of the input to the decoder
    k: key dimension
    v: value dimension
    d: embedding dimension (d_model)
    h: number of heads
    b: batch dimension

    Args:
    x: a vector with shape [b, d]
    K_prev: a matrix with shape [b, h, m, k]
    V_prev: a matrix with the shape [b, h, m, v]
    P_q: a tensor with shape [h, d, k]
    P_k: a tensor with shape [h, d, k]
    P_v: a tensor with shape [h, d, v]
    P_o: a tensor with shape [h, d, v]

    Returns : 
    y: a vector with shape [b, d]
    K_new: a matrix with shape [b, m+1, k]
    V_new: a matrix with the shape [b, m+a, v]
    """

    # compute q, K, V. Concat K and V
    q = torch.einsum('d, hdk -> hk', x, P_q)
    K_new = torch.concat([K_prev, torch.expand_dims(torch.einsum('bd, hdk -> bhk', M, P_k), 2)], 2)
    V_new = torch.concat([V_prev, torch.expand_dims(torch.einsum('bd, hdv -> bhv', M, P_v), 2)], 2)

    # perform attention along dimension h (heads)
    logits = torch.einsum('hk, bhmk -> bhm', q, K_new)
    key_weights = torch.nn.Softmax(logits)
    o = torch.einsum('bhmv, hm -> bhv', key_weights, V_new)

    # matmul with our linear layers at the end 
    y = torch.einsum('bhv, hdv -> bd', o, P_o)

    return y, K_new, V_new


def MultiQueryAttentionIncremental(
    x,
    M,
    P_q,
    P_k,
    P_v,
    P_o):
    """ 
    Multi Query Attention: Like the above, but only the query has heads. 
    P_v and P_k are shared across all the heads 

    Params:
    m: length of the input to the decoder
    k: key dimension
    v: value dimension
    d: embedding dimension (d_model)
    h: number of heads
    b: batch dimension

    Args:
    x: a vector with shape [b, d]
    K_prev: a matrix with shape [b, m, k]
    V_prev: a matrix with the shape [b, m, v]
    P_q: a tensor with shape [h, d, k]
    P_k: a tensor with shape [h, d, k]
    P_v: a tensor with shape [h, d, v]
    P_o: a tensor with shape [h, d, v]

    Returns : 
    y: a vector with shape [b, d]
    K_new: a matrix with shape [b, m+1, k]
    V_new: a matrix with the shape [b, m+a, v]
    """

    # compute q, K, V. Concat K and V
    q = torch.einsum('d, hdk -> hk', x, P_q)
    K_new = torch.concat([K_prev, torch.expand_dims(torch.einsum('bd, dk -> bk', M, P_k), 2)], 2)
    V_new = torch.concat([V_prev, torch.expand_dims(torch.einsum('bd, dv -> bv', M, P_v), 2)], 2)

    # perform attention along dimension h (heads)
    logits = torch.einsum('hk, bmk -> bhm', q, K_new)
    key_weights = torch.nn.Softmax(logits)
    o = torch.einsum('bmv, hm -> bhv', key_weights, V_new)

    # matmul with our linear layers at the end 
    y = torch.einsum('bhv, hdv -> bd', o, P_o)

    return y, K_new, V_new