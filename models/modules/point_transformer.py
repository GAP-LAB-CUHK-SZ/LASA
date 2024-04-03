from torch import nn, einsum
import torch
import torch.nn.functional as F
from einops import rearrange,repeat
from timm.models.layers import DropPath
from torch_cluster import fps
import numpy as np

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        if context is None:
            context = x

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(timestep)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = AdaLayerNorm(dim)
        self.norm2 = AdaLayerNorm(dim)
        self.norm3 = AdaLayerNorm(dim)
        self.checkpoint = checkpoint

        init_values = 0
        drop_path = 0.0


        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.ls3 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, t, context=None):
        x = self.drop_path1(self.ls1(self.attn1(self.norm1(x, t)))) + x
        x = self.drop_path2(self.ls2(self.attn2(self.norm2(x, t), context=context))) + x
        x = self.drop_path3(self.ls3(self.ff(self.norm3(x, t)))) + x
        return x

class LatentArrayTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, in_channels, t_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, out_channels=None, context_dim2=None,
                 block=BasicTransformerBlock):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.t_channels = t_channels

        self.proj_in = nn.Linear(in_channels, inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [block(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
             for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(inner_dim)

        if out_channels is None:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels, bias=False))
        else:
            self.num_cls = out_channels
            self.proj_out = zero_module(nn.Linear(inner_dim, out_channels, bias=False))

        self.context_dim = context_dim

        self.map_noise = PositionalEmbedding(t_channels)

        self.map_layer0 = nn.Linear(in_features=t_channels, out_features=inner_dim)
        self.map_layer1 = nn.Linear(in_features=inner_dim, out_features=inner_dim)

        # ###
        # self.pos_emb = nn.Embedding(512, inner_dim)
        # ###

    def forward(self, x, t, cond, class_emb):

        t_emb = self.map_noise(t)[:, None]
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))

        x = self.proj_in(x)
        #print(class_emb.shape,t_emb.shape)
        for block in self.transformer_blocks:
            x = block(x, t_emb+class_emb[:,None,:], context=cond)

        x = self.norm(x)

        x = self.proj_out(x)
        return x

class PointTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, in_channels, t_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, out_channels=None, context_dim2=None,
                 block=BasicTransformerBlock):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.t_channels = t_channels

        self.proj_in = nn.Linear(in_channels, inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [block(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
             for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(inner_dim)

        if out_channels is None:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels, bias=False))
        else:
            self.num_cls = out_channels
            self.proj_out = zero_module(nn.Linear(inner_dim, out_channels, bias=False))

        self.context_dim = context_dim

        self.map_noise = PositionalEmbedding(t_channels)

        self.map_layer0 = nn.Linear(in_features=t_channels, out_features=inner_dim)
        self.map_layer1 = nn.Linear(in_features=inner_dim, out_features=inner_dim)

        # ###
        # self.pos_emb = nn.Embedding(512, inner_dim)
        # ###

    def forward(self, x, t, cond=None):

        t_emb = self.map_noise(t)[:, None]
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))

        x = self.proj_in(x)

        for block in self.transformer_blocks:
            x = block(x, t_emb, context=cond)

        x = self.norm(x)

        x = self.proj_out(x)
        return x
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2))  # B x N x C
        return embed


class PointEncoder(nn.Module):
    def __init__(self,
        dim=512,
        num_inputs = 2048,
        num_latents = 512,
        latent_dim = 512):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim)
        self.proj=nn.Linear(dim,latent_dim)
    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs

        ###### fps
        flattened = pc.view(B * N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        ratio = 1.0 * self.num_latents / self.num_inputs

        idx = fps(pos, batch, ratio=ratio)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        ######

        sampled_pc_embeddings = self.point_embed(sampled_pc)

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        return self.proj(x)