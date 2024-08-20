import torch
from torch import nn
from argparse import Namespace
import xformers.ops as xops
from transformers.activations import ACT2FN

from torch import Tensor
from torch.nn import functional as F

import vllm._C.ops as ops
from flash_attn import flash_attn_varlen_func


def rms_norm(hidden_states: Tensor, normalized_shape, weight: Tensor, eps: float = 1e-6, residual: torch.Tensor = None):
    if residual is not None:
        ops.fused_add_rms_norm(
            hidden_states,
            residual,
            weight,
            eps,
        )
        return hidden_states, residual
    else:
        output = torch.empty_like(hidden_states)
        ops.rms_norm(output, hidden_states, weight, eps)
        return output


class LlamaRMSNorm(nn.LayerNorm):
    """Rewrite RMSNorm."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, hidden_states, residual: torch.Tensor = None):
        """forward."""
        # torch.nn.functional.normalize based implementation might leads
        # to wrong output
        ret = rms_norm(hidden_states, self.normalized_shape, self.weight, self.eps, residual)
        return ret


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)

    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim ** -0.5
        self.query_key_value = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, x: "tensor(B, L, D)", cu_seqlens_q) -> "tensor(B, L, D)":
        B, L, _ = x.shape
        # qkv = self.query_key_value(x)
        qkv = torch.matmul(x, self.query_key_value.weight.data)
        if self.query_key_value.bias is not None:
            qkv += self.query_key_value.bias
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)  # 3, B, L, H, D
        q, k, v = qkv[0], qkv[1], qkv[2]
        _, seq_len_q, H, D = q.shape

        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)
        # attn = attn.softmax(-1)
        # attn = F.dropout(attn, 0.0)
        # out = attn @ v
        win_size = (-1, -1)
        out = flash_attn_varlen_func(q.view(-1, H, D), 
                                     k.view(-1, H, D), 
                                     v.view(-1, H, D),
                                     cu_seqlens_q=cu_seqlens_q,
                                     cu_seqlens_k=cu_seqlens_q,
                                     max_seqlen_q=seq_len_q,
                                     max_seqlen_k=seq_len_q,
                                     softmax_scale=self.scale, 
                                     window_size=win_size)
        # out = xops.memory_efficient_attention(
        #     q, k, v, scale=self.scale,
        # )

        # output = self.dense(out.view(B, L, -1))
        output = torch.matmul(out.view(B, L, -1), self.dense.weight.data)
        if self.dense.bias is not None:
            output += self.dense.bias
        output = self.output_dropout(output)
        return output

    def attention(self, q, k, v):
        attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn_weights = attn_weights.softmax(dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.fc1(x)
        x = torch.matmul(x, self.fc1.weight.data)
        if self.fc1.bias is not None:
            x += self.fc1.bias
        x = self.activation_fn(x)
        # x = self.fc2(x)
        x = torch.matmul(x, self.fc2.weight.data)
        if self.fc2.bias is not None:
            x += self.fc2.bias
        return x


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, cu_seqlens_q):
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.attention(attention_input, cu_seqlens_q))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        seq_len = hidden_states.shape[1]
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=hidden_states.device)
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, cu_seqlens_q)
        return hidden_states


class GLU(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, config.hidden_size, bias=False)
        self.norm1 = LlamaRMSNorm(config.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        # x = self.linear_proj(x)
        x = torch.matmul(x, self.linear_proj.weight.data)
        if self.linear_proj.bias is not None:
            x += self.linear_proj.bias
        x = self.act1(self.norm1(x))
        # x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        # weight = torch.cat((self.gate_proj.weight.t(), self.dense_h_to_4h.weight.t()), dim=-1)
        # t = torch.matmul(x, weight)
        t = torch.matmul(x, self.gate_dense_weight)
        d = t.shape[-1] // 2
        output_shape = (t.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        ops.silu_and_mul(out, t)
        # x = self.dense_4h_to_h(out)
        x = torch.matmul(out, self.dense_4h_to_h.weight.data)
        if self.dense_4h_to_h.bias is not None:
            x += self.dense_4h_to_h.bias
        return x


class EVA2CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)
        self.transformer = Transformer(vision_config)
        self.linear_proj = GLU(config, in_features=vision_config.hidden_size)
        self.conv = nn.Conv2d(in_channels=vision_config.hidden_size, out_channels=vision_config.hidden_size, kernel_size=2, stride=2)
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]

        b, s, h = x.shape
        grid_size = int(s**0.5)
        x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
        x = self.conv(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.linear_proj(x)
        boi = self.boi.expand(x.shape[0], -1, -1)
        eoi = self.eoi.expand(x.shape[0], -1, -1)
        x = torch.cat((boi, x, eoi), dim=1)
        return x
