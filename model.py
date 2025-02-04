import torch
import torch.nn as nn
from typing import Optional
import math

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=16384, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]

class LlamaAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = 16
        self.head_dim = 128
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings
        rot_emb = self.rotary_emb(query_states, seq_length)
        query_states = query_states * torch.cos(rot_emb) + torch.roll(query_states, shifts=1, dims=-1) * torch.sin(rot_emb)
        key_states = key_states * torch.cos(rot_emb) + torch.roll(key_states, shifts=1, dims=-1) * torch.sin(rot_emb)
        
        # Compute attention
        attention_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        
        attention_weights = torch.softmax(attention_weights, dim=-1)
        hidden_states = torch.matmul(attention_weights, value_states)
        
        hidden_states = hidden_states.reshape(batch_size, seq_length, self.hidden_size)
        hidden_states = self.o_proj(hidden_states)
        
        return hidden_states

class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.self_attn = LlamaAttention(hidden_size)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = LlamaRMSNorm(hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config['hidden_size'], config['intermediate_size'])
            for _ in range(config['num_hidden_layers'])
        ])
        self.norm = LlamaRMSNorm(config['hidden_size'])
        self.rotary_emb = LlamaRotaryEmbedding(config['hidden_size'] // config['num_attention_heads'])

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        hidden_states = self.norm(hidden_states)
        return hidden_states

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        return loss if loss is not None else logits

def create_model():
    config = {
        'vocab_size': 32256,
        'hidden_size': 2048,
        'num_hidden_layers': 24,
        'num_attention_heads': 16,
        'intermediate_size': 5504,
        'max_position_embeddings': 16384,
        'rms_norm_eps': 1e-6,
        'bos_token_id': 32013,
        'eos_token_id': 32014,
    }
    
    model = LlamaForCausalLM(config)
    return model
