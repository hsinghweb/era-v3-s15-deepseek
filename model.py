import torch
import torch.nn as nn
from typing import Optional
import math

class LlamaRMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))
        self.eps = eps
        # Store shape for string representation
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
    
    def __repr__(self):
        return f"LlamaRMSNorm({self.shape}, eps={self.eps})"

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, seq_len=None):
        return x

class LlamaAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = 16  # Increased to match config
        self.head_dim = hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        # Simplified attention computation
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        scaling = 1.0 / math.sqrt(self.head_dim)
        attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scaling
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = torch.softmax(attention_scores, dim=-1)
        hidden_states = torch.matmul(attention_probs, value_states)
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
    def __init__(self, hidden_size=2048, intermediate_size=5504):
        super().__init__()
        self.self_attn = LlamaAttention(hidden_size)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = LlamaRMSNorm((hidden_size,), eps=1e-6)
        self.post_attention_layernorm = LlamaRMSNorm((hidden_size,), eps=1e-6)

    def forward(self, hidden_states, attention_mask=None):
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP
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
        self.norm = LlamaRMSNorm((config['hidden_size'],), eps=1e-6)
        self.rotary_emb = LlamaRotaryEmbedding()

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
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing=True):
        self.gradient_checkpointing = gradient_checkpointing
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Use torch.utils.checkpoint if gradient checkpointing is enabled
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.model),
                input_ids,
                attention_mask,
                use_reentrant=False  # Added explicit parameter
            )
        else:
            hidden_states = self.model(input_ids, attention_mask)
            
        logits = self.lm_head(hidden_states)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Ensure inputs require gradients
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = loss_fct(logits, labels)
            return loss
        return logits

def create_model():
    # Configuration for ~500M parameters
    config = {
        'vocab_size': 32256,
        'hidden_size': 1536,      # Increased from 512
        'num_hidden_layers': 16,  # Increased from 6
        'num_attention_heads': 16, # Increased from 8
        'intermediate_size': 4096, # Increased from 1024
        'max_position_embeddings': 512, # Keep this same for memory efficiency
        'rms_norm_eps': 1e-6,
        'bos_token_id': 32013,
        'eos_token_id': 32014,
    }
    
    model = LlamaForCausalLM(config)
    return model
