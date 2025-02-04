import torch
import torch.nn as nn
from typing import Optional
import math
import torch.nn.functional as F
from torch.distributions import Categorical

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

class MLHAttention(nn.Module):
    """Multi-head Linear Attention"""
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        # Linear projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to (batch_size, num_heads, seq_length, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply ELU + 1
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Linear attention
        # (batch_size, num_heads, seq_length, head_dim) @ (batch_size, num_heads, head_dim, seq_length)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # (batch_size, num_heads, seq_length, seq_length) @ (batch_size, num_heads, seq_length, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to (batch_size, seq_length, hidden_size)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.hidden_size)
        
        # Final linear projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output

class MoELayer(nn.Module):
    """Mixture of Experts with Loss-less Load Balancing"""
    def __init__(self, hidden_size, intermediate_size, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        
        # Create experts
        self.experts = nn.ModuleList([
            LlamaMLP(hidden_size, intermediate_size) 
            for _ in range(num_experts)
        ])
        
        # Router network
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Load balancing loss coefficient
        self.balance_coef = 0.01
        
    def forward(self, hidden_states):
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Get router logits
        router_logits = self.router(hidden_states)
        
        # Compute routing probabilities
        routing_weights = F.softmax(router_logits, dim=-1)
        
        # Compute load balancing loss
        # Ensure uniform expert utilization
        mean_routing = routing_weights.mean(dim=0)
        target_routing = torch.ones_like(mean_routing) / self.num_experts
        balance_loss = F.mse_loss(mean_routing, target_routing) * self.balance_coef
        
        # Select top-k experts (here k=2)
        top_k = 2
        top_k_weights, top_k_indices = torch.topk(routing_weights, top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Compute weighted sum of expert outputs
        expert_outputs = torch.zeros_like(hidden_states)
        for i, expert in enumerate(self.experts):
            # Compute mask for current expert
            expert_mask = (top_k_indices == i).any(dim=-1).float()
            if expert_mask.any():
                # Only process tokens routed to this expert
                expert_output = expert(hidden_states)
                expert_outputs += expert_output * routing_weights[..., i:i+1]
        
        return expert_outputs, balance_loss

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
    def __init__(self, hidden_size=256, intermediate_size=512):
        super().__init__()
        self.self_attn = MLHAttention(hidden_size)
        self.moe = MoELayer(hidden_size, intermediate_size)
        self.input_layernorm = LlamaRMSNorm((hidden_size,), eps=1e-6)
        self.post_attention_layernorm = LlamaRMSNorm((hidden_size,), eps=1e-6)

    def forward(self, hidden_states, attention_mask=None):
        # Self Attention with MLHA
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MoE layer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, balance_loss = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, balance_loss

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
        total_balance_loss = 0
        
        for layer in self.layers:
            hidden_states, balance_loss = layer(hidden_states, attention_mask)
            total_balance_loss += balance_loss
            
        hidden_states = self.norm(hidden_states)
        return hidden_states, total_balance_loss

class LlamaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing=True):
        self.gradient_checkpointing = gradient_checkpointing
        
    def generate(
        self,
        input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        pad_token_id=None,
        eos_token_id=None,
    ):
        batch_size = input_ids.shape[0]
        current_length = input_ids.shape[1]
        
        # Initialize generated sequences with input_ids
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - current_length):
                # Get model outputs
                outputs, _ = self.forward(generated)  # Ignore balance loss during generation
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Sample from the logits
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                # Append new tokens
                generated = torch.cat([generated, next_tokens], dim=1)
                
                # Check if EOS token was generated
                if eos_token_id is not None and (next_tokens == eos_token_id).any():
                    break
                
        return generated
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Use torch.utils.checkpoint if gradient checkpointing is enabled
        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            hidden_states, balance_loss = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.model),
                input_ids,
                attention_mask,
                use_reentrant=False
            )
        else:
            hidden_states, balance_loss = self.model(input_ids, attention_mask)
            
        logits = self.lm_head(hidden_states)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            loss = loss_fct(logits, labels) + balance_loss  # Add balance loss to total loss
            return loss
        return logits, balance_loss  # Return both logits and balance loss

def create_model():
    # Configuration for ~10M parameters
    config = {
        'vocab_size': 32256,
        'hidden_size': 256,       # Reduced from 512
        'num_hidden_layers': 4,   # Reduced from 8
        'num_attention_heads': 4,  # Reduced from 8
        'intermediate_size': 512,  # Reduced from 1536
        'max_position_embeddings': 512,
        'rms_norm_eps': 1e-6,
        'bos_token_id': 32013,
        'eos_token_id': 32014,
    }
    
    model = LlamaForCausalLM(config)
    return model
