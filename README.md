# DeepSeek Model Implementation and Training

This project implements and trains a lightweight version of the DeepSeek language model based on the LLaMA architecture with Multi-head Linear Attention (MLHA) and Mixture of Experts (MoE).

## Model Architecture

The model follows a modified LLaMA architecture with the following specifications:

- Base Model: LlamaForCausalLM
- Hidden Size: 128
- Number of Layers: 4
- Number of Attention Heads: 4
- Intermediate Size: 256
- Vocabulary Size: 32256
- Maximum Position Embeddings: 512
- Total Parameters: ~4M

### Key Components:
- Token Embedding Layer (32256 × 128)
- 4 Decoder Layers, each containing:
  - Multi-head Linear Attention (MLHA)
  - Mixture of Experts (MoE) with Load Balancing
  - Layer Normalization
- Final Layer Normalization
- Language Model Head

### Advanced Features:
- **MLHA (Multi-head Linear Attention)**:
  - 4 attention heads
  - Linear attention computation
  - ELU activation for positive features
  - Dropout for regularization

- **MoE (Mixture of Experts)**:
  - 4 expert networks
  - Loss-less load balancing
  - Top-k routing mechanism
  - Balance loss computation

## Training Configuration

### Dataset:
- Cosmopedia-100k from HuggingFace
- Maximum sequence length: 32 tokens
- Batch size: 1 with gradient accumulation

### Training Parameters:
- Learning rate: 5e-5
- Optimizer: AdamW
- Weight decay: 0.1
- Gradient accumulation steps: 8
- Mixed precision training
- Gradient checkpointing enabled

### Checkpointing:
- Saves model every 100 steps
- Maintains latest checkpoint only
- Includes model and optimizer states
- Automatic checkpoint recovery

### Memory Optimization:
- Gradient accumulation
- Mixed precision training
- Memory efficient attention
- Aggressive memory cleanup
- Device-agnostic training

## Usage

1. Install Dependencies:
```bash
pip install torch transformers datasets
```

2. Train the Model:
```bash
python train.py
```

## Model Verification
- Automatic parameter counting
- Architecture verification
- Sample text generation every 100 steps
- Training progress logging every 10 steps

## Features
- Efficient training on limited hardware
- Automatic device detection (CPU/GPU)
- Robust checkpoint management
- Progress monitoring
- Sample generation during training

## References
- DeepSeek Model Architecture
- LLaMA Paper
- Mixture of Experts Literature
- Linear Attention Mechanisms

## Training Logs
```
2025-02-05 16:55:19,550 - INFO - 
Model Architecture:
2025-02-05 16:55:19,550 - INFO - ===================
2025-02-05 16:55:19,550 - INFO - LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32256, 128)
    (layers): ModuleList(
      (0-3): 4 x LlamaDecoderLayer(
        (self_attn): MLHAttention(
          (q_proj): Linear(in_features=128, out_features=128, bias=False)
          (k_proj): Linear(in_features=128, out_features=128, bias=False)
          (v_proj): Linear(in_features=128, out_features=128, bias=False)
          (o_proj): Linear(in_features=128, out_features=128, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (moe): MoELayer(
          (experts): ModuleList(
            (0-3): 4 x LlamaMLP(
              (gate_proj): Linear(in_features=128, out_features=256, bias=False)
              (up_proj): Linear(in_features=128, out_features=256, bias=False)
              (down_proj): Linear(in_features=256, out_features=128, bias=False)
              (act_fn): SiLU()
            )
          )
          (router): Linear(in_features=128, out_features=4, bias=True)
        )
        (input_layernorm): LlamaRMSNorm((128,), eps=1e-06)
        (post_attention_layernorm): LlamaRMSNorm((128,), eps=1e-06)
      )
    )
    (norm): LlamaRMSNorm((128,), eps=1e-06)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=128, out_features=32256, bias=False)
)
2025-02-05 16:55:19,552 - INFO - 
Parameter Counts:
2025-02-05 16:55:19,552 - INFO - =================
2025-02-05 16:55:19,552 - INFO - Total Parameters: 10,095,760
2025-02-05 16:55:19,552 - INFO - Trainable Parameters: 10,095,760
2025-02-05 16:55:19,787 - INFO - Loading Cosmopedia-100k dataset...
2025-02-05 16:55:20,931 - INFO - Loaded 100000 examples
2025-02-05 16:55:20,934 - INFO - Using device: cuda
2025-02-05 16:55:20,966 - INFO - Gradient checkpointing enabled
/content/era-v3-s15-deepseek/train.py:170: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(checkpoint_file, map_location='cpu')
2025-02-05 16:55:21,084 - INFO - Found valid checkpoint: checkpoints/model_step_10000.pt
2025-02-05 16:55:21,226 - INFO - Successfully resumed training from step 10000
2025-02-05 16:55:21,227 - INFO - 
Training will run until step 10100
2025-02-05 16:55:21,227 - INFO - Current step: 10000
2025-02-05 16:55:49,696 - INFO - Progress: 99.1% (10010/10100) - Current Loss: 0.7229
2025-02-05 16:56:17,914 - INFO - Progress: 99.2% (10020/10100) - Current Loss: 0.4185
2025-02-05 16:56:45,890 - INFO - Progress: 99.3% (10030/10100) - Current Loss: 1.2548
2025-02-05 16:57:14,348 - INFO - Progress: 99.4% (10040/10100) - Current Loss: 0.9246
2025-02-05 16:57:42,429 - INFO - Progress: 99.5% (10050/10100) - Current Loss: 0.7827
2025-02-05 16:58:11,110 - INFO - Progress: 99.6% (10060/10100) - Current Loss: 0.1592
2025-02-05 16:58:40,515 - INFO - Progress: 99.7% (10070/10100) - Current Loss: 0.2277
2025-02-05 16:59:10,821 - INFO - Progress: 99.8% (10080/10100) - Current Loss: 0.8101
2025-02-05 16:59:40,942 - INFO - Progress: 99.9% (10090/10100) - Current Loss: 0.7396
2025-02-05 17:00:12,247 - INFO - Progress: 100.0% (10100/10100) - Current Loss: 0.0072
2025-02-05 17:00:12,247 - INFO - 
Step 10100: loss = 0.0072
2025-02-05 17:00:12,247 - INFO - Memory used: 211.3MB
2025-02-05 17:00:12,671 - INFO - Saved checkpoint at step 10100
2025-02-05 17:00:12,671 - INFO - Current Loss: 0.0072
2025-02-05 17:00:12,913 - INFO - 
Generating final samples...
2025-02-05 17:00:13,117 - INFO - 
Generated text:
2025-02-05 17:00:13,117 - INFO - Prompt: Here is an extract from a webpage: "Recording of Present Day:
2025-02-05 17:00:13,117 - INFO - Generated: Here is an extract from a webpage: "Recording of Present Day::::::::::::::::

2025-02-05 17:00:13,241 - INFO - 
Generated text:
2025-02-05 17:00:13,241 - INFO - Prompt: Course Unit: LISA Pathfinder Mission and Gravitational Wave Detection
2025-02-05 17:00:13,241 - INFO - Generated: Course Unit: LISA Pathfinder Mission and Gravitational Wave Detection tra tra tra Economic Economic Economic Economic Economic Economic Economic Economic

2025-02-05 17:00:13,492 - INFO - 
Generated text:
2025-02-05 17:00:13,492 - INFO - Prompt: Title: Making Mathematics Accessible: The Importance
2025-02-05 17:00:13,492 - INFO - Generated: Title: Making Mathematics Accessible: The Importanceanceanceanceanceanceanceanceanceanceanceanceanceanceanceanceanceanceanceanceance

2025-02-05 17:00:13,773 - INFO - 
Generated text:
2025-02-05 17:00:13,774 - INFO - Prompt: The Performing Arts encompass many different forms of artistic
2025-02-05 17:00:13,774 - INFO - Generated: The Performing Arts encompass many different forms of artistic artistic seeking seeking seeking seeking seeking seeking seeking seeking seeking seeking seeking seeking seeking seeking seeking seeking seeking seeking seeking

2025-02-05 17:00:14,059 - INFO - 
Generated text:
2025-02-05 17:00:14,059 - INFO - Prompt: It was a bright, sunny day and Maria was excited to wear
2025-02-05 17:00:14,059 - INFO - Generated: It was a bright, sunny day and Maria was excited to wear Blue Blue Blue Blue Blue Blue Blue Blue Blue Blue Blue Blueames opened opened equation equation equation
```

