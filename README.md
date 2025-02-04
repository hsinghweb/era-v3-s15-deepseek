# DeepSeek Model Implementation and Training

This project implements and trains a DeepSeek language model based on the LLaMA architecture. The implementation matches the exact architecture and parameters of the DeepSeek model.

## Model Architecture

The model follows the LLaMA architecture with the following specifications:

- Base Model: LlamaForCausalLM
- Hidden Size: 2048
- Number of Layers: 24
- Number of Attention Heads: 16
- Intermediate Size: 5504
- Vocabulary Size: 32256
- Maximum Position Embeddings: 16384

### Key Components:
- Embedding Layer
- 24 Decoder Layers, each containing:
  - Self Attention Layer
  - MLP Layer
  - Layer Normalization
- Final Layer Normalization
- Language Model Head

