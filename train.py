import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
import glob
from model import create_model
from transformers import AutoTokenizer
import torch.nn as nn
from torch.amp import autocast, GradScaler
import gc
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Memory optimization settings
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Add at the very top
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 32):  # Reduced to 32
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loading text from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        # Process text in smaller chunks
        tokens = self.tokenizer.encode(
            self.text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False
        )
        self.chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
        logger.info(f"Created {len(self.chunks)} text chunks")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        
        if len(chunk) < self.max_length:
            chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_length - len(chunk))
        
        input_ids = torch.tensor(chunk)
        labels = input_ids.clone()
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

def generate_sample_text(model, tokenizer, prompt, max_length=50, device='cuda'):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"\nGenerated text:")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Generated: {generated_text}\n")
    model.train()

def verify_architecture(model, reference_file):
    """Verify that model architecture matches reference"""
    # Get current model architecture as string
    current_arch = str(model).strip()
    
    # Read reference architecture
    with open(reference_file, 'r') as f:
        reference_arch = f.read().strip()
    
    # Compare architectures
    if current_arch == reference_arch:
        logger.info("✓ Model architecture exactly matches reference architecture")
        return True
    else:
        logger.error("✗ Model architecture differs from reference!")
        # Find and log differences
        current_lines = current_arch.split('\n')
        reference_lines = reference_arch.split('\n')
        
        for i, (curr, ref) in enumerate(zip(current_lines, reference_lines)):
            if curr != ref:
                logger.error(f"Line {i+1} differs:")
                logger.error(f"Expected: {ref}")
                logger.error(f"Got:      {curr}")
        
        return False

def main():
    # Memory optimization settings
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.set_per_process_memory_fraction(0.8)  # Limit GPU memory usage
    
    # Create model and verify architecture
    model = create_model()
    model = model.cpu()  # Keep on CPU initially
    
    # Verify architecture
    if not verify_architecture(model, "deepseek_v3_model_architecture.txt"):
        raise ValueError("Model architecture does not match reference architecture")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
    
    # Create dataset with smaller chunks
    dataset = TextDataset("input.txt", tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=1,
        shuffle=True,
        pin_memory=False,  # Disable pin_memory to reduce memory usage
        num_workers=0
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Move model to device layer by layer
    for name, module in model.named_children():
        module.to(device)
        torch.cuda.empty_cache()
    
    # Enable gradient checkpointing
    try:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    except AttributeError:
        logger.warning("Gradient checkpointing not available")
    
    # Optimizer with minimal memory usage
    optimizer = AdamW(
        model.parameters(),
        lr=5e-5,  # Further reduced learning rate
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=True
    )
    
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Training settings
    step = 0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    accumulated_steps = 0
    accumulation_steps = 8  # Increased for smaller memory footprint
    
    model.train()
    
    try:
        while step < 10000:
            for batch in dataloader:
                # Aggressive memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Process in smaller chunks
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                try:
                    with autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                        loss = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = loss / accumulation_steps
                    
                    if torch.cuda.is_available():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise e
                
                accumulated_steps += 1
                
                if accumulated_steps == accumulation_steps:
                    if torch.cuda.is_available():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Reduced from 1.0
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()
                    
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0
                    step += 1
                    
                    if step % 100 == 0:
                        logger.info(f"Step {step}: loss = {loss.item() * accumulation_steps:.4f}")
                    
                    if step % 2500 == 0:
                        # Save checkpoint to CPU first
                        model.cpu()
                        checkpoint = {
                            'step': step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                        }
                        checkpoint_path = checkpoint_dir / f"model_step_{step}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        del checkpoint
                        model.to(device)
                        logger.info(f"Saved checkpoint at step {step}")
                
                # Aggressive cleanup
                del loss
                del input_ids
                del attention_mask
                del labels
                torch.cuda.empty_cache()
                gc.collect()
                
                if step >= 10000:
                    break
                    
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise e
    
    # Final evaluation
    model.cpu()  # Move to CPU for final generation
    sample_prompts = [
        "KING RICHARD III: Now is the time",
        "RICHMOND: My noble friends,",
        "To be, or not to be,",
    ]
    
    for prompt in sample_prompts:
        generate_sample_text(model, tokenizer, max_length=32, device='cpu')  # Generate on CPU

if __name__ == "__main__":
    main()
