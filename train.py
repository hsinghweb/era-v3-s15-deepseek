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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loading text from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        # Split text into chunks of max_length tokens
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
    # Enable memory efficient attention
    if torch.cuda.is_available():
        torch.backends.cuda.max_memory_split_size = None
        torch.backends.cuda.max_memory_allocated = None
    
    # Create model and verify architecture
    model = create_model()
    logger.info("Model Architecture:")
    logger.info("===================")
    logger.info(model)
    
    if not verify_architecture(model, "deepseek_v3_model_architecture.txt"):
        raise ValueError("Model architecture does not match reference architecture")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
    
    # Set device and move model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Try to enable gradient checkpointing if available
    try:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    except AttributeError:
        logger.warning("Gradient checkpointing not available, continuing without it")
    
    # Create dataset and dataloader with smaller batch size
    dataset = TextDataset("input.txt", tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Set requires_grad for all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Initialize optimizer with lower memory usage
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # Only optimize parameters that require gradients
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=True
    )
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Training loop
    step = 0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    accumulated_steps = 0
    effective_batch_size = 4
    accumulation_steps = effective_batch_size
    
    model.train()  # Ensure model is in training mode
    
    while step < 10000:
        for batch in dataloader:
            if step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Use autocast only if CUDA is available
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
            
            accumulated_steps += 1
            
            if accumulated_steps == accumulation_steps:
                if torch.cuda.is_available():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                accumulated_steps = 0
                step += 1
                
                if step % 100 == 0:
                    logger.info(f"Step {step}: loss = {loss.item() * accumulation_steps:.4f}")
                
                if step % 2500 == 0:
                    checkpoint_path = checkpoint_dir / f"model_step_{step}.pt"
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint at step {step}")
                    
                    sample_prompts = [
                        "KING RICHARD III:",
                        "RICHMOND:",
                        "To be, or not",
                    ]
                    for prompt in sample_prompts:
                        generate_sample_text(model, tokenizer, prompt, device=device)
            
            del loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if step >= 10000:
                break
    
    logger.info("\nFinal Model Generation Samples:")
    sample_prompts = [
        "KING RICHARD III: Now is the time",
        "RICHMOND: My noble friends,",
        "To be, or not to be,",
    ]
    
    for prompt in sample_prompts:
        generate_sample_text(model, tokenizer, max_length=50, device=device)

if __name__ == "__main__":
    main()
