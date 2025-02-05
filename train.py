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
from datasets import load_dataset

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

class CosmopediaDataset(Dataset):
    def __init__(self, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info("Loading Cosmopedia-100k dataset...")
        self.dataset = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
        logger.info(f"Loaded {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get text from dataset
        text = self.dataset[idx]['text']
        
        # Tokenize with truncation and padding
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Squeeze to remove batch dimension added by tokenizer
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        
        # Create labels (same as input_ids for causal language modeling)
        labels = input_ids.clone()
        
        # Mask padding tokens in labels with -100 so they're ignored in loss computation
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def generate_sample_text(model, tokenizer, prompt, max_length=50, device=None):
    """Generate text with device fallback"""
    # Check if device is specified, otherwise detect automatically
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # If CUDA is specified but not available, fallback to CPU
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    
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

def generate_samples(model, tokenizer, device):
    """Generate text for sample prompts"""
    sample_prompts = [
        "KING RICHARD III: Now is the time",
        "RICHMOND: My noble friends,",
        "To be, or not to be,",
        "# Function to sort an array",
        "# Implement a simple REST API"
    ]
    
    logger.info("\n=== Sample Generations ===")
    model.eval()
    with torch.no_grad():
        for prompt in sample_prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                input_ids,
                max_length=128,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"\nPrompt: {prompt}")
            logger.info(f"Generated:\n{generated_text}\n")
            logger.info("-" * 50)
    model.train()

def get_latest_checkpoint(checkpoint_dir):
    """Get the latest valid checkpoint file from the checkpoint directory"""
    checkpoint_files = list(Path(checkpoint_dir).glob("model_step_*.pt"))
    if not checkpoint_files:
        return None
    
    # Sort checkpoints by step number
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]), reverse=True)
    
    # Try each checkpoint file until we find a valid one
    for checkpoint_file in checkpoint_files:
        try:
            # Try to load the checkpoint to verify it's not corrupted
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            return checkpoint_file
        except Exception as e:
            logger.warning(f"Checkpoint {checkpoint_file} is corrupted: {str(e)}")
            # Delete corrupted checkpoint
            try:
                checkpoint_file.unlink()
                logger.info(f"Deleted corrupted checkpoint: {checkpoint_file}")
            except Exception as e:
                logger.warning(f"Failed to delete corrupted checkpoint: {str(e)}")
            continue
    
    return None

def main():
    # Memory optimization settings
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Create model and print architecture
    model = create_model()
    model = model.cpu()  # Keep on CPU initially
    
    # Print model architecture and parameters
    logger.info("\nModel Architecture:")
    logger.info("===================")
    logger.info(model)
    
    # Count and print parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("\nParameter Counts:")
    logger.info("=================")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    
    # Remove architecture verification
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
    
    # Create dataset using Cosmopedia
    dataset = CosmopediaDataset(tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=1,
        shuffle=True,
        pin_memory=False,
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
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Check for existing checkpoints with error handling
    try:
        latest_checkpoint_path = get_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint_path is not None:
            logger.info(f"Found valid checkpoint: {latest_checkpoint_path}")
            try:
                checkpoint = torch.load(
                    latest_checkpoint_path,
                    map_location='cpu',
                    weights_only=True  # Add this to avoid pickle warning
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                step = checkpoint['step']
                best_loss = checkpoint.get('best_loss', float('inf'))
                logger.info(f"Successfully resumed training from step {step}")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                logger.info("Starting training from scratch")
                step = 0
                best_loss = float('inf')
        else:
            logger.info("No valid checkpoints found. Starting training from scratch")
            step = 0
            best_loss = float('inf')
    except Exception as e:
        logger.error(f"Error checking checkpoints: {str(e)}")
        logger.info("Starting training from scratch")
        step = 0
        best_loss = float('inf')
    
    accumulated_steps = 0
    accumulation_steps = 8
    last_checkpoint_path = latest_checkpoint_path
    
    # Add total steps for progress calculation
    total_steps = 10000
    
    model.train()
    logger.info(f"\nTraining will run until step {total_steps}")
    logger.info(f"Current step: {step}")
    
    try:
        while step < total_steps:
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
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()
                    
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_steps = 0
                    step += 1
                    
                    # Add progress logging every 10 steps
                    if step % 10 == 0:
                        progress = (step / total_steps) * 100
                        current_loss = loss.item() * accumulation_steps
                        logger.info(f"Progress: {progress:.1f}% ({step}/{total_steps}) - Current Loss: {current_loss:.4f}")
                    
                    # Generate samples after 100 steps
                    if step == 10:
                        logger.info("\nCompleted 10 steps - Generating sample outputs...")
                        generate_samples(model, tokenizer, device)
                    
                    # Detailed logging and checkpoint saving every 100 steps
                    if step % 100 == 0:
                        current_loss = loss.item() * accumulation_steps
                        logger.info(f"\nStep {step}: loss = {current_loss:.4f}")
                        logger.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1024**2:.1f}MB")
                        
                        # Delete previous checkpoint if it exists
                        if last_checkpoint_path and last_checkpoint_path.exists():
                            last_checkpoint_path.unlink()
                        
                        # Save new checkpoint
                        model.cpu()
                        checkpoint = {
                            'step': step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item() * accumulation_steps,
                            'best_loss': best_loss
                        }
                        
                        # Create checkpoint filename with step number
                        checkpoint_path = checkpoint_dir / f"model_step_{step}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        last_checkpoint_path = checkpoint_path
                        
                        logger.info(f"Saved checkpoint at step {step}")
                        logger.info(f"Current Loss: {loss.item() * accumulation_steps:.4f}")
                        
                        del checkpoint
                        model.to(device)
                
                # Aggressive cleanup
                del loss
                del input_ids
                del attention_mask
                del labels
                torch.cuda.empty_cache()
                gc.collect()
                
                if step >= total_steps:
                    break
                    
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise e
    
    # Final evaluation
    sample_prompts = [
        "Here is an extract from a webpage: \"Recording of Present Day:",
        "Course Unit: LISA Pathfinder Mission and Gravitational Wave Detection",
        "Title: Making Mathematics Accessible: The Importance",
        "The Performing Arts encompass many different forms of artistic",
        "It was a bright, sunny day and Maria was excited to wear"
    ]
    
    logger.info("\nGenerating final samples...")
    # Let generate_sample_text handle device selection
    for prompt in sample_prompts:
        generate_sample_text(
            model=model, 
            tokenizer=tokenizer, 
            prompt=prompt,
            max_length=32
        )

if __name__ == "__main__":
    main()
