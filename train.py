import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import logging
from pathlib import Path
import glob
from model import create_model
from transformers import AutoTokenizer
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
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
        
        # Ensure chunk is exactly max_length by padding
        if len(chunk) < self.max_length:
            chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_length - len(chunk))
        
        # Convert to tensor
        input_ids = torch.tensor(chunk)
        labels = input_ids.clone()
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }

def generate_sample_text(model, tokenizer, prompt, max_length=100, device='cuda'):
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
    # Create model and print architecture
    model = create_model()
    logger.info("Model Architecture:")
    logger.info("===================")
    logger.info(model)
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nTotal Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    
    # Verify architecture matches reference
    logger.info("\nVerifying Model Architecture...")
    if not verify_architecture(model, "deepseek_v3_model_architecture.txt"):
        raise ValueError("Model architecture does not match reference architecture")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create dataset and dataloader
    dataset = TextDataset("input.txt", tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=3e-4)
    
    # Training loop
    step = 0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    while step < 10000:
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            step += 1
            
            if step % 100 == 0:
                logger.info(f"Step {step}: loss = {loss.item():.4f}")
            
            if step % 2500 == 0:
                # Save checkpoint
                checkpoint_path = checkpoint_dir / f"model_step_{step}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint at step {step}")
                
                # Generate sample text
                sample_prompts = [
                    "KING RICHARD III:",
                    "RICHMOND:",
                    "To be, or not",
                    "Friends, Romans,",
                    "Now is the winter"
                ]
                for prompt in sample_prompts:
                    generate_sample_text(model, tokenizer, prompt, device=device)
            
            if step >= 10000:
                break
    
    # Final evaluation with 5 samples
    logger.info("\nFinal Model Generation Samples:")
    sample_prompts = [
        "KING RICHARD III: Now is the time for",
        "RICHMOND: My noble friends,",
        "To be, or not to be,",
        "Friends, Romans, countrymen,",
        "Now is the winter of our"
    ]
    
    for prompt in sample_prompts:
        generate_sample_text(model, tokenizer, prompt, max_length=200, device=device)

if __name__ == "__main__":
    main()
