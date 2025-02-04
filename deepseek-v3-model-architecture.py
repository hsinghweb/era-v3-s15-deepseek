import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_print_architecture():
    try:
        # Use the smallest DeepSeek model available on HuggingFace
        model_name = "deepseek-ai/deepseek-coder-1.3b-base"
        
        logger.info(f"Loading {model_name} model...")
        
        # Initialize the model in evaluation mode
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Load in half precision to save memory
            trust_remote_code=True
        )
        
        # Set model to eval mode
        model.eval()
        
        # Print model architecture
        logger.info("\nModel Architecture:")
        logger.info("===================")
        logger.info(model)
        
        
        # Print parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info("\nParameter Counts:")
        logger.info("=================")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        
        # Print model configuration
        logger.info("\nModel Configuration:")
        logger.info("===================")
        logger.info(model.config)
        
    except Exception as e:
        logger.error(f"Error loading or analyzing model: {str(e)}")
        raise

if __name__ == "__main__":
    load_and_print_architecture()
