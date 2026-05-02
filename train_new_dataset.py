#!/usr/bin/env python
"""
Train MobileNetV3 model on the new pharmaceutical dataset
Ready to use with the processed dataset
"""
import os
import sys
from pathlib import Path

# Add project to path
project_path = Path(__file__).parent
sys.path.insert(0, str(project_path))

from Users.utility.requirement import main, predictions
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_on_new_dataset():
    """Train model on the new pharmaceutical dataset"""
    
    logger.info("=" * 80)
    logger.info("🚀 Training MobileNetV3 on New Pharmaceutical Dataset")
    logger.info("=" * 80)
    
    try:
        # Run training with the new dataset
        accuracy, model, label_map, reverse_label_map = main()
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ Training Completed Successfully!")
        logger.info("=" * 80)
        logger.info(f"📊 Final Model Accuracy: {accuracy:.2%}")
        logger.info(f"📚 Number of Classes: {len(label_map)}")
        logger.info(f"💾 Model saved to: media/pilldata/model.keras")
        logger.info("=" * 80)
        
        return accuracy, model, label_map, reverse_label_map
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    accuracy, model, label_map, reverse_label_map = train_on_new_dataset()
    
    if model is not None:
        logger.info("\n✨ Model is ready for predictions!")
        logger.info("   You can now use: predictions('path/to/image.jpg')")
