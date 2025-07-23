import os
import sys
from loguru import logger
from bitext_processor import BitextDataProcessor
from data_processor import SIPDDataProcessor
import json

def setup_logging():
    """Setup logging configuration"""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/bitext_integration.log", rotation="10 MB", level="DEBUG")

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "data/bitext",
        "data/processed",
        "logs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def process_bitext_data():
    """Process Bitext dataset using BitextDataProcessor"""
    logger.info("Processing Bitext dataset...")
    processor = BitextDataProcessor()
    training_data = processor.process_all_data()
    
    if training_data:
        # Save to processed directory
        output_file = "data/processed/bitext_training_data.json"
        processor.save_training_data(training_data, output_file)
        logger.info(f"Saved {len(training_data)} Bitext training examples to {output_file}")
        return training_data
    else:
        logger.warning("No Bitext training data was processed")
        return []

def process_sipd_data():
    """Process SIPD dataset using SIPDDataProcessor"""
    logger.info("Processing SIPD dataset...")
    processor = SIPDDataProcessor()
    training_data = processor.process_all_data()
    
    if training_data:
        # Save to processed directory
        output_file = "data/processed/sipd_training_data.json"
        processor.save_training_data(training_data, output_file)
        logger.info(f"Saved {len(training_data)} SIPD training examples to {output_file}")
        return training_data
    else:
        logger.warning("No SIPD training data was processed")
        return []

def combine_datasets(bitext_data, sipd_data):
    """Combine both datasets into a single training file"""
    combined_data = bitext_data + sipd_data
    
    if combined_data:
        output_file = "data/processed/combined_training_data.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Combined dataset saved with {len(combined_data)} examples to {output_file}")
        except Exception as e:
            logger.error(f"Error saving combined dataset: {e}")
    else:
        logger.warning("No combined data to save")

def main():
    """Main function to orchestrate the integration process"""
    setup_logging()
    logger.info("Starting Bitext dataset integration process")
    
    ensure_directories()
    
    # Process Bitext data
    bitext_data = process_bitext_data()
    
    # Process SIPD data
    sipd_data = process_sipd_data()
    
    # Combine datasets
    if bitext_data or sipd_data:
        combine_datasets(bitext_data, sipd_data)
    
    logger.info("Integration process completed")

if __name__ == "__main__":
    main()