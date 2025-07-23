import os
import requests
import pandas as pd
from loguru import logger
import zipfile
import io

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")
    else:
        logger.info(f"Directory already exists: {directory_path}")

def download_bitext_dataset():
    """Download Bitext Gen AI Chatbot Customer Support Dataset from Hugging Face"""
    # Create data directory structure
    data_dir = os.path.join("data", "bitext")
    create_directory(data_dir)
    
    # URL for the dataset on Hugging Face
    # Note: This is a placeholder URL. The actual URL may be different.
    dataset_url = "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/data/bitext-customer-support-llm-chatbot-training-dataset.csv"
    
    try:
        logger.info(f"Downloading dataset from {dataset_url}")
        response = requests.get(dataset_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Save the dataset to a CSV file
        csv_path = os.path.join(data_dir, "bitext-customer-support-dataset.csv")
        with open(csv_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Dataset downloaded and saved to {csv_path}")
        
        # Verify the dataset
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {', '.join(df.columns)}")
            
            # Display a few sample rows
            logger.info("Sample data:")
            logger.info(df.head(3))
            
        except Exception as e:
            logger.error(f"Error verifying dataset: {e}")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.info("Alternative: Please download the dataset manually from Hugging Face and place it in the data/bitext directory")

def main():
    logger.info("Starting Bitext dataset download process")
    download_bitext_dataset()
    logger.info("Download process completed")

if __name__ == "__main__":
    main()