import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from collections import Counter
import numpy as np

def setup_logging():
    """Setup logging configuration"""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/dataset_comparison.log", rotation="10 MB", level="DEBUG")

def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "data/processed",
        "logs",
        "reports"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def load_json_data(file_path):
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} examples from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return []

def extract_text_from_training_data(training_data):
    """Extract user and assistant messages from training data"""
    user_messages = []
    assistant_messages = []
    
    for item in training_data:
        messages = item.get('messages', [])
        for message in messages:
            if message['role'] == 'user':
                user_messages.append(message['content'])
            elif message['role'] == 'assistant':
                assistant_messages.append(message['content'])
    
    return user_messages, assistant_messages

def calculate_text_statistics(messages):
    """Calculate statistics for a list of text messages"""
    if not messages:
        return {
            'count': 0,
            'avg_length': 0,
            'min_length': 0,
            'max_length': 0,
            'total_words': 0,
            'avg_words': 0
        }
    
    lengths = [len(msg) for msg in messages]
    word_counts = [len(msg.split()) for msg in messages]
    
    return {
        'count': len(messages),
        'avg_length': sum(lengths) / len(messages),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'total_words': sum(word_counts),
        'avg_words': sum(word_counts) / len(messages)
    }

def extract_menu_categories(user_messages):
    """Extract menu categories from user messages"""
    categories = []
    
    for msg in user_messages:
        if "Saya mengalami masalah di menu" in msg:
            parts = msg.split(":", 1)
            if len(parts) > 1:
                menu_parts = parts[0].split("menu", 1)
                if len(menu_parts) > 1:
                    menu = menu_parts[1].strip()
                    categories.append(menu)
    
    return categories

def generate_comparison_report(sipd_data, bitext_data):
    """Generate comparison report between SIPD and Bitext datasets"""
    # Extract messages
    sipd_user_msgs, sipd_assistant_msgs = extract_text_from_training_data(sipd_data)
    bitext_user_msgs, bitext_assistant_msgs = extract_text_from_training_data(bitext_data)
    
    # Calculate statistics
    sipd_user_stats = calculate_text_statistics(sipd_user_msgs)
    sipd_assistant_stats = calculate_text_statistics(sipd_assistant_msgs)
    bitext_user_stats = calculate_text_statistics(bitext_user_msgs)
    bitext_assistant_stats = calculate_text_statistics(bitext_assistant_msgs)
    
    # Extract categories
    sipd_categories = extract_menu_categories(sipd_user_msgs)
    
    # Create report
    report = {
        'sipd': {
            'total_examples': len(sipd_data),
            'user_messages': sipd_user_stats,
            'assistant_messages': sipd_assistant_stats,
            'unique_categories': len(set(sipd_categories)),
            'category_distribution': dict(Counter(sipd_categories))
        },
        'bitext': {
            'total_examples': len(bitext_data),
            'user_messages': bitext_user_stats,
            'assistant_messages': bitext_assistant_stats
        }
    }
    
    return report

def save_report_to_json(report, output_file):
    """Save report to JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving report: {e}")

def create_comparison_charts(report, output_dir):
    """Create comparison charts between SIPD and Bitext datasets"""
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Total examples comparison
    plt.figure(figsize=(10, 6))
    datasets = ['SIPD', 'Bitext']
    examples = [report['sipd']['total_examples'], report['bitext']['total_examples']]
    plt.bar(datasets, examples, color=['blue', 'orange'])
    plt.title('Total Training Examples Comparison')
    plt.ylabel('Number of Examples')
    plt.savefig(os.path.join(output_dir, 'total_examples_comparison.png'))
    plt.close()
    
    # 2. Message length comparison
    plt.figure(figsize=(12, 6))
    categories = ['User Avg Length', 'Assistant Avg Length', 'User Avg Words', 'Assistant Avg Words']
    sipd_values = [
        report['sipd']['user_messages']['avg_length'],
        report['sipd']['assistant_messages']['avg_length'],
        report['sipd']['user_messages']['avg_words'],
        report['sipd']['assistant_messages']['avg_words']
    ]
    bitext_values = [
        report['bitext']['user_messages']['avg_length'],
        report['bitext']['assistant_messages']['avg_length'],
        report['bitext']['user_messages']['avg_words'],
        report['bitext']['assistant_messages']['avg_words']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, sipd_values, width, label='SIPD')
    rects2 = ax.bar(x + width/2, bitext_values, width, label='Bitext')
    
    ax.set_title('Message Length Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'message_length_comparison.png'))
    plt.close()
    
    # 3. SIPD Category Distribution (if available)
    if report['sipd']['category_distribution']:
        plt.figure(figsize=(14, 8))
        categories = list(report['sipd']['category_distribution'].keys())
        counts = list(report['sipd']['category_distribution'].values())
        
        # Sort by count
        sorted_data = sorted(zip(categories, counts), key=lambda x: x[1], reverse=True)
        categories = [item[0] for item in sorted_data]
        counts = [item[1] for item in sorted_data]
        
        plt.bar(categories, counts)
        plt.title('SIPD Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sipd_category_distribution.png'))
        plt.close()

def main():
    """Main function to compare datasets"""
    setup_logging()
    logger.info("Starting dataset comparison")
    
    ensure_directories()
    
    # Load datasets
    sipd_data_path = "data/processed/sipd_training_data.json"
    bitext_data_path = "data/processed/bitext_training_data.json"
    
    sipd_data = load_json_data(sipd_data_path)
    bitext_data = load_json_data(bitext_data_path)
    
    if not sipd_data or not bitext_data:
        logger.error("One or both datasets are missing. Please run integrate_bitext_dataset.py first.")
        return
    
    # Generate comparison report
    report = generate_comparison_report(sipd_data, bitext_data)
    
    # Save report
    report_path = "reports/dataset_comparison_report.json"
    save_report_to_json(report, report_path)
    
    # Create charts
    create_comparison_charts(report, "reports")
    
    logger.info("Comparison completed. Check the reports directory for results.")

if __name__ == "__main__":
    main()