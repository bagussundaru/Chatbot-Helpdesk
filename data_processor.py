import pandas as pd
import json
import os
from typing import List, Dict, Any
from loguru import logger
import re

class SIPDDataProcessor:
    """Processor untuk data CSV aduan SIPD menjadi format training data"""
    
    def __init__(self, csv_directory: str = "./data/csv"):
        self.csv_directory = csv_directory
        self.processed_data = []
        
    def load_csv_files(self) -> List[pd.DataFrame]:
        """Load semua file CSV dari direktori"""
        csv_files = []
        
        if not os.path.exists(self.csv_directory):
            logger.warning(f"Directory {self.csv_directory} tidak ditemukan")
            return csv_files
            
        for filename in os.listdir(self.csv_directory):
            if filename.endswith('.csv'):
                try:
                    file_path = os.path.join(self.csv_directory, filename)
                    df = pd.read_csv(file_path, encoding='utf-8')
                    logger.info(f"Loaded {filename}: {len(df)} rows")
                    csv_files.append(df)
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
                    
        return csv_files
    
    def clean_text(self, text: str) -> str:
        """Bersihkan dan normalisasi teks"""
        if pd.isna(text) or text == "":
            return ""
            
        # Convert to string
        text = str(text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Indonesian characters
        text = re.sub(r'[^\w\s\-.,!?():]', '', text)
        
        return text.strip()
    
    def consolidate_dataframes(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Gabungkan semua dataframe dan standardisasi kolom"""
        consolidated_data = []
        
        for df in dataframes:
            # Standardisasi nama kolom
            df.columns = df.columns.str.strip().str.upper()
            
            # Mapping kolom yang mungkin berbeda
            column_mapping = {
                'ISSUE': ['ISSUE', 'MASALAH', 'PROBLEM', 'KELUHAN'],
                'EXPECTED': ['EXPECTED', 'SOLUSI', 'SOLUTION', 'JAWABAN'],
                'MENU': ['MENU', 'MODUL', 'MODULE', 'KATEGORI'],
                'NOTE_BY_DEV': ['NOTE BY DEV', 'NOTE_BY_DEV', 'DEV_NOTE', 'DEVELOPER_NOTE'],
                'NOTE_BY_QA': ['NOTE BY QA', 'NOTE_BY_QA', 'QA_NOTE', 'QUALITY_NOTE']
            }
            
            # Buat dataframe standar
            standard_df = pd.DataFrame()
            
            for standard_col, possible_cols in column_mapping.items():
                for col in possible_cols:
                    if col in df.columns:
                        standard_df[standard_col] = df[col]
                        break
                        
            if not standard_df.empty:
                consolidated_data.append(standard_df)
                
        if consolidated_data:
            result = pd.concat(consolidated_data, ignore_index=True)
            logger.info(f"Consolidated {len(result)} total records")
            return result
        else:
            logger.warning("No data could be consolidated")
            return pd.DataFrame()
    
    def create_training_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Konversi data ke format training untuk fine-tuning"""
        training_data = []
        
        for _, row in df.iterrows():
            issue = self.clean_text(row.get('ISSUE', ''))
            expected = self.clean_text(row.get('EXPECTED', ''))
            menu = self.clean_text(row.get('MENU', ''))
            note_dev = self.clean_text(row.get('NOTE_BY_DEV', ''))
            note_qa = self.clean_text(row.get('NOTE_BY_QA', ''))
            
            # Skip jika issue kosong
            if not issue:
                continue
                
            # Buat solusi komprehensif
            solution_parts = []
            if expected:
                solution_parts.append(f"Solusi: {expected}")
            if note_dev:
                solution_parts.append(f"Catatan Developer: {note_dev}")
            if note_qa:
                solution_parts.append(f"Catatan QA: {note_qa}")
                
            if not solution_parts:
                continue
                
            solution = " | ".join(solution_parts)
            
            # Format untuk fine-tuning
            training_example = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Saya mengalami masalah di menu {menu}: {issue}"
                    },
                    {
                        "role": "assistant", 
                        "content": f"Saya memahami masalah Anda di menu {menu}. {solution}"
                    }
                ]
            }
            
            training_data.append(training_example)
            
        logger.info(f"Created {len(training_data)} training examples")
        return training_data
    
    def save_training_data(self, training_data: List[Dict[str, Any]], output_file: str = "sipd_training_data.json"):
        """Simpan data training ke file JSON"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Training data saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
    
    def process_all_data(self) -> List[Dict[str, Any]]:
        """Proses semua data CSV menjadi training data"""
        logger.info("Starting data processing...")
        
        # Load CSV files
        dataframes = self.load_csv_files()
        if not dataframes:
            logger.error("No CSV files found")
            return []
            
        # Consolidate data
        consolidated_df = self.consolidate_dataframes(dataframes)
        if consolidated_df.empty:
            logger.error("No data could be consolidated")
            return []
            
        # Create training data
        training_data = self.create_training_data(consolidated_df)
        
        # Save training data
        self.save_training_data(training_data)
        
        return training_data

if __name__ == "__main__":
    processor = SIPDDataProcessor()
    training_data = processor.process_all_data()
    print(f"Processed {len(training_data)} training examples")