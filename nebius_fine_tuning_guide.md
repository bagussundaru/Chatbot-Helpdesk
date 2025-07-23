# Panduan Fine-tuning Model di Nebius AI Studio

Panduan lengkap untuk melakukan fine-tuning model AI di Nebius AI Studio untuk SIPD Chatbot.

## Persiapan Data Training

### 1. Proses Data CSV
```bash
# Jalankan data processor untuk mengkonversi CSV ke format JSON
python data_processor.py
```

Hasil: File `sipd_training_data.json` akan dibuat dengan format:
```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "Saya mengalami masalah di menu Login/Akses: Tidak bisa login ke SIPD, muncul error 500"
      },
      {
        "role": "assistant",
        "content": "Saya memahami masalah Anda di menu Login/Akses. Solusi: User dapat login dengan normal | Catatan Developer: Cek koneksi database dan clear browser cache | Catatan QA: Pastikan username dan password benar, coba browser lain"
      }
    ]
  }
]
```

### 2. Validasi Data Training
```python
import json

# Load dan validasi data training
with open('sipd_training_data.json', 'r', encoding='utf-8') as f:
    training_data = json.load(f)

print(f"Total training examples: {len(training_data)}")

# Cek format data
for i, example in enumerate(training_data[:3]):
    print(f"\nExample {i+1}:")
    print(f"User: {example['messages'][0]['content'][:100]}...")
    print(f"Assistant: {example['messages'][1]['content'][:100]}...")
```

## Setup Nebius AI Studio

### 1. Akses Nebius AI Studio
1. Buka https://studio.nebius.com/
2. Login atau buat akun baru
3. Buat project baru untuk SIPD Chatbot

### 2. Pilih Base Model
Rekomendasi model untuk fine-tuning:

**Option 1: Mistral 7B (Recommended)**
- Model: `mistralai/Mistral-7B-Instruct-v0.2`
- Pros: Cepat, efisien, good Indonesian support
- Cons: Kapasitas terbatas untuk konteks panjang

**Option 2: Llama 2 7B Chat**
- Model: `meta-llama/Llama-2-7b-chat-hf`
- Pros: Stabil, well-tested
- Cons: Perlu lebih banyak data training

**Option 3: Qwen 7B Chat**
- Model: `Qwen/Qwen-7B-Chat`
- Pros: Excellent multilingual support
- Cons: Lebih besar, butuh resource lebih

### 3. Upload Training Data
1. Di Nebius Studio, pilih "Fine-tuning"
2. Upload file `sipd_training_data.json`
3. Pilih format "Chat Completion"
4. Validasi data format

## Konfigurasi Fine-tuning

### 1. Hyperparameters
```yaml
# Recommended settings untuk SIPD Chatbot
learning_rate: 2e-5
batch_size: 4
num_epochs: 3
max_sequence_length: 2048
warmup_steps: 100
weight_decay: 0.01
lora_r: 16  # Jika menggunakan LoRA
lora_alpha: 32
lora_dropout: 0.1
```

### 2. Training Configuration
```json
{
  "model_name": "sipd-chatbot-v1",
  "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
  "training_type": "lora",  // atau "full" untuk full fine-tuning
  "dataset_format": "chat_completion",
  "hyperparameters": {
    "learning_rate": 2e-5,
    "batch_size": 4,
    "num_epochs": 3,
    "max_length": 2048
  },
  "validation_split": 0.1
}
```

## Proses Fine-tuning

### 1. Mulai Training
1. Review konfigurasi training
2. Estimasi biaya dan waktu
3. Mulai fine-tuning job
4. Monitor progress melalui dashboard

### 2. Monitoring Training
```python
# Script untuk monitor training (jika API tersedia)
import requests
import time

def monitor_training(job_id, api_key):
    headers = {'Authorization': f'Bearer {api_key}'}
    
    while True:
        response = requests.get(
            f'https://api.studio.nebius.ai/v1/fine-tuning/jobs/{job_id}',
            headers=headers
        )
        
        status = response.json()
        print(f"Status: {status['status']}")
        print(f"Progress: {status.get('progress', 0)}%")
        
        if status['status'] in ['completed', 'failed']:
            break
            
        time.sleep(60)  # Check every minute

# Usage
# monitor_training('your_job_id', 'your_api_key')
```

### 3. Evaluasi Model
Setelah training selesai:

```python
# Test model dengan sample queries
test_queries = [
    "Saya tidak bisa login ke SIPD",
    "Error saat input DPA",
    "Laporan tidak bisa di-export",
    "SPP tidak bisa dibuat",
    "Jurnal otomatis tidak terbentuk"
]

# Test dengan API (sesuaikan dengan endpoint Nebius)
for query in test_queries:
    response = test_model(query)
    print(f"Q: {query}")
    print(f"A: {response}\n")
```

## Deployment Model

### 1. Deploy ke Nebius Inference
1. Pilih model yang sudah di-fine-tune
2. Deploy ke inference endpoint
3. Konfigurasi scaling dan resource
4. Test endpoint

### 2. Update Aplikasi
```python
# Update config.py dengan model ID baru
NEBIUS_MODEL_ID = "your_fine_tuned_model_id"

# Test integrasi
from nebius_client import NebiusAIClient

client = NebiusAIClient()
if client.health_check():
    print("✓ Model integration successful")
else:
    print("✗ Model integration failed")
```

## Optimisasi dan Iterasi

### 1. Analisis Performance
```python
# Script untuk evaluasi model
def evaluate_model_performance():
    test_cases = [
        {
            "input": "Tidak bisa login SIPD",
            "expected_topics": ["login", "akses", "error"],
            "expected_sentiment": "negative"
        },
        # Tambah test cases lain
    ]
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for test in test_cases:
        response = client.generate_response([{
            "role": "user",
            "content": test["input"]
        }])
        
        # Evaluasi respons
        score = evaluate_response(response, test)
        if score > 0.7:  # Threshold
            correct_predictions += 1
    
    accuracy = correct_predictions / total_tests
    print(f"Model accuracy: {accuracy:.2%}")
    
    return accuracy
```

### 2. Continuous Improvement
1. **Collect Feedback**: Gunakan feedback dari users
2. **Expand Dataset**: Tambah data training baru
3. **Retrain**: Lakukan fine-tuning ulang dengan data yang diperluas
4. **A/B Testing**: Test model baru vs model lama

### 3. Data Augmentation
```python
# Script untuk augmentasi data
def augment_training_data(original_data):
    augmented_data = []
    
    for example in original_data:
        # Original example
        augmented_data.append(example)
        
        # Variasi 1: Paraphrase
        paraphrased = paraphrase_text(example['messages'][0]['content'])
        augmented_data.append({
            "messages": [
                {"role": "user", "content": paraphrased},
                example['messages'][1]  # Keep same response
            ]
        })
        
        # Variasi 2: Add context
        contextual = f"Selamat pagi, {example['messages'][0]['content']}"
        augmented_data.append({
            "messages": [
                {"role": "user", "content": contextual},
                example['messages'][1]
            ]
        })
    
    return augmented_data
```

## Best Practices

### 1. Data Quality
- **Konsistensi**: Pastikan format respons konsisten
- **Diversitas**: Variasikan cara penulisan masalah
- **Kelengkapan**: Sertakan semua jenis masalah SIPD
- **Akurasi**: Verifikasi solusi yang diberikan

### 2. Training Strategy
- **Start Small**: Mulai dengan dataset kecil untuk proof of concept
- **Iterative**: Lakukan fine-tuning bertahap
- **Validation**: Selalu gunakan validation set
- **Early Stopping**: Hindari overfitting

### 3. Cost Optimization
- **LoRA vs Full**: Gunakan LoRA untuk efisiensi
- **Batch Size**: Sesuaikan dengan budget
- **Model Size**: Pilih model terkecil yang memadai
- **Monitoring**: Track biaya training

### 4. Production Readiness
- **Latency**: Test response time
- **Throughput**: Test concurrent requests
- **Fallback**: Siapkan fallback mechanism
- **Monitoring**: Setup alerting dan logging

## Troubleshooting

### Common Issues

1. **Training Failed**
   - Cek format data JSON
   - Validasi encoding (UTF-8)
   - Periksa quota dan limits

2. **Poor Performance**
   - Tambah data training
   - Adjust hyperparameters
   - Coba model base yang berbeda

3. **High Latency**
   - Optimize model size
   - Use caching
   - Consider model quantization

4. **Inconsistent Responses**
   - Lower temperature
   - Add more training examples
   - Improve prompt engineering

### Debug Scripts

```python
# Debug training data
def debug_training_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total examples: {len(data)}")
    
    # Check for issues
    issues = []
    for i, example in enumerate(data):
        if 'messages' not in example:
            issues.append(f"Example {i}: Missing 'messages' key")
        elif len(example['messages']) != 2:
            issues.append(f"Example {i}: Should have exactly 2 messages")
        elif example['messages'][0]['role'] != 'user':
            issues.append(f"Example {i}: First message should be 'user'")
        elif example['messages'][1]['role'] != 'assistant':
            issues.append(f"Example {i}: Second message should be 'assistant'")
    
    if issues:
        print("Issues found:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
    else:
        print("✓ Data format looks good")

# Usage
debug_training_data('sipd_training_data.json')
```

## Kesimpulan

Fine-tuning model di Nebius AI Studio memungkinkan pembuatan chatbot SIPD yang:
- Memahami terminologi dan konteks SIPD
- Memberikan respons yang relevan dan akurat
- Dapat berkembang dengan data baru
- Terintegrasi dengan sistem RAG untuk konteks tambahan

Dengan mengikuti panduan ini, Anda dapat membuat model AI yang optimal untuk kebutuhan help desk SIPD.