import os
import sys
import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset
import math
import argparse
import psutil
import time
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from student_model.architecture import StudentGPT2Model

parser = argparse.ArgumentParser(description="Evaluate Optimized GPT-2 Model")
parser.add_argument('--checkpoint', type=str, default='checkpoint-5000', help='Checkpoint to use for evaluation')
parser.add_argument('--output_file', type=str, default='evaluation_results.txt', help='Output file to save results')
args = parser.parse_args()

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'student-fine-tuned'))
CHECKPOINT_DIR = os.path.join(MODEL_DIR, args.checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Завантаження токенізатора з базової GPT-2 моделі
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Завантажуємо модель
model = StudentGPT2Model.from_pretrained(CHECKPOINT_DIR).to(device)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_csv = os.path.join(base_dir, "data", "test_iso.csv")
print(f"Loading test dataset from: {test_csv}")
test_dataset = load_dataset("csv", data_files={"test": test_csv})["test"]

resource_usage = []

def monitor_resources(interval=1):
    process = psutil.Process(os.getpid())
    while True:
        cpu_usage = process.cpu_percent(interval=interval)
        memory_info = process.memory_info()
        resource_usage.append((cpu_usage, memory_info.rss / (1024 * 1024)))
        print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_info.rss / (1024 * 1024)} MB")
        time.sleep(interval)

resource_usage = []
monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()

def calculate_perplexity(model, tokenizer, dataset):
    model.eval()
    total_loss = 0
    num_samples = 0
    for example in dataset:
        inputs = tokenizer(
            example["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
            total_loss += loss.item()
            num_samples += 1
    return math.exp(total_loss / num_samples)

def measure_inference_speed(model, tokenizer, prompt, iterations=50):
    model.eval()
    total_time = 0
    for _ in range(iterations):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        start_time = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_length=50)
        total_time += time.time() - start_time
    return total_time / iterations

def generate_text(prompt, num_return=3):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=num_return,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

resource_usage = []
def monitor_resources(interval=1):
    process = psutil.Process(os.getpid())
    while True:
        cpu_usage = process.cpu_percent(interval=interval)
        memory_info = process.memory_info()
        resource_usage.append((cpu_usage, memory_info.rss / (1024 * 1024)))
        print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_info.rss / (1024 * 1024)} MB")
        time.sleep(interval)

monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()

test_dataset = load_dataset("csv", data_files={"test": test_csv})["test"]

# Calculate Perplexity
def calculate_perplexity(model, tokenizer, dataset):
    model.eval()
    total_loss, num_samples = 0, 0
    for sample in dataset:
        input_text = sample["requirement"]
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            num_samples += 1
    return math.exp(total_loss / num_samples)

print("Starting resource monitoring")
resource_usage = []
monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()

print("Evaluating student model...")
perplexity = calculate_perplexity(model, tokenizer, test_dataset)
print(f"Student Model Perplexity: {perplexity:.2f}")

# Вимірюємо швидкість інференсу
test_prompt_speed = "The software must"
inference_speed = measure_inference_speed(model, tokenizer, test_prompt_speed)

model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

generated_examples = {
    p: generate_text(p) for p in [
        "The software must be able to",
        "User authentication must include",
        "The system must provide real-time"
    ]
}

monitor_thread.join(timeout=1)

avg_cpu_usage = sum(u[0] for u in resource_usage) / len(resource_usage)
avg_memory_usage = sum(u[1] for u in resource_usage) / len(resource_usage)

results_dir = os.path.abspath(os.path.join(base_dir, '..', 'results'))
os.makedirs(results_dir, exist_ok=True)

with open(args.output_file, "w", encoding="utf-8") as f:
    f.write(f"Student Model Perplexity: {perplexity:.2f}\n")
    f.write(f"Inference Speed (sec/gen): {inference_speed:.4f}\n")
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    f.write(f"Model Size (MB): {model_size:.2f}\n")
    f.write(f"Avg CPU Usage (%): {avg_cpu_usage:.2f}\n")
    f.write(f"Avg Memory Usage (MB): {avg_memory_usage:.2f}\n\n")

    for prompt, examples in generated_examples.items():
        f.write(f"\nPrompt: {prompt}\n")
        for idx, text in enumerate(examples, 1):
            f.write(f"{idx}. {text}\n")

print(f"✅ Results saved to {args.output_file}")
