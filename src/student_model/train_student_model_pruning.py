import os
import sys
import torch
from transformers import Trainer, TrainingArguments, GPT2Config
from transformers import GPT2Tokenizer
import torch.nn.utils.prune as prune
import psutil
import time

# Додаємо шлях до модуля student_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from student_model.architecture_pruning import StudentGPT2ModelPruning
from student_model.utils import load_data, tokenize_data

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

resource_usage = []

def monitor_resources(interval=1):
    process = psutil.Process(os.getpid())
    while True:
        cpu_usage = process.cpu_percent(interval=interval)
        memory_info = process.memory_info()
        resource_usage.append((cpu_usage, memory_info.rss / (1024 * 1024)))
        print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_info.rss / (1024 * 1024)} MB")
        time.sleep(interval)

def train_student_model_pruning():
    model_name = "gpt2"
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading GPT-2 configuration for student model")
    config = GPT2Config.from_pretrained(model_name)
    config.n_layer = 6  # Зменшуємо кількість шарів до 6
    config.n_embd = 384  # Зменшуємо кількість нейронів у кожному шарі до 384

    print("Initializing StudentGPT2ModelPruning")
    model = StudentGPT2ModelPruning(config).to(device)

    print("Applying pruning to the model")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.4)  # використовуємо прунінг 

    print("Loading dataset")
    dataset = load_data()
    print("Tokenizing dataset")
    tokenized_datasets = tokenize_data(dataset, tokenizer)

    print("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir="./models/student-fine-tuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        logging_dir="./logs",
        logging_steps=100,
        fp16=True,
        report_to="none",
    )

    print("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
    )

    print("Starting resource monitoring")
    import threading
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    print("Starting training")
    trainer.train()
    print("Training completed!")

    # Збереження моделі
    model_save_path = os.path.join(training_args.output_dir, "checkpoint-600", "pytorch_model.bin")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Зупиняємо моніторинг ресурсів
    monitor_thread.join(timeout=1)

    # Виводимо підсумкове використання ресурсів
    avg_cpu_usage = sum([usage[0] for usage in resource_usage]) / len(resource_usage)
    avg_memory_usage = sum([usage[1] for usage in resource_usage]) / len(resource_usage)
    print(f"\nAverage CPU Usage: {avg_cpu_usage:.2f}%")
    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB")

if __name__ == "__main__":
    train_student_model_pruning()