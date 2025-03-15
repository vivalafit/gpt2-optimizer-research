import os
import sys
import torch
import threading
from transformers import Trainer, TrainingArguments, GPT2Config, GPT2Tokenizer
import torch.nn.utils.prune as prune
import psutil
import time

# Додаємо шлях до модулів student_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from student_model.architecture_pruning import StudentGPT2ModelPruning
from student_model.utils import load_data, tokenize_data

# Використовуємо CPU для всіх операцій
device = "cpu"
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
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config.from_pretrained(model_name)
    config.n_layer = 6
    config.n_embd = 384
    config.n_head = 6

    model = StudentGPT2ModelPruning(config).to(device)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.4)

    dataset = load_data()
    tokenized_datasets = tokenize_data(dataset, tokenizer)

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "student-fine-tuned"))
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=15,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-5,
        logging_dir="./logs",
        logging_steps=100,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
    )

    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    trainer.train()

    print("Removing pruning masks")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')

    model_save_path = os.path.join(training_args.output_dir, "pytorch_model.bin")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    monitor_thread.join(timeout=1)
    return model, tokenizer

def quantize_model(model, tokenizer):
    MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "student-fine-tuned"))
    state_dict = torch.load(os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location='cpu')

    model.load_state_dict(state_dict)
    model.to('cpu')
    model.eval()

    torch.backends.quantized.engine = "fbgemm"
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Exclude Embedding layers from quantization
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.qconfig = None

    torch.quantization.prepare(model, inplace=True)
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 10)).to('cpu')
    model(dummy_input)

    torch.quantization.convert(model, inplace=True)

    quant_model_path = os.path.join(MODEL_DIR, "student_model_quantized.pth")
    os.makedirs(os.path.dirname(quant_model_path), exist_ok=True)
    torch.save(model.state_dict(), quant_model_path)

    return model

if __name__ == "__main__":
    pruned_model, tokenizer = train_student_model_pruning()
    quantize_model(pruned_model, tokenizer)
