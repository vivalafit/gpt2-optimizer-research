import torch
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import math

# Завантажуємо натреновану модель
MODEL_PATH = "./models/fine-tuned/checkpoint-1200"  # Задаємо конкретний чекпойнт
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model and tokenizer from {MODEL_PATH}...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)  # Завантажуємо токенізатор
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)  # Завантажуємо модель

# Завантажуємо тестовий набір
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_dataset = load_dataset("csv", data_files={"test": os.path.join(base_dir, "test_iso.csv")})["test"]

# Оцінка перплексії
def calculate_perplexity(model, tokenizer, dataset):
    model.eval()
    total_loss = 0
    num_samples = 0

    for sample in dataset:
        input_text = sample["requirement"]
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item()
            num_samples += 1

    avg_loss = total_loss / num_samples
    perplexity = math.exp(avg_loss)  # Перетворення Loss у Perplexity
    return perplexity

# Генерація прикладів тексту
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Запускаємо оцінку
print("Evaluating model...")
perplexity = calculate_perplexity(model, tokenizer, test_dataset)
print(f"Model Perplexity: {perplexity:.2f}")

# Тестуємо генерацію
print("\n🔹 **Text Generation Examples:**")
test_prompts = [
    "The software must be able to",
    "User authentication must include",
    "The system must provide real-time"
]

for prompt in test_prompts:
    generated_text = generate_text(prompt)
    print(f"\n**Prompt:** {prompt}")
    print(f"**Generated:** {generated_text}")
