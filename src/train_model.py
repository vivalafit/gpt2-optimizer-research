import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# 🔹 Перевіряємо, чи доступний GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 🔹 Завантаження датасету
def load_data():
    """
    Функція завантажує датасети train, val, test із файлів CSV.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Отримуємо кореневу директорію
    print(f"Train file path: {os.path.join(base_dir, 'train_iso.csv')}")
    print(f"Validation file path: {os.path.join(base_dir, 'val_iso.csv')}")
    print(f"Test file path: {os.path.join(base_dir, 'test_iso.csv')}")

    # Завантажуємо файли в словник
    dataset = load_dataset(
        "csv", 
        data_files={
            "train": os.path.join(base_dir, "train_iso.csv"),
            "val": os.path.join(base_dir, "val_iso.csv"),
            "test": os.path.join(base_dir, "test_iso.csv"),
        }
    )
    print("Dataset loaded successfully!")
    return dataset

# 🔹 Токенізація датасету
def tokenize_data(dataset, tokenizer):
    """
    Токенізує дані, додаючи labels для підрахунку втрат (loss).
    """
    def preprocess_function(examples):
        # Токенізуємо текст із паддінгом, обрізанням і max_length
        model_inputs = tokenizer(
            examples["requirement"], 
            truncation=True, 
            padding="max_length", 
            max_length=128
        )
        
        # 🔹 Додаємо labels, щоб модель могла обчислювати втрати
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    # Токенізуємо кожен спліт (train, val, test)
    tokenized = dataset.map(preprocess_function, batched=True)
    print("Dataset tokenized successfully!")
    return tokenized

# 🔹 Навчання моделі
def train_model():
    """
    Головна функція для тренування моделі GPT-2.
    """
    # 1️⃣ Завантаження попередньо навченої моделі GPT-2
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 🔹 Встановлюємо паддінг-токен (GPT-2 не має його за замовчуванням)
    tokenizer.pad_token = tokenizer.eos_token

    # Завантажуємо модель і переміщуємо її на GPU (якщо доступний)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    print(f"Model {model_name} loaded successfully!")

    # 2️⃣ Завантаження і токенізація датасетів
    dataset = load_data()
    tokenized_datasets = tokenize_data(dataset, tokenizer)

    # 3️⃣ Налаштування параметрів тренування
    training_args = TrainingArguments(
        output_dir="./models/fine-tuned",  # Куди зберігати результати
        evaluation_strategy="epoch",     # Оцінювати модель після кожної епохи
        save_strategy="epoch",           # Зберігати чекпойнти після кожної епохи
        num_train_epochs=3,              # Кількість епох
        per_device_train_batch_size=4,   # Batch size для тренування
        per_device_eval_batch_size=4,    # Batch size для валідації
        logging_dir="./logs",            # Директорія для логів
        logging_steps=100,               # Як часто логувати прогрес
        fp16=True,                       # Використання Mixed Precision Training
        report_to="none",                # Вимкнення WandB або інших трекерів
    )

    # 4️⃣ Ініціалізація `Trainer`
    trainer = Trainer(
        model=model,                     # GPT-2 модель
        args=training_args,              # Параметри тренування
        train_dataset=tokenized_datasets["train"],  # Навчальний датасет
        eval_dataset=tokenized_datasets["val"],     # Валідаційний датасет
        tokenizer=tokenizer,             # Токенізатор
    )
    print("Trainer initialized successfully!")

    # 5️⃣ Запуск тренування
    print("Training started...")
    trainer.train()
    print("Training completed!")

# 🔹 Головна функція
if __name__ == "__main__":
    train_model()
