import os
import sys
import torch
from transformers import Trainer, TrainingArguments, GPT2Config, GPT2LMHeadModel
from transformers import GPT2Tokenizer
import psutil
import time
import torch.nn.functional as F

# Задаємо параметри температури та вагових коефіцієнтів для дистиляції
TEACHER_TEMP = 1.2   # Значення температури для teacher
ALPHA = 0.1        # Внесок KL-дистиляції логітів
BETA = 0.1           # Внесок дистиляції hidden states
GAMMA = 0.1          # Внесок дистиляції attention maps

# Додаємо шлях до модуля student_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from student_model.architecture_distillation import StudentGPT2ModelDistillation
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

def distillation_loss(student_outputs, teacher_outputs, temperature=TEACHER_TEMP):
    """
    Обчислює втрати дистиляції між кінцевими логітами student та teacher,
    масштабуючи їх за temperature**2.
    """
    student_logits = student_outputs.logits / temperature
    teacher_logits = teacher_outputs.logits / temperature
    loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    return loss

def compute_hidden_loss(teacher_hidden, student_hidden):
    """
    Обчислює MSE-втрату між відповідними hidden states.
    Для teacher вибираємо шари з індексами 1,3,5,7,9,11 (тобто teacher_hidden[1::2]),
    а для student використовуємо всі шари, окрім ембеддингів (student_hidden[1:]).
    Щоб привести teacher hidden state з 768 до 384, ми розбиваємо останню розмірність
    на (384, 2) та беремо середнє по останньому виміру.
    """
    loss = 0.0
    teacher_layers = teacher_hidden[1::2]  # 6 шарів teacher
    student_layers = student_hidden[1:]      # 6 шарів student
    for t_hidden, s_hidden in zip(teacher_layers, student_layers):
        # t_hidden: [batch, seq_len, 768]
        # Переформатовуємо до [batch, seq_len, 384, 2] та беремо середнє по останньому виміру, отримуючи [batch, seq_len, 384]
        t_proj = t_hidden.view(t_hidden.size(0), t_hidden.size(1), 384, 2).mean(dim=-1)
        loss += F.mse_loss(t_proj, s_hidden)
    loss = loss / len(teacher_layers)
    return loss

def compute_attention_loss(teacher_attentions, student_attentions):
    """
    Обчислює MSE-втрату між відповідними attention maps.
    Для teacher вибираємо шари з індексами 1,3,5,7,9,11 (тобто teacher_attentions[1::2]),
    а для student використовуємо всі attention maps student (які мають 6 голів).
    Для кожного teacher attention шару, який має форму [batch, 12, seq_len, seq_len],
    розбиваємо вимір голів на (6, 2) і беремо середнє по останньому виміру, щоб отримати [batch, 6, seq_len, seq_len].
    Порівнюємо отриманий teacher attention з відповідним student attention.
    """
    loss = 0.0
    teacher_layers = teacher_attentions[1::2]  # 6 attention шарів teacher
    student_layers = student_attentions           # 6 attention шарів student
    for t_attn, s_attn in zip(teacher_layers, student_layers):
        # t_attn: [batch, 12, seq_len, seq_len]; s_attn: [batch, 6, seq_len, seq_len]
        # Переформатовуємо t_attn до [batch, 6, 2, seq_len, seq_len] і беремо середнє по вимірі 2
        t_proj = t_attn.view(t_attn.size(0), 6, 2, t_attn.size(2), t_attn.size(3)).mean(dim=2)
        loss += F.mse_loss(t_proj, s_attn)
    loss = loss / len(teacher_layers)
    return loss

def distill_student_model():
    # Завантажуємо попередньо натреновану модель GPT-2 як teacher model
    teacher_model_name = "gpt2"
    print(f"Loading tokenizer for model: {teacher_model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(teacher_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading GPT-2 configuration for teacher model")
    teacher_model = GPT2LMHeadModel.from_pretrained(teacher_model_name).to(device)
    teacher_model.config.output_hidden_states = True  # повертаємо hidden_states
    teacher_model.config.output_attentions = True       # повертаємо attention maps
    teacher_model.eval()

    # Створюємо конфігурацію для student моделі
    print("Loading GPT-2 configuration and modifying for student model")
    config = GPT2Config.from_pretrained(teacher_model_name)
    config.n_layer = 6   # зменшуємо кількість шарів до 6
    config.n_embd = 384  # зменшуємо розмір ембеддингів до 384
    config.n_head = 6    # встановлюємо кількість голів для student моделі
    config.output_hidden_states = True  # student теж повертає hidden_states
    config.output_attentions = True       # student теж повертає attention maps

    print("Initializing StudentGPT2ModelDistillation")
    student_model = StudentGPT2ModelDistillation(config).to(device)

    # Завантажуємо і токенізуємо датасет
    print("Loading dataset")
    dataset = load_data()
    print("Tokenizing dataset")
    tokenized_datasets = tokenize_data(dataset, tokenizer)

    print("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir="./models/student-fine-tuned",
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=15,  # тренуємо 15 епох
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-5,   # знижено learning rate для більш плавного навчання
        logging_dir="./logs",
        logging_steps=100,
        fp16=True,
        report_to="none",
    )

    # Клас DistillationTrainer, який обчислює комбіновану втрату:
    # cross-entropy (з label smoothing 0.1) + KL-дивергенція між логітами + hidden state loss + attention loss
    class DistillationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if "num_items_in_batch" in inputs:
                inputs.pop("num_items_in_batch")
            labels = inputs.pop("labels")
            student_outputs = model(**inputs)  # повертає logits, hidden_states, attentions
            
            ce_loss = F.cross_entropy(
                student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
                labels.view(-1),
                label_smoothing=0.1
            )
            
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)  # також повертає hidden_states, attentions
            
            logit_loss = distillation_loss(student_outputs, teacher_outputs, temperature=TEACHER_TEMP)
            hidden_loss = compute_hidden_loss(teacher_outputs.hidden_states, student_outputs.hidden_states)
            attn_loss = compute_attention_loss(teacher_outputs.attentions, student_outputs.attentions)
            
            loss = ALPHA * logit_loss + (1 - ALPHA) * ce_loss + BETA * hidden_loss + GAMMA * attn_loss
            return (loss, student_outputs) if return_outputs else loss

    print("Initializing DistillationTrainer")
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
    )

    print("Starting resource monitoring")
    import threading
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()

    print("Starting distillation")
    trainer.train()
    print("Distillation completed!")

    model_save_path = os.path.join(training_args.output_dir, "checkpoint-600", "pytorch_model.bin")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(student_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    monitor_thread.join(timeout=1)
    avg_cpu_usage = sum([usage[0] for usage in resource_usage]) / len(resource_usage)
    avg_memory_usage = sum([usage[1] for usage in resource_usage]) / len(resource_usage)
    print(f"\nAverage CPU Usage: {avg_cpu_usage:.2f}%")
    print(f"Average Memory Usage: {avg_memory_usage:.2f} MB")

if __name__ == "__main__":
    distill_student_model()
