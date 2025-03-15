import torch
from transformers import GPT2Config, GPT2LMHeadModel

class StudentGPT2ModelDistillation(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # Зменшуємо кількість шарів до 6
        self.transformer.config.n_layer = 6
        # Зменшуємо кількість нейронів у кожному шарі до 384
        self.transformer.config.n_embd = 384

        # Оновлюємо маски для головок (head masks)
        self.transformer.h = self.transformer.h[:config.n_layer]

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, use_cache=None, **kwargs):
        # Видаляємо аргумент num_items_in_batch, якщо він присутній, щоб уникнути помилки
        if "num_items_in_batch" in kwargs:
            kwargs.pop("num_items_in_batch")
        # Викликаємо базовий метод forward з підтримкою inputs_embeds, use_cache та інших аргументів
        return super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            inputs_embeds=inputs_embeds, 
            use_cache=use_cache, 
            **kwargs
        )