import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel

class StudentGPT2Model(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer.config.n_layer = 6
        self.transformer.config.n_embd = 384

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, use_cache=None, **kwargs):
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)

# Завантаження конфігурації та моделі
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'student-fine-tuned', 'checkpoint-600'))
config = GPT2Config.from_pretrained(MODEL_DIR)
model = StudentGPT2Model(config)

# Завантаження стану моделі
state_dict = torch.load(os.path.join(MODEL_DIR, 'pytorch_model.bin'))

# Видалення ключів weight_orig та weight_mask
keys_to_remove = [key for key in state_dict.keys() if 'weight_orig' in key or 'weight_mask' in key]
for key in keys_to_remove:
    new_key = key.replace('_orig', '')
    state_dict[new_key] = state_dict.pop(key)

# Видалення ключів weight_mask
keys_to_remove = [key for key in state_dict.keys() if 'weight_mask' in key]
for key in keys_to_remove:
    del state_dict[key]

model.load_state_dict(state_dict)

# Квантизація моделі
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Використання float_qparams_weight_only_qconfig для шарів типу Embedding
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.qconfig = torch.quantization.float_qparams_weight_only_qconfig

torch.quantization.convert(model, inplace=True)

# Збереження квантизованої моделі
torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'student_model_quantized.pth'))
print("Model quantized and saved successfully!")