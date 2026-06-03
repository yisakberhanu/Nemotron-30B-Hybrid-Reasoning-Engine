import os, sys, stat, shutil, re, types, zipfile
import pandas as pd
import torch
import torch.nn.functional as F
import kagglehub
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script')
for _mod_name in ['mamba_ssm.modules.mamba3', 'mamba_ssm.ops.cute', 'mamba_ssm.ops.cute.mamba3', 'mamba_ssm.ops.cute.mamba3.mamba3_step_fn']:
    sys.modules[_mod_name] = types.ModuleType(_mod_name)
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    if upcast: x = x.float()
    out = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * weight.float()
    if bias is not None: out = out + bias.float()
    if z is not None: out = out * F.silu(z.float())
    return out.to(dtype)

train_df = pd.read_csv('/kaggle/working/train_100_high_quality.csv')

MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', trust_remote_code=True, dtype=torch.bfloat16)

for name, mod in sys.modules.items():
    if 'modeling_nemotron_h' in name and hasattr(mod, 'is_fast_path_available'): mod.is_fast_path_available = False
for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'rmsnorm_fn'): mod.rmsnorm_fn = _pure_rmsnorm_fn
if hasattr(model, "enable_input_require_grads"): model.enable_input_require_grads()

# Selective LoRA Targeting (12 Layers)
config = model.config
pattern = getattr(config, 'hybrid_override_pattern', '')
sensitive = sorted(set([i for i, c in enumerate(pattern) if c == '*'] + [i - 1 for i, c in enumerate(pattern) if c == '*' and i > 0]))

target_modules = [name for name, mod in model.named_modules() 
                  if isinstance(mod, torch.nn.Linear) and 'lora' not in name.lower() 
                  and 'router' not in name.lower() and 'gate' not in name.lower()
                  and getattr(re.search(r'backbone\.layers\.(\d+)', name), 'group', lambda x: -1)(1) in map(str, sensitive)
                  and (not re.search(r'experts\.(\d+)', name) or int(re.search(r'experts\.(\d+)', name).group(1)) < 2)]

model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias='none', task_type=TaskType.CAUSAL_LM))

def format_train(row):
    user_msg = row['prompt'] + '\nPut your final answer inside \\boxed{}.'
    assistant_msg = f"<think>\n{row['reasoning']}\n</think>\n\n\\boxed{{{row['answer']}}}"
    try:
        return tokenizer.apply_chat_template([{'role': 'user', 'content': user_msg}, {'role': 'assistant', 'content': assistant_msg}], tokenize=False, add_generation_prompt=False)
    except:
        return f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"

train_df['text'] = train_df.apply(format_train, axis=1)

OUTPUT_DIR = "/kaggle/working/adapter"
trainer = SFTTrainer(
    model=model, train_dataset=Dataset.from_pandas(train_df[['text']]), processing_class=tokenizer,
    args=SFTConfig(output_dir=OUTPUT_DIR, per_device_train_batch_size=1, gradient_accumulation_steps=8, num_train_epochs=2, learning_rate=5e-5, bf16=True, optim="adamw_torch", max_length=1024)
)
trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)

with zipfile.ZipFile("/kaggle/working/submission.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.isfile(fpath): zf.write(fpath, fname)
