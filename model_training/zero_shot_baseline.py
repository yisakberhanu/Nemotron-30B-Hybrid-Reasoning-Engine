import os, sys, stat, shutil, re, types
import pandas as pd
import torch
import torch.nn.functional as F
import kagglehub
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script')
for _mod_name in ['mamba_ssm.modules.mamba3', 'mamba_ssm.ops.cute', 'mamba_ssm.ops.cute.mamba3', 'mamba_ssm.ops.cute.mamba3.mamba3_step_fn']:
    sys.modules[_mod_name] = types.ModuleType(_mod_name)
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    if upcast: x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None: out = out + bias.float()
    if z is not None: out = out * F.silu(z.float())
    return out.to(dtype)

MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', trust_remote_code=True, dtype=torch.bfloat16).eval()
for name, mod in sys.modules.items():
    if 'modeling_nemotron_h' in name and hasattr(mod, 'is_fast_path_available'): mod.is_fast_path_available = False
for name, mod in list(sys.modules.items()):
    if hasattr(mod, 'rmsnorm_fn'): mod.rmsnorm_fn = _pure_rmsnorm_fn

df = pd.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv').sample(5, random_state=42)

for idx, row in df.iterrows():
    user_msg = row['prompt'] + '\nPut your final answer inside \\boxed{}.'
    try:
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': user_msg}], tokenize=False, add_generation_prompt=True)
    except:
        text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1500).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.0, do_sample=False, pad_token_id=tokenizer.pad_token_id)

    response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\n🎯 EXPECTED:  {row['answer']}")
    print(f"🤖 RAW TRACE: {response.strip()[:150]}")
