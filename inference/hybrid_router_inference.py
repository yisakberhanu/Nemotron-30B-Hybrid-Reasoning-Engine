import pandas as pd
import numpy as np
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. DETERMINISTIC SOLVERS (Fast Lane)
# ==========================================
def solve_roman(prompt):
    m = re.search(r'Now, write the number (\d+)', prompt)
    if not m: return None
    n = int(m.group(1))
    val = [1000,900,500,400,100,90,50,40,10,9,5,4,1]
    sym = ['M','CM','D','CD','C','XC','L','XL','X','IX','V','IV','I']
    result = ''
    for v, s in zip(val, sym):
        while n >= v: result += s; n -= v
    return result

def solve_physics(prompt):
    pairs = re.findall(r't\s*=\s*([\d.]+)s.*?distance\s*=\s*([\d.]+)\s*m', prompt)
    if not pairs: return None
    g = np.mean([2*float(d)/float(t)**2 for t, d in pairs])
    m = re.search(r'for t\s*=\s*([\d.]+)s', prompt.split('Now,')[-1])
    if not m: return None
    return round(0.5 * g * float(m.group(1))**2, 2)

def solve_unit(prompt):
    pairs = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    if not pairs: return None
    ratio = np.mean([float(o)/float(i) for i, o in pairs if float(i) != 0])
    m = re.search(r'(?:convert the following measurement:|measurement:)\s*([\d.]+)', prompt)
    if not m: m = re.search(r'([\d.]+)\s*m\s*$', prompt.strip())
    if not m: return None
    return round(float(m.group(1)) * ratio, 2)

def classify_puzzle(prompt):
    p = prompt.lower()
    if 'numeral system' in p: return 'numeral_system'
    elif 'unit conversion' in p: return 'unit_conversion'
    elif 'gravitational' in p or 'gravity' in p: return 'physics_gravity'
    return 'llm_required'

# ==========================================
# 2. LLM INFERENCE (Heavy Compute Lane)
# ==========================================
# Note: For actual Kaggle submission, adapt this loading block to match 
# the competition's specific vLLM/scoring container constraints.
MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
ADAPTER_PATH = "/kaggle/input/your-trained-adapter-dataset/adapter"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH).eval()

def solve_with_llm(prompt):
    user_msg = prompt + '\nPut your final answer inside \\boxed{}.'
    try:
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': user_msg}], tokenize=False, add_generation_prompt=True)
    except:
        text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=1500).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=768, temperature=0.0, do_sample=False)
    
    response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    return match.group(1).strip() if match else "FORMAT_FAILED"

# ==========================================
# 3. ROUTING ENGINE
# ==========================================
test_df = pd.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/test.csv')
submission = []

for idx, row in test_df.iterrows():
    p_type = classify_puzzle(row['prompt'])
    
    if p_type == 'numeral_system':
        ans = solve_roman(row['prompt'])
    elif p_type == 'physics_gravity':
        ans = solve_physics(row['prompt'])
    elif p_type == 'unit_conversion':
        ans = solve_unit(row['prompt'])
    else:
        ans = solve_with_llm(row['prompt'])
        
    submission.append({'id': row['id'], 'answer': ans})

pd.DataFrame(submission).to_csv('submission.csv', index=False)
