import pandas as pd
import time
import concurrent.futures
import google.generativeai as genai
from tqdm import tqdm

# Configure Gemini API
API_KEY = "YOUR_GEMINI_API_KEY" 
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro') # Using Pro for flawless logic extraction

print("Loading original competition data...")
df = pd.read_csv('/kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv')

def get_category(ans):
    ans_str = str(ans).strip()
    if "." in ans_str or (ans_str.replace("-", "").isnumeric() and len(ans_str) < 6): return "Math/Numeric"
    elif all(c in "01" for c in ans_str) and len(ans_str) >= 4: return "Binary/Bitwise"
    elif all(c in "IVXLCDM" for c in ans_str) and len(ans_str) >= 1 and not ans_str.isnumeric(): return "Roman Numeral"
    else: return "Symbolic/Cipher"

df['cat'] = df['answer'].apply(get_category)

# Isolate LLM-dependent puzzles
bitwise_df = df[df['cat'] == 'Binary/Bitwise'].sample(n=500, random_state=42)
symbolic_df = df[df['cat'] == 'Symbolic/Cipher'].sample(n=500, random_state=42)
target_df = pd.concat([bitwise_df, symbolic_df]).reset_index(drop=True)

def generate_reasoning(row):
    prompt_text, target_answer, category = row['prompt'], row['answer'], row['cat']
    
    system_instruction = f"""
    You are an expert logic puzzle reverse-engineer.
    Category: {category}
    
    CRITICAL RULES:
    1. The CORRECT FINAL ANSWER is provided. Do NOT guess it.
    2. Write the step-by-step logical reasoning trace that connects the puzzle to that exact answer.
    3. Keep it concise. Break down bit shifts, XORs, or letter-mappings explicitly based on the given examples.
    """
    
    user_message = f"PUZZLE PROMPT:\n{prompt_text}\n\nCORRECT FINAL ANSWER: {target_answer}\n\nWrite the step-by-step reasoning trace."
    
    try:
        response = model.generate_content(
            system_instruction + "\n\n" + user_message,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        return row['id'], response.text.strip()
    except Exception:
        time.sleep(5)
        return row['id'], "ERROR"

print(f"Generating synthetic traces for {len(target_df)} puzzles...")
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(generate_reasoning, row) for _, row in target_df.iterrows()]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_df)):
        results.append(future.result())

results_df = pd.DataFrame(results, columns=['id', 'reasoning'])
final_data = pd.merge(target_df, results_df, on='id')
final_data = final_data[final_data['reasoning'] != 'ERROR']
final_data.to_csv('train_100_high_quality.csv', index=False)
print(f"Saved {len(final_data)} traces to train_100_high_quality.csv")
