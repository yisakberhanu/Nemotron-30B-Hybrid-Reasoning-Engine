# 🏆 NVIDIA Nemotron-30B Hybrid Reasoning Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-F9AB00)
![Optimization](https://img.shields.io/badge/Optimization-Selective_LoRA-success)

An end-to-end machine learning pipeline built for the **NVIDIA Nemotron Model Reasoning Challenge**. 

Rather than relying purely on brute-force LLM compute, this project demonstrates a highly optimized **Hybrid AI Architecture**. It dynamically routes logic puzzles to deterministic Python algorithms (for ~99% accuracy at zero compute cost) while focusing a heavily optimized 30B-parameter LLM exclusively on complex pattern recognition (Bitwise and Symbolic logic) using Synthetic Chain-of-Thought (CoT) injection.

---

## 🧠 Key Innovations & Engineering Highlights

This repository highlights several advanced ML engineering techniques required to deploy massive models in constrained environments (like Kaggle's 9-hour GPU limits):

### 1. Smart-Routing Architecture (Algorithmic Bypass)
Through deep EDA, I identified that 56% of the dataset consisted of deterministic math puzzles (Physics, Unit Conversions, Roman Numerals). Instead of fine-tuning the LLM to learn basic math, I built a pre-inference regex classifier that routes these directly to custom Python solvers, achieving **99.4% accuracy in microseconds**.

### 2. Selective LoRA Targeting (99% Adapter Size Reduction)
Training a 30B parameter model with standard `target_modules="all-linear"` produces a bloated 3+ GB adapter and risks catastrophic forgetting. By analyzing the Nemotron-3-Nano architecture paper, I built a dynamic regex scanner to target **only the 12 specific reasoning layers** (6 GQA Attention layers + 6 Mamba layers). 
* **Result:** Reduced trainable parameters to ~7.5M and shrank the final adapter from **2.8 GB to 26 MB**, massively speeding up training and preventing OOM crashes.

### 3. Synthetic Logic Injection (Chain-of-Thought)
Base LLMs fail at out-of-distribution Bitwise operations due to zero-shot formatting traps. I built a multi-threaded data generation pipeline using the Gemini API to reverse-engineer the correct answers and generate high-quality `<think>` step-by-step reasoning traces. Fine-tuning the model on this synthetic data forces the weights to adopt a rigorous, step-by-step "muscle memory" before outputting the strict `\boxed{answer}` format.

### 4. Deep Environment & Hardware Patching
The Nemotron-3 model utilizes NVIDIA Blackwell-native Cutlass and Triton kernels that natively crash on older hardware (T4/P100 GPUs). I engineered a robust environment patch that:
* Uses Python's `unittest.mock.MagicMock` to build a "Deep Bypass" for missing `mamba_ssm` and `cutlass` dependencies.
* Forces the model to fall back to pure PyTorch `rmsnorm` implementations, ensuring stability across any GPU cluster.

---

## 📂 Repository Structure

```text
├── data_generation/
│   └── synthetic_cot_generator.py   # Multi-threaded Gemini API script for <think> traces
├── EDA/
│   └── nemotron_eda_and_routing.ipynb # Discovery of deterministic vs LLM puzzle splits
├── model_training/
│   ├── zero_shot_baseline.py        # Micro-batch script to test raw model capabilities
│   └── optimized_selective_lora.py  # Main SFT pipeline targeting the 12 sensitive layers
├── inference/
│   └── hybrid_router_inference.py   # Final submission script: Python Solvers + LLM
└── README.md
