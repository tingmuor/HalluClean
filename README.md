# 🧹 **HalluClean**

This repository contains the code for our AAAI 2026 paper "**HalluClean: A Unified Framework to Combat Hallucinations in LLMs**"

Extended version (with additional experiments and analyses):  
[https://arxiv.org/pdf/2511.08916](https://arxiv.org/pdf/2511.08916)
<img width="2760" height="1574" alt="image" src="https://github.com/user-attachments/assets/a7a7809b-19a7-4c85-8efc-85ba4c3acbf4" />

---

## ✨ Overview

HalluClean is a lightweight and generalizable framework for **detecting and correcting hallucinations** in large language model (LLM) outputs.

- **Unified framework across tasks**  
  Question answering (QA), summarization (SUM), dialogue (DA), self-contradiction (TSC), and math word problems (MWP).

- **Plan → Reason → Judge detection**  
  HalluClean decomposes hallucination detection into three structured stages:
  1. **Plan**: understand the task and propose a checking strategy.
  2. **Reason**: follow the plan and analyze the answer step by step.
  3. **Judge**: make a final Yes/No decision on hallucination.

- **Analysis-aware revision**  
  The **analysis** produced in the detection stage is reused in the **revision** stage to guide a more faithful correction.

- **Zero-shot and model-agnostic**  
  No task-specific training or labeled hallucination data is required. HalluClean can run with OpenAI-style APIs or local models.

- **Simple Python & CLI interfaces**  
  A small, clean API that can be plugged into existing pipelines.

---

## 📦 Installation

Clone this repository:

```bash
git clone https://github.com/your-name/HalluClean.git
cd HalluClean
```
Install dependencies:
```bash
pip install openai
pip install transformers
```

API configuration:
```bash
export OPENAI_API_KEY="YOUR_API_KEY"
# If you are using a proxy endpoint (e.g., gptsapi):
# export OPENAI_BASE_URL="https://api.gptsapi.net/v1"
```
## 🧪 Quick Test (Python API)
From the repository root (where halluclean/ lives), open Python:
```bash
python

from halluclean import hallu_clean_qa

question = "Which conference accepted the HalluClean paper?"
answer = "It was accepted by ICML 2019."  # intentionally wrong

res = hallu_clean_qa(
    question=question,
    answer=answer,
    detect_model="chatgpt",
)

print("=== PLAN ===")
print(res["detection"]["plan"])
print("\n=== ANALYSIS ===")
print(res["detection"]["analysis"])
print("\n=== RAW JUDGEMENT ===")
print(res["detection"]["raw_judgement"])
print("\n=== IS HALLUCINATED ===")
print(res["detection"]["is_hallucinated"])
print("\n=== REVISED ANSWER ===")
print(res["revised_answer"])
```

## 🖥 Command-line Interface (CLI)
We also provide a simple CLI in halluclean/cli.py.

General usage:

-- mode detect: only run detection (Plan → Reason → Judge).

-- mode revise: only run revision (expects hallucinated inputs).

-- mode pipeline: detection + (if needed) revision.

This is the default recommended mode.

```bash
python -m halluclean.cli \
  --task {qa,sum,da,tsc,mwp} \
  --mode {detect,revise,pipeline} \
  --input path/to/input.jsonl \
  --output path/to/output.jsonl \
  --detect-model chatgpt \
  [--revise-model chatgpt]
```

## 📖 Citation

If you find this repository or our paper useful in your research, please consider citing:

```bibtex
@article{Zhao_Zhang_2026, title={HalluClean: A Unified Framework to Combat Hallucinations in LLMs}, volume={40}, url={https://ojs.aaai.org/index.php/AAAI/article/view/40926}, DOI={10.1609/aaai.v40i42.40926}, abstractNote={Large language models (LLMs) have achieved impressive performance across a wide range of natural language processing tasks, yet they often produce hallucinated content that undermines factual reliability. To address this challenge, we introduce HalluClean, a lightweight and task-agnostic framework for detecting and correcting hallucinations in LLM-generated text. HalluClean adopts a reasoning-enhanced paradigm, explicitly decomposing the process into planning, execution, and revision stages to identify and refine unsupported claims. It employs minimal task-routing prompts to enable zero-shot generalization across diverse domains, without relying on external knowledge sources or supervised detectors. We conduct extensive evaluations on five representative tasks—question answering, dialogue, summarization, math word problems, and contradiction detection. Experimental results show that HalluClean significantly improves factual consistency and outperforms competitive baselines, demonstrating its potential to enhance the trustworthiness of LLM outputs in real-world applications.}, number={42}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Zhao, Yaxin and Zhang, Yu}, year={2026}, month={Mar.}, pages={36092-36100} }



