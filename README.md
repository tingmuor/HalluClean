# HalluClean

Code and data of our AAAI 2026 paper:

> **HalluClean: A Unified Framework to Combat Hallucinations in LLMs**

Extended version (with additional experiments and analyses):  
[https://arxiv.org/pdf/2511.08916](https://arxiv.org/pdf/2511.08916)

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
