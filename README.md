# HalluClean: A Unified Framework to Combat Hallucinations in LLMs

This repository provides the code and data for our AAAI 2026 paper:

> **HalluClean: A Unified Framework to Combat Hallucinations in LLMs**

HalluClean is a lightweight, training-free framework that detects and corrects hallucinations in large language model (LLM) outputs via structured reasoning, without external knowledge or task-specific supervision.

## Paper & Extended Version

- **AAAI 2026 conference paper** – *HalluClean: A Unified Framework to Combat Hallucinations in LLMs* (to appear).
- **Extended version (with additional experiments and details)**: [arXiv PDF](https://arxiv.org/pdf/2511.08916)

## Highlights

- **Unified detection + revision** pipeline: plan → reason → judge → revise.
- **Task-agnostic & zero-shot**: supports QA, dialogue, summarization, math word problems, and self-contradiction.
- **Model-agnostic & practical**: works with both proprietary and open-source LLMs, suitable for privacy-sensitive or resource-constrained deployment.

<!-- You can add installation / usage sections below -->
