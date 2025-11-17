"""
HalluClean: A Unified Framework to Combat Hallucinations in LLMs.

Python package public API.

按任务划分：

QA (Question Answering)
-----------------------
- detect_qa(question, answer, model_name="chatgpt", ...)
- revise_qa(question, hallucinated_answer, model_name="chatgpt", ...)
- hallu_clean_qa(question, answer, detect_model="chatgpt", revise_model=None, ...)

Summarization (SUM)
-------------------
- detect_sum(source_text, summary, model_name="chatgpt", ...)
- revise_sum(source_text, hallucinated_summary, model_name="chatgpt", ...)
- hallu_clean_sum(source_text, summary, detect_model="chatgpt", revise_model=None, ...)

Dialogue / Conversation (DA)
----------------------------
- detect_da(context, response, model_name="chatgpt", ...)
- revise_da(context, hallucinated_response, model_name="chatgpt", ...)
- hallu_clean_da(context, response, detect_model="chatgpt", revise_model=None, ...)

Self-Contradiction (TSC)
------------------------
- detect_tsc(text, model_name="chatgpt", ...)
- revise_tsc(text, model_name="chatgpt", ...)
- hallu_clean_tsc(text, detect_model="chatgpt", revise_model=None, ...)

Math Word Problems (MWP)
------------------------
- detect_mwp(problem, solution, model_name="chatgpt", ...)
- revise_mwp(problem, hallucinated_solution, model_name="chatgpt", ...)
- hallu_clean_mwp(problem, solution, detect_model="chatgpt", revise_model=None, ...)

底层模型调用
------------
- run_model(model_name, prompt, pipe=None, max_new_tokens=512, timeout=1000)

示例
----
>>> from halluclean import hallu_clean_qa
>>> res = hallu_clean_qa("What is HalluClean?", "It is a novel from 1850.")
>>> print(res["revised_answer"])
"""

from .api import (
    # QA
    detect_qa,
    revise_qa,
    hallu_clean_qa,
    # Summarization
    detect_sum,
    revise_sum,
    hallu_clean_sum,
    # Dialogue
    detect_da,
    revise_da,
    hallu_clean_da,
    # Self-Contradiction
    detect_tsc,
    revise_tsc,
    hallu_clean_tsc,
    # Math Word Problems
    detect_mwp,
    revise_mwp,
    hallu_clean_mwp,
)

from .model_client import run_model

__all__ = [
    # QA
    "detect_qa",
    "revise_qa",
    "hallu_clean_qa",
    # Summarization
    "detect_sum",
    "revise_sum",
    "hallu_clean_sum",
    # Dialogue
    "detect_da",
    "revise_da",
    "hallu_clean_da",
    # Self-Contradiction
    "detect_tsc",
    "revise_tsc",
    "hallu_clean_tsc",
    # Math Word Problems
    "detect_mwp",
    "revise_mwp",
    "hallu_clean_mwp",
    # low-level model call
    "run_model",
]
