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

# halluclean/prompts/__init__.py

from .qa import (
    QA_PLAN_PROMPT,
    QA_REASON_PROMPT,
    QA_JUDGE_PROMPT,
    QA_REVISE_PROMPT,
)
from .sum import (
    SUM_PLAN_PROMPT,
    SUM_REASON_PROMPT,
    SUM_JUDGE_PROMPT,
    SUM_REVISE_PROMPT,
)
from .da import (
    DA_PLAN_PROMPT,
    DA_REASON_PROMPT,
    DA_JUDGE_PROMPT,
    DA_REVISE_PROMPT,
)
from .tsc import (
    TSC_PLAN_PROMPT,
    TSC_REASON_PROMPT,
    TSC_JUDGE_PROMPT,
    TSC_REVISE_PROMPT,
)
from .mwp import (
    MWP_PLAN_PROMPT,
    MWP_REASON_PROMPT,
    MWP_JUDGE_PROMPT,
    MWP_REVISE_PROMPT,
)

__all__ = [
    # QA
    "QA_PLAN_PROMPT",
    "QA_REASON_PROMPT",
    "QA_JUDGE_PROMPT",
    "QA_REVISE_PROMPT",
    # SUM
    "SUM_PLAN_PROMPT",
    "SUM_REASON_PROMPT",
    "SUM_JUDGE_PROMPT",
    "SUM_REVISE_PROMPT",
    # DA
    "DA_PLAN_PROMPT",
    "DA_REASON_PROMPT",
    "DA_JUDGE_PROMPT",
    "DA_REVISE_PROMPT",
    # TSC
    "TSC_PLAN_PROMPT",
    "TSC_REASON_PROMPT",
    "TSC_JUDGE_PROMPT",
    "TSC_REVISE_PROMPT",
    # MWP
    "MWP_PLAN_PROMPT",
    "MWP_REASON_PROMPT",
    "MWP_JUDGE_PROMPT",
    "MWP_REVISE_PROMPT",
]
