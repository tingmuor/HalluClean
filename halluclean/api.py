"""
High-level public API for HalluClean.

对外暴露一个统一、简洁的 Python 接口，覆盖以下任务：
- QA   : question answering
- SUM  : summarization
- DA   : dialogue / conversation
- TSC  : self-contradiction detection
- MWP  : math word problems

每个任务提供三个接口层级：
- detect_xxx       : 只做幻觉 / 错误检测（Plan → Reason → Judge）
- revise_xxx       : 只做修订（可利用 detection 阶段的 analysis）
- hallu_clean_xxx  : Pipeline = 检测 +（必要时）修订

所有函数都只依赖 .model_client.run_model，便于统一管理 API key / base_url 等。
"""

from typing import Dict, Any, Optional

from .model_client import run_model
from .prompts import (
    # QA
    QA_PLAN_PROMPT,
    QA_REASON_PROMPT,
    QA_JUDGE_PROMPT,
    QA_REVISE_PROMPT,
    # SUM
    SUM_PLAN_PROMPT,
    SUM_REASON_PROMPT,
    SUM_JUDGE_PROMPT,
    SUM_REVISE_PROMPT,
    # DA
    DA_PLAN_PROMPT,
    DA_REASON_PROMPT,
    DA_JUDGE_PROMPT,
    DA_REVISE_PROMPT,
    # TSC
    TSC_PLAN_PROMPT,
    TSC_REASON_PROMPT,
    TSC_JUDGE_PROMPT,
    TSC_REVISE_PROMPT,
    # MWP
    MWP_PLAN_PROMPT,
    MWP_REASON_PROMPT,
    MWP_JUDGE_PROMPT,
    MWP_REVISE_PROMPT,
)


# ----------------- 通用小工具 -----------------


def _parse_yes_no(raw: str) -> bool:
    """
    从模型回答里抽取 Yes / No。

    返回 True  表示 Yes（有幻觉 / 有错误 / 有矛盾）
    返回 False 表示 No  （没有）

    如果同时出现 yes 和 no，或者都没出现，视为不确定 → 保守地当作 True。
    """
    lower = raw.lower()
    has_yes = "yes" in lower
    has_no = "no" in lower

    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return True  # 不确定时保守处理为“有问题”


# =========================================================
# 1. QA: Question Answering
# =========================================================


def detect_qa(
    question: str,
    answer: str,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    QA 幻觉检测：三阶段 CoT（Plan → Reason → Judge）。
    """

    # 1) Plan
    plan_prompt = QA_PLAN_PROMPT.format(
        question=question,
        answer=answer,
    )
    plan = run_model(
        model_name,
        plan_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    # 2) Reason
    reason_prompt = QA_REASON_PROMPT.format(
        question=question,
        answer=answer,
        plan=plan,
    )
    analysis = run_model(
        model_name,
        reason_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    # 3) Judge
    judge_prompt = QA_JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        analysis=analysis,
    )
    raw = run_model(
        model_name,
        judge_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    is_halu = _parse_yes_no(raw)

    return {
        "task": "qa",
        "question": question,
        "answer": answer,
        "plan": plan,
        "analysis": analysis,
        "raw_judgement": raw,
        "is_hallucinated": is_halu,
    }


def revise_qa(
    question: str,
    hallucinated_answer: str,
    analysis: Optional[str] = None,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    QA 修订。

    使用 detection 阶段的 analysis（如果提供）辅助修订。
    """
    prompt = QA_REVISE_PROMPT.format(
        question=question,
        answer=hallucinated_answer,
        analysis=analysis or "",
    )

    revised = run_model(
        model_name,
        prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    return {
        "task": "qa",
        "question": question,
        "original_answer": hallucinated_answer,
        "analysis": analysis,
        "revised_answer": revised,
    }


def hallu_clean_qa(
    question: str,
    answer: str,
    detect_model: str = "chatgpt",
    revise_model: Optional[str] = None,
    detect_pipe=None,
    revise_pipe=None,
    max_new_tokens_detect: int = 512,
    max_new_tokens_revise: int = 512,
) -> Dict[str, Any]:
    """
    QA 一站式 pipeline。
    """
    if revise_model is None:
        revise_model = detect_model

    det = detect_qa(
        question=question,
        answer=answer,
        model_name=detect_model,
        pipe=detect_pipe,
        max_new_tokens=max_new_tokens_detect,
    )

    if not det["is_hallucinated"]:
        return {
            "task": "qa",
            "question": question,
            "original_answer": answer,
            "revised_answer": answer,
            "detection": det,
        }

    rev = revise_qa(
        question=question,
        hallucinated_answer=answer,
        analysis=det.get("analysis"),
        model_name=revise_model,
        pipe=revise_pipe,
        max_new_tokens=max_new_tokens_revise,
    )

    return {
        "task": "qa",
        "question": question,
        "original_answer": answer,
        "revised_answer": rev["revised_answer"],
        "detection": det,
    }


# =========================================================
# 2. SUM: Summarization
# =========================================================


def detect_sum(
    source_text: str,
    summary: str,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    摘要幻觉检测：三阶段 CoT。
    """
    # 1) Plan
    plan_prompt = SUM_PLAN_PROMPT.format(
        source_text=source_text,
        summary=summary,
    )
    plan = run_model(
        model_name,
        plan_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    # 2) Reason
    reason_prompt = SUM_REASON_PROMPT.format(
        source_text=source_text,
        summary=summary,
        plan=plan,
    )
    analysis = run_model(
        model_name,
        reason_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    # 3) Judge
    judge_prompt = SUM_JUDGE_PROMPT.format(
        source_text=source_text,
        summary=summary,
        analysis=analysis,
    )
    raw = run_model(
        model_name,
        judge_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    is_halu = _parse_yes_no(raw)

    return {
        "task": "sum",
        "source_text": source_text,
        "summary": summary,
        "plan": plan,
        "analysis": analysis,
        "raw_judgement": raw,
        "is_hallucinated": is_halu,
    }


def revise_sum(
    source_text: str,
    hallucinated_summary: str,
    analysis: Optional[str] = None,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    摘要修订。

    使用 detection 阶段的 analysis（如果提供）辅助修订。
    """
    prompt = SUM_REVISE_PROMPT.format(
        source_text=source_text,
        summary=hallucinated_summary,
        analysis=analysis or "",
    )

    revised = run_model(
        model_name,
        prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    return {
        "task": "sum",
        "source_text": source_text,
        "original_summary": hallucinated_summary,
        "analysis": analysis,
        "revised_summary": revised,
    }


def hallu_clean_sum(
    source_text: str,
    summary: str,
    detect_model: str = "chatgpt",
    revise_model: Optional[str] = None,
    detect_pipe=None,
    revise_pipe=None,
    max_new_tokens_detect: int = 512,
    max_new_tokens_revise: int = 512,
) -> Dict[str, Any]:
    """
    Summarization pipeline。
    """
    if revise_model is None:
        revise_model = detect_model

    det = detect_sum(
        source_text=source_text,
        summary=summary,
        model_name=detect_model,
        pipe=detect_pipe,
        max_new_tokens=max_new_tokens_detect,
    )

    if not det["is_hallucinated"]:
        return {
            "task": "sum",
            "source_text": source_text,
            "original_summary": summary,
            "revised_summary": summary,
            "detection": det,
        }

    rev = revise_sum(
        source_text=source_text,
        hallucinated_summary=summary,
        analysis=det.get("analysis"),
        model_name=revise_model,
        pipe=revise_pipe,
        max_new_tokens=max_new_tokens_revise,
    )

    return {
        "task": "sum",
        "source_text": source_text,
        "original_summary": summary,
        "revised_summary": rev["revised_summary"],
        "detection": det,
    }


# =========================================================
# 3. DA: Dialogue / Conversation
# =========================================================


def detect_da(
    context: str,
    response: str,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    对话场景幻觉检测：三阶段 CoT。
    """
    # 1) Plan
    plan_prompt = DA_PLAN_PROMPT.format(
        context=context,
        response=response,
    )
    plan = run_model(
        model_name,
        plan_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    # 2) Reason
    reason_prompt = DA_REASON_PROMPT.format(
        context=context,
        response=response,
        plan=plan,
    )
    analysis = run_model(
        model_name,
        reason_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    # 3) Judge
    judge_prompt = DA_JUDGE_PROMPT.format(
        context=context,
        response=response,
        analysis=analysis,
    )
    raw = run_model(
        model_name,
        judge_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    is_halu = _parse_yes_no(raw)

    return {
        "task": "da",
        "context": context,
        "response": response,
        "plan": plan,
        "analysis": analysis,
        "raw_judgement": raw,
        "is_hallucinated": is_halu,
    }


def revise_da(
    context: str,
    hallucinated_response: str,
    analysis: Optional[str] = None,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    对话修订。

    使用 detection 阶段的 analysis（如果提供）辅助修订。
    """
    prompt = DA_REVISE_PROMPT.format(
        context=context,
        response=hallucinated_response,
        analysis=analysis or "",
    )

    revised = run_model(
        model_name,
        prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    return {
        "task": "da",
        "context": context,
        "original_response": hallucinated_response,
        "analysis": analysis,
        "revised_response": revised,
    }


def hallu_clean_da(
    context: str,
    response: str,
    detect_model: str = "chatgpt",
    revise_model: Optional[str] = None,
    detect_pipe=None,
    revise_pipe=None,
    max_new_tokens_detect: int = 512,
    max_new_tokens_revise: int = 512,
) -> Dict[str, Any]:
    """
    Dialogue pipeline。
    """
    if revise_model is None:
        revise_model = detect_model

    det = detect_da(
        context=context,
        response=response,
        model_name=detect_model,
        pipe=detect_pipe,
        max_new_tokens=max_new_tokens_detect,
    )

    if not det["is_hallucinated"]:
        return {
            "task": "da",
            "context": context,
            "original_response": response,
            "revised_response": response,
            "detection": det,
        }

    rev = revise_da(
        context=context,
        hallucinated_response=response,
        analysis=det.get("analysis"),
        model_name=revise_model,
        pipe=revise_pipe,
        max_new_tokens=max_new_tokens_revise,
    )

    return {
        "task": "da",
        "context": context,
        "original_response": response,
        "revised_response": rev["revised_response"],
        "detection": det,
    }


# =========================================================
# 4. TSC: Self-Contradiction Detection
# =========================================================


def detect_tsc(
    text: str,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    自相矛盾检测：三阶段 CoT。
    """
    # 1) Plan
    plan_prompt = TSC_PLAN_PROMPT.format(text=text)
    plan = run_model(
        model_name,
        plan_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    # 2) Reason
    reason_prompt = TSC_REASON_PROMPT.format(
        text=text,
        plan=plan,
    )
    analysis = run_model(
        model_name,
        reason_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    # 3) Judge
    judge_prompt = TSC_JUDGE_PROMPT.format(
        text=text,
        analysis=analysis,
    )
    raw = run_model(
        model_name,
        judge_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    is_contra = _parse_yes_no(raw)

    return {
        "task": "tsc",
        "text": text,
        "plan": plan,
        "analysis": analysis,
        "raw_judgement": raw,
        "is_self_contradictory": is_contra,
        "is_hallucinated": is_contra,
    }


def revise_tsc(
    text: str,
    analysis: Optional[str] = None,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    自相矛盾文本修订。

    使用 detection 阶段的 analysis（如果提供）辅助修订。
    """
    prompt = TSC_REVISE_PROMPT.format(
        text=text,
        analysis=analysis or "",
    )

    revised = run_model(
        model_name,
        prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    return {
        "task": "tsc",
        "original_text": text,
        "analysis": analysis,
        "revised_text": revised,
    }


def hallu_clean_tsc(
    text: str,
    detect_model: str = "chatgpt",
    revise_model: Optional[str] = None,
    detect_pipe=None,
    revise_pipe=None,
    max_new_tokens_detect: int = 512,
    max_new_tokens_revise: int = 512,
) -> Dict[str, Any]:
    """
    TSC pipeline。
    """
    if revise_model is None:
        revise_model = detect_model

    det = detect_tsc(
        text=text,
        model_name=detect_model,
        pipe=detect_pipe,
        max_new_tokens=max_new_tokens_detect,
    )

    if not det["is_self_contradictory"]:
        return {
            "task": "tsc",
            "original_text": text,
            "revised_text": text,
            "detection": det,
        }

    rev = revise_tsc(
        text=text,
        analysis=det.get("analysis"),
        model_name=revise_model,
        pipe=revise_pipe,
        max_new_tokens=max_new_tokens_revise,
    )

    return {
        "task": "tsc",
        "original_text": text,
        "revised_text": rev["revised_text"],
        "detection": det,
    }


# =========================================================
# 5. MWP: Math Word Problems
# =========================================================


def detect_mwp(
    problem: str,
    solution: str,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    数学应用题幻觉检测：三阶段 CoT。
    """
    # 1) Plan
    plan_prompt = MWP_PLAN_PROMPT.format(
        problem=problem,
        solution=solution,
    )
    plan = run_model(
        model_name,
        plan_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    # 2) Reason
    reason_prompt = MWP_REASON_PROMPT.format(
        problem=problem,
        solution=solution,
        plan=plan,
    )
    analysis = run_model(
        model_name,
        reason_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    # 3) Judge
    judge_prompt = MWP_JUDGE_PROMPT.format(
        problem=problem,
        solution=solution,
        analysis=analysis,
    )
    raw = run_model(
        model_name,
        judge_prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    is_halu = _parse_yes_no(raw)

    return {
        "task": "mwp",
        "problem": problem,
        "solution": solution,
        "plan": plan,
        "analysis": analysis,
        "raw_judgement": raw,
        "is_hallucinated": is_halu,
    }


def revise_mwp(
    problem: str,
    hallucinated_solution: str,
    analysis: Optional[str] = None,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    数学应用题修订。

    使用 detection 阶段的 analysis（如果提供）辅助修订。
    """
    prompt = MWP_REVISE_PROMPT.format(
        problem=problem,
        solution=hallucinated_solution,
        analysis=analysis or "",
    )

    revised = run_model(
        model_name,
        prompt,
        pipe=pipe,
        max_new_tokens=max_new_tokens,
    )

    return {
        "task": "mwp",
        "problem": problem,
        "original_solution": hallucinated_solution,
        "analysis": analysis,
        "revised_solution": revised,
    }


def hallu_clean_mwp(
    problem: str,
    solution: str,
    detect_model: str = "chatgpt",
    revise_model: Optional[str] = None,
    detect_pipe=None,
    revise_pipe=None,
    max_new_tokens_detect: int = 512,
    max_new_tokens_revise: int = 512,
) -> Dict[str, Any]:
    """
    MWP pipeline。
    """
    if revise_model is None:
        revise_model = detect_model

    det = detect_mwp(
        problem=problem,
        solution=solution,
        model_name=detect_model,
        pipe=detect_pipe,
        max_new_tokens=max_new_tokens_detect,
    )

    if not det["is_hallucinated"]:
        return {
            "task": "mwp",
            "problem": problem,
            "original_solution": solution,
            "revised_solution": solution,
            "detection": det,
        }

    rev = revise_mwp(
        problem=problem,
        hallucinated_solution=solution,
        analysis=det.get("analysis"),
        model_name=revise_model,
        pipe=revise_pipe,
        max_new_tokens=max_new_tokens_revise,
    )

    return {
        "task": "mwp",
        "problem": problem,
        "original_solution": solution,
        "revised_solution": rev["revised_solution"],
        "detection": det,
    }
