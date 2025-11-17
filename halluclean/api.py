"""
覆盖以下任务：
- QA   : question answering
- SUM  : summarization
- DA   : dialogue / conversation
- TSC  : self-contradiction detection
- MWP  : math word problems

每个任务提供三个接口层级：
- detect_xxx       : 只做幻觉 / 错误检测（Yes/No）
- revise_xxx       : 只做修订
- hallu_clean_xxx  : Pipeline = 检测 +（必要时）修订

所有函数都只依赖 .model_client.run_model，便于统一管理 API key / base_url 等。
"""

from typing import Dict, Any, Optional

from .model_client import run_model

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
    QA 幻觉检测：判断 answer 是否包含明显错误或虚构内容。
    """
    prompt = f"""\
You are provided with a question and its corresponding answer.

[Question]
{question}

[Answer]
{answer}

Your task:
1. Carefully check whether the answer contains hallucinated content,
   i.e., statements that are likely false or unsupported by reliable knowledge.
2. Output only a single word:
   - "Yes"  if the answer contains hallucinations or factual errors.
   - "No"   if the answer is factually sound and does not hallucinate.
"""

    raw = run_model(model_name, prompt, pipe=pipe, max_new_tokens=max_new_tokens)
    is_halu = _parse_yes_no(raw)

    return {
        "task": "qa",
        "question": question,
        "answer": answer,
        "raw_judgement": raw,
        "is_hallucinated": is_halu,
    }


def revise_qa(
    question: str,
    hallucinated_answer: str,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    QA 修订：在知道原答案可能有幻觉的前提下，重新给出尽量可靠的新答案。
    """
    prompt = f"""\
Given a question and its corresponding hallucinated answer.

[Question]
{question}

[Hallucinated Answer]
{hallucinated_answer}

Your task:
- Answer the question again WITHOUT introducing any hallucinations.
- If you are uncertain, you MUST explicitly say you are uncertain
  instead of fabricating facts.
- Just output the final answer text, without any additional explanation.
"""

    revised = run_model(model_name, prompt, pipe=pipe, max_new_tokens=max_new_tokens)

    return {
        "task": "qa",
        "question": question,
        "original_answer": hallucinated_answer,
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
    QA 一站式 pipeline：
    - Step 1: detect_qa
    - Step 2: 如果检测为 hallucinated，则调用 revise_qa；否则直接返回原答案。
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
    摘要幻觉检测：判断 summary 是否包含源文档中不存在或被否定的信息。
    """
    prompt = f"""\
You are given a source document and a model-generated summary.

[Source Document]
{source_text}

[Summary]
{summary}

Your task:
1. Check whether the summary contains hallucinated content, i.e.,
   statements that are not supported by the source document or
   contradicted by it.
2. Output only a single word:
   - "Yes" if the summary contains any hallucinations.
   - "No"  if the summary is fully supported by the source.
"""

    raw = run_model(model_name, prompt, pipe=pipe, max_new_tokens=max_new_tokens)
    is_halu = _parse_yes_no(raw)

    return {
        "task": "sum",
        "source_text": source_text,
        "summary": summary,
        "raw_judgement": raw,
        "is_hallucinated": is_halu,
    }


def revise_sum(
    source_text: str,
    hallucinated_summary: str,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    摘要修订：给定源文档和可能含有幻觉的摘要，生成一个忠实的摘要。
    """
    prompt = f"""\
You are given a source document and a hallucinated summary.

[Source Document]
{source_text}

[Hallucinated Summary]
{hallucinated_summary}

Your task:
- Rewrite the summary so that:
  * It is faithful to the source document.
  * It does NOT introduce any new facts that are not stated or
    clearly implied by the source.
- Keep the length roughly similar to the original summary.
- Just output the revised summary text without any extra explanation.
"""

    revised = run_model(model_name, prompt, pipe=pipe, max_new_tokens=max_new_tokens)

    return {
        "task": "sum",
        "source_text": source_text,
        "original_summary": hallucinated_summary,
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
    Summarization pipeline：先检测再必要时修订。
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
    对话场景幻觉检测：context 是对话历史，response 是当前回复。
    """
    prompt = f"""\
You are given a dialogue context and an assistant response.

[Dialogue Context]
{context}

[Assistant Response]
{response}

Your task:
1. Decide whether the assistant response contains hallucinated factual
   statements, i.e., plausible-looking statements that are likely false
   or unsupported by widely accepted knowledge.
2. Ignore harmless small talk ("How are you?", etc.) — focus on factual claims.
3. Output only a single word:
   - "Yes" if the response contains hallucinations.
   - "No"  if the response is factually reliable.
"""

    raw = run_model(model_name, prompt, pipe=pipe, max_new_tokens=max_new_tokens)
    is_halu = _parse_yes_no(raw)

    return {
        "task": "da",
        "context": context,
        "response": response,
        "raw_judgement": raw,
        "is_hallucinated": is_halu,
    }


def revise_da(
    context: str,
    hallucinated_response: str,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    对话修订：保持对话风格自然，把回复改成更可靠、不乱编事实的版本。
    """
    prompt = f"""\
You are given a dialogue context and a hallucinated assistant response.

[Dialogue Context]
{context}

[Hallucinated Assistant Response]
{hallucinated_response}

Your task:
- Rewrite the assistant response so that:
  * It is factually reliable.
  * It does NOT invent unsupported facts.
  * When uncertain, it explicitly states uncertainty instead of fabricating.
- Keep the style conversational and helpful.
- Just output the revised assistant response text.
"""

    revised = run_model(model_name, prompt, pipe=pipe, max_new_tokens=max_new_tokens)

    return {
        "task": "da",
        "context": context,
        "original_response": hallucinated_response,
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
    Dialogue pipeline：context + response → detect →（必要时）revise。
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
    自相矛盾 (TSC) 检测：判断文本内部是否存在事实层面的矛盾陈述。
    """
    prompt = f"""\
You are given a single passage.

[Text]
{text}

Your task:
1. Check whether the text contains internal factual contradictions, i.e.,
   the text makes incompatible claims about the same entities, time, or quantities.
2. Ignore minor stylistic issues; focus on factual self-contradiction.
3. Output only a single word:
   - "Yes" if the text is self-contradictory.
   - "No"  if the text is internally consistent.
"""

    raw = run_model(model_name, prompt, pipe=pipe, max_new_tokens=max_new_tokens)
    is_contra = _parse_yes_no(raw)

    return {
        "task": "tsc",
        "text": text,
        "raw_judgement": raw,
        "is_self_contradictory": is_contra,
        "is_hallucinated": is_contra,
    }


def revise_tsc(
    text: str,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    自相矛盾文本修订：输入是一段可能自相矛盾的文本，输出内部一致版本。
    """
    prompt = f"""\
You are given a passage that may contain internal factual contradictions.

[Original Text]
{text}

Your task:
- Rewrite the text so that:
  * It becomes internally consistent.
  * It does not contain factual self-contradictions.
- Keep the overall meaning and key information as close as possible
  to the original, while resolving contradictions.
- Just output the revised text.
"""

    revised = run_model(model_name, prompt, pipe=pipe, max_new_tokens=max_new_tokens)

    return {
        "task": "tsc",
        "original_text": text,
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
    TSC pipeline：先检测是否自相矛盾，再视情况修订。
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
    数学应用题 (MWP) 幻觉检测：判断解答是否存在数学错误或不合理推理。
    """
    prompt = f"""\
You are given a math word problem and a model-generated solution.

[Problem]
{problem}

[Proposed Solution]
{solution}

Your task:
1. Carefully verify the solution step by step.
2. If there is ANY mathematical error, unjustified step, or incorrect
   final answer, treat the solution as hallucinated.
3. Output only a single word:
   - "Yes" if the solution is incorrect or contains hallucinated reasoning.
   - "No"  if the solution is fully correct and logically sound.
"""

    raw = run_model(model_name, prompt, pipe=pipe, max_new_tokens=max_new_tokens)
    is_halu = _parse_yes_no(raw)

    return {
        "task": "mwp",
        "problem": problem,
        "solution": solution,
        "raw_judgement": raw,
        "is_hallucinated": is_halu,
    }


def revise_mwp(
    problem: str,
    hallucinated_solution: str,
    model_name: str = "chatgpt",
    pipe=None,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    数学应用题修订：重新解题并给出正确且有条理的解答。
    """
    prompt = f"""\
You are given a math word problem and a hallucinated solution.

[Problem]
{problem}

[Hallucinated Solution]
{hallucinated_solution}

Your task:
- Solve the problem from scratch.
- Provide a clear and correct step-by-step solution.
- You may reuse correct parts from the hallucinated solution,
  but you MUST fix any errors.
- At the end, clearly state the final numerical/short answer.
"""

    revised = run_model(model_name, prompt, pipe=pipe, max_new_tokens=max_new_tokens)

    return {
        "task": "mwp",
        "problem": problem,
        "original_solution": hallucinated_solution,
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
    MWP pipeline：先判解答是否“幻觉”，再决定是否重解。
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
