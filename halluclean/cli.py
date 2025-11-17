#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for HalluClean.

Usage examples
--------------

# 1) QA, full pipeline, JSONL 输入输出
python -m halluclean.cli \
  --task qa \
  --mode pipeline \
  --input data/qa_input.jsonl \
  --output out/qa_hallu_clean.jsonl \
  --detect-model chatgpt

# 2) Summarization, 只做检测，从 stdin 读，写到 stdout
cat data/sum_input.jsonl | python -m halluclean.cli \
  --task sum \
  --mode detect \
  --detect-model chatgpt

Input / output format
---------------------

默认使用 JSONL（每行一个 JSON 对象），不同 task 对应字段如下：

- task = qa
  input:  {"question": "...", "answer": "..."}
- task = sum
  input:  {"source_text": "...", "summary": "..."}
- task = da
  input:  {"context": "...", "response": "..."}
- task = tsc
  input:  {"text": "..."}
- task = mwp
  input:  {"problem": "...", "solution": "..."}

输出：在原有字段基础上 merge HalluClean 的结果（detection / revised_* 等），
仍然是一行一个 JSON。
"""

import argparse
import json
import sys
from typing import Dict, Any, Iterable, Optional

from . import (
    # QA
    detect_qa,
    revise_qa,
    hallu_clean_qa,
    # SUM
    detect_sum,
    revise_sum,
    hallu_clean_sum,
    # DA
    detect_da,
    revise_da,
    hallu_clean_da,
    # TSC
    detect_tsc,
    revise_tsc,
    hallu_clean_tsc,
    # MWP
    detect_mwp,
    revise_mwp,
    hallu_clean_mwp,
)


# ---------------------- IO helpers ----------------------


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Read JSONL from a file or stdin ('-')."""
    if path == "-":
        f = sys.stdin
    else:
        f = open(path, "r", encoding="utf-8")
    try:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
    finally:
        if path != "-":
            f.close()


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    """Write JSONL to a file or stdout ('-')."""
    if path == "-":
        f = sys.stdout
    else:
        f = open(path, "w", encoding="utf-8")
    try:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if path != "-":
            f.close()


def _get_field(rec: Dict[str, Any], *candidates: str) -> Any:
    """Try multiple field names and return the first that exists."""
    for name in candidates:
        if name in rec:
            return rec[name]
    raise KeyError(f"None of fields {candidates} found in record: {rec}")


# ---------------------- core dispatch ----------------------


def process_record(
    task: str,
    mode: str,
    rec: Dict[str, Any],
    detect_model: str,
    revise_model: Optional[str],
    max_new_tokens_detect: int,
    max_new_tokens_revise: int,
) -> Dict[str, Any]:
    """
    对单条记录执行指定 task & mode，并返回新的记录（原记录 + 结果）。
    """
    task = task.lower()
    mode = mode.lower()

    if revise_model is None:
        revise_model = detect_model

    # 先拷贝一份原始记录，避免直接修改输入
    out = dict(rec)

    # ---------------- QA ----------------
    if task == "qa":
        question = _get_field(rec, "question", "q")
        answer = _get_field(rec, "answer", "a", "response")

        if mode == "detect":
            det = detect_qa(
                question=question,
                answer=answer,
                model_name=detect_model,
                max_new_tokens=max_new_tokens_detect,
            )
            out["hallu_detection"] = det
            return out

        if mode == "revise":
            rev = revise_qa(
                question=question,
                hallucinated_answer=answer,
                model_name=revise_model,
                max_new_tokens=max_new_tokens_revise,
            )
            out["hallu_revision"] = rev
            out["revised_answer"] = rev["revised_answer"]
            return out

        # pipeline
        res = hallu_clean_qa(
            question=question,
            answer=answer,
            detect_model=detect_model,
            revise_model=revise_model,
            max_new_tokens_detect=max_new_tokens_detect,
            max_new_tokens_revise=max_new_tokens_revise,
        )
        out["hallu_result"] = res
        out["revised_answer"] = res["revised_answer"]
        return out

    # ---------------- SUM ----------------
    if task == "sum":
        source_text = _get_field(rec, "source_text", "source", "document")
        summary = _get_field(rec, "summary")

        if mode == "detect":
            det = detect_sum(
                source_text=source_text,
                summary=summary,
                model_name=detect_model,
                max_new_tokens=max_new_tokens_detect,
            )
            out["hallu_detection"] = det
            return out

        if mode == "revise":
            rev = revise_sum(
                source_text=source_text,
                hallucinated_summary=summary,
                model_name=revise_model,
                max_new_tokens=max_new_tokens_revise,
            )
            out["hallu_revision"] = rev
            out["revised_summary"] = rev["revised_summary"]
            return out

        res = hallu_clean_sum(
            source_text=source_text,
            summary=summary,
            detect_model=detect_model,
            revise_model=revise_model,
            max_new_tokens_detect=max_new_tokens_detect,
            max_new_tokens_revise=max_new_tokens_revise,
        )
        out["hallu_result"] = res
        out["revised_summary"] = res["revised_summary"]
        return out

    # ---------------- DA ----------------
    if task == "da":
        context = _get_field(rec, "context", "history")
        response = _get_field(rec, "response", "answer")

        if mode == "detect":
            det = detect_da(
                context=context,
                response=response,
                model_name=detect_model,
                max_new_tokens=max_new_tokens_detect,
            )
            out["hallu_detection"] = det
            return out

        if mode == "revise":
            rev = revise_da(
                context=context,
                hallucinated_response=response,
                model_name=revise_model,
                max_new_tokens=max_new_tokens_revise,
            )
            out["hallu_revision"] = rev
            out["revised_response"] = rev["revised_response"]
            return out

        res = hallu_clean_da(
            context=context,
            response=response,
            detect_model=detect_model,
            revise_model=revise_model,
            max_new_tokens_detect=max_new_tokens_detect,
            max_new_tokens_revise=max_new_tokens_revise,
        )
        out["hallu_result"] = res
        out["revised_response"] = res["revised_response"]
        return out

    # ---------------- TSC ----------------
    if task == "tsc":
        text = _get_field(rec, "text", "content")

        if mode == "detect":
            det = detect_tsc(
                text=text,
                model_name=detect_model,
                max_new_tokens=max_new_tokens_detect,
            )
            out["hallu_detection"] = det
            return out

        if mode == "revise":
            rev = revise_tsc(
                text=text,
                model_name=revise_model,
                max_new_tokens=max_new_tokens_revise,
            )
            out["hallu_revision"] = rev
            out["revised_text"] = rev["revised_text"]
            return out

        res = hallu_clean_tsc(
            text=text,
            detect_model=detect_model,
            revise_model=revise_model,
            max_new_tokens_detect=max_new_tokens_detect,
            max_new_tokens_revise=max_new_tokens_revise,
        )
        out["hallu_result"] = res
        out["revised_text"] = res["revised_text"]
        return out

    # ---------------- MWP ----------------
    if task == "mwp":
        problem = _get_field(rec, "problem", "question")
        solution = _get_field(rec, "solution", "answer")

        if mode == "detect":
            det = detect_mwp(
                problem=problem,
                solution=solution,
                model_name=detect_model,
                max_new_tokens=max_new_tokens_detect,
            )
            out["hallu_detection"] = det
            return out

        if mode == "revise":
            rev = revise_mwp(
                problem=problem,
                hallucinated_solution=solution,
                model_name=revise_model,
                max_new_tokens=max_new_tokens_revise,
            )
            out["hallu_revision"] = rev
            out["revised_solution"] = rev["revised_solution"]
            return out

        res = hallu_clean_mwp(
            problem=problem,
            solution=solution,
            detect_model=detect_model,
            revise_model=revise_model,
            max_new_tokens_detect=max_new_tokens_detect,
            max_new_tokens_revise=max_new_tokens_revise,
        )
        out["hallu_result"] = res
        out["revised_solution"] = res["revised_solution"]
        return out

    # 未知 task
    raise ValueError(f"Unsupported task: {task}")


# ---------------------- main ----------------------


def main():
    parser = argparse.ArgumentParser(
        description="HalluClean command-line interface (JSONL in / JSONL out)."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["qa", "sum", "da", "tsc", "mwp"],
        help="Task type: qa / sum / da / tsc / mwp",
    )
    parser.add_argument(
        "--mode",
        default="pipeline",
        choices=["detect", "revise", "pipeline"],
        help="Run mode: detect / revise / pipeline (default: pipeline)",
    )
    parser.add_argument(
        "--input",
        default="-",
        help="Input JSONL file path (default: '-' = stdin).",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output JSONL file path (default: '-' = stdout).",
    )
    parser.add_argument(
        "--detect-model",
        default="chatgpt",
        help="Model name for detection (default: chatgpt).",
    )
    parser.add_argument(
        "--revise-model",
        default=None,
        help="Model name for revision (default: same as --detect-model).",
    )
    parser.add_argument(
        "--max-new-tokens-detect",
        type=int,
        default=512,
        help="max_new_tokens for detection calls (default: 512).",
    )
    parser.add_argument(
        "--max-new-tokens-revise",
        type=int,
        default=512,
        help="max_new_tokens for revision calls (default: 512).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of records to process.",
    )

    args = parser.parse_args()

    records_in = read_jsonl(args.input)
    records_out = []

    for idx, rec in enumerate(records_in):
        if args.limit is not None and idx >= args.limit:
            break
        try:
            processed = process_record(
                task=args.task,
                mode=args.mode,
                rec=rec,
                detect_model=args.detect_model,
                revise_model=args.revise_model,
                max_new_tokens_detect=args.max_new_tokens_detect,
                max_new_tokens_revise=args.max_new_tokens_revise,
            )
        except Exception as e:  # noqa
            # 出错时也写出一条记录，方便排查
            processed = dict(rec)
            processed["hallu_error"] = str(e)
        records_out.append(processed)

    write_jsonl(args.output, records_out)


if __name__ == "__main__":
    main()
