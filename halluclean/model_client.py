"""
Unified model interface for HalluClean.

这个模块对外提供一个统一的 LLM 调用入口：
    run_model(model_name, prompt, pipe=None, max_new_tokens=512, timeout=1000)

支持两类模型：
1. 远程模型（OpenAI / DeepSeek / 其他 OpenAI 兼容服务）
2. 本地模型（通过 transformers 的 pipeline，由用户自行加载）

- "chatgpt"     : 走 OpenAI / OPENAI_BASE_URL，默认 gpt-3.5-turbo-0125
- "gpt4o"       : 默认 gpt-4o
- "gpt4o-mini"  : 默认 gpt-4o-mini
- "deepseek"    : 走 DeepSeek OpenAI 兼容接口
- "local", "hf" : 使用本地 transformers pipeline（需要传 pipe）

如需扩展其他远程模型，可以直接在本文件中加一个分支。
"""

import os
from typing import Optional

from openai import OpenAI


# ========== 基础配置 ==========

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def _openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """
    构造一个 OpenAI 兼容客户端（包括官方/代理/gptsapi 等）。
    """
    key = api_key or OPENAI_API_KEY
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY 未设置，"
            "请先在环境中配置：export OPENAI_API_KEY=sk-..."
        )
    return OpenAI(
        api_key=key,
        base_url=base_url or OPENAI_BASE_URL,
    )


def _deepseek_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    """
    DeepSeek 使用 OpenAI 兼容接口。
    """
    key = api_key or DEEPSEEK_API_KEY
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未设置，"
            "使用 model_name='deepseek' 前请先配置：export DEEPSEEK_API_KEY=..."
        )
    return OpenAI(
        api_key=key,
        base_url=base_url or DEEPSEEK_BASE_URL,
    )


# ========== 统一入口函数 ==========


def run_model(
    model_name: str,
    prompt: str,
    pipe=None,
    max_new_tokens: int = 512,
    timeout: int = 1000,
) -> str:
    name = model_name.lower().strip()

    # ---------- 远程模型：OpenAI / 兼容接口 ----------

    if name in {"chatgpt", "gpt4o", "gpt4o-mini"}:
        model_id_map = {
            "chatgpt": os.getenv("HALLUCLEAN_CHATGPT_MODEL", "gpt-3.5-turbo-0125"),
            "gpt4o": os.getenv("HALLUCLEAN_GPT4O_MODEL", "gpt-4o"),
            "gpt4o-mini": os.getenv("HALLUCLEAN_GPT4O_MINI_MODEL", "gpt-4o-mini"),
        }
        model_id = model_id_map[name]

        client = _openai_client()
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": prompt}],
            timeout=timeout,
        )
        return resp.choices[0].message.content

    if name == "deepseek":
        model_id = os.getenv("HALLUCLEAN_DEEPSEEK_MODEL", "deepseek-chat")
        client = _deepseek_client()
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": prompt}],
            timeout=timeout,
        )
        return resp.choices[0].message.content

    # ---------- 本地模型：transformers pipeline ----------

    if name in {"local", "hf"}:
        if pipe is None:
            raise ValueError(
                "使用本地模型 (model_name='local' 或 'hf') 时，必须传入 pipe（transformers pipeline）。"
            )

        # 通用的 text-generation 调用方式：
        outputs = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
        )

        # 兼容几种常见返回格式
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            # transformers text-generation pipeline: [{"generated_text": "..."}]
            if isinstance(first, dict):
                if "generated_text" in first:
                    return first["generated_text"]
                if "summary_text" in first:
                    return first["summary_text"]
                if "text" in first:
                    return first["text"]
            # 如果直接是字符串列表
            if isinstance(first, str):
                return first

        # 若到这里还没返回，说明格式比较特殊，交给用户自己调试
        raise RuntimeError(
            "本地 pipeline 返回格式无法解析，请检查 transformers pipeline 的输出结构。"
        )
