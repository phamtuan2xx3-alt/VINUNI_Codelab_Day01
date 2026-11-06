"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"

GEMINI_COMPARISON_MODELS = (
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
)

# Standard paid-tier text prices (USD / 1M tokens), checked 2026-11-06.
# https://ai.google.dev/gemini-api/docs/pricing
# 3.6/3.8 Flash promotional rates expire 2026-12-31. Free-tier bills may be $0.
GEMINI_COMPARISON_PRICING = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
}


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Chưa điền OPENAI_API_KEY trong .env")

    client = OpenAI(api_key=api_key)
    try:
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_tokens,
        )
        latency = time.perf_counter() - start
        metadata = response.usage
        usage = {
            "input_tokens": (metadata.prompt_tokens or 0) if metadata else 0,
            "output_tokens": (metadata.completion_tokens or 0) if metadata else 0,
        }
        return response.choices[0].message.content or "", latency, usage
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    from google import genai
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Chưa điền GEMINI_API_KEY trong .env")

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
        thinking_config=(
            types.ThinkingConfig(thinking_budget=0)
            if model == "gemini-2.5-flash" else None
        ),
    )
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1beta"),
    )
    try:
        start = time.perf_counter()
        response = client.models.generate_content(
            model=model, contents=prompt, config=config,
        )
        latency = time.perf_counter() - start
        metadata = response.usage_metadata
        usage = {
            "input_tokens": (metadata.prompt_token_count or 0) if metadata else 0,
            "output_tokens": (metadata.candidates_token_count or 0) if metadata else 0,
        }
        thinking_tokens = getattr(metadata, "thoughts_token_count", None)
        usage["thinking_tokens"] = thinking_tokens if isinstance(thinking_tokens, int) else 0
        return response.text or "", latency, usage
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude Haiku 4.5 as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-haiku-4-5-20251001).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold. For Haiku 4.5, a non-default
                     value takes precedence over temperature; only one is sent.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Chưa điền ANTHROPIC_API_KEY trong .env")

    client = anthropic.Anthropic(api_key=api_key)
    try:
        # Haiku 4.5 accepts only one sampling control per request.
        sampling = {"top_p": top_p} if top_p != 0.9 else {"temperature": temperature}
        start = time.perf_counter()
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            # SDK 2.x removed sampling kwargs; the older API still supports
            # them for Haiku 4.5, preserving the lab's function parameters.
            extra_headers={"anthropic-version": "2023-06-01"},
            extra_body=sampling,
        )
        latency = time.perf_counter() - start
        text = "".join(
            block.text for block in response.content
            if isinstance(getattr(block, "text", None), str)
        )
        metadata = response.usage
        usage = {
            "input_tokens": metadata.input_tokens or 0,
            "output_tokens": metadata.output_tokens or 0,
        }
        return text, latency, usage
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Task 4 — Compare six Gemini Flash / Flash-Lite models
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Send the same prompt to GEMINI_COMPARISON_MODELS, sequentially.

    Return one entry per model ID with response, latency, token usage, estimated
    paid-tier USD cost, and status. Output cost includes thinking tokens.
    Failed requests have cost=None (unknown), an error, and status="error";
    remaining models are still evaluated. No automatic quality score is assigned.
    """
    results = {}
    for model in GEMINI_COMPARISON_MODELS:
        started = time.perf_counter()
        try:
            text, latency, usage = call_gemini(prompt, model=model, max_tokens=1024)
        except Exception as exc:
            # Do not echo API exception messages: they can contain credentials.
            results[model] = {
                "response": "", "latency": time.perf_counter() - started,
                "input_tokens": None, "output_tokens": None, "thinking_tokens": None,
                "cost": None, "status": "error",
                "error": f"{type(exc).__name__} (code={getattr(exc, 'code', 'unknown')})",
            }
            continue

        rates = GEMINI_COMPARISON_PRICING[model]
        thinking = usage.get("thinking_tokens", 0)
        cost = (
            usage["input_tokens"] * rates["input"]
            + (usage["output_tokens"] + thinking) * rates["output"]
        ) / 1_000_000
        results[model] = {
            "response": text, "latency": latency,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"], "thinking_tokens": thinking,
            "cost": cost, "status": "ok" if text else "empty",
            "cost_basis": "estimated standard paid-tier text pricing, 2026-11-06",
        }
    return results


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """Chat streaming với Gemini 2.5 Flash, giữ 3 lượt hội thoại."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = []

    print("Gemini chatbot — nhập 'quit' hoặc 'exit' để thoát.")

    try:
        while True:
            try:
                user_input = input("\nBạn: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nĐã kết thúc.")
                break

            if user_input.lower() in ("quit", "exit"):
                print("Đã kết thúc.")
                break

            if not user_input:
                continue

            user_message = types.Content(
                role="user",
                parts=[types.Part(text=user_input)],
            )

            print("Gemini: ", end="", flush=True)
            response_parts = []

            try:
                stream = client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=history + [user_message],
                )

                for chunk in stream:
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        response_parts.append(chunk.text)

                print()

            except Exception as exc:
                print(f"\nLỗi gọi API: {exc}")
                continue

            response_text = "".join(response_parts)

            if response_text:
                history.extend([
                    user_message,
                    types.Content(
                        role="model",
                        parts=[types.Part(text=response_text)],
                    ),
                ])

                # 3 lượt = 3 cặp user/model = 6 message.
                history = history[-6:]

    finally:
        client.close()


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """Gọi fn(), thử lại khi lỗi với thời gian chờ tăng gấp đôi."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise

            time.sleep(base_delay * (2 ** attempt))

# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []

    for prompt in prompts:
        result = dict(compare_models(prompt))
        result["prompt"] = prompt
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    def cell(value):
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    model_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-Mini",
        "gemini_flash": "Gemini-Flash",
    }
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        for model, stats in result.items():
            if model == "prompt":
                continue
            response = str(stats.get("response", stats.get("error", "")))
            response = response.replace("\r", " ").replace("\n", " ")
            if len(response) > 50:
                response = response[:47] + "..."
            values = [
                cell(result.get("prompt", "")),
                cell(model_names.get(model, model)),
                cell(response),
                f"{stats.get('latency', 0.0):.3f}s",
                f"{stats.get('input_tokens', 0)}/{stats.get('output_tokens', 0)}",
                f"${stats.get('cost', 0.0):.6f}",
            ]
            lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            if stats["status"] == "error":
                print(f"Failed: {stats['error']}")
                continue
            print(f"Latency: {stats['latency']:.2f}s | Estimated paid cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
