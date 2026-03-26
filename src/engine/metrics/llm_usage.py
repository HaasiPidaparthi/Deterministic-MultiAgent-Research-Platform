from typing import Any, Dict

# TODO: placeholder pricing
MODEL_PRICING = {
    # model: (usd_per_1k_input, usd_per_1k_output)
    "llama-3.1-70b-versatile": (0.0, 0.0),  # put real values for accurate $$
}

def extract_token_usage(msg: Any) -> tuple[int, int]:
    """
    Best-effort extraction across adapters.
    """
    meta = getattr(msg, "response_metadata", None) or {}
    usage = (
        meta.get("token_usage")
        or meta.get("usage")
        or meta.get("usage_metadata")
        or {}
    )

    prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    return prompt, completion


def add_llm_usage(
        metrics: Dict[str, Any], 
        model_name: str, 
        msg: Any
    ) -> None:
    """
    Updates metrics in-place:
    - prompt tokens
    - completion tokens
    - total tokens
    - estimated cost
    """
    if metrics is None:
        return
    
    prompt_tokens, completion_tokens = extract_token_usage(msg)
    if not prompt_tokens and not completion_tokens:
        return
    
    metrics["llm_prompt_tokens"] = int(metrics.get("llm_prompt_tokens", 0)) + prompt_tokens
    metrics["llm_completion_tokens"] = int(metrics.get("llm_completion_tokens", 0)) + completion_tokens
    metrics["llm_total_tokens"] = int(metrics.get("llm_total_tokens", 0)) + prompt_tokens + completion_tokens

    in_price, out_price = MODEL_PRICING.get(model_name, (0.0, 0.0))
    cost = (prompt_tokens / 1000.0) * in_price + (completion_tokens / 1000.0) * out_price
    metrics["cost_usd"] = float(metrics.get("cost_usd", 0.0)) + float(cost)