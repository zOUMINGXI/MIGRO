"""OpenAI-compatible HTTP adapter; credentials come from environment variables."""
import json, os, time, urllib.request

def connection(args) -> dict:
    return {"base_url": args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "api_key": os.environ[args.api_key_env or "OPENAI_API_KEY"]}

def request_body(provider, prompt, model, temperature, max_tokens, reasoning_effort=None) -> dict:
    value = {"model": model, "messages": [{"role": "user", "content": prompt}],
             "max_completion_tokens": max_tokens}
    if temperature is not None:
        value["temperature"] = temperature
    if reasoning_effort is not None:
        value["reasoning_effort"] = reasoning_effort
    return value

def post_json(url, headers, body, timeout, retries):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                    headers={**headers, "Content-Type": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 8))
